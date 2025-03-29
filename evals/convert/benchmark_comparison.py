"""
Benchmark comparison between original transformers and BSBR-converted models.

This script conducts comprehensive benchmarks to compare:
1. Inference speed & memory usage at different sequence lengths
2. Model quality metrics (perplexity, prediction accuracy)
3. Scaling behavior analysis
"""
import argparse
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer
from datasets import load_dataset

from bsbr_transformers.gpt2_converter import convert_to_bsbr

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12
})


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark comparison between original and BSBR models")
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="gpt2", 
        help="HuggingFace model name (default: gpt2)"
    )
    parser.add_argument(
        "--chunk_size", 
        type=int, 
        default=128, 
        help="Chunk size for BSBR (default: 128)"
    )
    parser.add_argument(
        "--seq_lengths", 
        type=str, 
        default="128,256,512,1024,2048,4096",
        help="Comma-separated sequence lengths to test (default: 128,256,512,1024,2048,4096)"
    )
    parser.add_argument(
        "--batch_sizes", 
        type=str, 
        default="16,8,4,2,1,1",
        help="Comma-separated batch sizes for each sequence length (default: 16,8,4,2,1,1)"
    )
    parser.add_argument(
        "--num_repeats", 
        type=int, 
        default=5, 
        help="Number of measurement repeats for each configuration (default: 5)"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./benchmark_results", 
        help="Directory to save results (default: ./benchmark_results)"
    )
    parser.add_argument(
        "--quality_eval", 
        action="store_true",
        help="Whether to evaluate model quality metrics (default: False)"
    )
    parser.add_argument(
        "--eval_dataset", 
        type=str, 
        default="wikitext",
        help="Dataset to use for quality evaluation (default: wikitext)"
    )
    parser.add_argument(
        "--profile_memory", 
        action="store_true",
        help="Profile peak memory usage (requires pytorch_memlab, default: False)"
    )
    
    return parser.parse_args()


def create_model_wrapper(bsbr_model, original_lm_head):
    """Create wrapper for BSBR model to make it compatible with language modeling head."""
    class BSBRModelWrapper(torch.nn.Module):
        def __init__(self, bsbr_model, original_lm_head):
            super().__init__()
            self.bsbr_model = bsbr_model
            self.lm_head = original_lm_head
            
        def forward(self, input_ids, attention_mask=None, **kwargs):
            hidden_states = self.bsbr_model(input_ids, attention_mask)
            lm_logits = self.lm_head(hidden_states)
            return type('obj', (object,), {'logits': lm_logits, 'last_hidden_state': hidden_states})
    
    return BSBRModelWrapper(bsbr_model, original_lm_head)


def benchmark_inference_speed(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    num_repeats: int = 5,
    warmup: int = 3,
    profile_memory: bool = False,
    device: str = "cpu"
) -> Dict[str, float]:
    """Benchmark inference speed and memory usage."""
    # Optional memory profiling
    if profile_memory and device == "cuda":
        try:
            from pytorch_memlab import MemReporter
            reporter = MemReporter()
        except ImportError:
            print("pytorch_memlab not installed, skipping memory profiling")
            profile_memory = False
    
    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(input_ids, attention_mask=attention_mask)
    
    # Actual timing
    if device == "cuda":
        torch.cuda.synchronize()
    timings = []
    peak_memory = 0
    
    for _ in range(num_repeats):
        if profile_memory and device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
        
        start_time = time.time()
        with torch.no_grad():
            _ = model(input_ids, attention_mask=attention_mask)
        
        if device == "cuda":
            torch.cuda.synchronize()
        end_time = time.time()
        timings.append(end_time - start_time)
        
        if profile_memory and device == "cuda":
            current_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
            peak_memory = max(peak_memory, current_memory)
    
    results = {
        "mean_time": np.mean(timings),
        "std_time": np.std(timings),
        "median_time": np.median(timings),
        "min_time": np.min(timings),
        "max_time": np.max(timings),
    }
    
    if profile_memory and device == "cuda":
        results["peak_memory_mb"] = peak_memory
    
    return results


def calculate_perplexity(
    model: torch.nn.Module,
    tokenizer,
    dataset,
    max_samples: int = 50,
    device: str = "cuda"
) -> float:
    """Calculate perplexity on a text dataset."""
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for i, example in enumerate(tqdm(dataset, desc="Calculating perplexity")):
            if i >= max_samples:
                break
            
            text = example["text"] if "text" in example else example
            inputs = tokenizer(text, return_tensors="pt").to(device)
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            
            # Handle potential long sequences by chunking
            max_length = model.config.max_position_embeddings if hasattr(model, 'config') else 1024
            for j in range(0, input_ids.shape[1], max_length):
                chunk_input_ids = input_ids[:, j:j+max_length]
                chunk_attention_mask = attention_mask[:, j:j+max_length]
                
                if chunk_input_ids.shape[1] < 2:  # Need at least 2 tokens for loss
                    continue
                
                outputs = model(chunk_input_ids, attention_mask=chunk_attention_mask)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                
                # Shift for language modeling loss
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = chunk_input_ids[:, 1:].contiguous()
                
                loss_fct = torch.nn.CrossEntropyLoss(reduction='sum')
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
                
                total_loss += loss.item()
                total_tokens += shift_labels.numel()
    
    # Perplexity = exp(average negative log-likelihood)
    return torch.exp(torch.tensor(total_loss / total_tokens)).item()


def run_scaling_analysis(
    model_name: str,
    chunk_size: int,
    seq_lengths: List[int],
    batch_sizes: List[int],
    num_repeats: int = 5,
    output_dir: str = "./benchmark_results",
    profile_memory: bool = False,
):
    """Run scaling analysis with different sequence lengths."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load models
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    original_model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    
    print(f"Converting to BSBR (chunk_size={chunk_size})")
    bsbr_base_model = convert_to_bsbr(
        model_name,
        chunk_size=chunk_size
    ).to(device)
    
    bsbr_model = create_model_wrapper(bsbr_base_model, original_model.get_output_embeddings()).to(device)
    
    # Run benchmarks for each sequence length
    results = {
        "config": {
            "model_name": model_name,
            "chunk_size": chunk_size,
            "device": device,
            "num_repeats": num_repeats,
        },
        "original": [],
        "bsbr": []
    }
    
    for seq_len, batch_size in zip(seq_lengths, batch_sizes):
        print(f"\nBenchmarking sequence length {seq_len} with batch size {batch_size}")
        
        # Create inputs
        input_ids = torch.randint(0, tokenizer.vocab_size, (batch_size, seq_len), device=device)
        attention_mask = torch.ones(batch_size, seq_len, device=device)
        
        # Benchmark original model
        print("Benchmarking original model...")
        original_results = benchmark_inference_speed(
            original_model,
            input_ids,
            attention_mask,
            num_repeats=num_repeats,
            profile_memory=profile_memory,
            device=device
        )
        original_results.update({
            "seq_length": seq_len,
            "batch_size": batch_size,
            "tokens_per_batch": seq_len * batch_size,
        })
        results["original"].append(original_results)
        
        print(f"Mean time: {original_results['mean_time']:.4f}s")
        if profile_memory and device == "cuda":
            print(f"Peak memory: {original_results['peak_memory_mb']:.2f} MB")
        
        # Benchmark BSBR model
        print("Benchmarking BSBR model...")
        bsbr_results = benchmark_inference_speed(
            bsbr_model,
            input_ids,
            attention_mask,
            num_repeats=num_repeats,
            profile_memory=profile_memory,
            device=device
        )
        bsbr_results.update({
            "seq_length": seq_len,
            "batch_size": batch_size,
            "tokens_per_batch": seq_len * batch_size,
        })
        results["bsbr"].append(bsbr_results)
        
        print(f"Mean time: {bsbr_results['mean_time']:.4f}s")
        if profile_memory and device == "cuda":
            print(f"Peak memory: {bsbr_results['peak_memory_mb']:.2f} MB")
        
        # Calculate speedup
        speedup = original_results["mean_time"] / bsbr_results["mean_time"]
        print(f"Speedup: {speedup:.2f}x")
        
        # Save intermediate results
        with open(os.path.join(output_dir, "scaling_results.json"), "w") as f:
            json.dump(results, f, indent=2)
    
    # Plot results
    plot_scaling_results(results, output_dir)
    
    return results


def plot_scaling_results(results, output_dir):
    """Plot scaling results comparing original and BSBR models."""
    
    # Extract data for plotting
    seq_lengths = [r["seq_length"] for r in results["original"]]
    original_times = [r["mean_time"] for r in results["original"]]
    bsbr_times = [r["mean_time"] for r in results["bsbr"]]
    
    # Calculate tokens per second
    tokens_per_batch = [r["tokens_per_batch"] for r in results["original"]]
    original_tps = [tokens / time for tokens, time in zip(tokens_per_batch, original_times)]
    bsbr_tps = [tokens / time for tokens, time in zip(tokens_per_batch, bsbr_times)]
    
    # Speedup ratio
    speedups = [orig / bsbr for orig, bsbr in zip(original_times, bsbr_times)]
    
    # Memory usage if available
    has_memory = "peak_memory_mb" in results["original"][0]
    if has_memory:
        original_memory = [r["peak_memory_mb"] for r in results["original"]]
        bsbr_memory = [r["peak_memory_mb"] for r in results["bsbr"]]
        memory_savings = [(orig - bsbr) / orig * 100 for orig, bsbr in zip(original_memory, bsbr_memory)]
    
    # Create plots
    # 1. Inference time vs sequence length
    plt.figure(figsize=(10, 6))
    plt.plot(seq_lengths, original_times, 'o-', label='Original Transformer')
    plt.plot(seq_lengths, bsbr_times, 's-', label='BSBR Transformer')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Sequence Length')
    plt.ylabel('Inference Time (s)')
    plt.title('Inference Time vs Sequence Length')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "inference_time_vs_seq_length.png"))
    
    # 2. Tokens per second vs sequence length
    plt.figure(figsize=(10, 6))
    plt.plot(seq_lengths, original_tps, 'o-', label='Original Transformer')
    plt.plot(seq_lengths, bsbr_tps, 's-', label='BSBR Transformer')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Sequence Length')
    plt.ylabel('Tokens per Second')
    plt.title('Throughput vs Sequence Length')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "throughput_vs_seq_length.png"))
    
    # 3. Speedup ratio vs sequence length
    plt.figure(figsize=(10, 6))
    plt.plot(seq_lengths, speedups, 'o-', color='green')
    plt.axhline(y=1.0, color='r', linestyle='--')
    plt.xscale('log')
    plt.xlabel('Sequence Length')
    plt.ylabel('Speedup (Original / BSBR)')
    plt.title('BSBR Speedup vs Sequence Length')
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "speedup_vs_seq_length.png"))
    
    # 4. Memory usage if available
    if has_memory:
        plt.figure(figsize=(10, 6))
        plt.plot(seq_lengths, original_memory, 'o-', label='Original Transformer')
        plt.plot(seq_lengths, bsbr_memory, 's-', label='BSBR Transformer')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Sequence Length')
        plt.ylabel('Peak Memory Usage (MB)')
        plt.title('Memory Usage vs Sequence Length')
        plt.grid(True, which="both", ls="--")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "memory_usage_vs_seq_length.png"))
        
        # 5. Memory savings percentage
        plt.figure(figsize=(10, 6))
        plt.plot(seq_lengths, memory_savings, 'o-', color='purple')
        plt.xscale('log')
        plt.xlabel('Sequence Length')
        plt.ylabel('Memory Savings (%)')
        plt.title('BSBR Memory Savings vs Sequence Length')
        plt.grid(True, which="both", ls="--")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "memory_savings_vs_seq_length.png"))
    
    # 6. Fit power law to scaling curves
    # For original model
    log_seq = np.log(seq_lengths)
    log_time_orig = np.log(original_times)
    log_time_bsbr = np.log(bsbr_times)
    
    # Linear regression in log-log space to find scaling exponent
    orig_poly = np.polyfit(log_seq, log_time_orig, 1)
    bsbr_poly = np.polyfit(log_seq, log_time_bsbr, 1)
    
    orig_exponent = orig_poly[0]
    bsbr_exponent = bsbr_poly[0]
    
    plt.figure(figsize=(10, 6))
    plt.scatter(log_seq, log_time_orig, label=f'Original: O(n^{orig_exponent:.2f})')
    plt.scatter(log_seq, log_time_bsbr, label=f'BSBR: O(n^{bsbr_exponent:.2f})')
    
    # Add trend lines
    seq_range = np.linspace(min(log_seq), max(log_seq), 100)
    plt.plot(seq_range, orig_poly[0] * seq_range + orig_poly[1], '--')
    plt.plot(seq_range, bsbr_poly[0] * seq_range + bsbr_poly[1], '--')
    
    plt.xlabel('log(Sequence Length)')
    plt.ylabel('log(Inference Time)')
    plt.title('Log-Log Scaling Analysis')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "scaling_exponent_analysis.png"))
    
    # Save scaling exponents
    with open(os.path.join(output_dir, "scaling_exponents.json"), "w") as f:
        json.dump({
            "original_exponent": float(orig_exponent),
            "bsbr_exponent": float(bsbr_exponent),
            "speedup_at_max_length": float(speedups[-1]),
            "memory_savings_at_max_length": float(memory_savings[-1]) if has_memory else None
        }, f, indent=2)
    
    print(f"\nScaling Analysis Results:")
    print(f"Original Transformer scaling: O(n^{orig_exponent:.2f})")
    print(f"BSBR Transformer scaling: O(n^{bsbr_exponent:.2f})")
    print(f"Maximum speedup (at seq_len={seq_lengths[-1]}): {speedups[-1]:.2f}x")
    if has_memory:
        print(f"Maximum memory savings: {memory_savings[-1]:.2f}%")


def evaluate_model_quality(
    model_name: str,
    chunk_size: int,
    dataset_name: str = "wikitext",
    output_dir: str = "./benchmark_results",
):
    """Evaluate model quality using perplexity on standard datasets."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nEvaluating model quality on {dataset_name}...")
    
    # Load models
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    original_model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    
    bsbr_base_model = convert_to_bsbr(
        model_name,
        chunk_size=chunk_size
    ).to(device)
    
    bsbr_model = create_model_wrapper(bsbr_base_model, original_model.get_output_embeddings()).to(device)
    
    # Load evaluation dataset
    if dataset_name == "wikitext":
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    else:
        try:
            dataset = load_dataset(dataset_name, split="test")
        except Exception as e:
            print(f"Error loading dataset {dataset_name}: {e}")
            print("Falling back to wikitext-2")
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    
    # Calculate perplexity
    print("Calculating perplexity for original model...")
    original_ppl = calculate_perplexity(original_model, tokenizer, dataset, device=device)
    
    print("Calculating perplexity for BSBR model...")
    bsbr_ppl = calculate_perplexity(bsbr_model, tokenizer, dataset, device=device)
    
    # Calculate relative difference
    ppl_diff_percent = ((bsbr_ppl - original_ppl) / original_ppl) * 100
    
    quality_results = {
        "dataset": dataset_name,
        "original_perplexity": original_ppl,
        "bsbr_perplexity": bsbr_ppl,
        "perplexity_difference_percent": ppl_diff_percent
    }
    
    print(f"\nQuality Evaluation Results:")
    print(f"Original model perplexity: {original_ppl:.2f}")
    print(f"BSBR model perplexity: {bsbr_ppl:.2f}")
    print(f"Relative difference: {ppl_diff_percent:.2f}%")
    
    # Save results
    with open(os.path.join(output_dir, "quality_results.json"), "w") as f:
        json.dump(quality_results, f, indent=2)
    
    # Plot perplexity comparison
    plt.figure(figsize=(8, 6))
    plt.bar(['Original', 'BSBR'], [original_ppl, bsbr_ppl])
    plt.axhline(y=original_ppl, color='r', linestyle='--')
    plt.ylabel('Perplexity (lower is better)')
    plt.title(f'Perplexity Comparison on {dataset_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"perplexity_comparison_{dataset_name}.png"))
    
    return quality_results


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Parse sequence lengths and batch sizes
    seq_lengths = [int(x) for x in args.seq_lengths.split(',')]
    batch_sizes = [int(x) for x in args.batch_sizes.split(',')]
    
    if len(batch_sizes) != len(seq_lengths):
        # If batch sizes don't match sequence lengths, use the first batch size for all
        if len(batch_sizes) == 1:
            batch_sizes = [batch_sizes[0]] * len(seq_lengths)
        else:
            raise ValueError("Number of batch sizes must match number of sequence lengths")
    
    # Run scaling analysis
    scaling_results = run_scaling_analysis(
        args.model_name,
        args.chunk_size,
        seq_lengths,
        batch_sizes,
        args.num_repeats,
        args.output_dir,
        args.profile_memory
    )
    
    # Optionally evaluate model quality
    if args.quality_eval:
        quality_results = evaluate_model_quality(
            args.model_name,
            args.chunk_size,
            args.eval_dataset,
            args.output_dir
        )
    
    print(f"\nAll benchmark results saved to {args.output_dir}")


if __name__ == "__main__":
    main() 