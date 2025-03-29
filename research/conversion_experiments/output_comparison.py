"""
Compare output distributions between original and BSBR-converted models.
"""
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import entropy
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import json

# Make sure bsbr_transformers is importable
try:
    from bsbr_transformers.gpt2_converter import convert_to_bsbr
except ImportError:
    print("Error: bsbr_transformers package not found. Make sure it's installed.")
    import sys
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Compare outputs between original and BSBR models")
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
        "--num_samples", 
        type=int, 
        default=20, 
        help="Number of samples to test (default: 20)"
    )
    parser.add_argument(
        "--seq_len", 
        type=int, 
        default=512, 
        help="Sequence length for tests (default: 512)"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="research/conversion_experiments/results", 
        help="Directory to save results (default: research/conversion_experiments/results)"
    )
    parser.add_argument(
        "--plot_attention",
        action="store_true",
        help="Attempt to plot attention patterns (requires functional attention extraction in BSBR model, default: False)"
    )
    
    return parser.parse_args()


def compute_similarity_metrics(
    original_outputs: torch.Tensor, 
    bsbr_outputs: torch.Tensor
) -> Dict[str, float]:
    """Compute various similarity metrics between two output tensors."""
    # ##################### BEGIN TENSOR EXTRACTION #####################
    try:
        # Ensure tensors are on CPU and in numpy format
        orig = original_outputs.detach().cpu().numpy()
        bsbr = bsbr_outputs.detach().cpu().numpy()
    except AttributeError:
        # Handle case where inputs are not tensors but model outputs
        print("Extracting hidden states from model outputs...")
        
        # Try to extract hidden states from model outputs - different models might use different attributes
        if hasattr(original_outputs, 'hidden_states'):
            orig = original_outputs.hidden_states[-1].detach().cpu().numpy()  # Get last layer
        elif hasattr(original_outputs, 'last_hidden_state'):
            orig = original_outputs.last_hidden_state.detach().cpu().numpy()
        elif hasattr(original_outputs, 'logits'):
            # For language models, we can use logits as a proxy for hidden states
            orig = original_outputs.logits.detach().cpu().numpy()
        else:
            raise ValueError(f"Could not extract tensor from original_outputs. Available attributes: {dir(original_outputs)}")
            
        if hasattr(bsbr_outputs, 'hidden_states'):
            bsbr = bsbr_outputs.hidden_states[-1].detach().cpu().numpy()  # Get last layer
        elif hasattr(bsbr_outputs, 'last_hidden_state'):
            bsbr = bsbr_outputs.last_hidden_state.detach().cpu().numpy()
        elif hasattr(bsbr_outputs, 'logits'):
            bsbr = bsbr_outputs.logits.detach().cpu().numpy()
        else:
            raise ValueError(f"Could not extract tensor from bsbr_outputs. Available attributes: {dir(bsbr_outputs)}")
    # ##################### END TENSOR EXTRACTION #####################
    
    # ##################### BEGIN SIMILARITY CALCULATIONS #####################
    # Cosine similarity
    batch_size, seq_len, hidden_dim = orig.shape
    orig_flat = orig.reshape(batch_size * seq_len, hidden_dim)
    bsbr_flat = bsbr.reshape(batch_size * seq_len, hidden_dim)
    
    # Reshape to handle potential singleton dimensions
    if len(orig_flat.shape) == 1:
        orig_flat = orig_flat.reshape(1, -1)
    if len(bsbr_flat.shape) == 1:
        bsbr_flat = bsbr_flat.reshape(1, -1)
    
    cos_sim = np.mean([
        cosine_similarity([orig_flat[i]], [bsbr_flat[i]])[0][0]
        for i in range(len(orig_flat))
    ])
    
    # Mean squared error
    mse = np.mean((orig - bsbr) ** 2)
    
    # Normalize for visualizing probability distributions
    orig_norm = F.softmax(torch.tensor(orig), dim=-1).numpy()
    bsbr_norm = F.softmax(torch.tensor(bsbr), dim=-1).numpy()
    
    # KL divergence (average over batch and sequence)
    kl_divs = []
    for i in range(batch_size):
        for j in range(seq_len):
            # Add small epsilon to avoid division by zero
            p = orig_norm[i, j] + 1e-10
            q = bsbr_norm[i, j] + 1e-10
            # Normalize to ensure they sum to 1
            p = p / p.sum()
            q = q / q.sum()
            kl_divs.append(entropy(p, q))
    
    kl_div = np.mean(kl_divs)
    # ##################### END SIMILARITY CALCULATIONS #####################
    
    return {
        "cosine_similarity": float(cos_sim),
        "mse": float(mse),
        "kl_divergence": float(kl_div)
    }


def plot_attention_patterns(
    original_model: torch.nn.Module,
    bsbr_model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    layer_idx: int,
    head_idx: int,
    output_path: str
):
    """Plot attention patterns from both models for comparison."""
    # Extract attention matrices
    # Note: This assumes we can access attention weights, may need model-specific hooks
    with torch.no_grad():
        # For original model
        orig_attn = None
        orig_outputs = original_model(
            input_ids, 
            attention_mask=attention_mask, 
            output_attentions=True
        )
        if hasattr(orig_outputs, 'attentions') and orig_outputs.attentions is not None:
            orig_attn = orig_outputs.attentions[layer_idx][0, head_idx].cpu().numpy()
        
        # BSBR model doesn't directly output attention weights, so we need to instrument it
        # This is simplified and would need to be adapted for actual implementation
        bsbr_attn = extract_bsbr_attention(bsbr_model, input_ids, attention_mask, layer_idx, head_idx)
    
    if orig_attn is not None and bsbr_attn is not None:
        # Plot both attention matrices
        fig, axs = plt.subplots(1, 2, figsize=(16, 7))
        
        im0 = axs[0].imshow(orig_attn, cmap='viridis')
        axs[0].set_title('Original Transformer Attention')
        axs[0].set_xlabel('Key Position')
        axs[0].set_ylabel('Query Position')
        fig.colorbar(im0, ax=axs[0])
        
        im1 = axs[1].imshow(bsbr_attn, cmap='viridis')
        axs[1].set_title('BSBR Transformer Attention')
        axs[1].set_xlabel('Key Position')
        axs[1].set_ylabel('Query Position')
        fig.colorbar(im1, ax=axs[1])
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()


def extract_bsbr_attention(
    bsbr_model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    layer_idx: int,
    head_idx: int
) -> np.ndarray:
    """
    Extract attention patterns from a BSBR model.
    
    Note: This is a placeholder. The actual implementation would depend on
    how to access attention weights within the BSBR model, which may require
    adding hooks or instrumentation to the model.
    """
    # Placeholder implementation that creates a simulated BSBR attention pattern
    seq_len = input_ids.shape[1]
    
    # Get chunk size - handle different model wrapper structures
    if hasattr(bsbr_model, 'bsbr_model'):
        # Access through wrapper
        chunk_size = 128  # Default fallback
        if hasattr(bsbr_model.bsbr_model, 'chunk_size'):
            chunk_size = bsbr_model.bsbr_model.chunk_size
        elif hasattr(bsbr_model.bsbr_model, 'layers') and hasattr(bsbr_model.bsbr_model.layers[0], 'attention'):
            chunk_size = bsbr_model.bsbr_model.layers[0].attention.chunk_size
    elif hasattr(bsbr_model, 'layers'):
        # Direct access
        chunk_size = bsbr_model.layers[0].attention.chunk_size
    else:
        # Fallback to default
        chunk_size = 128
        
    num_chunks = -(-seq_len // chunk_size)  # Ceiling division
    
    # Create a simulated attention pattern with block structure
    attn = np.zeros((seq_len, seq_len))
    
    # Set within-chunk attention (block diagonal)
    for i in range(num_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, seq_len)
        chunk_len = end - start
        
        # Create a causal mask within the chunk
        for j in range(chunk_len):
            for k in range(j + 1):
                attn[start + j, start + k] = 0.8 * np.random.random() + 0.2
    
    # Normalize rows
    for i in range(seq_len):
        if np.sum(attn[i]) > 0:
            attn[i] = attn[i] / np.sum(attn[i])
    
    return attn


def compare_top_k_predictions(
    original_model: torch.nn.Module,
    bsbr_model: torch.nn.Module,
    tokenizer,
    texts: List[str],
    k_values: List[int] = [1, 5, 10]
) -> Dict[int, float]:
    """
    Compute agreement rates for top-k predictions between models.
    """
    agreement_counts = {k: 0 for k in k_values}
    total_positions = 0
    
    for text in tqdm(texts, desc="Processing texts"):
        # Tokenize text
        inputs = tokenizer(text, return_tensors="pt")
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        # Get logits from both models
        with torch.no_grad():
            # Original model
            orig_outputs = original_model(input_ids, attention_mask=attention_mask)
            if hasattr(orig_outputs, 'logits'):
                orig_logits = orig_outputs.logits
            else:
                # Apply language modeling head if needed
                lm_head = original_model.get_output_embeddings()
                orig_logits = lm_head(orig_outputs.last_hidden_state)
            
            # BSBR model
            bsbr_outputs = bsbr_model(input_ids, attention_mask=attention_mask)
            if hasattr(bsbr_outputs, 'logits'):
                bsbr_logits = bsbr_outputs.logits
            else:
                # Apply language modeling head if needed
                lm_head = original_model.get_output_embeddings()
                if hasattr(bsbr_outputs, 'last_hidden_state'):
                    bsbr_logits = lm_head(bsbr_outputs.last_hidden_state)
                else:
                    bsbr_logits = lm_head(bsbr_outputs)
        
        # For each position (except the last one), compare top-k predictions
        seq_len = input_ids.shape[1]
        for pos in range(seq_len - 1):
            # Get top-k indices for both models at this position
            orig_topk = torch.topk(orig_logits[0, pos], max(k_values)).indices
            bsbr_topk = torch.topk(bsbr_logits[0, pos], max(k_values)).indices
            
            # Check agreement for each k
            for k in k_values:
                orig_set = set(orig_topk[:k].tolist())
                bsbr_set = set(bsbr_topk[:k].tolist())
                if len(orig_set.intersection(bsbr_set)) > 0:
                    agreement_counts[k] += 1
            
            total_positions += 1
    
    # Calculate agreement rates
    agreement_rates = {k: count / total_positions for k, count in agreement_counts.items()}
    return agreement_rates


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load models
    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    original_model = GPT2LMHeadModel.from_pretrained(args.model_name).to(device)
    
    print(f"Converting to BSBR (chunk_size={args.chunk_size})")
    bsbr_base_model = convert_to_bsbr(
        args.model_name,
        chunk_size=args.chunk_size
    ).to(device)
    
    # Create wrapper for BSBR model
    class BSBRModelWrapper(torch.nn.Module):
        def __init__(self, bsbr_model, original_lm_head):
            super().__init__()
            self.bsbr_model = bsbr_model
            self.lm_head = original_lm_head
            
        def forward(self, input_ids, attention_mask=None, output_hidden_states=False, **kwargs):
            # Get hidden states from BSBR model
            hidden_states = self.bsbr_model(input_ids, attention_mask)
            
            # Apply language modeling head to get logits
            lm_logits = self.lm_head(hidden_states)
            
            # Create a custom output object with both logits and hidden_states
            output = type('BSBROutput', (object,), {
                'logits': lm_logits,
                'hidden_states': [hidden_states],  # Wrapped in list to mimic transformer models
                'last_hidden_state': hidden_states
            })
            
            return output
    
    bsbr_model = BSBRModelWrapper(bsbr_base_model, original_model.get_output_embeddings()).to(device)
    
    # Generate random input sequences
    print(f"Generating {args.num_samples} random input sequences")
    all_metrics = []
    
    for i in tqdm(range(args.num_samples), desc="Processing samples"):
        # Create a random input of specified length
        input_ids = torch.randint(0, tokenizer.vocab_size, (1, args.seq_len), device=device)
        attention_mask = torch.ones(1, args.seq_len, device=device)
        
        # ##################### BEGIN MODEL OUTPUT EXTRACTION #####################
        # Get outputs from both models
        with torch.no_grad():
            # Get outputs from original model with hidden states
            original_outputs = original_model(
                input_ids, 
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            
            # Get outputs from BSBR model
            bsbr_outputs = bsbr_model(input_ids, attention_mask=attention_mask)
        
        # Compute similarity metrics directly from model outputs
        # The compute_similarity_metrics function will extract hidden states as needed
        metrics = compute_similarity_metrics(original_outputs, bsbr_outputs)
        all_metrics.append(metrics)
        # ##################### END MODEL OUTPUT EXTRACTION #####################
        
        # Generate attention visualizations (conditionally)
        if args.plot_attention and i < 5:  # Only plot if requested and for first few samples
            print(f"Plotting attention for sample {i}... (Note: BSBR attention is a placeholder)")
            for layer_idx in [0, len(original_model.transformer.h) // 2, len(original_model.transformer.h) - 1]:
                for head_idx in [0, 1]:
                    plot_attention_patterns(
                        original_model,
                        bsbr_model,
                        input_ids,
                        attention_mask,
                        layer_idx,
                        head_idx,
                        os.path.join(args.output_dir, f"attn_sample{i}_layer{layer_idx}_head{head_idx}.png")
                    )
    
    # Calculate average metrics
    avg_metrics = {
        metric: np.mean([m[metric] for m in all_metrics]) 
        for metric in all_metrics[0].keys()
    }
    std_metrics = {
        metric: np.std([m[metric] for m in all_metrics]) 
        for metric in all_metrics[0].keys()
    }
    
    print("\nAverage Similarity Metrics:")
    for metric, value in avg_metrics.items():
        print(f"{metric}: {value:.4f} ± {std_metrics[metric]:.4f}")
    
    # Save metrics to file
    similarity_metrics_path = os.path.join(args.output_dir, "similarity_metrics.json")
    try:
        with open(similarity_metrics_path, "w") as f:
            json.dump({
                "average": avg_metrics,
                "std_dev": std_metrics,
                "all_samples": all_metrics
            }, f, indent=2)
        print(f"Saved similarity metrics to {similarity_metrics_path}")
    except Exception as e:
        print(f"Error saving similarity metrics to {similarity_metrics_path}: {e}")
    
    # Plot distribution of metrics
    plt.figure(figsize=(15, 5))
    
    for i, metric in enumerate(avg_metrics.keys()):
        plt.subplot(1, 3, i+1)
        values = [m[metric] for m in all_metrics]
        sns.histplot(values, kde=True)
        plt.axvline(avg_metrics[metric], color='r', linestyle='--')
        plt.title(f"{metric}\nMean: {avg_metrics[metric]:.4f}")
    
    plt.tight_layout()
    metrics_dist_path = os.path.join(args.output_dir, "metrics_distribution.png")
    plt.savefig(metrics_dist_path)
    print(f"Saved metrics distribution plot to {metrics_dist_path}")
    plt.close()
    
    # Compare next token predictions
    print("\nComparing next token predictions...")
    
    # Load or create sample texts
    sample_texts = []
    
    # You could load texts from a dataset, but for simplicity we'll use the tokenizer
    # to create some random-ish text by sampling from the model
    for _ in tqdm(range(min(args.num_samples, 10)), desc="Generating sample texts"):
        prompt = tokenizer.decode(
            tokenizer.encode("The quick brown fox jumps over the lazy dog.")[:10]
        )
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        output = original_model.generate(
            inputs.input_ids, 
            max_length=min(100, args.seq_len), 
            do_sample=True,
            temperature=0.7
        )
        
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        sample_texts.append(text)
    
    # Compare top-k predictions
    agreement_rates = compare_top_k_predictions(
        original_model,
        bsbr_model,
        tokenizer,
        sample_texts,
        k_values=[1, 5, 10]
    )
    
    print("\nTop-K Agreement Rates:")
    for k, rate in agreement_rates.items():
        print(f"Top-{k}: {rate*100:.2f}%")
    
    # Save agreement rates
    agreement_rates_path = os.path.join(args.output_dir, "agreement_rates.json")
    try:
        with open(agreement_rates_path, "w") as f:
            json.dump(agreement_rates, f, indent=2)
        print(f"Saved agreement rates to {agreement_rates_path}")
    except Exception as e:
        print(f"Error saving agreement rates to {agreement_rates_path}: {e}")
    
    # Plot agreement rates
    plt.figure(figsize=(8, 5))
    ks = list(agreement_rates.keys())
    rates = [agreement_rates[k] * 100 for k in ks]
    
    plt.bar(range(len(ks)), rates)
    plt.xticks(range(len(ks)), [f"Top-{k}" for k in ks])
    plt.ylabel("Agreement Rate (%)")
    plt.title("Agreement Rate Between Original and BSBR Models")
    
    plt.tight_layout()
    agreement_plot_path = os.path.join(args.output_dir, "agreement_rates.png")
    plt.savefig(agreement_plot_path)
    print(f"Saved agreement rates plot to {agreement_plot_path}")
    plt.close()
    
    print(f"\nAll results saved to {args.output_dir}")


if __name__ == "__main__":
    main() 