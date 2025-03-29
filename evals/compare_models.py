import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import argparse
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.bsbr import BSBRModel
from src.bsbr_extras import (
    LinearTransformerModel, 
    DeltaNetModel, 
    StandardTransformerModel, 
    SlidingWindowTransformerModel,
    HopfieldNetworkModel,
    GAUModel
)


class AutoregressiveEvaluator:
    """
    Class to evaluate different models on an autoregressive text generation task.
    
    Args:
        vocab_size (int): Vocabulary size for the models
        hidden_dim (int): Hidden dimension size
        num_layers (int): Number of layers
        num_heads (int): Number of attention heads
        ff_dim (int): Feed-forward dimension
        chunk_size (int): Chunk size for BSBR model and GAU
        window_size (int): Window size for SlidingWindowTransformer
        beta (float): Beta parameter for DeltaNet model
        temperature (float): Temperature parameter for HopfieldNetwork
        expansion_factor (int): Expansion factor for GAU
        dropout (float): Dropout probability
        device (str): Device to use for computation
        model_selection (list): List of model names to evaluate
    """
    def __init__(
        self,
        vocab_size: int = 10000,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        ff_dim: int = 512,
        chunk_size: int = 32,
        window_size: int = 32,
        beta: float = 0.9,
        temperature: float = 1.0,
        expansion_factor: int = 2,
        dropout: float = 0.1,
        device: str = 'cpu',
        model_selection: List[str] = None
    ):
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.chunk_size = chunk_size
        self.window_size = window_size
        self.beta = beta
        self.temperature = temperature
        self.expansion_factor = expansion_factor
        self.dropout = dropout
        self.device = torch.device(device)
        self.model_selection = model_selection or ["BSBR", "Linear", "DeltaNet", "Standard", "SlidingWindow", "Hopfield", "GAU"]
        
        # Create models
        self.models = self._create_models()
        
        # Create a simple language model head
        self.lm_head = nn.Linear(hidden_dim, vocab_size).to(self.device)
        
    def _create_models(self) -> Dict[str, nn.Module]:
        """Create the different model variants."""
        available_models = {
            "BSBR": BSBRModel(
                vocab_size=self.vocab_size,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                num_heads=self.num_heads,
                chunk_size=self.chunk_size,
                ff_dim=self.ff_dim,
                dropout=self.dropout,
                compression_factor=2  # Compression for BSBR
            ).to(self.device),
            
            "Linear": LinearTransformerModel(
                vocab_size=self.vocab_size,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                num_heads=self.num_heads,
                ff_dim=self.ff_dim,
                dropout=self.dropout
            ).to(self.device),
            
            "DeltaNet": DeltaNetModel(
                vocab_size=self.vocab_size,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                num_heads=self.num_heads,
                ff_dim=self.ff_dim,
                beta=self.beta,
                dropout=self.dropout
            ).to(self.device),
            
            "Standard": StandardTransformerModel(
                vocab_size=self.vocab_size,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                num_heads=self.num_heads,
                ff_dim=self.ff_dim,
                dropout=self.dropout
            ).to(self.device),
            
            "SlidingWindow": SlidingWindowTransformerModel(
                vocab_size=self.vocab_size,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                num_heads=self.num_heads,
                window_size=self.window_size,
                ff_dim=self.ff_dim,
                dropout=self.dropout
            ).to(self.device),
            
            "Hopfield": HopfieldNetworkModel(
                vocab_size=self.vocab_size,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                num_heads=self.num_heads,
                ff_dim=self.ff_dim,
                temperature=self.temperature,
                dropout=self.dropout
            ).to(self.device),
            
            "GAU": GAUModel(
                vocab_size=self.vocab_size,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                chunk_size=self.chunk_size,
                ff_dim=self.ff_dim,
                expansion_factor=self.expansion_factor,
                dropout=self.dropout
            ).to(self.device)
        }
        
        # Filter models based on selection
        return {name: model for name, model in available_models.items() if name in self.model_selection}
    
    def count_parameters(self) -> Dict[str, int]:
        """Count the number of parameters in each model."""
        return {
            name: sum(p.numel() for p in model.parameters() if p.requires_grad)
            for name, model in self.models.items()
        }
    
    def _generate_dummy_data(self, seq_len: int, batch_size: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate dummy input data."""
        input_ids = torch.randint(0, self.vocab_size, (batch_size, seq_len), device=self.device)
        attention_mask = torch.ones(batch_size, seq_len, device=self.device)
        return input_ids, attention_mask
    
    def evaluate_inference_time(
        self, 
        seq_lengths: List[int], 
        n_tokens: int = 100,
        batch_size: int = 1
    ) -> Dict[str, Dict[int, float]]:
        """
        Evaluate inference time for generating n_tokens with different sequence lengths.
        
        Args:
            seq_lengths: List of sequence lengths to test
            n_tokens: Number of tokens to generate autoregressively
            batch_size: Batch size
            
        Returns:
            Dictionary mapping model names to dictionaries of sequence length -> time
        """
        results = {name: {} for name in self.models.keys()}
        
        for seq_len in seq_lengths:
            print(f"Testing with sequence length: {seq_len}")
            
            for name, model in self.models.items():
                model.eval()
                
                # Generate initial sequence
                input_ids, attention_mask = self._generate_dummy_data(seq_len, batch_size)
                
                # For recurrent models, initialize states
                states = None
                if name in ["Linear", "DeltaNet", "Hopfield"]:
                    # Warm-up to initialize states
                    with torch.no_grad():
                        _, states = model(input_ids, attention_mask)
                
                # Time the generation of n_tokens
                start_time = time.time()
                
                with torch.no_grad():
                    for _ in range(n_tokens):
                        if name in ["BSBR", "Standard", "SlidingWindow", "GAU"]:
                            # These models don't use states
                            hidden_states = model(input_ids, attention_mask)
                        else:
                            # Models with stateful layers
                            hidden_states, states = model(input_ids, attention_mask, states)
                        
                        # Get next token prediction
                        logits = self.lm_head(hidden_states[:, -1])
                        next_token = torch.argmax(logits, dim=-1).unsqueeze(-1)
                        
                        # Append new token to input_ids and update attention_mask
                        input_ids = torch.cat([input_ids, next_token], dim=1)
                        attention_mask = torch.cat([
                            attention_mask, 
                            torch.ones(batch_size, 1, device=self.device)
                        ], dim=1)
                
                end_time = time.time()
                generation_time = end_time - start_time
                
                results[name][seq_len] = generation_time
                print(f"  {name}: {generation_time:.4f} seconds")
        
        return results
    
    def evaluate_memory_usage(
        self, 
        seq_lengths: List[int], 
        batch_size: int = 1
    ) -> Dict[str, Dict[int, float]]:
        """
        Evaluate memory usage for forward pass with different sequence lengths.
        
        Args:
            seq_lengths: List of sequence lengths to test
            batch_size: Batch size
            
        Returns:
            Dictionary mapping model names to dictionaries of sequence length -> memory usage
        """
        results = {name: {} for name in self.models.keys()}
        
        for seq_len in seq_lengths:
            print(f"Testing with sequence length: {seq_len}")
            
            for name, model in self.models.items():
                model.eval()
                
                # Clear GPU memory cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                
                # Generate sequence
                input_ids, attention_mask = self._generate_dummy_data(seq_len, batch_size)
                
                # Forward pass
                with torch.no_grad():
                    if name in ["Linear", "DeltaNet", "Hopfield"]:
                        model(input_ids, attention_mask)
                    else:
                        model(input_ids, attention_mask)
                
                # Measure memory usage
                if torch.cuda.is_available():
                    memory_usage = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
                else:
                    # Rough estimate on CPU (not accurate)
                    memory_usage = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)
                    memory_usage += (input_ids.numel() * input_ids.element_size()) / (1024 ** 2)
                
                results[name][seq_len] = memory_usage
                print(f"  {name}: {memory_usage:.2f} MB")
        
        return results
    
    def plot_results(
        self, 
        time_results: Dict[str, Dict[int, float]], 
        memory_results: Dict[str, Dict[int, float]],
        param_counts: Dict[str, int],
        output_path: str = None
    ):
        """Plot the evaluation results."""
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Set colors for each model
        colors = {
            "BSBR": "blue",
            "Linear": "green", 
            "DeltaNet": "red",
            "Standard": "purple",
            "SlidingWindow": "orange",
            "Hopfield": "brown",
            "GAU": "pink"
        }
        
        # Plot inference time
        for name, results in time_results.items():
            seq_lengths = list(results.keys())
            times = list(results.values())
            ax1.plot(seq_lengths, times, marker='o', label=f"{name} ({param_counts[name]/1e6:.1f}M params)", color=colors[name])
        
        ax1.set_xlabel("Sequence Length")
        ax1.set_ylabel("Inference Time (seconds)")
        ax1.set_title("Inference Time vs Sequence Length")
        ax1.legend()
        ax1.grid(True)
        
        # Plot memory usage
        for name, results in memory_results.items():
            seq_lengths = list(results.keys())
            memory = list(results.values())
            ax2.plot(seq_lengths, memory, marker='o', label=f"{name} ({param_counts[name]/1e6:.1f}M params)", color=colors[name])
        
        ax2.set_xlabel("Sequence Length")
        ax2.set_ylabel("Memory Usage (MB)")
        ax2.set_title("Memory Usage vs Sequence Length")
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path)
        
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Compare different transformer architectures")
    parser.add_argument("--vocab_size", type=int, default=10000, help="Vocabulary size")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension size")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of layers")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--ff_dim", type=int, default=512, help="Feed-forward dimension")
    parser.add_argument("--chunk_size", type=int, default=32, help="Chunk size for BSBR and GAU")
    parser.add_argument("--window_size", type=int, default=32, help="Window size for SlidingWindowTransformer")
    parser.add_argument("--beta", type=float, default=0.9, help="Beta parameter for DeltaNet")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature parameter for Hopfield Network")
    parser.add_argument("--expansion_factor", type=int, default=2, help="Expansion factor for GAU")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to use (cuda or cpu)")
    parser.add_argument("--output", type=str, default="model_comparison.png", help="Output file for plots")
    parser.add_argument("--seq_lengths", type=int, nargs="+", default=[64, 128, 256, 512, 1024], 
                        help="Sequence lengths to test")
    parser.add_argument("--n_tokens", type=int, default=50, 
                        help="Number of tokens to generate for inference time test")
    parser.add_argument("--models", type=str, nargs="+", 
                        default=["BSBR", "Linear", "DeltaNet", "Standard", "SlidingWindow", "Hopfield", "GAU"],
                        help="Models to evaluate: BSBR, Linear, DeltaNet, Standard, SlidingWindow, Hopfield, GAU")
    
    args = parser.parse_args()
    
    evaluator = AutoregressiveEvaluator(
        vocab_size=args.vocab_size,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        chunk_size=args.chunk_size,
        window_size=args.window_size,
        beta=args.beta,
        temperature=args.temperature,
        expansion_factor=args.expansion_factor,
        dropout=args.dropout,
        device=args.device,
        model_selection=args.models
    )
    
    # Count parameters
    param_counts = evaluator.count_parameters()
    print("Parameter counts:")
    for name, count in param_counts.items():
        print(f"  {name}: {count:,} parameters")
    
    # Evaluate inference time
    print("\nEvaluating inference time...")
    time_results = evaluator.evaluate_inference_time(
        seq_lengths=args.seq_lengths,
        n_tokens=args.n_tokens
    )
    
    # Evaluate memory usage
    print("\nEvaluating memory usage...")
    memory_results = evaluator.evaluate_memory_usage(
        seq_lengths=args.seq_lengths
    )
    
    # Plot results
    evaluator.plot_results(
        time_results=time_results,
        memory_results=memory_results,
        param_counts=param_counts,
        output_path=args.output
    )


if __name__ == "__main__":
    main() 