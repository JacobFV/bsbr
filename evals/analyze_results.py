import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import argparse
from typing import Dict, List, Tuple


def plot_complexity_curves(seq_lengths: List[int], times: Dict[str, List[float]], title: str = "Inference Time"):
    """
    Plot complexity curves for each model.
    
    Args:
        seq_lengths: List of sequence lengths
        times: Dictionary mapping model names to lists of times
        title: Title for the plot
    """
    plt.figure(figsize=(12, 8))
    
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
    
    # Plot the actual times
    for model_name, model_times in times.items():
        plt.plot(seq_lengths, model_times, marker='o', label=f"{model_name} (Actual)", color=colors.get(model_name, "gray"))
    
    # Plot theoretical complexity curves
    x = np.array(seq_lengths)
    
    # Linear complexity O(n) - Linear Transformer theoretical
    if "Linear" in times:
        linear_scale = times["Linear"][-1] / seq_lengths[-1]
        plt.plot(x, linear_scale * x, '--', color='green', alpha=0.5, label="O(n) - Linear")
    
    # Linear * Window complexity O(n·w) - Sliding Window Transformer theoretical
    if "SlidingWindow" in times:
        sliding_scale = times["SlidingWindow"][-1] / (seq_lengths[-1])
        plt.plot(x, sliding_scale * x, '--', color='orange', alpha=0.5, label="O(n·w) - Sliding Window")
    
    # Quadratic complexity O(n²) - Standard Transformer theoretical
    if "Standard" in times:
        quadratic_scale = times["Standard"][-1] / (seq_lengths[-1] ** 2)
        plt.plot(x, quadratic_scale * x**2, '--', color='purple', alpha=0.5, label="O(n²) - Quadratic")
    
    # Hopfield complexity (similar to Linear)
    if "Hopfield" in times:
        hopfield_scale = times["Hopfield"][-1] / seq_lengths[-1]
        plt.plot(x, hopfield_scale * x, '--', color='brown', alpha=0.5, label="O(n) - Hopfield")
    
    # GAU complexity (should be efficient with chunks)
    if "GAU" in times:
        gau_scale = times["GAU"][-1] / (seq_lengths[-1] * np.log(seq_lengths[-1]))
        plt.plot(x, gau_scale * x * np.log(x), '--', color='pink', alpha=0.5, label="O(n log n) - GAU")
    
    plt.xlabel("Sequence Length")
    plt.ylabel("Time (seconds)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    
    return plt


def estimate_complexity(seq_lengths: List[int], times: List[float]) -> Tuple[str, float]:
    """
    Estimate the computational complexity of a model based on its inference times.
    
    Args:
        seq_lengths: List of sequence lengths
        times: List of corresponding inference times
        
    Returns:
        Complexity order (O(n), O(n²), etc.) and fit quality
    """
    log_seq = np.log(seq_lengths)
    log_times = np.log(times)
    
    # Linear regression on log-log scale
    coeffs = np.polyfit(log_seq, log_times, 1)
    slope = coeffs[0]
    
    # Compute R-squared to measure fit quality
    y_pred = coeffs[0] * log_seq + coeffs[1]
    ss_total = np.sum((log_times - np.mean(log_times))**2)
    ss_residual = np.sum((log_times - y_pred)**2)
    r_squared = 1 - (ss_residual / ss_total)
    
    # Determine complexity based on slope
    if slope < 0.2:
        complexity = "O(1)"
    elif 0.2 <= slope < 1.2:
        complexity = f"O(n^{slope:.2f}) ≈ O(n)"
    elif 1.2 <= slope < 1.5:
        complexity = f"O(n^{slope:.2f}) ≈ O(n log n)"
    elif 1.5 <= slope < 1.8:
        complexity = f"O(n^{slope:.2f}) ≈ O(n·w)"
    elif 1.8 <= slope < 2.2:
        complexity = f"O(n^{slope:.2f}) ≈ O(n²)"
    else:
        complexity = f"O(n^{slope:.2f})"
    
    return complexity, r_squared


def analyze_results(seq_lengths: List[int], time_results: Dict[str, List[float]], memory_results: Dict[str, List[float]]):
    """
    Analyze the performance results of different models.
    
    Args:
        seq_lengths: List of sequence lengths
        time_results: Dictionary mapping model names to lists of inference times
        memory_results: Dictionary mapping model names to lists of memory usage
    """
    print("\n===== COMPLEXITY ANALYSIS =====")
    
    results = []
    for model_name, times in time_results.items():
        complexity, r_squared = estimate_complexity(seq_lengths, times)
        results.append({
            "Model": model_name,
            "Complexity": complexity,
            "R-squared": f"{r_squared:.4f}",
            f"Time at n={seq_lengths[-1]}": f"{times[-1]:.4f}s",
            f"Memory at n={seq_lengths[-1]}": f"{memory_results[model_name][-1]:.2f} MB"
        })
    
    # Convert to pandas DataFrame for nice display
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    
    # Calculate relative performance
    print("\n===== RELATIVE PERFORMANCE =====")
    
    # Find the fastest model for the largest sequence length
    max_seq_idx = -1  # Last item is the largest sequence length
    fastest_model = min(time_results.items(), key=lambda x: x[1][max_seq_idx])[0]
    baseline_times = time_results[fastest_model]
    
    # Find model with best memory efficiency
    most_memory_efficient = min(memory_results.items(), key=lambda x: x[1][max_seq_idx])[0]
    
    print(f"Fastest model at sequence length {seq_lengths[-1]}: {fastest_model}")
    print(f"Most memory efficient model at sequence length {seq_lengths[-1]}: {most_memory_efficient}")
    print()
    
    rel_results = []
    for model_name, times in time_results.items():
        if model_name != fastest_model:
            # Calculate relative performance compared to the fastest model
            slowdowns = [model_time / baseline_time for baseline_time, model_time in zip(baseline_times, times)]
            rel_results.append({
                "Model": model_name,
                f"Avg Slowdown vs {fastest_model}": f"{np.mean(slowdowns):.2f}x",
                "Min Slowdown": f"{min(slowdowns):.2f}x",
                "Max Slowdown": f"{max(slowdowns):.2f}x",
                f"Slowdown at n={seq_lengths[-1]}": f"{times[-1] / baseline_times[-1]:.2f}x"
            })
            
    # Convert to pandas DataFrame for nice display
    rel_df = pd.DataFrame(rel_results)
    print(rel_df.to_string(index=False))
    
    # Create complexity plot
    plt = plot_complexity_curves(seq_lengths, time_results, "Inference Time vs Sequence Length")
    plt.savefig("complexity_analysis.png")
    print("\nSaved complexity plot to complexity_analysis.png")
    
    # Create log-log plot
    plt.figure(figsize=(12, 8))
    for model_name, times in time_results.items():
        plt.loglog(seq_lengths, times, marker='o', label=model_name)
    
    plt.xlabel("Sequence Length (log scale)")
    plt.ylabel("Time (seconds, log scale)")
    plt.title("Inference Time vs Sequence Length (Log-Log Scale)")
    plt.legend()
    plt.grid(True)
    plt.savefig("complexity_loglog.png")
    print("Saved log-log plot to complexity_loglog.png")


def main():
    parser = argparse.ArgumentParser(description="Analyze model comparison results")
    parser.add_argument("--seq_lengths", type=int, nargs="+", default=[64, 256, 512, 1024], 
                        help="Sequence lengths used in evaluation")
    parser.add_argument("--use_example_data", action="store_true", 
                        help="Use example/synthetic data instead of loading from files")
    args = parser.parse_args()
    
    if args.use_example_data:
        # Example results using synthetic data
        # These are example values - replace with actual results from your runs
        time_results = {
            "BSBR": [0.043, 0.058, 0.156, 0.428],
            "Linear": [0.213, 0.490, 1.096, 1.862],
            "DeltaNet": [1.273, 2.366, 4.837, 9.960],
            "Standard": [0.056, 0.215, 0.836, 3.285],
            "SlidingWindow": [0.062, 0.196, 0.592, 1.834],
            "Hopfield": [0.254, 0.568, 1.245, 2.143],
            "GAU": [0.138, 0.342, 0.731, 1.528]
        }
        
        memory_results = {
            "BSBR": [7.66, 7.66, 7.66, 7.67],
            "Linear": [6.40, 6.40, 6.40, 6.41],
            "DeltaNet": [6.40, 6.40, 6.40, 6.41],
            "Standard": [7.20, 8.34, 11.58, 23.92],
            "SlidingWindow": [7.14, 7.98, 9.54, 12.65],
            "Hopfield": [6.62, 6.63, 6.64, 6.68],
            "GAU": [7.82, 7.85, 7.92, 8.12]
        }
    else:
        # TODO: Add code to load results from files if needed
        print("Loading results from files not implemented yet. Using example data.")
        # Use the same example data as above
        time_results = {
            "BSBR": [0.043, 0.058, 0.156, 0.428],
            "Linear": [0.213, 0.490, 1.096, 1.862],
            "DeltaNet": [1.273, 2.366, 4.837, 9.960],
            "Standard": [0.056, 0.215, 0.836, 3.285],
            "SlidingWindow": [0.062, 0.196, 0.592, 1.834],
            "Hopfield": [0.254, 0.568, 1.245, 2.143],
            "GAU": [0.138, 0.342, 0.731, 1.528]
        }
        
        memory_results = {
            "BSBR": [7.66, 7.66, 7.66, 7.67],
            "Linear": [6.40, 6.40, 6.40, 6.41],
            "DeltaNet": [6.40, 6.40, 6.40, 6.41],
            "Standard": [7.20, 8.34, 11.58, 23.92],
            "SlidingWindow": [7.14, 7.98, 9.54, 12.65],
            "Hopfield": [6.62, 6.63, 6.64, 6.68],
            "GAU": [7.82, 7.85, 7.92, 8.12]
        }
    
    seq_lengths = args.seq_lengths
    analyze_results(seq_lengths, time_results, memory_results)


if __name__ == "__main__":
    main() 