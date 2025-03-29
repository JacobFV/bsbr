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
    plt.figure(figsize=(10, 6))
    
    # Set colors for each model
    colors = {
        "BSBR": "blue",
        "LinearTransformer": "green", 
        "DeltaNet": "red"
    }
    
    # Plot the actual times
    for model_name, model_times in times.items():
        plt.plot(seq_lengths, model_times, marker='o', label=f"{model_name} (Actual)", color=colors[model_name])
    
    # Plot theoretical complexity curves
    x = np.array(seq_lengths)
    
    # Linear complexity O(n) - LinearTransformer theoretical
    linear_scale = times["LinearTransformer"][-1] / seq_lengths[-1]
    plt.plot(x, linear_scale * x, '--', color='green', alpha=0.5, label="O(n) - Linear")
    
    # Quadratic complexity O(n²) - Standard Transformer theoretical
    quadratic_scale = times["BSBR"][-1] / (seq_lengths[-1] ** 2)
    plt.plot(x, quadratic_scale * x**2, '--', color='purple', alpha=0.5, label="O(n²) - Quadratic")
    
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
    elif 1.2 <= slope < 1.8:
        complexity = f"O(n^{slope:.2f}) ≈ O(n log n)"
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
            "Time at n=1024": f"{times[-1]:.4f}s",
            "Memory at n=1024": f"{memory_results[model_name][-1]:.2f} MB"
        })
    
    # Convert to pandas DataFrame for nice display
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    
    # Calculate relative speedup
    print("\n===== RELATIVE PERFORMANCE =====")
    baseline_model = "BSBR"
    baseline_times = time_results[baseline_model]
    
    rel_results = []
    for model_name, times in time_results.items():
        if model_name != baseline_model:
            # Calculate how much slower other models are compared to BSBR
            # BSBR appears to be faster based on the raw numbers
            slowdowns = [model_time / baseline_time for baseline_time, model_time in zip(baseline_times, times)]
            rel_results.append({
                "Model": model_name,
                "Avg Slowdown vs BSBR": f"{np.mean(slowdowns):.2f}x",
                "Min Slowdown": f"{min(slowdowns):.2f}x",
                "Max Slowdown": f"{max(slowdowns):.2f}x",
                "Slowdown at n=1024": f"{times[-1] / baseline_times[-1]:.2f}x"
            })
            
    # Convert to pandas DataFrame for nice display
    rel_df = pd.DataFrame(rel_results)
    print(rel_df.to_string(index=False))
    
    # Create complexity plot
    plt = plot_complexity_curves(seq_lengths, time_results, "Inference Time vs Sequence Length")
    plt.savefig("complexity_analysis.png")
    print("\nSaved complexity plot to complexity_analysis.png")
    
    # Create log-log plot
    plt.figure(figsize=(10, 6))
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
    args = parser.parse_args()
    
    # Example results from the evaluation - replace with actual results
    # These are values from our latest run with sequence lengths [256, 512, 1024, 2048]
    time_results = {
        "BSBR": [0.043, 0.058, 0.156, 0.428],
        "LinearTransformer": [0.213, 0.490, 1.096, 1.862],
        "DeltaNet": [1.273, 2.366, 4.837, 9.960]
    }
    
    memory_results = {
        "BSBR": [7.66, 7.66, 7.66, 7.67],
        "LinearTransformer": [6.40, 6.40, 6.40, 6.41],
        "DeltaNet": [6.40, 6.40, 6.40, 6.41]
    }
    
    seq_lengths = [256, 512, 1024, 2048]
    analyze_results(seq_lengths, time_results, memory_results)


if __name__ == "__main__":
    main() 