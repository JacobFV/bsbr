import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import argparse
import json
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


def analyze_results(seq_lengths: List[int], time_results: Dict[str, List[float]], memory_results: Dict[str, List[float]], args):
    """
    Analyze the performance results of different models.
    
    Args:
        seq_lengths: List of sequence lengths
        time_results: Dictionary mapping model names to lists of inference times
        memory_results: Dictionary mapping model names to lists of memory usage
        args: ArgumentParser object
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
    plt_complexity = plot_complexity_curves(seq_lengths, time_results, "Inference Time vs Sequence Length")
    complexity_plot_path = os.path.join(args.output_dir, "complexity_analysis.png")
    plt_complexity.savefig(complexity_plot_path)
    print(f"\nSaved complexity plot to {complexity_plot_path}")
    plt.close(plt_complexity.gcf()) # Close the plot figure

    # Create log-log plot
    plt.figure(figsize=(12, 8))
    for model_name, times in time_results.items():
        plt.loglog(seq_lengths, times, marker='o', label=model_name)
    
    plt.xlabel("Sequence Length (log scale)")
    plt.ylabel("Time (seconds, log scale)")
    plt.title("Inference Time vs Sequence Length (Log-Log Scale)")
    plt.legend()
    plt.grid(True)
    loglog_plot_path = os.path.join(args.output_dir, "complexity_loglog.png")
    plt.savefig(loglog_plot_path)
    print(f"Saved log-log plot to {loglog_plot_path}")
    plt.close() # Close the current plot figure


def main():
    parser = argparse.ArgumentParser(description="Analyze model comparison results")
    parser.add_argument("--results_file", type=str, default="research/architecture_comparisons/results/comparison_results.json",
                        help="Path to the comparison_results.json file")
    parser.add_argument("--output_dir", type=str, default="research/architecture_comparisons/results",
                        help="Directory to save analysis plots")
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Load results from JSON file
    try:
        with open(args.results_file, 'r') as f:
            results_data = json.load(f)
        print(f"Loaded results from {args.results_file}")
    except FileNotFoundError:
        print(f"Error: Results file not found at {args.results_file}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {args.results_file}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while loading the results file: {e}")
        sys.exit(1)

    # Extract data
    time_results = results_data.get("time_results")
    memory_results = results_data.get("memory_results")
    seq_lengths = results_data.get("seq_lengths")
    # param_counts = results_data.get("param_counts") # Not used in this script

    if not time_results or not memory_results or not seq_lengths:
        print("Error: Missing required data (time_results, memory_results, seq_lengths) in the results file.")
        sys.exit(1)

    # Pass output_dir to the analysis function
    analyze_results(seq_lengths, time_results, memory_results, args)


if __name__ == "__main__":
    main() 