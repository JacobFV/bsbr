import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import argparse
import json
import sys
from typing import Dict, List, Tuple
from matplotlib.ticker import FuncFormatter

# Configure seaborn
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 12


def plot_inference_heatmap(time_results, seq_lengths, output_path=None):
    """Create a heatmap showing normalized inference times across models and sequence lengths"""
    # Create DataFrame from time_results (which should now be list-based)
    df = pd.DataFrame(time_results, index=seq_lengths)
    
    # Normalize each row (sequence length) to show relative performance
    df_normalized = df.div(df.min(axis=1), axis=0)
    
    # Create the heatmap
    plt.figure(figsize=(14, 8))
    heatmap = sns.heatmap(df_normalized.T, annot=True, fmt=".2f", cmap="YlGnBu", 
                         cbar_kws={'label': 'Relative Slowdown (vs. Fastest)'})
    
    # Set labels and title
    plt.xlabel("Sequence Length")
    plt.ylabel("Model Architecture")
    plt.title("Relative Inference Time Across Models (Lower is Better)")
    
    # Adjust y-axis labels
    plt.yticks(rotation=0)
    
    if output_path:
        plt.savefig(output_path)
        print(f"Saved inference heatmap to {output_path}")
    
    plt.tight_layout()
    plt.close()
    
    return plt


def plot_radar_chart(time_results, memory_results, param_counts, seq_lengths, output_path=None):
    """Create a radar chart comparing different architectures across metrics"""
    # Select the last sequence length for comparison
    final_idx = -1
    
    # Prepare data (time_results and memory_results are dicts of {model: list_of_values})
    models = list(time_results.keys())
    metrics = ['Inference Speed', 'Memory Efficiency', 'Parameter Efficiency', 'Scaling Behavior']
    
    # Normalize metrics (higher value = better performance)
    inference_speed = [1.0 / time_results[model][final_idx] for model in models]
    inference_speed = [val / max(inference_speed) for val in inference_speed]
    
    memory_efficiency = [1.0 / memory_results[model][final_idx] for model in models]
    memory_efficiency = [val / max(memory_efficiency) for val in memory_efficiency]
    
    param_efficiency = [1.0 / param_counts[model] for model in models]
    param_efficiency = [val / max(param_efficiency) for val in param_efficiency]
    
    # Calculate scaling as ratio between largest and smallest sequence length performance
    scaling_factor = [time_results[model][final_idx] / time_results[model][0] / 
                     (seq_lengths[final_idx] / seq_lengths[0]) for model in models]
    scaling_factor = [1.0 / val for val in scaling_factor]  # Invert so lower is better
    scaling_factor = [val / max(scaling_factor) for val in scaling_factor]
    
    # Combine metrics
    data = {
        model: [inference_speed[i], memory_efficiency[i], param_efficiency[i], scaling_factor[i]]
        for i, model in enumerate(models)
    }
    
    # Number of metrics
    N = len(metrics)
    
    # Create angles for radar chart
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Create the radar chart
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(polar=True))
    
    # Colors for each model
    colors = ["blue", "green", "red", "purple", "orange", "brown", "pink"]
    
    # Plot each model
    for i, model in enumerate(models):
        values = data[model]
        values += values[:1]  # Close the loop
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=model, color=colors[i])
        ax.fill(angles, values, alpha=0.1, color=colors[i])
    
    # Set ticks and labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    
    # Draw y-axis labels
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75, 1.0], ["0.25", "0.5", "0.75", "1.0"], color="grey", size=10)
    plt.ylim(0, 1)
    
    # Title and legend
    plt.title(f"Model Architecture Comparison (Sequence Length {seq_lengths[final_idx]})", size=15)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    if output_path:
        plt.savefig(output_path)
        print(f"Saved radar chart to {output_path}")
    
    plt.tight_layout()
    plt.close()
    
    return plt


def plot_scaling_curves(time_results, seq_lengths, output_path=None):
    """Plot scaling behavior with theoretical curves for comparison"""
    plt.figure(figsize=(14, 8))
    
    # Define theoretical curves (use processed time_results which are lists)
    x = np.array(seq_lengths)
    constant = np.ones_like(x) * time_results["BSBR"][0]
    linear = x * (time_results["Linear"][0] / x[0])
    nlogn = x * np.log2(x) * (time_results["SlidingWindow"][0] / (x[0] * np.log2(x[0])))
    quadratic = x**2 * (time_results["Standard"][0] / x[0]**2)
    
    # Plot theoretical curves
    plt.plot(x, constant, '--', color='gray', alpha=0.5, label="O(1)")
    plt.plot(x, linear, '--', color='green', alpha=0.5, label="O(n)")
    plt.plot(x, nlogn, '--', color='orange', alpha=0.5, label="O(n log n)")
    plt.plot(x, quadratic, '--', color='purple', alpha=0.5, label="O(n²)")
    
    # Plot actual data
    markers = ['o', 's', '^', 'D', 'v', 'p', '*']
    colors = ["blue", "green", "red", "purple", "orange", "brown", "pink"]
    
    for i, (model, times) in enumerate(time_results.items()):
        plt.plot(seq_lengths, times, marker=markers[i], color=colors[i], label=model)
    
    # Set log scales for better visualization
    plt.xscale('log', base=2)
    plt.yscale('log', base=2)
    
    # Add labels and title
    plt.xlabel("Sequence Length")
    plt.ylabel("Inference Time (seconds)")
    plt.title("Scaling Behavior Compared to Theoretical Bounds")
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend(loc='upper left')
    
    if output_path:
        plt.savefig(output_path)
        print(f"Saved scaling curves to {output_path}")
    
    plt.tight_layout()
    plt.close()
    
    return plt


def plot_memory_scaling(memory_results, seq_lengths, output_path=None):
    """Plot memory scaling behavior across models"""
    plt.figure(figsize=(14, 8))
    
    # Configure y-axis to show MB
    def format_mb(x, pos):
        return f'{x:.1f} MB'
    
    formatter = FuncFormatter(format_mb)
    
    # Plot actual data (memory_results should be dict of {model: list_of_values})
    markers = ['o', 's', '^', 'D', 'v', 'p', '*']
    colors = ["blue", "green", "red", "purple", "orange", "brown", "pink"]
    
    for i, (model, memory) in enumerate(memory_results.items()):
        plt.plot(seq_lengths, memory, marker=markers[i], color=colors[i], label=model)
    
    # Set log scale for x-axis
    plt.xscale('log', base=2)
    
    # Add labels and title
    plt.xlabel("Sequence Length")
    plt.ylabel("Memory Usage")
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.title("Memory Scaling by Architecture")
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend(loc='upper left')
    
    if output_path:
        plt.savefig(output_path)
        print(f"Saved memory scaling plot to {output_path}")
    
    plt.tight_layout()
    plt.close()
    
    return plt


def plot_combined_performance(time_results, memory_results, param_counts, seq_lengths, output_path=None):
    """Create a bubble chart showing three dimensions: time, memory, and parameters"""
    # Use the largest sequence length for comparison
    seq_idx = -1
    
    # Extract data for the plot (use processed results which are lists)
    models = list(time_results.keys())
    x = [memory_results[model][seq_idx] for model in models]  # Memory usage
    y = [time_results[model][seq_idx] for model in models]  # Inference time
    size = [param_counts[model] / 100000 for model in models]  # Parameter count
    colors = ["blue", "green", "red", "purple", "orange", "brown", "pink"]
    
    plt.figure(figsize=(14, 10))
    
    # Create scatter plot
    for i, model in enumerate(models):
        plt.scatter(x[i], y[i], s=size[i], c=colors[i], alpha=0.7, edgecolors='w', linewidths=1.5, label=model)
    
    # Add model names as annotations
    for i, model in enumerate(models):
        plt.annotate(model, (x[i], y[i]), xytext=(5, 5), textcoords='offset points')
    
    # Add labels and title
    plt.xlabel("Memory Usage (MB)")
    plt.ylabel("Inference Time (seconds)")
    plt.title(f"Model Performance Comparison (n={seq_lengths[seq_idx]})")
    
    # Add legend explaining bubble size
    # Create a fake plot for the legend
    sizes = [1000000, 4000000, 7000000]
    labels = ["1M", "4M", "7M"]
    for i, size in enumerate(sizes):
        plt.scatter([], [], s=size/100000, c='gray', alpha=0.5, label=f"{labels[i]} params")
    
    plt.legend(title="Parameters", loc='upper right')
    
    # Add an "ideal point" in the lower left corner
    plt.scatter(x[0] * 0.8, y[0] * 0.8, marker='*', s=300, c='gold', edgecolors='k', linewidths=1.5)
    plt.annotate("Ideal", (x[0] * 0.8, y[0] * 0.8), xytext=(10, 10), textcoords='offset points')
    
    plt.grid(True, alpha=0.3)
    
    if output_path:
        plt.savefig(output_path)
        print(f"Saved combined performance plot to {output_path}")
    
    plt.tight_layout()
    plt.close()
    
    return plt


def plot_summary_dashboard(time_results, memory_results, param_counts, seq_lengths, output_dir=None):
    """Create a comprehensive dashboard with multiple visualizations"""
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create individual plots
    inference_plt = plot_inference_heatmap(
        time_results, seq_lengths, 
        os.path.join(output_dir, "inference_heatmap.png") if output_dir else None
    )
    
    radar_plt = plot_radar_chart(
        time_results, memory_results, param_counts, seq_lengths,
        os.path.join(output_dir, "radar_chart.png") if output_dir else None
    )
    
    scaling_plt = plot_scaling_curves(
        time_results, seq_lengths,
        os.path.join(output_dir, "scaling_curves.png") if output_dir else None
    )
    
    memory_plt = plot_memory_scaling(
        memory_results, seq_lengths,
        os.path.join(output_dir, "memory_scaling.png") if output_dir else None
    )
    
    combined_plt = plot_combined_performance(
        time_results, memory_results, param_counts, seq_lengths,
        os.path.join(output_dir, "combined_performance.png") if output_dir else None
    )
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, "summary_dashboard.png"))
        print(f"Saved summary dashboard to {os.path.join(output_dir, 'summary_dashboard.png')}")

    plt.tight_layout()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize model comparison results")
    parser.add_argument("--results_file", type=str, default="research/architecture_comparisons/results/comparison_results.json",
                        help="Path to the comparison_results.json file")
    parser.add_argument("--output_dir", type=str, default="research/architecture_comparisons/results",
                        help="Directory to save visualization plots")
    parser.add_argument("--plot_types", type=str, nargs="+", 
                        default=["heatmap", "radar", "scaling", "memory", "combined", "dashboard"],
                        help="Types of plots to generate: heatmap, radar, scaling, memory, combined, dashboard")
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
    raw_time_results = results_data.get("time_results")
    raw_memory_results = results_data.get("memory_results")
    param_counts = results_data.get("param_counts")
    seq_lengths = results_data.get("seq_lengths") # This should be a list of ints

    if not all([raw_time_results, raw_memory_results, param_counts, seq_lengths]):
        print("Error: Missing required data (time_results, memory_results, param_counts, seq_lengths) in the results file.")
        sys.exit(1)

    # Process raw results into lists ordered by seq_lengths
    processed_time_results = {}
    processed_memory_results = {}
    valid_models = list(raw_time_results.keys())

    for model_name in valid_models:
        try:
            # Convert time dict to ordered list
            model_time_data = raw_time_results[model_name]
            ordered_times = [model_time_data[str(sl)] for sl in seq_lengths]
            processed_time_results[model_name] = ordered_times

            # Convert memory dict to ordered list
            model_memory_data = raw_memory_results[model_name]
            ordered_memory = [model_memory_data[str(sl)] for sl in seq_lengths]
            processed_memory_results[model_name] = ordered_memory

        except KeyError as e:
            print(f"Warning: Missing data for sequence length {e} in model {model_name}. Excluding model from visualizations.")
            # Remove model from processed results if data is incomplete
            if model_name in processed_time_results:
                del processed_time_results[model_name]
            if model_name in processed_memory_results:
                del processed_memory_results[model_name]
            if model_name in param_counts:
                 del param_counts[model_name] # Also remove from params if excluding model
            continue
    
    if not processed_time_results:
        print("Error: No models have complete data for visualization.")
        sys.exit(1)

    # Generate selected plots using processed data
    print("\nGenerating plots...")
    
    if "heatmap" in args.plot_types:
        plot_inference_heatmap(processed_time_results, seq_lengths, 
                               output_path=os.path.join(args.output_dir, "inference_heatmap.png"))
    
    if "radar" in args.plot_types:
        plot_radar_chart(processed_time_results, processed_memory_results, param_counts, seq_lengths, 
                         output_path=os.path.join(args.output_dir, "radar_chart.png"))
    
    if "scaling" in args.plot_types:
        plot_scaling_curves(processed_time_results, seq_lengths, 
                              output_path=os.path.join(args.output_dir, "scaling_curves.png"))
        
    if "memory" in args.plot_types:
        plot_memory_scaling(processed_memory_results, seq_lengths, 
                              output_path=os.path.join(args.output_dir, "memory_scaling.png"))
    
    if "combined" in args.plot_types:
        plot_combined_performance(processed_time_results, processed_memory_results, param_counts, seq_lengths, 
                                  output_path=os.path.join(args.output_dir, "combined_performance.png"))
        
    if "dashboard" in args.plot_types:
        # Dashboard calls the individual plot functions, ensure it gets processed data
        plot_summary_dashboard(processed_time_results, processed_memory_results, param_counts, seq_lengths, 
                               output_dir=args.output_dir)

    print("\nVisualization complete.")


if __name__ == "__main__":
    main() 