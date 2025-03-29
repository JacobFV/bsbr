import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import argparse
from typing import Dict, List, Tuple
import json
from matplotlib.ticker import FuncFormatter

# Configure seaborn
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 12


def load_example_data():
    """Load example data for visualization"""
    time_results = {
        "BSBR": [0.043, 0.058, 0.156, 0.428, 1.25],
        "Linear": [0.213, 0.490, 1.096, 1.862, 3.54],
        "DeltaNet": [1.273, 2.366, 4.837, 9.960, 18.42],
        "Standard": [0.056, 0.215, 0.836, 3.285, 12.87],
        "SlidingWindow": [0.062, 0.196, 0.592, 1.834, 4.98],
        "Hopfield": [0.254, 0.568, 1.245, 2.143, 4.25],
        "GAU": [0.138, 0.342, 0.731, 1.528, 3.18]
    }
    
    memory_results = {
        "BSBR": [7.66, 7.66, 7.66, 7.67, 7.68],
        "Linear": [6.40, 6.40, 6.40, 6.41, 6.45],
        "DeltaNet": [6.40, 6.40, 6.40, 6.41, 6.45],
        "Standard": [7.20, 8.34, 11.58, 23.92, 86.45],
        "SlidingWindow": [7.14, 7.98, 9.54, 12.65, 21.32],
        "Hopfield": [6.62, 6.63, 6.64, 6.68, 6.72],
        "GAU": [7.82, 7.85, 7.92, 8.12, 8.36]
    }
    
    param_counts = {
        "BSBR": 5983488,
        "Linear": 3614720,
        "DeltaNet": 3614720,
        "Standard": 3614720,
        "SlidingWindow": 3614720,
        "Hopfield": 3614720,
        "GAU": 4403712
    }
    
    seq_lengths = [64, 256, 512, 1024, 2048]
    
    return time_results, memory_results, param_counts, seq_lengths


def plot_inference_heatmap(time_results, seq_lengths, output_path=None):
    """Create a heatmap showing normalized inference times across models and sequence lengths"""
    # Create DataFrame from time_results
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
    
    plt.tight_layout()
    plt.show()
    
    return plt


def plot_radar_chart(time_results, memory_results, param_counts, seq_lengths, output_path=None):
    """Create a radar chart comparing different architectures across metrics"""
    # Select the last sequence length for comparison
    final_idx = -1
    
    # Prepare data
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
    
    plt.tight_layout()
    plt.show()
    
    return plt


def plot_scaling_curves(time_results, seq_lengths, output_path=None):
    """Plot scaling behavior with theoretical curves for comparison"""
    plt.figure(figsize=(14, 8))
    
    # Define theoretical curves
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
    
    plt.tight_layout()
    plt.show()
    
    return plt


def plot_memory_scaling(memory_results, seq_lengths, output_path=None):
    """Plot memory scaling behavior across models"""
    plt.figure(figsize=(14, 8))
    
    # Configure y-axis to show MB
    def format_mb(x, pos):
        return f'{x:.1f} MB'
    
    formatter = FuncFormatter(format_mb)
    
    # Plot actual data
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
    
    plt.tight_layout()
    plt.show()
    
    return plt


def plot_combined_performance(time_results, memory_results, param_counts, seq_lengths, output_path=None):
    """Create a bubble chart showing three dimensions: time, memory, and parameters"""
    # Use the largest sequence length for comparison
    seq_idx = -1
    
    # Extract data for the plot
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
    
    plt.tight_layout()
    plt.show()
    
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
    
    print(f"All visualizations have been generated and saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Visualize transformer model comparisons")
    parser.add_argument("--data_file", type=str, default=None, 
                        help="JSON file containing evaluation results")
    parser.add_argument("--output_dir", type=str, default="visualization_results", 
                        help="Directory to save visualization outputs")
    parser.add_argument("--plot_type", type=str, choices=[
                        "heatmap", "radar", "scaling", "memory", "combined", "all"], 
                        default="all", help="Type of plot to generate")
    parser.add_argument("--use_example_data", action="store_true", 
                        help="Use example data for visualization")
    
    args = parser.parse_args()
    
    # Load data
    if args.use_example_data or args.data_file is None:
        time_results, memory_results, param_counts, seq_lengths = load_example_data()
        print("Using example data for visualization")
    else:
        # Load data from JSON file
        try:
            with open(args.data_file, 'r') as f:
                data = json.load(f)
                time_results = data.get("time_results", {})
                memory_results = data.get("memory_results", {})
                param_counts = data.get("param_counts", {})
                seq_lengths = data.get("seq_lengths", [])
            print(f"Loaded data from {args.data_file}")
        except Exception as e:
            print(f"Error loading data from file: {e}")
            print("Falling back to example data")
            time_results, memory_results, param_counts, seq_lengths = load_example_data()
    
    # Create output directory if needed
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate requested plots
    if args.plot_type == "heatmap" or args.plot_type == "all":
        plot_inference_heatmap(
            time_results, seq_lengths, 
            os.path.join(args.output_dir, "inference_heatmap.png") if args.output_dir else None
        )
    
    if args.plot_type == "radar" or args.plot_type == "all":
        plot_radar_chart(
            time_results, memory_results, param_counts, seq_lengths,
            os.path.join(args.output_dir, "radar_chart.png") if args.output_dir else None
        )
    
    if args.plot_type == "scaling" or args.plot_type == "all":
        plot_scaling_curves(
            time_results, seq_lengths,
            os.path.join(args.output_dir, "scaling_curves.png") if args.output_dir else None
        )
    
    if args.plot_type == "memory" or args.plot_type == "all":
        plot_memory_scaling(
            memory_results, seq_lengths,
            os.path.join(args.output_dir, "memory_scaling.png") if args.output_dir else None
        )
    
    if args.plot_type == "combined" or args.plot_type == "all":
        plot_combined_performance(
            time_results, memory_results, param_counts, seq_lengths,
            os.path.join(args.output_dir, "combined_performance.png") if args.output_dir else None
        )
    
    if args.plot_type == "all":
        print(f"All visualizations have been generated and saved to {args.output_dir}")
    else:
        print(f"{args.plot_type.capitalize()} visualization has been generated and saved to {args.output_dir}")


if __name__ == "__main__":
    main() 