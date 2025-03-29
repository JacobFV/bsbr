# Research Benchmarks

This document provides detailed benchmark results comparing BSBR with other transformer architectures based on the experiments run in the `research/architecture_comparisons/` directory.

## Benchmark Setup

### Environment

-   **Hardware**: CPU (Intel Core i9)
-   **Software**: Python 3.12, PyTorch, Transformers, etc. (see `requirements.txt`)
-   **BSBR Version**: 0.1.2

### Models Compared

Models were configured with comparable hyperparameters (hidden dim: 256, heads: 4, layers: 2 where applicable) for evaluation.

1.  **BSBR (Block Sparse Attention with Block Retrieval)**
2.  **Standard Transformer**
3.  **Linear Transformer**
4.  **DeltaNet**
5.  **Sliding Window Transformer**
6.  **Hopfield Network**
7.  **GAU (Gated Attention Unit)**

### Parameter Counts

| Model         | Parameters (Millions) | Relative to Base (Standard) |
|---------------|----------------------|-----------------------------|
| BSBR          | 6.0M                 | 1.66x                       |
| Standard      | 3.6M                 | 1.0x                        |
| Linear        | 3.6M                 | 1.0x                        |
| DeltaNet      | 3.6M                 | 1.0x                        |
| SlidingWindow | 3.6M                 | 1.0x                        |
| Hopfield      | 3.6M                 | 1.0x                        |
| GAU           | 4.4M                 | 1.22x                       |

*Data source: `research/architecture_comparisons/results/comparison_results.json`*

## Performance Benchmarks (CPU)

Results are based on runs with sequence lengths [64, 128, 256, 512, 1024].
*Data source: `research/architecture_comparisons/results/comparison_results.json`*

### Inference Time (seconds)

| Model         | n=64  | n=128 | n=256 | n=512  | n=1024  |
|---------------|-------|-------|-------|--------|---------|
| BSBR          | 0.462 | 0.560 | 0.753 | 1.570  | 3.092   |
| Linear        | 1.570 | 2.742 | 4.896 | 8.879  | 17.322  |
| DeltaNet      | 8.085 | 13.31 | 23.71 | 46.166 | 92.276  |
| Standard      | 0.254 | 0.334 | 0.453 | 0.908  | 2.538   |
| SlidingWindow | 0.514 | 0.748 | 1.289 | 2.442  | 5.568   |
| Hopfield      | 0.255 | 0.365 | 0.478 | 0.937  | 2.568   |
| GAU           | 0.488 | 0.880 | 1.950 | 5.381  | 17.649  |

### Peak Memory Usage (MB)

| Model         | n=64   | n=128  | n=256  | n=512  | n=1024  |
|---------------|--------|--------|--------|--------|---------|
| BSBR          | 22.826 | 22.826 | 22.827 | 22.829 | 22.833  |
| Linear        | 13.790 | 13.790 | 13.791 | 13.793 | 13.797  |
| DeltaNet      | 13.790 | 13.790 | 13.791 | 13.793 | 13.797  |
| Standard      | 13.790 | 13.790 | 13.791 | 13.793 | 13.797  |
| SlidingWindow | 13.790 | 13.790 | 13.791 | 13.793 | 13.797  |
| Hopfield      | 13.790 | 13.790 | 13.791 | 13.793 | 13.797  |
| GAU           | 16.799 | 16.800 | 16.801 | 16.803 | 16.807  |

### Complexity Analysis

Based on fitting power-law curves to the inference time data.

| Model         | Empirical Complexity | R-squared | Time at n=1024 (seconds) | Memory at n=1024 (MB) |
|---------------|----------------------|-----------|--------------------------|----------------------|
| BSBR          | O(n^0.70) ≈ O(n)     | 0.9380    | 3.0916                   | 22.83                |
| Standard      | O(n^0.81) ≈ O(n)     | 0.9212    | 2.5382                   | 13.80                |
| Linear        | O(n^0.86) ≈ O(n)     | 0.9988    | 17.3223                  | 13.80                |
| DeltaNet      | O(n^0.88) ≈ O(n)     | 0.9956    | 92.2763                  | 13.80                |
| SlidingWindow | O(n^0.86) ≈ O(n)     | 0.9804    | 5.5680                   | 13.80                |
| Hopfield      | O(n^0.80) ≈ O(n)     | 0.9308    | 2.5681                   | 13.80                |
| GAU           | O(n^1.30) ≈ O(n log n)| 0.9826    | 17.6486                  | 16.81                |

*Note: Empirical complexity measured on CPU for n <= 1024 may differ from theoretical asymptotic behavior, especially for Standard attention.*

### Visualizations

*Note: Paths are relative to the `docs/` directory.*

**Complexity & Scaling:**

![Complexity Analysis](../research/architecture_comparisons/results/complexity_analysis.png)
![Log-Log Complexity](../research/architecture_comparisons/results/complexity_loglog.png)
![Scaling Curves](../research/architecture_comparisons/results/scaling_curves.png)

**Performance Comparison:**

![Inference Heatmap](../research/architecture_comparisons/results/inference_heatmap.png)
![Radar Chart](../research/architecture_comparisons/results/radar_chart.png)
![Memory Scaling](../research/architecture_comparisons/results/memory_scaling.png)
![Combined Performance](../research/architecture_comparisons/results/combined_performance.png)
![Basic Comparison](../research/architecture_comparisons/results/model_comparison.png)
![Summary Dashboard](../research/architecture_comparisons/results/summary_dashboard.png)

---

*Sections below contain placeholder data and are for illustrative purposes only.*

## Training Benchmarks

### Convergence Speed

```python
convergence_results = {
    'BSBR': {
        'epochs_to_converge': 50,
        'final_loss': 0.15,
        'validation_accuracy': 0.92
    },
    'Standard': {
        'epochs_to_converge': 45,
        'final_loss': 0.18,
        'validation_accuracy': 0.89
    },
    'Linear': {
        'epochs_to_converge': 55,
        'final_loss': 0.20,
        'validation_accuracy': 0.87
    }
}
```

### Training Memory

```python
training_memory = {
    'BSBR': {
        'peak_memory': 2.5,  # GB
        'gradient_memory': 1.4,  # GB
        'activation_memory': 0.8  # GB
    },
    'Standard': {
        'peak_memory': 16.0,  # GB
        'gradient_memory': 12.8,  # GB
        'activation_memory': 8.0  # GB
    },
    'Linear': {
        'peak_memory': 2.5,  # GB
        'gradient_memory': 1.4,  # GB
        'activation_memory': 0.8  # GB
    }
}
```

## Task-Specific Benchmarks

### Document Classification

```python
document_results = {
    'sequence_lengths': [512, 1024, 2048],
    'BSBR': {
        'accuracy': [0.94, 0.92, 0.89],
        'inference_time': [0.4, 0.7, 1.2]  # seconds
    },
    'Standard': {
        'accuracy': [0.95, 0.88, 0.75],
        'inference_time': [1.5, 5.0, 18.0]  # seconds
    },
    'Linear': {
        'accuracy': [0.92, 0.89, 0.85],
        'inference_time': [0.4, 0.7, 1.2]  # seconds
    }
}
```

### Language Modeling

```python
language_modeling_results = {
    'sequence_lengths': [512, 1024, 2048],
    'BSBR': {
        'perplexity': [15.2, 16.8, 18.5],
        'inference_time': [0.4, 0.7, 1.2]  # seconds
    },
    'Standard': {
        'perplexity': [14.8, 16.5, 18.2],
        'inference_time': [1.5, 5.0, 18.0]  # seconds
    },
    'Linear': {
        'perplexity': [15.5, 17.2, 19.0],
        'inference_time': [0.4, 0.7, 1.2]  # seconds
    }
}
```

### Question Answering

```python
qa_results = {
    'sequence_lengths': [512, 1024, 2048],
    'BSBR': {
        'f1_score': [0.82, 0.80, 0.77],
        'inference_time': [0.4, 0.7, 1.2]  # seconds
    },
    'Standard': {
        'f1_score': [0.83, 0.79, 0.74],
        'inference_time': [1.5, 5.0, 18.0]  # seconds
    },
    'Linear': {
        'f1_score': [0.81, 0.78, 0.75],
        'inference_time': [0.4, 0.7, 1.2]  # seconds
    }
}
```

## Hardware Utilization

### GPU Utilization

```python
gpu_utilization = {
    'BSBR': {
        'gpu_util': 85,  # percentage
        'memory_util': 60,  # percentage
        'power_usage': 250  # watts
    },
    'Standard': {
        'gpu_util': 95,  # percentage
        'memory_util': 90,  # percentage
        'power_usage': 300  # watts
    },
    'Linear': {
        'gpu_util': 80,  # percentage
        'memory_util': 55,  # percentage
        'power_usage': 230  # watts
    }
}
```

### CPU Utilization

```python
cpu_utilization = {
    'BSBR': {
        'cpu_util': 60,  # percentage
        'peak_memory': 2.5  # GB
    },
    'Standard': {
        'cpu_util': 50,
        'peak_memory': 16.0  # GB
    },
    'Linear': {
        'cpu_util': 55,
        'peak_memory': 2.5  # GB
    }
}
``` 