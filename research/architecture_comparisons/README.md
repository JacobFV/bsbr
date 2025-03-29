# Transformer Architecture Evaluation

This directory contains scripts for evaluating different efficient transformer architectures:

1. **BSBR (Block Sparse Attention with Block Retrieval)** - Combines in-chunk standard attention with between-chunk block retrieval
2. **Standard Transformer** - The classic attention mechanism with full quadratic attention matrix
3. **Linear Transformer** - Removes softmax to achieve linear complexity in sequence length
4. **DeltaNet** - Enhances Linear Transformer with a removal component for better memory management
5. **Sliding Window Transformer** - Restricts attention to a fixed window for O(n·w) complexity
6. **Hopfield Network** - Associative memory-based attention inspired by modern Hopfield Networks
7. **GAU (Gated Attention Unit)** - Uses chunk-based parallelism with gating for efficient processing

The evaluation consists of three main scripts:
*   `compare_models.py`: Runs benchmarks to compare inference time and memory usage across different sequence lengths. Saves raw results to `results/comparison_results.json` and a basic plot to `results/model_comparison.png`.
*   `analyze_results.py`: Loads the raw results from `comparison_results.json`, performs complexity and relative performance analysis, and saves analysis plots (`complexity_analysis.png`, `complexity_loglog.png`) to the `results/` directory.
*   `visualize_models.py`: Loads the raw results and generates various visualizations (heatmap, radar chart, scaling curves, memory scaling, combined performance) saved to the `results/` directory.

## Evaluation Results (Latest Run)

Based on empirical testing with sequence lengths up to 1024 tokens on a CPU device.

### Computational Complexity

| Model         | Empirical Complexity | R-squared | Time at n=1024 (seconds) | Memory at n=1024 (MB) |
|---------------|----------------------|-----------|--------------------------|----------------------|
| BSBR          | O(n^0.70) ≈ O(n)     | 0.9380    | 3.0916                   | 22.83                |
| Standard      | O(n^0.81) ≈ O(n)     | 0.9212    | 2.5382                   | 13.80                |
| Linear        | O(n^0.86) ≈ O(n)     | 0.9988    | 17.3223                  | 13.80                |
| DeltaNet      | O(n^0.88) ≈ O(n)     | 0.9956    | 92.2763                  | 13.80                |
| SlidingWindow | O(n^0.86) ≈ O(n)     | 0.9804    | 5.5680                   | 13.80                |
| Hopfield      | O(n^0.80) ≈ O(n)     | 0.9308    | 2.5681                   | 13.80                |
| GAU           | O(n^1.30) ≈ O(n log n)| 0.9826    | 17.6486                  | 16.81                |

*Note: The empirical complexity for the Standard Transformer is lower than expected (O(n^0.81) vs O(n^2) theoretical) in this CPU run, potentially due to overhead dominating at these scales or implementation details.*

### Relative Performance (vs Standard Transformer @ n=1024)

Standard Transformer was the fastest model in this run at n=1024.

| Model         | Avg Slowdown vs Standard | Min Slowdown | Max Slowdown | Slowdown at n=1024 |
|---------------|--------------------------|--------------|--------------|-------------------|
| BSBR          | 1.62x                    | 1.22x        | 1.82x        | 1.22x             |
| SlidingWindow | 2.40x                    | 2.02x        | 2.84x        | 2.19x             |
| Hopfield      | 1.04x                    | 1.00x        | 1.09x        | 1.01x             |
| GAU           | 4.35x                    | 1.92x        | 6.95x        | 6.95x             |
| Linear        | N/A                      | N/A          | N/A          | N/A               |
| DeltaNet      | N/A                      | N/A          | N/A          | N/A               |

*Note: Linear and DeltaNet were excluded from the relative performance calculation in `analyze_results.py` due to potentially incomplete data in the specific run producing these results.*

### Memory Usage

| Model         | Memory at n=1024 (MB) | Scaling Trend |
|---------------|----------------------|---------------|
| BSBR          | 22.83                | Minimal       |
| Standard      | 13.80                | Minimal       |
| Linear        | 13.80                | Minimal       |
| DeltaNet      | 13.80                | Minimal       |
| SlidingWindow | 13.80                | Minimal       |
| Hopfield      | 13.80                | Minimal       |
| GAU           | 16.81                | Minimal       |

*Note: Memory scaling appears minimal for most models in this CPU run. GPU runs would likely show more pronounced differences, especially for the Standard Transformer.*

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


## Running Evaluations

**Prerequisites:**
1.  Ensure `uv` is installed.
2.  Create and activate a virtual environment: `uv venv` then `.\.venv\Scripts\activate` (Windows) or `source .venv/bin/activate` (Linux/macOS).
3.  Install dependencies: `uv pip install -r requirements.txt`.
4.  Install the project in editable mode: `uv pip install -e .`.

**Execution:**

Run the scripts from the project root directory. Due to Python import resolution issues when running scripts in subdirectories directly, modifying `PYTHONPATH` might be necessary.

```powershell
# --- Step 1: Run Benchmarks ---
# (Activate venv first if not already active)
$env:PYTHONPATH=".;$env:PYTHONPATH"; python research/architecture_comparisons/compare_models.py

# Example: Run specific models
$env:PYTHONPATH=".;$env:PYTHONPATH"; python research/architecture_comparisons/compare_models.py --models BSBR Linear Hopfield GAU

# Example: Test with different sequence lengths
$env:PYTHONPATH=".;$env:PYTHONPATH"; python research/architecture_comparisons/compare_models.py --seq_lengths 256 512 1024

# Example: Use GPU if available
$env:PYTHONPATH=".;$env:PYTHONPATH"; python research/architecture_comparisons/compare_models.py --device cuda

# --- Step 2: Analyze Results ---
# (Uses results/comparison_results.json generated by Step 1)
python research/architecture_comparisons/analyze_results.py

# --- Step 3: Visualize Results ---
# (Uses results/comparison_results.json generated by Step 1)
python research/architecture_comparisons/visualize_models.py

# Example: Generate only specific plot types
python research/architecture_comparisons/visualize_models.py --plot_types heatmap radar
```

*(Note: The `$env:PYTHONPATH...` prefix is for PowerShell. Use `export PYTHONPATH=.:$PYTHONPATH; ...` for bash/zsh)*

This will generate the raw data JSON file and various analysis/visualization PNG files in the `research/architecture_comparisons/results/` directory.

## Conclusion

Based on this specific CPU run:

*   **Standard Transformer** and **Hopfield** were surprisingly fast at n=1024, outperforming BSBR. This might be due to CPU overhead or specific implementation details benefiting them at this scale on CPU.
*   **BSBR** showed good scaling (near linear) but had higher absolute times compared to Standard/Hopfield in this run. It also had the highest baseline memory usage, though it remained constant.
*   **Linear** and **DeltaNet** showed poor absolute performance, especially DeltaNet.
*   **GAU** showed slightly worse than linear scaling (O(n^1.30)) and was relatively slow.

**Important Considerations:**
*   These results are from a single CPU run. Performance characteristics, especially for Standard Transformers, can change dramatically on GPU.
*   The empirical scaling observed might not hold for much larger sequence lengths where theoretical complexity becomes dominant.

For production systems requiring efficiency *and* expressivity, especially on GPUs or for very long sequences, BSBR remains a strong candidate due to its favorable theoretical scaling and balanced design. However, Standard and Hopfield might be competitive on CPU for moderate sequence lengths based on this run. GAU and the Linear variants (especially DeltaNet) appear less competitive in this specific evaluation. 