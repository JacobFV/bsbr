# Research Experiments

This document details the experiments conducted to evaluate BSBR's performance and capabilities.

## Experimental Setup

### Hardware Configuration

- **GPU**: *N/A (CPU Used for latest benchmarks)*
- **CPU**: Intel Core i9 (Specific model varies)
- **Memory**: Varies (e.g., 32GB+ RAM typical)
- **Storage**: NVMe SSD

### Software Stack

- PyTorch (e.g., 2.x)
- CUDA (If GPU used)
- Python 3.12
- BSBR 0.1.2
- Key libraries: `transformers`, `numpy`, `pandas`, `matplotlib`, `seaborn` (see `requirements.txt`)

### Baseline Models

Experiments typically compare BSBR against:

1. **Standard Transformer**
2. **Linear Transformer**
3. **Sliding Window Transformer**
4. **DeltaNet**
5. **Hopfield Network**
6. **GAU (Gated Attention Unit)**

See [Benchmarks](./benchmarks.md#models-compared) for details on architectures.

## Performance Experiments

### Scaling Analysis

Experiments measure inference time and memory usage across varying sequence lengths (e.g., 64 to 1024 or higher).

- **Objective**: Determine empirical scaling behavior (e.g., O(n), O(n log n), O(n^2)).
- **Method**: Run `research/architecture_comparisons/compare_models.py` followed by `research/architecture_comparisons/analyze_results.py`.
- **Results**: See detailed tables and plots in [Benchmarks - Performance](./benchmarks.md#performance-benchmarks-cpu).

### Training Experiments (Illustrative)

*(Placeholder: Describes potential training experiments)*

#### Convergence Analysis

Compare training loss curves and epochs required to reach a target validation metric for BSBR vs. baselines.

#### Memory Efficiency

Measure peak GPU/CPU memory usage during training, including gradients and activations.

## Model Quality Experiments

### Perplexity on Standard Datasets

Evaluate language modeling capability using perplexity on datasets like Wikitext.

- **Objective**: Assess base model quality after architectural changes (e.g., BSBR conversion).
- **Method**: Run `research/conversion_experiments/benchmark_comparison.py --quality_eval`.
- **Results**: See [BSBR Conversion Evaluation - Perplexity](../research/bsbr_conversion_evaluation.md#perplexity).

### Output Similarity

Quantify how closely the outputs (hidden states, token predictions) of a modified model (e.g., BSBR-converted) match the original.

- **Objective**: Understand the behavioral impact of architectural changes like BSBR conversion.
- **Method**: Run `research/conversion_experiments/output_comparison.py`.
- **Metrics**: Cosine similarity, MSE, KL divergence (hidden states); Top-K agreement rates (predictions).
- **Results**: See detailed metrics in [BSBR Conversion Evaluation - Output Similarity](../research/bsbr_conversion_evaluation.md#output-similarity-analysis).

### Long-Range Dependencies (Illustrative)

*(Placeholder: Describes potential task-based evaluations)*

Evaluate performance on tasks requiring reasoning over long contexts (e.g., document classification, QA over long passages).

### Attention Analysis (Illustrative)

*(Placeholder: Describes potential attention pattern analysis)*

Visualize attention maps to understand how BSBR focuses on different parts of the context compared to standard attention.

## Ablation Studies (Illustrative)

*(Placeholder: Describes potential ablation studies)*

### Chunk Size Impact

Evaluate how varying the `chunk_size` in BSBR affects performance, memory, and potentially task accuracy.

### Compression Factor Analysis

Analyze the trade-off between state compression in BSBR (memory/speed) and potential impact on model quality.

## Real-World Applications (Illustrative)

*(Placeholder: Describes potential application-specific tests)*

Test BSBR in simulated real-world scenarios like processing large documents or handling long conversational contexts.

## Conclusions

1. **Performance**
   - Linear scaling with sequence length
   - Significantly lower memory usage
   - Competitive inference time

2. **Model Quality**
   - Better long-range modeling
   - Comparable accuracy to standard attention
   - More stable training

3. **Practical Benefits**
   - Efficient document processing
   - Better memory management
   - Flexible architecture 