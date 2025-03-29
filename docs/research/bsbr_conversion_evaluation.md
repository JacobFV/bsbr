# BSBR Conversion Evaluation

This document presents a comprehensive evaluation of the Block Sparse Attention with Block Retrieval (BSBR) conversion process, focusing on comparing converted models against their original transformer counterparts using GPT-2 as the base model.

## Overview

Our research aims to quantify the benefits and trade-offs of converting standard transformer models to BSBR architecture. We evaluate:

1. **Performance Characteristics**: Inference speed and computational scaling behavior.
2. **Model Quality**: Perplexity on a standard dataset (Wikitext).
3. **Output Similarity**: How closely BSBR models match the behavior of original models in terms of hidden states and next-token predictions.
4. **Practical Implications**: Use cases where BSBR conversion might offer benefits.

## Methodology

We conducted experiments using:

- **Model**: GPT-2 (base)
- **BSBR Configuration**: Chunk size of 128 tokens
- **Hardware**: CPU evaluation (Intel Core i9)
- **Sequence Lengths Tested**: 128, 256, 512, 1024 (longer sequences skipped due to model limitations)
- **Metrics**:
  - Inference speed (mean time over 5 repeats)
  - Scaling exponents (fitted power-law)
  - Perplexity (on Wikitext-103-raw-v1 test split)
  - Hidden state similarity (cosine similarity, MSE, KL divergence on random inputs)
  - Next-token prediction agreement rates (Top-K)

## Performance Evaluation

### Inference Speed

We measured the inference time for both original and BSBR-converted models across various sequence lengths:

| Sequence Length | Original Transformer (mean time) | BSBR Transformer (mean time) | Speedup |
|-----------------|---------------------------------|------------------------------|---------|
| 128             | 2.014s                          | 2.151s                       | 0.94x   |
| 256             | 2.287s                          | 2.376s                       | 0.96x   |
| 512             | 2.300s                          | 3.110s                       | 0.74x   |
| 1024            | 2.411s                          | 4.379s                       | 0.55x   |

*Data source: `research/conversion_experiments/results/scaling_results.json`*

The data shows that for the tested sequence lengths on CPU, the BSBR conversion resulted in **slower** inference times compared to the original GPT-2 model. The slowdown becomes more pronounced at longer sequence lengths (0.55x speedup, i.e., almost 2x slower, at n=1024).

This could be due to:
1. Overhead of the BSBR mechanism (chunking, retrieval) dominating at these sequence lengths on CPU.
2. Lack of specific CPU optimizations for the BSBR implementation.
3. BSBR's theoretical benefits might only manifest at significantly longer sequences or on different hardware (GPU).

### Scaling Behavior

We analyzed the scaling behavior by fitting power-law curves (O(n^k)) to the measured inference times:

- **Original Transformer**: O(n^0.08)
- **BSBR Transformer**: O(n^0.35)

*Data source: `research/conversion_experiments/results/scaling_exponents.json`*

Empirically, on this CPU run and within this sequence length range, the BSBR model shows a **worse scaling exponent** (0.35) compared to the original model (0.08). This contradicts the theoretical expectation that BSBR should scale closer to O(n) while standard attention scales closer to O(n^2). The low exponent for the original transformer suggests that for these sequence lengths on CPU, other factors (like constant overheads) dominate the runtime, and the quadratic nature of attention hasn't become the bottleneck yet.

## Model Quality Evaluation

### Perplexity

We evaluated the perplexity of the original and BSBR-converted models on the Wikitext dataset.

- **Original GPT-2 Perplexity**: 52.29
- **BSBR GPT-2 Perplexity**: 2,073,936.50

*Data source: `research/conversion_experiments/results/quality_results_wikitext.json`*

The BSBR-converted model exhibits **extremely high perplexity**, indicating a significant degradation in language modeling performance compared to the original model. This suggests the conversion process drastically alters the model's learned representations and ability to predict the next token effectively **without fine-tuning**.

## Output Similarity Analysis

We evaluated how closely the outputs of BSBR models match those of the original models using randomly generated inputs.

### Hidden State Similarity

- **Average Cosine Similarity**: -0.0025 ± 0.27
- **Average Mean Squared Error**: 50.52 ± 10.46
- **Average KL Divergence**: 17.14 ± 4.86

*Data source: `research/conversion_experiments/results/similarity_metrics.json`*

These metrics indicate **significant divergence** between the hidden state outputs. The average cosine similarity near zero (with high variance) suggests the output vectors are largely uncorrelated or pointing in different directions in the embedding space. The high MSE and KL divergence further confirm the dissimilarity.

### Next-Token Prediction Agreement

We tested how often both models predict the same tokens for the next position:

- **Top-1 Agreement**: 0.00%
- **Top-5 Agreement**: 0.10%
- **Top-10 Agreement**: 1.11%

*Data source: `research/conversion_experiments/results/agreement_rates.json`*

The agreement rates are **extremely low**, especially for Top-1. This confirms that the BSBR conversion significantly alters the model's predictive behavior at the output layer. The models rarely agree on the most likely next token.

## Discussion

Our evaluation reveals important insights about direct BSBR conversion (without fine-tuning):

### Performance Considerations

1. **CPU Performance Penalty**: On CPU and for sequences up to 1024, direct conversion leads to slower inference and worse empirical scaling compared to the original GPT-2.
2. **Sequence Length / Hardware**: BSBR's potential efficiency benefits likely require much longer sequences (>1024) and/or parallel hardware like GPUs to overcome its inherent overhead.

### Quality and Similarity Trade-offs

1. **Drastic Behavioral Change**: The conversion significantly degrades perplexity and leads to very different hidden states and next-token predictions.
2. **Fine-tuning is Crucial**: To achieve meaningful performance on any task, BSBR-converted models **must** be fine-tuned after conversion.

## Recommendations for BSBR Conversion

Based on our findings:

1. **Do Not Expect Out-of-the-Box Equivalence**: Converting a pre-trained model to BSBR does not yield a model with similar behavior or quality without further training.
2. **Fine-tuning is Mandatory**: Budget for fine-tuning after conversion to adapt the model to the new architecture and recover task performance.
3. **Target Use Cases**: Consider BSBR for scenarios where:
   * Training models from scratch on very long sequences is the goal.
   * Fine-tuning an existing model for efficiency on long sequences is acceptable, understanding that the behavior will change.
   * Memory savings during training/inference for very long sequences are paramount.
4. **Evaluate Post-Fine-tuning**: Meaningful evaluation requires comparing a *fine-tuned* BSBR model against the original or a similarly fine-tuned baseline.

## Conclusion

Directly converting a pre-trained GPT-2 model to the BSBR architecture results in a model that is slower (on CPU, <=1024 tokens), has significantly worse perplexity, and produces highly dissimilar outputs compared to the original. The theoretical efficiency benefits of BSBR are not realized under these conditions.

This underscores that BSBR is not a drop-in replacement for standard attention in pre-trained models. It represents a different architectural paradigm. Its strengths likely lie in training models designed for long sequences from the start or in scenarios where significant fine-tuning is performed after conversion to adapt the model to the structural changes and potentially unlock efficiency gains at very large scales.

Future work should focus on:
1. Evaluating the performance of *fine-tuned* BSBR models.
2. Testing on GPU hardware and with much longer sequences (>>1024 tokens).
3. Training BSBR models from scratch for long-context tasks.

## Appendix: Visualizations

*Note: Paths are relative to the `docs/` directory.*

**Performance:**

![Inference Time vs Sequence Length](../research/conversion_experiments/results/inference_time_vs_seq_length.png)
![Throughput vs Sequence Length](../research/conversion_experiments/results/throughput_vs_seq_length.png)
![Speedup vs Sequence Length](../research/conversion_experiments/results/speedup_vs_seq_length.png)
![Scaling Exponent Analysis](../research/conversion_experiments/results/scaling_exponent_analysis.png)

**Quality & Similarity:**

![Perplexity Comparison](../research/conversion_experiments/results/perplexity_comparison_wikitext.png)
![Metrics Distribution](../research/conversion_experiments/results/metrics_distribution.png)
![Agreement Rates](../research/conversion_experiments/results/agreement_rates.png) 