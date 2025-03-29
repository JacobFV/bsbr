# Transformer Architecture Evaluation

This directory contains scripts for evaluating different efficient transformer architectures:

1. **BSBR (Block Sparse Attention with Block Retrieval)** - Combines in-chunk standard attention with between-chunk block retrieval
2. **Linear Transformer** - Removes softmax to achieve linear complexity in sequence length
3. **DeltaNet** - Enhances Linear Transformer with a removal component for better memory management

## Evaluation Results

### Computational Complexity

Based on our empirical testing with sequence lengths up to 2048 tokens:

| Model | Empirical Complexity | R-squared | Time at n=2048 (seconds) |
|-------|----------------------|-----------|--------------------------|
| BSBR | O(n^1.14) ≈ O(n) | 0.9544 | 0.428 |
| LinearTransformer | O(n^0.95) ≈ O(n) | 0.9965 | 1.862 |  
| DeltaNet | O(n^0.99) ≈ O(n) | 0.9981 | 9.960 |

### Relative Performance

BSBR significantly outperforms both Linear Transformer and DeltaNet in inference time:

| Model | Avg Slowdown vs BSBR | Min Slowdown | Max Slowdown | Slowdown at n=2048 |
|-------|----------------------|--------------|--------------|-------------------|
| LinearTransformer | 6.19x | 4.35x | 8.45x | 4.35x |
| DeltaNet | 31.17x | 23.27x | 40.79x | 23.27x |

### Memory Usage

Memory usage is similar across all models, with BSBR using slightly more memory due to storing block states:

| Model | Memory at n=2048 (MB) |
|-------|----------------------|
| BSBR | 7.67 |
| LinearTransformer | 6.41 |
| DeltaNet | 6.41 |

### Analysis

- **BSBR**: Offers the best inference performance, with a complexity close to linear. The block structure provides efficient computation while maintaining effective attention.

- **LinearTransformer**: Shows true linear scaling but at a constant factor penalty compared to BSBR. The absence of softmax simplifies computation but may affect expressiveness.

- **DeltaNet**: Also shows linear scaling but with a much larger constant factor penalty. The removal component improves memory management but adds significant computational overhead.

## Running Evaluations

To run the model comparison:

```bash
python evals/compare_models.py --hidden_dim 128 --seq_lengths 256 512 1024 2048 --n_tokens 3
```

To analyze results:

```bash
python evals/analyze_results.py
```

This will generate complexity analysis plots:

1. `complexity_analysis.png`: Shows the inference time vs sequence length with theoretical curves
2. `complexity_loglog.png`: Log-log plot to help visualize the asymptotic complexity

## Conclusion

BSBR achieves the best performance among the three architectures for autoregressive generation tasks. Its block structure provides a good balance between the computational efficiency of linear transformers and the expressiveness of standard attention mechanisms.

For very long sequences (beyond what we tested), the theoretical advantages of the Linear Transformer and DeltaNet may become more apparent, but for practical sequence lengths up to thousands of tokens, BSBR is the most efficient choice. 