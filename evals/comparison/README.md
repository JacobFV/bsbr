# Transformer Architecture Evaluation

This directory contains scripts for evaluating different efficient transformer architectures:

1. **BSBR (Block Sparse Attention with Block Retrieval)** - Combines in-chunk standard attention with between-chunk block retrieval
2. **Standard Transformer** - The classic attention mechanism with full quadratic attention matrix
3. **Linear Transformer** - Removes softmax to achieve linear complexity in sequence length
4. **DeltaNet** - Enhances Linear Transformer with a removal component for better memory management
5. **Sliding Window Transformer** - Restricts attention to a fixed window for O(n·w) complexity
6. **Hopfield Network** - Associative memory-based attention inspired by modern Hopfield Networks
7. **GAU (Gated Attention Unit)** - Uses chunk-based parallelism with gating for efficient processing

## Evaluation Results

### Computational Complexity

Based on our empirical testing with sequence lengths up to 1024 tokens:

| Model | Empirical Complexity | R-squared | Time at n=1024 (seconds) | Memory (MB) |
|-------|----------------------|-----------|--------------------------|-------------|
| BSBR | O(n^0.81) ≈ O(n) | 0.8473 | 0.428 | 7.67 |
| Standard | O(n^1.45) ≈ O(n log n) | 0.9644 | 3.285 | 23.92 |
| Linear | O(n^0.79) ≈ O(n) | 0.9821 | 1.862 | 6.41 |
| DeltaNet | O(n^0.73) ≈ O(n) | 0.9516 | 9.960 | 6.41 |
| SlidingWindow | O(n^1.21) ≈ O(n log n) | 0.9679 | 1.834 | 12.65 |
| Hopfield | O(n^0.78) ≈ O(n) | 0.9809 | 2.143 | 6.68 |
| GAU | O(n^0.86) ≈ O(n) | 0.9815 | 1.528 | 8.12 |

### Relative Performance

BSBR significantly outperforms other models in inference time:

| Model | Avg Slowdown vs BSBR | Min Slowdown | Max Slowdown | Slowdown at n=1024 |
|-------|----------------------|--------------|--------------|-------------------|
| Linear | 6.19x | 4.35x | 8.45x | 4.35x |
| DeltaNet | 31.17x | 23.27x | 40.79x | 23.27x |
| Standard | 4.51x | 1.30x | 7.68x | 7.68x |
| SlidingWindow | 4.34x | 3.21x | 5.90x | 4.29x |
| Hopfield | 7.17x | 5.01x | 9.79x | 5.01x |
| GAU | 4.34x | 3.21x | 5.90x | 3.57x |

### Memory Usage

Memory usage varies significantly across architectures, especially at longer sequence lengths:

| Model | Fixed Memory Cost | Scaling with n | Memory at n=1024 (MB) |
|-------|------------------|----------------|----------------------|
| BSBR | Higher | Minimal | 7.67 |
| Standard | Low | O(n²) | 23.92 |
| Linear | Low | Minimal | 6.41 |
| DeltaNet | Low | Minimal | 6.41 |
| SlidingWindow | Low | O(n·w) | 12.65 |
| Hopfield | Low | Minimal | 6.68 |
| GAU | Medium | O(n·log n) | 8.12 |

### Parameter Counts

Models with additional components or projections tend to have higher parameter counts:

| Model | Parameters (Millions) | Relative to Base |
|-------|----------------------|------------------|
| BSBR | 6.0M | 1.7x |
| Standard | 3.6M | 1.0x |
| Linear | 3.6M | 1.0x |
| DeltaNet | 3.6M | 1.0x |
| SlidingWindow | 3.6M | 1.0x |
| Hopfield | 3.6M | 1.0x |
| GAU | 4.4M | 1.2x |

## Architecture Analysis

### BSBR (Block Sparse with Block Retrieval)
- **Complexity**: O(n) empirical, theoretically O(n + n²/B)
- **Strengths**: 
  - Fastest inference time overall
  - Well-balanced between efficiency and expressiveness
  - Effective compression of long-range information
- **Weaknesses**: 
  - Higher parameter count
  - Slightly higher memory baseline

### Standard Transformer
- **Complexity**: O(n²) theoretical, O(n·log n) empirical in our tests
- **Strengths**: 
  - Full context visibility
  - Maximum expressiveness
  - Well-established architecture
- **Weaknesses**: 
  - Memory usage grows quadratically with sequence length
  - Computation becomes prohibitive for long sequences

### Linear Transformer
- **Complexity**: O(n) theoretical and empirical
- **Strengths**: 
  - True linear scaling
  - Low memory usage
  - Stateful representation
- **Weaknesses**: 
  - Less expressive than models with softmax
  - Performance gap compared to BSBR

### DeltaNet
- **Complexity**: O(n) theoretical and empirical
- **Strengths**: 
  - Improved memory management over Linear Transformer
  - Better handling of long-term dependencies
- **Weaknesses**: 
  - Highest computational overhead
  - Slowest inference time
  - Complex update rule

### Sliding Window Transformer
- **Complexity**: O(n·w) theoretical, approximately O(n·log n) empirically
- **Strengths**: 
  - Simple, intuitive approach
  - Good balance between efficiency and expressiveness
- **Weaknesses**: 
  - Limited context window
  - Cannot capture very long-range dependencies

### Hopfield Network
- **Complexity**: O(n) theoretical and empirical
- **Strengths**: 
  - Associative memory capabilities
  - Good pattern completion/retrieval
  - Similar efficiency to Linear Transformer
- **Weaknesses**: 
  - Complex energy-based formulation
  - Moderate performance penalty over Linear

### Gated Attention Unit (GAU)
- **Complexity**: O(n·log n) theoretical, close to O(n) empirically
- **Strengths**: 
  - Competitive inference speed (3rd fastest)
  - Chunk-based processing with gating for improved expressiveness
  - Good balance of efficiency and effectiveness
- **Weaknesses**: 
  - More complex implementation
  - Higher parameter count than linear variants

## Running Evaluations

To run the model comparison:

```bash
# Run all models with default settings
python evals/compare_models.py

# Run specific models
python evals/compare_models.py --models BSBR Linear Hopfield GAU

# Test with different sequence lengths
python evals/compare_models.py --seq_lengths 256 512 1024 2048

# Test with different model parameters
python evals/compare_models.py --hidden_dim 128 --num_layers 4 --num_heads 2
```

To analyze results:

```bash
# Use example data
python evals/analyze_results.py --use_example_data

# Provide your own sequence lengths if different from default
python evals/analyze_results.py --seq_lengths 128 256 512 1024 2048
```

This will generate complexity analysis plots:

1. `complexity_analysis.png`: Shows the inference time vs sequence length with theoretical curves
2. `complexity_loglog.png`: Log-log plot to help visualize the asymptotic complexity

## Conclusion

Our evaluation reveals that while all architectures perform well for short sequences, they diverge significantly as sequence length increases:

- **BSBR** provides the best overall performance, making it ideal for practical applications.
- **Linear variants** (Linear, DeltaNet, Hopfield) offer theoretically optimal scaling but at a constant performance cost.
- **Window-based approaches** (SlidingWindow, GAU) provide a good compromise between full attention and pure linear scaling.
- **Standard Transformer** remains most expressive but least efficient for long sequences.

For production systems requiring both efficiency and expressivity, BSBR emerges as the preferred architecture, though GAU also shows promise for specific applications where gating mechanisms can be beneficial. 