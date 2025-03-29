# Core Concepts

This guide explains the fundamental concepts behind BSBR (Block Sparse Attention with Block Retrieval). Understanding these concepts will help you make better use of the library and customize it for your needs.

## Attention Mechanisms

### Standard Transformer Attention

The standard transformer attention mechanism computes attention scores between all pairs of tokens in a sequence:

```math
Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V
```

This leads to O(n²) complexity in both computation and memory, where n is the sequence length.

### BSBR's Approach

BSBR addresses this scalability issue by combining two types of attention:

1. **Within-chunk attention**: Standard attention within fixed-size chunks
2. **Between-chunk attention**: Efficient block retrieval between chunks

```math
O = Q ⊙ softmax(RH^T · M_out)F.repeat(B) + softmax(QK^T · M_in)V
```

Where:
- Q, K, V: Query, Key, Value matrices
- R, H: Meta queries and keys for chunk-level attention
- F: State vectors (flattened K^T·V for each chunk)
- M_in: Block diagonal mask
- M_out: Causal mask for chunk-level attention

## Chunking Strategy

### Chunk Size Selection

The chunk size (B) is a crucial hyperparameter that affects:

1. **Memory Usage**: Larger chunks use more memory but provide better local context
2. **Computation Time**: Smaller chunks are faster but may miss important long-range dependencies
3. **Model Expressivity**: Chunk size affects how well the model can capture different types of relationships

### Chunk Overlap

BSBR supports overlapping chunks to improve information flow between adjacent chunks:

```python
model = BSBRModel(
    chunk_size=128,
    chunk_overlap=32,  # 25% overlap between chunks
    ...
)
```

## Block Retrieval

### Meta Attention

Between chunks, BSBR uses a meta-attention mechanism to efficiently retrieve information:

1. **State Compression**: Each chunk's information is compressed into a state vector
2. **Meta Queries**: Special queries that operate at the chunk level
3. **Efficient Retrieval**: O(n/B) complexity for chunk-level attention

### Compression Factor

The compression factor (c) controls how much information is preserved in chunk states:

```python
model = BSBRModel(
    compression_factor=4,  # Compress chunk states by 4x
    ...
)
```

Higher compression factors:
- Reduce memory usage
- Speed up computation
- May lose some fine-grained information

## Memory Management

### Gradient Checkpointing

BSBR supports gradient checkpointing to trade computation for memory:

```python
model = BSBRModel(
    use_checkpointing=True,  # Enable gradient checkpointing
    ...
)
```

### Memory-Efficient Attention

The implementation includes several memory optimizations:

1. **Sparse Attention**: Only compute attention for relevant token pairs
2. **State Reuse**: Reuse chunk states across layers
3. **Efficient Masking**: Optimized attention masks for causal language modeling

## Performance Characteristics

### Computational Complexity

BSBR achieves near-linear complexity in sequence length:

- Within-chunk: O(n·B) where B is chunk size
- Between-chunk: O(n + n²/B)
- Overall: O(n) for fixed chunk size

### Memory Usage

Memory consumption scales linearly with sequence length:

- Within-chunk: O(n·B)
- Between-chunk: O(n/B)
- State vectors: O(n/c) where c is compression factor

## Best Practices

### Model Configuration

Recommended configurations for different use cases:

1. **Short Sequences** (n < 512):
   ```python
   model = BSBRModel(
       chunk_size=64,
       compression_factor=2,
       ...
   )
   ```

2. **Medium Sequences** (512 ≤ n < 2048):
   ```python
   model = BSBRModel(
       chunk_size=128,
       compression_factor=4,
       ...
   )
   ```

3. **Long Sequences** (n ≥ 2048):
   ```python
   model = BSBRModel(
       chunk_size=256,
       compression_factor=8,
       use_checkpointing=True,
       ...
   )
   ```

### Training Tips

1. **Learning Rate**: Use slightly higher learning rates than standard transformers
2. **Warmup**: Longer warmup periods may be needed for very long sequences
3. **Gradient Clipping**: Monitor gradients and clip if necessary
4. **Batch Size**: Adjust based on available memory and sequence length

## Next Steps

1. Learn about the [BSBR Model](bsbr-model.md) implementation details
2. Explore [Additional Models](additional-models.md) for comparison
3. See [Evaluation](evaluation.md) results and benchmarks 