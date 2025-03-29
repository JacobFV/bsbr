# BSBR Extras Module

This module contains additional transformer architectures for evaluation and research purposes.

## StandardTransformer

```python
class StandardTransformer(nn.Module):
    """Standard transformer with full attention mechanism.
    
    Implements the classic transformer architecture with O(n²) complexity.
    
    Args:
        vocab_size (int): Size of the vocabulary
        hidden_dim (int): Hidden dimension of the model
        num_layers (int): Number of transformer layers
        num_heads (int): Number of attention heads
        ff_dim (int): Feed-forward network dimension
        dropout (float): Dropout rate
    """
```

## LinearTransformer

```python
class LinearTransformer(nn.Module):
    """Linear complexity transformer using reformulated attention.
    
    Implements a transformer with O(n) complexity by removing softmax and using
    the associative property of matrix multiplication.
    
    Args:
        vocab_size (int): Size of the vocabulary
        hidden_dim (int): Hidden dimension of the model
        num_layers (int): Number of transformer layers
        num_heads (int): Number of attention heads
        ff_dim (int): Feed-forward network dimension
        dropout (float): Dropout rate
    """
```

## DeltaNet

```python
class DeltaNet(nn.Module):
    """Enhanced linear transformer with removal component.
    
    Implements a linear transformer with additional memory management through
    a removal component.
    
    Args:
        vocab_size (int): Size of the vocabulary
        hidden_dim (int): Hidden dimension of the model
        num_layers (int): Number of transformer layers
        num_heads (int): Number of attention heads
        ff_dim (int): Feed-forward network dimension
        dropout (float): Dropout rate
    """
```

## SlidingWindowTransformer

```python
class SlidingWindowTransformer(nn.Module):
    """Transformer with fixed context window attention.
    
    Implements a transformer that restricts attention to a fixed window size
    for O(n·w) complexity.
    
    Args:
        vocab_size (int): Size of the vocabulary
        hidden_dim (int): Hidden dimension of the model
        num_layers (int): Number of transformer layers
        num_heads (int): Number of attention heads
        window_size (int): Size of the attention window
        ff_dim (int): Feed-forward network dimension
        dropout (float): Dropout rate
    """
```

## HopfieldNetwork

```python
class HopfieldNetwork(nn.Module):
    """Memory-based attention inspired by modern Hopfield Networks.
    
    Implements a transformer using associative memory-based attention
    for pattern completion.
    
    Args:
        vocab_size (int): Size of the vocabulary
        hidden_dim (int): Hidden dimension of the model
        num_layers (int): Number of transformer layers
        num_heads (int): Number of attention heads
        ff_dim (int): Feed-forward network dimension
        dropout (float): Dropout rate
    """
```

## GAU

```python
class GAU(nn.Module):
    """Gated Attention Unit with chunk-based parallelism.
    
    Implements a transformer using gated attention units with chunk-based
    parallel processing.
    
    Args:
        vocab_size (int): Size of the vocabulary
        hidden_dim (int): Hidden dimension of the model
        num_layers (int): Number of transformer layers
        num_heads (int): Number of attention heads
        chunk_size (int): Size of chunks for parallel processing
        ff_dim (int): Feed-forward network dimension
        dropout (float): Dropout rate
    """
``` 