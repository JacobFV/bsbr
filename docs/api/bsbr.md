# BSBR Module

## BSBRModel

```python
class BSBRModel(nn.Module):
    """Block Sparse Attention with Block Retrieval model.
    
    A transformer model that combines standard attention within chunks with efficient
    block retrieval between chunks for processing long sequences.
    
    Args:
        vocab_size (int): Size of the vocabulary
        hidden_dim (int): Hidden dimension of the model
        num_layers (int): Number of transformer layers
        num_heads (int): Number of attention heads
        chunk_size (int): Size of chunks for block sparse attention
        ff_dim (int): Feed-forward network dimension
        dropout (float): Dropout rate
        compression_factor (int, optional): Factor to compress chunk states. Defaults to 1.
    """
```

### Methods

#### forward

```python
def forward(
    self,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    **kwargs
) -> torch.Tensor:
    """Forward pass of the model.
    
    Args:
        input_ids (torch.Tensor): Input token IDs of shape (batch_size, seq_len)
        attention_mask (torch.Tensor, optional): Attention mask of shape (batch_size, seq_len).
            Defaults to None.
        **kwargs: Additional arguments passed to the model.
    
    Returns:
        torch.Tensor: Output logits of shape (batch_size, seq_len, vocab_size)
    """
```

## BSBRAttention

```python
class BSBRAttention(nn.Module):
    """Block Sparse Attention with Block Retrieval.
    
    Implements the core attention mechanism combining standard attention within chunks
    with efficient block retrieval between chunks.
    
    Args:
        hidden_dim (int): Hidden dimension of the model
        num_heads (int): Number of attention heads
        chunk_size (int): Size of chunks for block sparse attention
        dropout (float): Dropout rate
        compression_factor (int, optional): Factor to compress chunk states. Defaults to 1.
    """
```

### Methods

#### forward

```python
def forward(
    self,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    **kwargs
) -> torch.Tensor:
    """Forward pass of the attention mechanism.
    
    Args:
        query (torch.Tensor): Query tensor of shape (batch_size, seq_len, hidden_dim)
        key (torch.Tensor): Key tensor of shape (batch_size, seq_len, hidden_dim)
        value (torch.Tensor): Value tensor of shape (batch_size, seq_len, hidden_dim)
        mask (torch.Tensor, optional): Attention mask. Defaults to None.
        **kwargs: Additional arguments passed to the attention mechanism.
    
    Returns:
        torch.Tensor: Output tensor of shape (batch_size, seq_len, hidden_dim)
    """
```

## BSBRLayer

```python
class BSBRLayer(nn.Module):
    """A complete transformer layer with BSBR attention and feed-forward network.
    
    Args:
        hidden_dim (int): Hidden dimension of the model
        num_heads (int): Number of attention heads
        chunk_size (int): Size of chunks for block sparse attention
        ff_dim (int): Feed-forward network dimension
        dropout (float): Dropout rate
        compression_factor (int, optional): Factor to compress chunk states. Defaults to 1.
    """
```

### Methods

#### forward

```python
def forward(
    self,
    x: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    **kwargs
) -> torch.Tensor:
    """Forward pass of the transformer layer.
    
    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, seq_len, hidden_dim)
        mask (torch.Tensor, optional): Attention mask. Defaults to None.
        **kwargs: Additional arguments passed to the layer.
    
    Returns:
        torch.Tensor: Output tensor of shape (batch_size, seq_len, hidden_dim)
    """
``` 