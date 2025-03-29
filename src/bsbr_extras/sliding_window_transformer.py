import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class SlidingWindowAttention(nn.Module):
    """
    Sliding Window Attention implementation.
    
    This attention mechanism limits the attention context to a fixed window size,
    achieving O(n·w) complexity where w is the window size.
    
    Args:
        hidden_dim (int): Hidden dimension size
        num_heads (int): Number of attention heads
        window_size (int): Size of the attention window to each side
        dropout (float): Dropout probability
    """
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        window_size: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.window_size = window_size
        self.dropout = dropout
        
        # Standard attention projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout_layer = nn.Dropout(dropout)

    def _reshape_for_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape input for multi-head attention."""
        batch_size, seq_len, _ = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
    def forward(
        self, 
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of the Sliding Window Attention mechanism.
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_dim]
            attention_mask: Optional attention mask of shape [batch_size, seq_len]
            
        Returns:
            output: Processed tensor of shape [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len, _ = hidden_states.size()
        
        # Standard projections for Q, K, V
        q = self._reshape_for_heads(self.q_proj(hidden_states))  # [batch, num_heads, seq_len, head_dim]
        k = self._reshape_for_heads(self.k_proj(hidden_states))  # [batch, num_heads, seq_len, head_dim]
        v = self._reshape_for_heads(self.v_proj(hidden_states))  # [batch, num_heads, seq_len, head_dim]
        
        # Apply attention mask to k and v if provided
        if attention_mask is not None:
            # Expand mask for broadcasting
            mask = attention_mask.unsqueeze(1).unsqueeze(-1)  # [batch, 1, seq_len, 1]
            k = k * mask
            v = v * mask
        
        # Calculate attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [batch, num_heads, seq_len, seq_len]
        
        # Create window mask
        window_mask = torch.ones_like(attn_scores, dtype=torch.bool)
        for i in range(seq_len):
            # Allow attention to a window of window_size tokens before the current token
            start = max(0, i - self.window_size)
            end = i + 1  # Include current token (causal)
            window_mask[:, :, i, start:end] = False
        
        # Apply sliding window mask (set everything outside window to -inf)
        attn_scores = attn_scores.masked_fill(window_mask, float('-inf'))
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Expand mask for broadcasting
            attn_mask = attention_mask.unsqueeze(1).unsqueeze(2).expand(batch_size, self.num_heads, seq_len, seq_len)
            # Create inverted mask (1 -> attend, 0 -> don't attend)
            attn_mask = (1.0 - attn_mask) * -10000.0  # Convert 0s to -inf and 1s to 0
            attn_scores = attn_scores + attn_mask
        
        # Apply softmax and dropout
        # Add a small epsilon to prevent NaN in softmax when all values in a row are -inf
        attn_scores = attn_scores.clamp(min=-1e4, max=1e4)  # Clamp to prevent overflow
        attn_probs = F.softmax(attn_scores, dim=-1)
        
        # If all attention weights are 0 (all -inf before softmax), then all will be NaN
        # Replace NaN values with 0s
        attn_probs = torch.nan_to_num(attn_probs, nan=0.0, posinf=0.0, neginf=0.0)
        
        attn_probs = self.dropout_layer(attn_probs)
        
        # Calculate output
        output = torch.matmul(attn_probs, v)  # [batch, num_heads, seq_len, head_dim]
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        
        # Final projection
        output = self.out_proj(output)
        output = self.dropout_layer(output)
        
        return output


class SlidingWindowTransformerLayer(nn.Module):
    """
    A single Sliding Window Transformer layer with attention and feed-forward networks.
    
    Args:
        hidden_dim (int): Hidden dimension size
        num_heads (int): Number of attention heads
        window_size (int): Size of the attention window
        ff_dim (int): Feed-forward intermediate dimension
        dropout (float): Dropout probability
    """
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        window_size: int,
        ff_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.attention = SlidingWindowAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            window_size=window_size,
            dropout=dropout
        )
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        
        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass for a single Sliding Window Transformer layer."""
        # Pre-LayerNorm architecture
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.attention(hidden_states, attention_mask)
        hidden_states = residual + hidden_states
        
        # Feed-forward network
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.ff(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


class SlidingWindowTransformerModel(nn.Module):
    """
    Full Sliding Window Transformer model stacking multiple transformer layers.
    
    Args:
        vocab_size (int): Vocabulary size for embedding layer
        hidden_dim (int): Hidden dimension size
        num_layers (int): Number of transformer layers
        num_heads (int): Number of attention heads
        window_size (int): Size of the attention window
        ff_dim (int): Feed-forward intermediate dimension
        dropout (float): Dropout probability
    """
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        window_size: int,
        ff_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_encoding = PositionalEncoding(hidden_dim, dropout)
        
        self.layers = nn.ModuleList([
            SlidingWindowTransformerLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                window_size=window_size,
                ff_dim=ff_dim,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for the full Sliding Window Transformer model.
        
        Args:
            input_ids: Token IDs of shape [batch_size, seq_len]
            attention_mask: Optional attention mask of shape [batch_size, seq_len]
            
        Returns:
            output: Processed tensor of shape [batch_size, seq_len, hidden_dim]
        """
        hidden_states = self.embedding(input_ids)
        hidden_states = self.pos_encoding(hidden_states)
        
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
            
        hidden_states = self.layer_norm(hidden_states)
        return hidden_states


class PositionalEncoding(nn.Module):
    """
    Positional encoding using sine and cosine functions.
    
    Args:
        hidden_dim (int): Hidden dimension size
        dropout (float): Dropout probability
        max_len (int): Maximum sequence length
    """
    def __init__(self, hidden_dim: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2) * -(math.log(10000.0) / hidden_dim)
        )
        
        pe = torch.zeros(max_len, hidden_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input tensor."""
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x) 