import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List


class ChunkGatedAttentionUnit(nn.Module):
    """
    Chunk-based Gated Attention Unit (GAU) implementation.
    
    This implements the Gated Attention Unit with chunk-based parallelism as mentioned in the thesis.
    It's based on the paper: https://arxiv.org/pdf/2202.10447
    
    Args:
        hidden_dim (int): Hidden dimension size
        chunk_size (int): Size of chunks for parallel processing
        expansion_factor (int): Expansion factor for intermediate computations
        dropout (float): Dropout probability
    """
    def __init__(
        self,
        hidden_dim: int,
        chunk_size: int,
        expansion_factor: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.chunk_size = chunk_size
        self.expansion_factor = expansion_factor
        self.dropout = dropout
        
        # Expansion dimension
        self.expanded_dim = hidden_dim * expansion_factor
        
        # Projections for queries, keys, values and gates
        self.q_proj = nn.Linear(hidden_dim, self.expanded_dim)
        self.k_proj = nn.Linear(hidden_dim, self.expanded_dim)
        self.v_proj = nn.Linear(hidden_dim, self.expanded_dim)
        self.gate_proj = nn.Linear(hidden_dim, self.expanded_dim)
        
        self.out_proj = nn.Linear(self.expanded_dim, hidden_dim)
        self.dropout_layer = nn.Dropout(dropout)
        
    def _chunk_sequence(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """Split sequence into chunks of size chunk_size."""
        batch_size, seq_len, _ = x.size()
        num_chunks = math.ceil(seq_len / self.chunk_size)
        
        # Pad sequence if needed
        if seq_len % self.chunk_size != 0:
            padding = self.chunk_size - (seq_len % self.chunk_size)
            x = F.pad(x, (0, 0, 0, padding))
        
        # Reshape and chunk
        # [batch_size, num_chunks, chunk_size, dim]
        chunked = x.view(batch_size, num_chunks, self.chunk_size, -1)
        return chunked, num_chunks, seq_len
        
    def forward(
        self, 
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of the Gated Attention Unit.
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_dim]
            attention_mask: Optional attention mask of shape [batch_size, seq_len]
            
        Returns:
            output: Processed tensor of shape [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len, _ = hidden_states.size()
        
        # Project inputs
        q = self.q_proj(hidden_states)  # [batch, seq_len, expanded_dim]
        k = self.k_proj(hidden_states)  # [batch, seq_len, expanded_dim]
        v = self.v_proj(hidden_states)  # [batch, seq_len, expanded_dim]
        gate = self.gate_proj(hidden_states)  # [batch, seq_len, expanded_dim]
        gate = torch.sigmoid(gate)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Expand mask for broadcasting
            mask = attention_mask.unsqueeze(-1)  # [batch, seq_len, 1]
            k = k * mask
            v = v * mask
        
        # Chunk the inputs for parallel processing
        chunked_q, num_chunks, original_seq_len = self._chunk_sequence(q)
        chunked_k, _, _ = self._chunk_sequence(k)
        chunked_v, _, _ = self._chunk_sequence(v)
        chunked_gate, _, _ = self._chunk_sequence(gate)
        
        # Create causal mask for chunks
        chunk_mask = torch.triu(
            torch.ones(num_chunks, num_chunks, dtype=torch.bool, device=hidden_states.device),
            diagonal=1
        )
        
        # Process within each chunk and between chunks
        outputs = []
        
        # For each chunk of queries
        for i in range(num_chunks):
            # Current chunk of queries
            chunk_q = chunked_q[:, i]  # [batch, chunk_size, expanded_dim]
            chunk_gate = chunked_gate[:, i]  # [batch, chunk_size, expanded_dim]
            
            # Initialize chunk output
            chunk_output = torch.zeros_like(chunk_q)
            
            # Process within-chunk attention (parallelly)
            # Attend to all positions within the same chunk
            local_k = chunked_k[:, i]  # [batch, chunk_size, expanded_dim]
            local_v = chunked_v[:, i]  # [batch, chunk_size, expanded_dim]
            
            # Create causal mask within chunk
            local_causal_mask = torch.triu(
                torch.ones(self.chunk_size, self.chunk_size, dtype=torch.bool, device=hidden_states.device),
                diagonal=1
            )
            
            # Compute local attention (standard QK^T V pattern)
            # [batch, chunk_size, chunk_size]
            local_attn = torch.bmm(chunk_q, local_k.transpose(1, 2)) / math.sqrt(self.expanded_dim)
            local_attn = local_attn.masked_fill(local_causal_mask.unsqueeze(0), float('-inf'))
            local_attn = F.softmax(local_attn, dim=-1)
            local_attn = self.dropout_layer(local_attn)
            
            # [batch, chunk_size, expanded_dim]
            local_output = torch.bmm(local_attn, local_v)
            
            # Add local outputs to chunk output
            chunk_output = chunk_output + local_output
            
            # Process cross-chunk attention
            # For chunks before the current chunk
            for j in range(i):
                # Skip if masked
                if chunk_mask[i, j]:
                    continue
                
                # Get previous chunk keys and values
                prev_k = chunked_k[:, j]  # [batch, chunk_size, expanded_dim]
                prev_v = chunked_v[:, j]  # [batch, chunk_size, expanded_dim]
                
                # Compute cross-chunk attention (K^T V first for efficiency)
                # [batch, expanded_dim, expanded_dim]
                kv_state = torch.bmm(prev_k.transpose(1, 2), prev_v)
                
                # [batch, chunk_size, expanded_dim]
                cross_output = torch.bmm(chunk_q, kv_state)
                
                # Add cross-chunk outputs to chunk output
                chunk_output = chunk_output + cross_output
            
            # Apply gating
            gated_output = chunk_output * chunk_gate
            
            # Collect chunk output
            outputs.append(gated_output)
        
        # Concatenate all chunk outputs
        output = torch.cat(outputs, dim=1)
        
        # Trim to original sequence length if padded
        if seq_len != original_seq_len:
            output = output[:, :original_seq_len, :]
        else:
            # Make sure we're always returning exactly the original sequence length
            output = output[:, :original_seq_len, :]
        
        # Final projection
        output = self.out_proj(output)
        output = self.dropout_layer(output)
        
        return output


class GAULayer(nn.Module):
    """
    A single Gated Attention Unit layer with feed-forward networks.
    
    Args:
        hidden_dim (int): Hidden dimension size
        chunk_size (int): Size of chunks for parallel processing
        ff_dim (int): Feed-forward intermediate dimension
        expansion_factor (int): Expansion factor for GAU
        dropout (float): Dropout probability
    """
    def __init__(
        self,
        hidden_dim: int,
        chunk_size: int,
        ff_dim: int,
        expansion_factor: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.attention = ChunkGatedAttentionUnit(
            hidden_dim=hidden_dim,
            chunk_size=chunk_size,
            expansion_factor=expansion_factor,
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
        """Forward pass for a single GAU layer."""
        # Pre-LayerNorm architecture
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.attention(hidden_states, attention_mask)
        # Shape check to ensure attention output matches residual
        assert hidden_states.shape == residual.shape, f"Attention output shape {hidden_states.shape} doesn't match residual shape {residual.shape}"
        hidden_states = residual + hidden_states
        
        # Feed-forward network
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.ff(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


class GAUModel(nn.Module):
    """
    Full Gated Attention Unit model stacking multiple GAU layers.
    
    Args:
        vocab_size (int): Vocabulary size for embedding layer
        hidden_dim (int): Hidden dimension size
        num_layers (int): Number of GAU layers
        chunk_size (int): Size of chunks for parallel processing
        ff_dim (int): Feed-forward intermediate dimension
        expansion_factor (int): Expansion factor for GAU
        dropout (float): Dropout probability
    """
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        num_layers: int,
        chunk_size: int,
        ff_dim: int,
        expansion_factor: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_encoding = PositionalEncoding(hidden_dim, dropout)
        
        self.layers = nn.ModuleList([
            GAULayer(
                hidden_dim=hidden_dim,
                chunk_size=chunk_size,
                ff_dim=ff_dim,
                expansion_factor=expansion_factor,
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
        Forward pass for the full GAU model.
        
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