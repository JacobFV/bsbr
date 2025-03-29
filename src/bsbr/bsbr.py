import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class BSBRAttention(nn.Module):
    """
    Block Sparse Attention with Block Retrieval (BSBR) implementation.
    
    This attention mechanism splits the input sequence into chunks and processes them in two ways:
    1. Within each chunk: Standard attention with softmax
    2. Between chunks: Block retrieval using meta queries and keys
    
    Args:
        hidden_dim (int): Hidden dimension size
        num_heads (int): Number of attention heads
        chunk_size (int): Size of each chunk (B)
        dropout (float): Dropout probability
        compression_factor (Optional[int]): If provided, compresses the state dimension from d_0^2 to d_1^2
    """
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        chunk_size: int,
        dropout: float = 0.1,
        compression_factor: Optional[int] = None
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.chunk_size = chunk_size
        self.dropout = dropout
        
        # Standard attention projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Meta query and key projections for chunk-level attention
        self.meta_r_proj = nn.Linear(hidden_dim, hidden_dim)  # R (retriever)
        self.meta_h_proj = nn.Linear(hidden_dim, hidden_dim)  # H (hash)
        
        # Optional compression for state vectors
        self.compression_factor = compression_factor
        if compression_factor is not None:
            compressed_dim = hidden_dim // compression_factor
            # Fix: The compress_proj should map from head_dim*head_dim to compressed_dim
            state_dim = self.head_dim * self.head_dim
            self.compress_proj = nn.Linear(state_dim, compressed_dim)
            self.decompress_proj = nn.Linear(compressed_dim, state_dim)
        
        self.dropout_layer = nn.Dropout(dropout)
        
    def _reshape_for_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape input for multi-head attention."""
        batch_size, seq_len, _ = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
    
    def _create_masks(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create M_in (block diagonal) and M_out (causal) masks."""
        # Calculate number of chunks
        num_chunks = math.ceil(seq_len / self.chunk_size)
        
        # Create block diagonal mask for within-chunk attention
        m_in = torch.zeros(seq_len, seq_len, device=self.q_proj.weight.device)
        
        for i in range(num_chunks):
            start_idx = i * self.chunk_size
            end_idx = min((i + 1) * self.chunk_size, seq_len)
            # Create causal mask within each chunk
            chunk_size = end_idx - start_idx
            chunk_mask = torch.triu(torch.ones(chunk_size, chunk_size, device=m_in.device), diagonal=0)
            m_in[start_idx:end_idx, start_idx:end_idx] = chunk_mask
        
        # Create causal mask for between-chunk attention
        chunk_indices = torch.arange(num_chunks, device=m_in.device)
        m_out = torch.triu(torch.ones(num_chunks, num_chunks, device=m_in.device), diagonal=0)
        
        return m_in, m_out
    
    def compute_chunk_states(
        self, 
        keys: torch.Tensor, 
        values: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute F states by flattening K^T·V for each chunk.
        
        Args:
            keys: [batch, num_heads, num_chunks, chunk_size, head_dim]
            values: [batch, num_heads, num_chunks, chunk_size, head_dim]
            
        Returns:
            states: [batch, num_heads, num_chunks, state_dim]
        """
        batch_size, num_heads, num_chunks, chunk_size, head_dim = keys.size()
        
        # For each chunk, compute K^T·V
        # Reshape to [batch*num_heads*num_chunks, chunk_size, head_dim]
        keys_flat = keys.reshape(-1, chunk_size, head_dim)
        values_flat = values.reshape(-1, chunk_size, head_dim)
        
        # Compute K^T·V for each chunk
        # [chunk_size, head_dim] @ [chunk_size, head_dim] -> [head_dim, head_dim]
        states = torch.bmm(
            keys_flat.transpose(1, 2),  # [B*H*C, head_dim, chunk_size]
            values_flat                 # [B*H*C, chunk_size, head_dim]
        )
        
        # Flatten the state matrices to vectors
        states_flat = states.reshape(batch_size, num_heads, num_chunks, -1)
        
        # Apply optional compression
        if self.compression_factor is not None:
            states_flat = self.compress_proj(states_flat)
        
        return states_flat
    
    def forward(
        self, 
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of the BSBR attention mechanism.
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_dim]
            attention_mask: Optional attention mask of shape [batch_size, seq_len]
            
        Returns:
            output: Processed tensor of shape [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len, _ = hidden_states.size()
        num_chunks = math.ceil(seq_len / self.chunk_size)
        
        # Create masks for within-chunk and between-chunk attention
        m_in, m_out = self._create_masks(seq_len)
        if attention_mask is not None:
            # Fix: Expand attention_mask to match m_in dimensions
            attention_mask_expanded = attention_mask.unsqueeze(1).unsqueeze(2).expand(-1, -1, seq_len, -1)
            m_in = m_in.unsqueeze(0).unsqueeze(0).expand(batch_size, self.num_heads, -1, -1)
            m_in = m_in * attention_mask_expanded
        
        # Standard projections for Q, K, V
        q = self._reshape_for_heads(self.q_proj(hidden_states))  # [batch, num_heads, seq_len, head_dim]
        k = self._reshape_for_heads(self.k_proj(hidden_states))  # [batch, num_heads, seq_len, head_dim]
        v = self._reshape_for_heads(self.v_proj(hidden_states))  # [batch, num_heads, seq_len, head_dim]
        
        # Reshape to chunks
        # Pad sequence length to be divisible by chunk_size
        padding = (self.chunk_size - seq_len % self.chunk_size) % self.chunk_size
        if padding > 0:
            q = F.pad(q, (0, 0, 0, padding))
            k = F.pad(k, (0, 0, 0, padding))
            v = F.pad(v, (0, 0, 0, padding))
            padded_seq_len = seq_len + padding
        else:
            padded_seq_len = seq_len
            
        # Reshape to chunks [batch, num_heads, num_chunks, chunk_size, head_dim]
        q_chunks = q.view(batch_size, self.num_heads, num_chunks, self.chunk_size, self.head_dim)
        k_chunks = k.view(batch_size, self.num_heads, num_chunks, self.chunk_size, self.head_dim)
        v_chunks = v.view(batch_size, self.num_heads, num_chunks, self.chunk_size, self.head_dim)
        
        # Meta projections for chunk-level attention
        # Fix: First, get a representative hidden state for each chunk
        # Create a padded version of hidden_states for safe reshaping
        if padding > 0:
            padded_hidden = F.pad(hidden_states, (0, 0, 0, padding))
        else:
            padded_hidden = hidden_states
            
        # Reshape to [batch, num_chunks, chunk_size, hidden_dim]
        chunk_hidden = padded_hidden.view(batch_size, num_chunks, self.chunk_size, self.hidden_dim)
        
        # Get the last token of each chunk as representative
        # For the final chunk, make sure we don't index beyond valid tokens
        chunk_repr = []
        for i in range(num_chunks):
            if i == num_chunks - 1 and padding > 0:
                # For the last chunk, get the last valid token
                last_valid_idx = self.chunk_size - padding - 1
                if last_valid_idx < 0:  # In case the last chunk is all padding
                    last_valid_idx = 0
                repr_i = chunk_hidden[:, i, last_valid_idx, :]
            else:
                # For full chunks, get the last token
                repr_i = chunk_hidden[:, i, -1, :]
            chunk_repr.append(repr_i)
            
        # Stack to [batch, num_chunks, hidden_dim]
        chunk_repr = torch.stack(chunk_repr, dim=1)
        
        # Project to meta queries and keys
        r = self.meta_r_proj(chunk_repr)  # [batch, num_chunks, hidden_dim]
        h = self.meta_h_proj(chunk_repr)  # [batch, num_chunks, hidden_dim]
        
        # Reshape for multi-head attention
        r = r.view(batch_size, num_chunks, self.num_heads, self.head_dim).transpose(1, 2)  # [batch, num_heads, num_chunks, head_dim]
        h = h.view(batch_size, num_chunks, self.num_heads, self.head_dim).transpose(1, 2)  # [batch, num_heads, num_chunks, head_dim]
        
        # Compute chunk states (F)
        f = self.compute_chunk_states(k_chunks, v_chunks)  # [batch, num_heads, num_chunks, state_dim]
        
        # Between-chunk attention: softmax(R·H^T)·F
        # Calculate R·H^T
        chunk_attn_scores = torch.matmul(r, h.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [batch, num_heads, num_chunks, num_chunks]
        
        # Apply causal mask for between-chunk attention
        m_out = m_out.unsqueeze(0).unsqueeze(0).expand(batch_size, self.num_heads, -1, -1)
        chunk_attn_scores = chunk_attn_scores.masked_fill(m_out == 0, -1e9)
        
        # Apply softmax
        chunk_attn_probs = F.softmax(chunk_attn_scores, dim=-1)  # [batch, num_heads, num_chunks, num_chunks]
        chunk_attn_probs = self.dropout_layer(chunk_attn_probs)
        
        # Calculate retrieved states
        retrieved_states = torch.matmul(chunk_attn_probs, f)  # [batch, num_heads, num_chunks, state_dim]
        
        # Decompress if needed
        if self.compression_factor is not None:
            retrieved_states = self.decompress_proj(retrieved_states)
            
        # Expand retrieved states to match each position in the chunk
        # First reshape retrieved_states to [batch, num_heads, num_chunks, head_dim, head_dim]
        retrieved_states = retrieved_states.view(batch_size, self.num_heads, num_chunks, self.head_dim, self.head_dim)
        
        # Multiply query with retrieved states: q_chunks @ retrieved_states
        # For each position in each chunk, calculate q @ retrieved_state
        # [batch, num_heads, num_chunks, chunk_size, head_dim] @ [batch, num_heads, num_chunks, head_dim, head_dim]
        # -> [batch, num_heads, num_chunks, chunk_size, head_dim]
        long_term_output = torch.matmul(q_chunks, retrieved_states)
        
        # Within-chunk attention
        # Calculate local attention scores
        local_attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [batch, num_heads, padded_seq_len, padded_seq_len]
        
        # Fix: Apply the correct block diagonal causal mask
        # First create a mask for the padded sequence
        padded_m_in = torch.zeros(padded_seq_len, padded_seq_len, device=m_in.device)
        
        for i in range(num_chunks):
            start_idx = i * self.chunk_size
            end_idx = (i + 1) * self.chunk_size
            # Create causal mask within each chunk
            chunk_mask = torch.triu(torch.ones(self.chunk_size, self.chunk_size, device=padded_m_in.device), diagonal=0)
            padded_m_in[start_idx:end_idx, start_idx:end_idx] = chunk_mask
            
        # Expand dimensions for broadcasting
        expanded_mask = padded_m_in.unsqueeze(0).unsqueeze(0).expand(batch_size, self.num_heads, -1, -1)
        
        # Apply the mask
        local_attn_scores = local_attn_scores.masked_fill(expanded_mask == 0, -1e9)
        
        # Apply softmax
        local_attn_probs = F.softmax(local_attn_scores, dim=-1)
        local_attn_probs = self.dropout_layer(local_attn_probs)
        
        # Apply attention
        local_output = torch.matmul(local_attn_probs, v)  # [batch, num_heads, padded_seq_len, head_dim]
        
        # Reshape long_term_output to match local_output
        long_term_output = long_term_output.view(batch_size, self.num_heads, padded_seq_len, self.head_dim)
        
        # Combine long-term and local outputs
        output = long_term_output + local_output
        
        # Remove padding if added
        if padding > 0:
            output = output[:, :, :seq_len, :]
            
        # Reshape back to [batch, seq_len, hidden_dim]
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        
        # Final projection
        output = self.out_proj(output)
        
        return output


class BSBRLayer(nn.Module):
    """
    A single BSBR layer with attention and feed-forward networks.
    
    Args:
        hidden_dim (int): Hidden dimension size
        num_heads (int): Number of attention heads
        chunk_size (int): Size of each chunk
        ff_dim (int): Feed-forward intermediate dimension
        dropout (float): Dropout probability
        compression_factor (Optional[int]): If provided, compresses the state dimension
    """
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        chunk_size: int,
        ff_dim: int,
        dropout: float = 0.1,
        compression_factor: Optional[int] = None
    ):
        super().__init__()
        self.attention = BSBRAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            chunk_size=chunk_size,
            dropout=dropout,
            compression_factor=compression_factor
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
        """Forward pass for a single BSBR layer."""
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


class BSBRModel(nn.Module):
    """
    Full BSBR model stacking multiple BSBR layers.
    
    Args:
        vocab_size (int): Vocabulary size for embedding layer
        hidden_dim (int): Hidden dimension size
        num_layers (int): Number of BSBR layers
        num_heads (int): Number of attention heads
        chunk_size (int): Size of each chunk
        ff_dim (int): Feed-forward intermediate dimension
        dropout (float): Dropout probability
        compression_factor (Optional[int]): If provided, compresses the state dimension
    """
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        chunk_size: int,
        ff_dim: int,
        dropout: float = 0.1,
        compression_factor: Optional[int] = None
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_encoding = PositionalEncoding(hidden_dim, dropout)
        
        self.layers = nn.ModuleList([
            BSBRLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                chunk_size=chunk_size,
                ff_dim=ff_dim,
                dropout=dropout,
                compression_factor=compression_factor
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
        Forward pass for the full BSBR model.
        
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
