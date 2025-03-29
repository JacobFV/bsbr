import pytest
import torch
from bsbr.bsbr import BSBRAttention


def test_bsbr_attention_init(device, model_config):
    """Test the initialization of BSBRAttention."""
    attention = BSBRAttention(
        hidden_dim=model_config["hidden_dim"],
        num_heads=model_config["num_heads"],
        chunk_size=model_config["chunk_size"],
        dropout=model_config["dropout"],
        compression_factor=model_config["compression_factor"]
    )
    
    assert attention.hidden_dim == model_config["hidden_dim"]
    assert attention.num_heads == model_config["num_heads"]
    assert attention.head_dim == model_config["hidden_dim"] // model_config["num_heads"]
    assert attention.chunk_size == model_config["chunk_size"]
    assert attention.dropout == model_config["dropout"]
    assert attention.compression_factor == model_config["compression_factor"]


def test_bsbr_attention_forward(device, sample_batch, model_config):
    """Test the forward pass of BSBRAttention."""
    hidden_states = sample_batch["hidden_states"]
    attention_mask = sample_batch["attention_mask"]
    
    attention = BSBRAttention(
        hidden_dim=model_config["hidden_dim"],
        num_heads=model_config["num_heads"],
        chunk_size=model_config["chunk_size"],
        dropout=model_config["dropout"],
        compression_factor=model_config["compression_factor"]
    ).to(device)
    
    # Forward pass
    output = attention(hidden_states, attention_mask)
    
    # Check output shape
    assert output.shape == hidden_states.shape
    
    # Check output is not NaN or inf
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_create_masks(device, model_config):
    """Test the mask creation in BSBRAttention."""
    attention = BSBRAttention(
        hidden_dim=model_config["hidden_dim"],
        num_heads=model_config["num_heads"],
        chunk_size=model_config["chunk_size"],
        dropout=model_config["dropout"]
    ).to(device)
    
    seq_len = 128
    m_in, m_out = attention._create_masks(seq_len)
    
    # Check dimensions
    assert m_in.shape == (seq_len, seq_len)
    
    # Calculate expected number of chunks
    num_chunks = (seq_len + attention.chunk_size - 1) // attention.chunk_size
    assert m_out.shape == (num_chunks, num_chunks)
    
    # Check causal property (upper triangular)
    for i in range(num_chunks):
        for j in range(num_chunks):
            if i > j:  # Should be masked (0)
                assert m_out[i, j] == 0
            else:  # Should be visible (1)
                assert m_out[i, j] == 1


def test_compute_chunk_states(device, sample_batch, model_config):
    """Test the computation of chunk states in BSBRAttention."""
    batch_size = sample_batch["batch_size"]
    seq_len = sample_batch["seq_len"]
    hidden_dim = sample_batch["hidden_dim"]
    
    attention = BSBRAttention(
        hidden_dim=hidden_dim,
        num_heads=model_config["num_heads"],
        chunk_size=model_config["chunk_size"],
        dropout=model_config["dropout"],
        compression_factor=model_config["compression_factor"]
    ).to(device)
    
    # Create fake chunked keys and values
    head_dim = hidden_dim // model_config["num_heads"]
    num_chunks = (seq_len + model_config["chunk_size"] - 1) // model_config["chunk_size"]
    
    keys = torch.randn(
        batch_size, 
        model_config["num_heads"], 
        num_chunks, 
        model_config["chunk_size"], 
        head_dim, 
        device=device
    )
    values = torch.randn(
        batch_size, 
        model_config["num_heads"], 
        num_chunks, 
        model_config["chunk_size"], 
        head_dim, 
        device=device
    )
    
    # Compute chunk states
    states = attention.compute_chunk_states(keys, values)
    
    # Check output shape
    compressed_dim = hidden_dim // model_config["compression_factor"]
    expected_shape = (batch_size, model_config["num_heads"], num_chunks, compressed_dim)
    assert states.shape == expected_shape
    
    # Check output is not NaN or inf
    assert not torch.isnan(states).any()
    assert not torch.isinf(states).any()


def test_various_sequence_lengths(device, model_config):
    """Test BSBRAttention with various sequence lengths relative to chunk size."""
    attention = BSBRAttention(
        hidden_dim=model_config["hidden_dim"],
        num_heads=model_config["num_heads"],
        chunk_size=model_config["chunk_size"],
        dropout=model_config["dropout"]
    ).to(device)
    
    batch_size = 2
    
    # Test cases: sequence lengths shorter than, equal to, and longer than chunk size
    seq_lengths = [
        model_config["chunk_size"] // 2,  # Shorter
        model_config["chunk_size"],       # Equal
        model_config["chunk_size"] * 2,   # Longer
        model_config["chunk_size"] * 2 + 1  # Not divisible
    ]
    
    for seq_len in seq_lengths:
        hidden_states = torch.randn(batch_size, seq_len, model_config["hidden_dim"], device=device)
        attention_mask = torch.ones(batch_size, seq_len, device=device)
        
        # Forward pass
        output = attention(hidden_states, attention_mask)
        
        # Check output shape
        assert output.shape == hidden_states.shape
        
        # Check output is not NaN or inf
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


def test_attention_with_mask(device, model_config):
    """Test BSBRAttention with a non-trivial attention mask."""
    attention = BSBRAttention(
        hidden_dim=model_config["hidden_dim"],
        num_heads=model_config["num_heads"],
        chunk_size=model_config["chunk_size"],
        dropout=model_config["dropout"]
    ).to(device)
    
    batch_size = 2
    seq_len = model_config["chunk_size"] * 3
    hidden_states = torch.randn(batch_size, seq_len, model_config["hidden_dim"], device=device)
    
    # Create a mask with some tokens masked out
    attention_mask = torch.ones(batch_size, seq_len, device=device)
    attention_mask[:, seq_len // 3:2 * seq_len // 3] = 0
    
    # Forward pass
    output = attention(hidden_states, attention_mask)
    
    # Check output shape
    assert output.shape == hidden_states.shape
    
    # Check output is not NaN or inf
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any() 