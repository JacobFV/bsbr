import pytest
import torch
from bsbr_extras.gau import ChunkGatedAttentionUnit, GAULayer, GAUModel


def test_chunk_gated_attention_unit_init(device, model_config):
    """Test the initialization of ChunkGatedAttentionUnit."""
    chunk_size = 32
    expansion_factor = 2
    gau = ChunkGatedAttentionUnit(
        hidden_dim=model_config["hidden_dim"],
        chunk_size=chunk_size,
        expansion_factor=expansion_factor,
        dropout=model_config["dropout"]
    )
    
    assert gau.hidden_dim == model_config["hidden_dim"]
    assert gau.chunk_size == chunk_size
    assert gau.expansion_factor == expansion_factor
    assert gau.expanded_dim == model_config["hidden_dim"] * expansion_factor
    assert gau.dropout == model_config["dropout"]


def test_chunk_gated_attention_unit_forward(device, sample_batch, model_config):
    """Test the forward pass of ChunkGatedAttentionUnit."""
    hidden_states = sample_batch["hidden_states"]
    attention_mask = sample_batch["attention_mask"]
    chunk_size = 32
    expansion_factor = 2
    
    gau = ChunkGatedAttentionUnit(
        hidden_dim=model_config["hidden_dim"],
        chunk_size=chunk_size,
        expansion_factor=expansion_factor,
        dropout=model_config["dropout"]
    ).to(device)
    
    # Forward pass
    output = gau(hidden_states, attention_mask)
    
    # Check output shape
    assert output.shape == hidden_states.shape
    
    # Check output is not NaN or inf
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_gau_layer_init(device, model_config):
    """Test the initialization of GAULayer."""
    chunk_size = 32
    expansion_factor = 2
    layer = GAULayer(
        hidden_dim=model_config["hidden_dim"],
        chunk_size=chunk_size,
        ff_dim=model_config["ff_dim"],
        expansion_factor=expansion_factor,
        dropout=model_config["dropout"]
    )
    
    # Check that the layer components are properly initialized
    assert hasattr(layer, "attention")
    assert hasattr(layer, "layer_norm1")
    assert hasattr(layer, "layer_norm2")
    assert hasattr(layer, "ff")
    assert layer.attention.chunk_size == chunk_size
    assert layer.attention.expansion_factor == expansion_factor


def test_gau_layer_forward(device, sample_batch, model_config):
    """Test the forward pass of GAULayer."""
    hidden_states = sample_batch["hidden_states"]
    attention_mask = sample_batch["attention_mask"]
    chunk_size = 32
    expansion_factor = 2
    
    layer = GAULayer(
        hidden_dim=model_config["hidden_dim"],
        chunk_size=chunk_size,
        ff_dim=model_config["ff_dim"],
        expansion_factor=expansion_factor,
        dropout=model_config["dropout"]
    ).to(device)
    
    # Forward pass
    output = layer(hidden_states, attention_mask)
    
    # Check output shape
    assert output.shape == hidden_states.shape
    
    # Check output is not NaN or inf
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_gau_model_init(device, model_config):
    """Test the initialization of GAUModel."""
    chunk_size = 32
    expansion_factor = 2
    model = GAUModel(
        vocab_size=model_config["vocab_size"],
        hidden_dim=model_config["hidden_dim"],
        num_layers=model_config["num_layers"],
        chunk_size=chunk_size,
        ff_dim=model_config["ff_dim"],
        expansion_factor=expansion_factor,
        dropout=model_config["dropout"]
    )
    
    # Check that the model components are properly initialized
    assert hasattr(model, "embedding")
    assert hasattr(model, "pos_encoding")
    assert hasattr(model, "layers")
    assert hasattr(model, "layer_norm")
    
    # Check number of layers
    assert len(model.layers) == model_config["num_layers"]
    
    # Check chunk size and expansion factor were passed correctly
    for layer in model.layers:
        assert layer.attention.chunk_size == chunk_size
        assert layer.attention.expansion_factor == expansion_factor


def test_gau_model_forward(device, sample_batch, model_config):
    """Test the forward pass of GAUModel."""
    input_ids = sample_batch["input_ids"]
    attention_mask = sample_batch["attention_mask"]
    chunk_size = 32
    expansion_factor = 2
    
    model = GAUModel(
        vocab_size=model_config["vocab_size"],
        hidden_dim=model_config["hidden_dim"],
        num_layers=model_config["num_layers"],
        chunk_size=chunk_size,
        ff_dim=model_config["ff_dim"],
        expansion_factor=expansion_factor,
        dropout=model_config["dropout"]
    ).to(device)
    
    # Forward pass
    output = model(input_ids, attention_mask)
    
    # Check output shape
    batch_size, seq_len = input_ids.shape
    expected_shape = (batch_size, seq_len, model_config["hidden_dim"])
    assert output.shape == expected_shape
    
    # Check output is not NaN or inf
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_gau_with_different_expansion_factors(device, sample_batch, model_config):
    """Test GAUModel with different expansion factors."""
    input_ids = sample_batch["input_ids"]
    attention_mask = sample_batch["attention_mask"]
    chunk_size = 32
    
    # Try different expansion factors
    expansion_factors = [1, 2, 4]
    
    for expansion_factor in expansion_factors:
        model = GAUModel(
            vocab_size=model_config["vocab_size"],
            hidden_dim=model_config["hidden_dim"],
            num_layers=model_config["num_layers"],
            chunk_size=chunk_size,
            ff_dim=model_config["ff_dim"],
            expansion_factor=expansion_factor,
            dropout=model_config["dropout"]
        ).to(device)
        
        # Forward pass
        output = model(input_ids, attention_mask)
        
        # Check output is not NaN or inf
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


def test_gau_with_different_chunk_sizes(device, sample_batch, model_config):
    """Test GAUModel with different chunk sizes."""
    input_ids = sample_batch["input_ids"]
    attention_mask = sample_batch["attention_mask"]
    expansion_factor = 2
    
    # Try different chunk sizes
    chunk_sizes = [16, 32, 64]
    
    for chunk_size in chunk_sizes:
        model = GAUModel(
            vocab_size=model_config["vocab_size"],
            hidden_dim=model_config["hidden_dim"],
            num_layers=model_config["num_layers"],
            chunk_size=chunk_size,
            ff_dim=model_config["ff_dim"],
            expansion_factor=expansion_factor,
            dropout=model_config["dropout"]
        ).to(device)
        
        # Forward pass
        output = model(input_ids, attention_mask)
        
        # Check output is not NaN or inf
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


def test_gau_with_different_sequence_lengths(device, model_config):
    """Test GAUModel with different sequence lengths."""
    chunk_size = 32
    expansion_factor = 2
    model = GAUModel(
        vocab_size=model_config["vocab_size"],
        hidden_dim=model_config["hidden_dim"],
        num_layers=model_config["num_layers"],
        chunk_size=chunk_size,
        ff_dim=model_config["ff_dim"],
        expansion_factor=expansion_factor,
        dropout=model_config["dropout"]
    ).to(device)
    
    batch_size = 2
    
    # Test with different sequence lengths, including those that are not divisible by chunk_size
    seq_lengths = [24, chunk_size, chunk_size * 2, chunk_size + 5]
    
    for seq_len in seq_lengths:
        input_ids = torch.randint(0, model_config["vocab_size"], (batch_size, seq_len), device=device)
        attention_mask = torch.ones(batch_size, seq_len, device=device)
        
        # Forward pass
        output = model(input_ids, attention_mask)
        
        # Check output shape
        expected_shape = (batch_size, seq_len, model_config["hidden_dim"])
        assert output.shape == expected_shape
        
        # Check output is not NaN or inf
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


def test_gau_chunking_logic(device, model_config):
    """Test the chunking logic in ChunkGatedAttentionUnit."""
    hidden_dim = model_config["hidden_dim"]
    chunk_size = 32
    expansion_factor = 2
    batch_size = 2
    
    gau = ChunkGatedAttentionUnit(
        hidden_dim=hidden_dim,
        chunk_size=chunk_size,
        expansion_factor=expansion_factor,
        dropout=model_config["dropout"]
    ).to(device)
    
    # Test with a sequence length that's not divisible by chunk_size
    seq_len = chunk_size * 2 + 5
    hidden_states = torch.randn(batch_size, seq_len, hidden_dim, device=device)
    
    # Call _chunk_sequence method
    chunked, num_chunks, original_seq_len = gau._chunk_sequence(hidden_states)
    
    # Check chunking results
    assert original_seq_len == seq_len
    assert num_chunks == 3  # Math.ceil(69 / 32) = 3
    
    # Check chunked shape (should include padding)
    expected_padded_length = num_chunks * chunk_size
    assert chunked.shape == (batch_size, num_chunks, chunk_size, hidden_dim)
    
    # Process through the model and verify it handles the padding correctly
    output = gau(hidden_states)
    
    # Check that the output has the original sequence length
    assert output.shape == (batch_size, seq_len, hidden_dim)
    
    # Check output is not NaN or inf
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any() 