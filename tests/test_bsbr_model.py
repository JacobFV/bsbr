import pytest
import torch
from bsbr.bsbr import BSBRLayer, BSBRModel


def test_bsbr_layer_init(device, model_config):
    """Test the initialization of BSBRLayer."""
    layer = BSBRLayer(
        hidden_dim=model_config["hidden_dim"],
        num_heads=model_config["num_heads"],
        chunk_size=model_config["chunk_size"],
        ff_dim=model_config["ff_dim"],
        dropout=model_config["dropout"],
        compression_factor=model_config["compression_factor"]
    )
    
    # Check that the layer components are properly initialized
    assert hasattr(layer, "attention")
    assert hasattr(layer, "layer_norm1")
    assert hasattr(layer, "layer_norm2")
    assert hasattr(layer, "ff")


def test_bsbr_layer_forward(device, sample_batch, model_config):
    """Test the forward pass of BSBRLayer."""
    hidden_states = sample_batch["hidden_states"]
    attention_mask = sample_batch["attention_mask"]
    
    layer = BSBRLayer(
        hidden_dim=model_config["hidden_dim"],
        num_heads=model_config["num_heads"],
        chunk_size=model_config["chunk_size"],
        ff_dim=model_config["ff_dim"],
        dropout=model_config["dropout"],
        compression_factor=model_config["compression_factor"]
    ).to(device)
    
    # Forward pass
    output = layer(hidden_states, attention_mask)
    
    # Check output shape
    assert output.shape == hidden_states.shape
    
    # Check output is not NaN or inf
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_bsbr_model_init(device, model_config):
    """Test the initialization of BSBRModel."""
    model = BSBRModel(
        vocab_size=model_config["vocab_size"],
        hidden_dim=model_config["hidden_dim"],
        num_layers=model_config["num_layers"],
        num_heads=model_config["num_heads"],
        chunk_size=model_config["chunk_size"],
        ff_dim=model_config["ff_dim"],
        dropout=model_config["dropout"],
        compression_factor=model_config["compression_factor"]
    )
    
    # Check that the model components are properly initialized
    assert hasattr(model, "embedding")
    assert hasattr(model, "pos_encoding")
    assert hasattr(model, "layers")
    assert hasattr(model, "layer_norm")
    
    # Check number of layers
    assert len(model.layers) == model_config["num_layers"]


def test_bsbr_model_forward(device, sample_batch, model_config):
    """Test the forward pass of BSBRModel."""
    input_ids = sample_batch["input_ids"]
    attention_mask = sample_batch["attention_mask"]
    
    model = BSBRModel(
        vocab_size=model_config["vocab_size"],
        hidden_dim=model_config["hidden_dim"],
        num_layers=model_config["num_layers"],
        num_heads=model_config["num_heads"],
        chunk_size=model_config["chunk_size"],
        ff_dim=model_config["ff_dim"],
        dropout=model_config["dropout"],
        compression_factor=model_config["compression_factor"]
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


def test_bsbr_model_with_different_sequence_lengths(device, model_config):
    """Test BSBRModel with different sequence lengths."""
    model = BSBRModel(
        vocab_size=model_config["vocab_size"],
        hidden_dim=model_config["hidden_dim"],
        num_layers=model_config["num_layers"],
        num_heads=model_config["num_heads"],
        chunk_size=model_config["chunk_size"],
        ff_dim=model_config["ff_dim"],
        dropout=model_config["dropout"],
        compression_factor=model_config["compression_factor"]
    ).to(device)
    
    batch_size = 2
    
    # Test with different sequence lengths
    seq_lengths = [
        model_config["chunk_size"] // 2,  # Shorter than chunk
        model_config["chunk_size"],       # Equal to chunk
        model_config["chunk_size"] * 2,   # Multiple chunks
        model_config["chunk_size"] * 2 + 1  # Not divisible by chunk size
    ]
    
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


def test_bsbr_model_with_attention_mask(device, model_config):
    """Test BSBRModel with a non-trivial attention mask."""
    model = BSBRModel(
        vocab_size=model_config["vocab_size"],
        hidden_dim=model_config["hidden_dim"],
        num_layers=model_config["num_layers"],
        num_heads=model_config["num_heads"],
        chunk_size=model_config["chunk_size"],
        ff_dim=model_config["ff_dim"],
        dropout=model_config["dropout"],
        compression_factor=model_config["compression_factor"]
    ).to(device)
    
    batch_size = 2
    seq_len = model_config["chunk_size"] * 3
    
    input_ids = torch.randint(0, model_config["vocab_size"], (batch_size, seq_len), device=device)
    
    # Create a mask with some tokens masked out
    attention_mask = torch.ones(batch_size, seq_len, device=device)
    attention_mask[:, seq_len // 3:2 * seq_len // 3] = 0
    
    # Forward pass
    output = model(input_ids, attention_mask)
    
    # Check output shape
    expected_shape = (batch_size, seq_len, model_config["hidden_dim"])
    assert output.shape == expected_shape
    
    # Check output is not NaN or inf
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_model_without_compression(device, model_config):
    """Test BSBRModel without state compression."""
    # Create a copy of the config without compression
    config_no_compression = model_config.copy()
    config_no_compression["compression_factor"] = None
    
    model = BSBRModel(
        vocab_size=config_no_compression["vocab_size"],
        hidden_dim=config_no_compression["hidden_dim"],
        num_layers=config_no_compression["num_layers"],
        num_heads=config_no_compression["num_heads"],
        chunk_size=config_no_compression["chunk_size"],
        ff_dim=config_no_compression["ff_dim"],
        dropout=config_no_compression["dropout"],
        compression_factor=config_no_compression["compression_factor"]
    ).to(device)
    
    batch_size = 2
    seq_len = config_no_compression["chunk_size"] * 2
    
    input_ids = torch.randint(0, config_no_compression["vocab_size"], (batch_size, seq_len), device=device)
    attention_mask = torch.ones(batch_size, seq_len, device=device)
    
    # Forward pass
    output = model(input_ids, attention_mask)
    
    # Check output shape
    expected_shape = (batch_size, seq_len, config_no_compression["hidden_dim"])
    assert output.shape == expected_shape
    
    # Check output is not NaN or inf
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any() 