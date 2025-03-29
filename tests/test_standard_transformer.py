import pytest
import torch
from bsbr_extras.standard_transformer import StandardAttention, StandardTransformerLayer, StandardTransformerModel


def test_standard_attention_init(device, model_config):
    """Test the initialization of StandardAttention."""
    attention = StandardAttention(
        hidden_dim=model_config["hidden_dim"],
        num_heads=model_config["num_heads"],
        dropout=model_config["dropout"]
    )
    
    assert attention.hidden_dim == model_config["hidden_dim"]
    assert attention.num_heads == model_config["num_heads"]
    assert attention.head_dim == model_config["hidden_dim"] // model_config["num_heads"]
    assert attention.dropout == model_config["dropout"]


def test_standard_attention_forward(device, sample_batch, model_config):
    """Test the forward pass of StandardAttention."""
    hidden_states = sample_batch["hidden_states"]
    attention_mask = sample_batch["attention_mask"]
    
    attention = StandardAttention(
        hidden_dim=model_config["hidden_dim"],
        num_heads=model_config["num_heads"],
        dropout=model_config["dropout"]
    ).to(device)
    
    # Forward pass
    output = attention(hidden_states, attention_mask)
    
    # Check output shape
    assert output.shape == hidden_states.shape
    
    # Check output is not NaN or inf
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_standard_transformer_layer_init(device, model_config):
    """Test the initialization of StandardTransformerLayer."""
    layer = StandardTransformerLayer(
        hidden_dim=model_config["hidden_dim"],
        num_heads=model_config["num_heads"],
        ff_dim=model_config["ff_dim"],
        dropout=model_config["dropout"]
    )
    
    # Check that the layer components are properly initialized
    assert hasattr(layer, "attention")
    assert hasattr(layer, "layer_norm1")
    assert hasattr(layer, "layer_norm2")
    assert hasattr(layer, "ff")


def test_standard_transformer_layer_forward(device, sample_batch, model_config):
    """Test the forward pass of StandardTransformerLayer."""
    hidden_states = sample_batch["hidden_states"]
    attention_mask = sample_batch["attention_mask"]
    
    layer = StandardTransformerLayer(
        hidden_dim=model_config["hidden_dim"],
        num_heads=model_config["num_heads"],
        ff_dim=model_config["ff_dim"],
        dropout=model_config["dropout"]
    ).to(device)
    
    # Forward pass
    output = layer(hidden_states, attention_mask)
    
    # Check output shape
    assert output.shape == hidden_states.shape
    
    # Check output is not NaN or inf
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_standard_transformer_model_init(device, model_config):
    """Test the initialization of StandardTransformerModel."""
    model = StandardTransformerModel(
        vocab_size=model_config["vocab_size"],
        hidden_dim=model_config["hidden_dim"],
        num_layers=model_config["num_layers"],
        num_heads=model_config["num_heads"],
        ff_dim=model_config["ff_dim"],
        dropout=model_config["dropout"]
    )
    
    # Check that the model components are properly initialized
    assert hasattr(model, "embedding")
    assert hasattr(model, "pos_encoding")
    assert hasattr(model, "layers")
    assert hasattr(model, "layer_norm")
    
    # Check number of layers
    assert len(model.layers) == model_config["num_layers"]


def test_standard_transformer_model_forward(device, sample_batch, model_config):
    """Test the forward pass of StandardTransformerModel."""
    input_ids = sample_batch["input_ids"]
    attention_mask = sample_batch["attention_mask"]
    
    model = StandardTransformerModel(
        vocab_size=model_config["vocab_size"],
        hidden_dim=model_config["hidden_dim"],
        num_layers=model_config["num_layers"],
        num_heads=model_config["num_heads"],
        ff_dim=model_config["ff_dim"],
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


def test_standard_transformer_with_different_sequence_lengths(device, model_config):
    """Test StandardTransformerModel with different sequence lengths."""
    model = StandardTransformerModel(
        vocab_size=model_config["vocab_size"],
        hidden_dim=model_config["hidden_dim"],
        num_layers=model_config["num_layers"],
        num_heads=model_config["num_heads"],
        ff_dim=model_config["ff_dim"],
        dropout=model_config["dropout"]
    ).to(device)
    
    batch_size = 2
    
    # Test with different sequence lengths
    seq_lengths = [32, 64, 128]
    
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


def test_standard_transformer_with_attention_mask(device, model_config):
    """Test StandardTransformerModel with a non-trivial attention mask."""
    model = StandardTransformerModel(
        vocab_size=model_config["vocab_size"],
        hidden_dim=model_config["hidden_dim"],
        num_layers=model_config["num_layers"],
        num_heads=model_config["num_heads"],
        ff_dim=model_config["ff_dim"],
        dropout=model_config["dropout"]
    ).to(device)
    
    batch_size = 2
    seq_len = 64
    
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