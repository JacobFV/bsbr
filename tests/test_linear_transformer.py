import pytest
import torch
from bsbr_extras.linear_transformer import LinearAttention, LinearTransformerLayer, LinearTransformerModel


def test_linear_attention_init(device, model_config):
    """Test the initialization of LinearAttention."""
    attention = LinearAttention(
        hidden_dim=model_config["hidden_dim"],
        num_heads=model_config["num_heads"],
        dropout=model_config["dropout"]
    )
    
    assert attention.hidden_dim == model_config["hidden_dim"]
    assert attention.num_heads == model_config["num_heads"]
    assert attention.head_dim == model_config["hidden_dim"] // model_config["num_heads"]
    assert attention.dropout == model_config["dropout"]


def test_linear_attention_forward(device, sample_batch, model_config):
    """Test the forward pass of LinearAttention."""
    hidden_states = sample_batch["hidden_states"]
    attention_mask = sample_batch["attention_mask"]
    
    attention = LinearAttention(
        hidden_dim=model_config["hidden_dim"],
        num_heads=model_config["num_heads"],
        dropout=model_config["dropout"]
    ).to(device)
    
    # Forward pass
    output, state = attention(hidden_states, attention_mask)
    
    # Check output shape
    assert output.shape == hidden_states.shape
    
    # Check state shape
    batch_size = sample_batch["batch_size"]
    head_dim = model_config["hidden_dim"] // model_config["num_heads"]
    assert state.shape == (batch_size, model_config["num_heads"], head_dim, head_dim)
    
    # Check output is not NaN or inf
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_linear_transformer_layer_init(device, model_config):
    """Test the initialization of LinearTransformerLayer."""
    layer = LinearTransformerLayer(
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


def test_linear_transformer_layer_forward(device, sample_batch, model_config):
    """Test the forward pass of LinearTransformerLayer."""
    hidden_states = sample_batch["hidden_states"]
    attention_mask = sample_batch["attention_mask"]
    
    layer = LinearTransformerLayer(
        hidden_dim=model_config["hidden_dim"],
        num_heads=model_config["num_heads"],
        ff_dim=model_config["ff_dim"],
        dropout=model_config["dropout"]
    ).to(device)
    
    # Forward pass
    output, state = layer(hidden_states, attention_mask)
    
    # Check output shape
    assert output.shape == hidden_states.shape
    
    # Check state shape
    batch_size = sample_batch["batch_size"]
    head_dim = model_config["hidden_dim"] // model_config["num_heads"]
    assert state.shape == (batch_size, model_config["num_heads"], head_dim, head_dim)
    
    # Check output is not NaN or inf
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_linear_transformer_model_init(device, model_config):
    """Test the initialization of LinearTransformerModel."""
    model = LinearTransformerModel(
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


def test_linear_transformer_model_forward(device, sample_batch, model_config):
    """Test the forward pass of LinearTransformerModel."""
    input_ids = sample_batch["input_ids"]
    attention_mask = sample_batch["attention_mask"]
    
    model = LinearTransformerModel(
        vocab_size=model_config["vocab_size"],
        hidden_dim=model_config["hidden_dim"],
        num_layers=model_config["num_layers"],
        num_heads=model_config["num_heads"],
        ff_dim=model_config["ff_dim"],
        dropout=model_config["dropout"]
    ).to(device)
    
    # Forward pass
    output, states = model(input_ids, attention_mask)
    
    # Check output shape
    batch_size, seq_len = input_ids.shape
    expected_shape = (batch_size, seq_len, model_config["hidden_dim"])
    assert output.shape == expected_shape
    
    # Check states
    assert len(states) == model_config["num_layers"]
    head_dim = model_config["hidden_dim"] // model_config["num_heads"]
    for state in states:
        assert state.shape == (batch_size, model_config["num_heads"], head_dim, head_dim)
    
    # Check output is not NaN or inf
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_linear_transformer_with_state(device, sample_batch, model_config):
    """Test LinearTransformerModel with provided initial states."""
    input_ids = sample_batch["input_ids"]
    attention_mask = sample_batch["attention_mask"]
    batch_size = sample_batch["batch_size"]
    
    model = LinearTransformerModel(
        vocab_size=model_config["vocab_size"],
        hidden_dim=model_config["hidden_dim"],
        num_layers=model_config["num_layers"],
        num_heads=model_config["num_heads"],
        ff_dim=model_config["ff_dim"],
        dropout=model_config["dropout"]
    ).to(device)
    
    # First get states from a forward pass
    _, initial_states = model(input_ids, attention_mask)
    
    # Then do another forward pass with the obtained states
    output, new_states = model(input_ids, attention_mask, initial_states)
    
    # Check output shape
    batch_size, seq_len = input_ids.shape
    expected_shape = (batch_size, seq_len, model_config["hidden_dim"])
    assert output.shape == expected_shape
    
    # Check that states have changed
    for i in range(len(initial_states)):
        assert not torch.allclose(initial_states[i], new_states[i])
    
    # Check output is not NaN or inf
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any() 