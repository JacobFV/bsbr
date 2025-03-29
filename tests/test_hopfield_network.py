import pytest
import torch
from bsbr_extras.hopfield_network import HopfieldAttention, HopfieldNetworkLayer, HopfieldNetworkModel


def test_hopfield_attention_init(device, model_config):
    """Test the initialization of HopfieldAttention."""
    temperature = 1.0
    attention = HopfieldAttention(
        hidden_dim=model_config["hidden_dim"],
        num_heads=model_config["num_heads"],
        temperature=temperature,
        dropout=model_config["dropout"]
    )
    
    assert attention.hidden_dim == model_config["hidden_dim"]
    assert attention.num_heads == model_config["num_heads"]
    assert attention.head_dim == model_config["hidden_dim"] // model_config["num_heads"]
    assert attention.temperature == temperature
    assert attention.dropout == model_config["dropout"]


def test_hopfield_attention_forward(device, sample_batch, model_config):
    """Test the forward pass of HopfieldAttention."""
    hidden_states = sample_batch["hidden_states"]
    attention_mask = sample_batch["attention_mask"]
    temperature = 1.0
    
    attention = HopfieldAttention(
        hidden_dim=model_config["hidden_dim"],
        num_heads=model_config["num_heads"],
        temperature=temperature,
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


def test_hopfield_network_layer_init(device, model_config):
    """Test the initialization of HopfieldNetworkLayer."""
    temperature = 1.0
    layer = HopfieldNetworkLayer(
        hidden_dim=model_config["hidden_dim"],
        num_heads=model_config["num_heads"],
        ff_dim=model_config["ff_dim"],
        temperature=temperature,
        dropout=model_config["dropout"]
    )
    
    # Check that the layer components are properly initialized
    assert hasattr(layer, "attention")
    assert hasattr(layer, "layer_norm1")
    assert hasattr(layer, "layer_norm2")
    assert hasattr(layer, "ff")
    assert layer.attention.temperature == temperature


def test_hopfield_network_layer_forward(device, sample_batch, model_config):
    """Test the forward pass of HopfieldNetworkLayer."""
    hidden_states = sample_batch["hidden_states"]
    attention_mask = sample_batch["attention_mask"]
    temperature = 1.0
    
    layer = HopfieldNetworkLayer(
        hidden_dim=model_config["hidden_dim"],
        num_heads=model_config["num_heads"],
        ff_dim=model_config["ff_dim"],
        temperature=temperature,
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


def test_hopfield_network_model_init(device, model_config):
    """Test the initialization of HopfieldNetworkModel."""
    temperature = 1.0
    model = HopfieldNetworkModel(
        vocab_size=model_config["vocab_size"],
        hidden_dim=model_config["hidden_dim"],
        num_layers=model_config["num_layers"],
        num_heads=model_config["num_heads"],
        ff_dim=model_config["ff_dim"],
        temperature=temperature,
        dropout=model_config["dropout"]
    )
    
    # Check that the model components are properly initialized
    assert hasattr(model, "embedding")
    assert hasattr(model, "pos_encoding")
    assert hasattr(model, "layers")
    assert hasattr(model, "layer_norm")
    
    # Check number of layers
    assert len(model.layers) == model_config["num_layers"]
    
    # Check temperature parameter was passed correctly
    for layer in model.layers:
        assert layer.attention.temperature == temperature


def test_hopfield_network_model_forward(device, sample_batch, model_config):
    """Test the forward pass of HopfieldNetworkModel."""
    input_ids = sample_batch["input_ids"]
    attention_mask = sample_batch["attention_mask"]
    temperature = 1.0
    
    model = HopfieldNetworkModel(
        vocab_size=model_config["vocab_size"],
        hidden_dim=model_config["hidden_dim"],
        num_layers=model_config["num_layers"],
        num_heads=model_config["num_heads"],
        ff_dim=model_config["ff_dim"],
        temperature=temperature,
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


def test_hopfield_network_with_state(device, sample_batch, model_config):
    """Test HopfieldNetworkModel with provided initial states."""
    input_ids = sample_batch["input_ids"]
    attention_mask = sample_batch["attention_mask"]
    batch_size = sample_batch["batch_size"]
    temperature = 1.0
    
    model = HopfieldNetworkModel(
        vocab_size=model_config["vocab_size"],
        hidden_dim=model_config["hidden_dim"],
        num_layers=model_config["num_layers"],
        num_heads=model_config["num_heads"],
        ff_dim=model_config["ff_dim"],
        temperature=temperature,
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
    
    # Check that states have been updated
    for i in range(len(initial_states)):
        # States should be different after seeing the same data again in associative memory
        assert not torch.allclose(initial_states[i], new_states[i], rtol=1e-3, atol=1e-3)
    
    # Check output is not NaN or inf
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_hopfield_with_different_temperatures(device, sample_batch, model_config):
    """Test HopfieldNetworkModel with different temperature values."""
    input_ids = sample_batch["input_ids"]
    attention_mask = sample_batch["attention_mask"]
    
    # Try different temperature values
    temperature_values = [0.1, 0.5, 1.0, 2.0]
    
    for temperature in temperature_values:
        model = HopfieldNetworkModel(
            vocab_size=model_config["vocab_size"],
            hidden_dim=model_config["hidden_dim"],
            num_layers=model_config["num_layers"],
            num_heads=model_config["num_heads"],
            ff_dim=model_config["ff_dim"],
            temperature=temperature,
            dropout=model_config["dropout"]
        ).to(device)
        
        # Forward pass
        output, _ = model(input_ids, attention_mask)
        
        # Check output is not NaN or inf
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


def test_hopfield_with_different_sequence_lengths(device, model_config):
    """Test HopfieldNetworkModel with different sequence lengths."""
    temperature = 1.0
    model = HopfieldNetworkModel(
        vocab_size=model_config["vocab_size"],
        hidden_dim=model_config["hidden_dim"],
        num_layers=model_config["num_layers"],
        num_heads=model_config["num_heads"],
        ff_dim=model_config["ff_dim"],
        temperature=temperature,
        dropout=model_config["dropout"]
    ).to(device)
    
    batch_size = 2
    
    # Test with different sequence lengths
    seq_lengths = [32, 64, 128]
    
    for seq_len in seq_lengths:
        input_ids = torch.randint(0, model_config["vocab_size"], (batch_size, seq_len), device=device)
        attention_mask = torch.ones(batch_size, seq_len, device=device)
        
        # Forward pass
        output, _ = model(input_ids, attention_mask)
        
        # Check output shape
        expected_shape = (batch_size, seq_len, model_config["hidden_dim"])
        assert output.shape == expected_shape
        
        # Check output is not NaN or inf
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any() 