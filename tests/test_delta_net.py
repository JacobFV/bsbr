import pytest
import torch
from bsbr_extras.delta_net import DeltaNetAttention, DeltaNetLayer, DeltaNetModel


def test_delta_net_attention_init(device, model_config):
    """Test the initialization of DeltaNetAttention."""
    beta = 0.9
    attention = DeltaNetAttention(
        hidden_dim=model_config["hidden_dim"],
        num_heads=model_config["num_heads"],
        beta=beta,
        dropout=model_config["dropout"]
    )
    
    assert attention.hidden_dim == model_config["hidden_dim"]
    assert attention.num_heads == model_config["num_heads"]
    assert attention.head_dim == model_config["hidden_dim"] // model_config["num_heads"]
    assert attention.beta == beta
    assert attention.dropout == model_config["dropout"]


def test_delta_net_attention_forward(device, sample_batch, model_config):
    """Test the forward pass of DeltaNetAttention."""
    hidden_states = sample_batch["hidden_states"]
    attention_mask = sample_batch["attention_mask"]
    beta = 0.9
    
    attention = DeltaNetAttention(
        hidden_dim=model_config["hidden_dim"],
        num_heads=model_config["num_heads"],
        beta=beta,
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


def test_delta_net_layer_init(device, model_config):
    """Test the initialization of DeltaNetLayer."""
    beta = 0.9
    layer = DeltaNetLayer(
        hidden_dim=model_config["hidden_dim"],
        num_heads=model_config["num_heads"],
        ff_dim=model_config["ff_dim"],
        beta=beta,
        dropout=model_config["dropout"]
    )
    
    # Check that the layer components are properly initialized
    assert hasattr(layer, "attention")
    assert hasattr(layer, "layer_norm1")
    assert hasattr(layer, "layer_norm2")
    assert hasattr(layer, "ff")
    assert layer.attention.beta == beta


def test_delta_net_layer_forward(device, sample_batch, model_config):
    """Test the forward pass of DeltaNetLayer."""
    hidden_states = sample_batch["hidden_states"]
    attention_mask = sample_batch["attention_mask"]
    beta = 0.9
    
    layer = DeltaNetLayer(
        hidden_dim=model_config["hidden_dim"],
        num_heads=model_config["num_heads"],
        ff_dim=model_config["ff_dim"],
        beta=beta,
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


def test_delta_net_model_init(device, model_config):
    """Test the initialization of DeltaNetModel."""
    beta = 0.9
    model = DeltaNetModel(
        vocab_size=model_config["vocab_size"],
        hidden_dim=model_config["hidden_dim"],
        num_layers=model_config["num_layers"],
        num_heads=model_config["num_heads"],
        ff_dim=model_config["ff_dim"],
        beta=beta,
        dropout=model_config["dropout"]
    )
    
    # Check that the model components are properly initialized
    assert hasattr(model, "embedding")
    assert hasattr(model, "pos_encoding")
    assert hasattr(model, "layers")
    assert hasattr(model, "layer_norm")
    
    # Check number of layers
    assert len(model.layers) == model_config["num_layers"]
    
    # Check beta parameter was passed correctly
    for layer in model.layers:
        assert layer.attention.beta == beta


def test_delta_net_model_forward(device, sample_batch, model_config):
    """Test the forward pass of DeltaNetModel."""
    input_ids = sample_batch["input_ids"]
    attention_mask = sample_batch["attention_mask"]
    beta = 0.9
    
    model = DeltaNetModel(
        vocab_size=model_config["vocab_size"],
        hidden_dim=model_config["hidden_dim"],
        num_layers=model_config["num_layers"],
        num_heads=model_config["num_heads"],
        ff_dim=model_config["ff_dim"],
        beta=beta,
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


def test_delta_net_with_state(device, sample_batch, model_config):
    """Test DeltaNetModel with provided initial states."""
    input_ids = sample_batch["input_ids"]
    attention_mask = sample_batch["attention_mask"]
    batch_size = sample_batch["batch_size"]
    
    # Use a larger beta to ensure state changes are more noticeable
    beta = 0.99
    
    model = DeltaNetModel(
        vocab_size=model_config["vocab_size"],
        hidden_dim=model_config["hidden_dim"],
        num_layers=model_config["num_layers"],
        num_heads=model_config["num_heads"],
        ff_dim=model_config["ff_dim"],
        beta=beta,
        dropout=0.0  # Disable dropout for deterministic results
    ).to(device)
    
    # Create perturbed inputs for second pass to ensure state changes
    perturbed_ids = input_ids.clone()
    # Change a few random tokens to ensure different processing
    idx = torch.randint(0, input_ids.size(1), (batch_size, 5))
    for b in range(batch_size):
        for i in idx[b]:
            perturbed_ids[b, i] = (perturbed_ids[b, i] + 1) % model_config["vocab_size"]
    
    # First get states from a forward pass
    _, initial_states = model(input_ids, attention_mask)
    
    # Then do another forward pass with the obtained states but different inputs
    output, new_states = model(perturbed_ids, attention_mask, initial_states)
    
    # Check output shape
    batch_size, seq_len = input_ids.shape
    expected_shape = (batch_size, seq_len, model_config["hidden_dim"])
    assert output.shape == expected_shape
    
    # Check that at least one state has changed significantly
    # We don't check every element because some states might not change much
    # due to stability measures
    has_changed = False
    for i in range(len(initial_states)):
        # Calculate absolute difference between initial and new states
        diff = torch.abs(initial_states[i] - new_states[i])
        # If max difference exceeds threshold, consider it changed
        if diff.max() > 1e-3:
            has_changed = True
            break
    
    assert has_changed, "None of the states changed significantly"
    
    # Check output is not NaN or inf
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_delta_net_different_betas(device, sample_batch, model_config):
    """Test DeltaNetModel with different beta values."""
    input_ids = sample_batch["input_ids"]
    attention_mask = sample_batch["attention_mask"]
    
    # Try different beta values
    beta_values = [0.1, 0.5, 0.9, 0.99]
    
    for beta in beta_values:
        model = DeltaNetModel(
            vocab_size=model_config["vocab_size"],
            hidden_dim=model_config["hidden_dim"],
            num_layers=model_config["num_layers"],
            num_heads=model_config["num_heads"],
            ff_dim=model_config["ff_dim"],
            beta=beta,
            dropout=model_config["dropout"]
        ).to(device)
        
        # Forward pass
        output, _ = model(input_ids, attention_mask)
        
        # Check output is not NaN or inf
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any() 