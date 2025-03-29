import pytest
import torch
from bsbr_extras.sliding_window_transformer import SlidingWindowAttention, SlidingWindowTransformerLayer, SlidingWindowTransformerModel


def test_sliding_window_attention_init(device, model_config):
    """Test the initialization of SlidingWindowAttention."""
    window_size = 32
    attention = SlidingWindowAttention(
        hidden_dim=model_config["hidden_dim"],
        num_heads=model_config["num_heads"],
        window_size=window_size,
        dropout=model_config["dropout"]
    )
    
    assert attention.hidden_dim == model_config["hidden_dim"]
    assert attention.num_heads == model_config["num_heads"]
    assert attention.head_dim == model_config["hidden_dim"] // model_config["num_heads"]
    assert attention.window_size == window_size
    assert attention.dropout == model_config["dropout"]


def test_sliding_window_attention_forward(device, sample_batch, model_config):
    """Test the forward pass of SlidingWindowAttention."""
    hidden_states = sample_batch["hidden_states"]
    attention_mask = sample_batch["attention_mask"]
    window_size = 16
    
    attention = SlidingWindowAttention(
        hidden_dim=model_config["hidden_dim"],
        num_heads=model_config["num_heads"],
        window_size=window_size,
        dropout=model_config["dropout"]
    ).to(device)
    
    # Forward pass
    output = attention(hidden_states, attention_mask)
    
    # Check output shape
    assert output.shape == hidden_states.shape
    
    # Check output is not NaN or inf
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_sliding_window_transformer_layer_init(device, model_config):
    """Test the initialization of SlidingWindowTransformerLayer."""
    window_size = 32
    layer = SlidingWindowTransformerLayer(
        hidden_dim=model_config["hidden_dim"],
        num_heads=model_config["num_heads"],
        window_size=window_size,
        ff_dim=model_config["ff_dim"],
        dropout=model_config["dropout"]
    )
    
    # Check that the layer components are properly initialized
    assert hasattr(layer, "attention")
    assert hasattr(layer, "layer_norm1")
    assert hasattr(layer, "layer_norm2")
    assert hasattr(layer, "ff")
    assert layer.attention.window_size == window_size


def test_sliding_window_transformer_layer_forward(device, sample_batch, model_config):
    """Test the forward pass of SlidingWindowTransformerLayer."""
    hidden_states = sample_batch["hidden_states"]
    attention_mask = sample_batch["attention_mask"]
    window_size = 16
    
    layer = SlidingWindowTransformerLayer(
        hidden_dim=model_config["hidden_dim"],
        num_heads=model_config["num_heads"],
        window_size=window_size,
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


def test_sliding_window_transformer_model_init(device, model_config):
    """Test the initialization of SlidingWindowTransformerModel."""
    window_size = 32
    model = SlidingWindowTransformerModel(
        vocab_size=model_config["vocab_size"],
        hidden_dim=model_config["hidden_dim"],
        num_layers=model_config["num_layers"],
        num_heads=model_config["num_heads"],
        window_size=window_size,
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
    
    # Check window size was passed correctly
    for layer in model.layers:
        assert layer.attention.window_size == window_size


def test_sliding_window_transformer_model_forward(device, sample_batch, model_config):
    """Test the forward pass of SlidingWindowTransformerModel."""
    input_ids = sample_batch["input_ids"]
    attention_mask = sample_batch["attention_mask"]
    window_size = 16
    
    model = SlidingWindowTransformerModel(
        vocab_size=model_config["vocab_size"],
        hidden_dim=model_config["hidden_dim"],
        num_layers=model_config["num_layers"],
        num_heads=model_config["num_heads"],
        window_size=window_size,
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


def test_sliding_window_with_different_window_sizes(device, sample_batch, model_config):
    """Test SlidingWindowTransformerModel with different window sizes."""
    input_ids = sample_batch["input_ids"]
    attention_mask = sample_batch["attention_mask"]
    
    # Try different window sizes
    window_sizes = [4, 8, 16, 32]
    
    for window_size in window_sizes:
        model = SlidingWindowTransformerModel(
            vocab_size=model_config["vocab_size"],
            hidden_dim=model_config["hidden_dim"],
            num_layers=model_config["num_layers"],
            num_heads=model_config["num_heads"],
            window_size=window_size,
            ff_dim=model_config["ff_dim"],
            dropout=model_config["dropout"]
        ).to(device)
        
        # Forward pass
        output = model(input_ids, attention_mask)
        
        # Check output is not NaN or inf
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


def test_sliding_window_with_different_sequence_lengths(device, model_config):
    """Test SlidingWindowTransformerModel with different sequence lengths."""
    window_size = 16
    model = SlidingWindowTransformerModel(
        vocab_size=model_config["vocab_size"],
        hidden_dim=model_config["hidden_dim"],
        num_layers=model_config["num_layers"],
        num_heads=model_config["num_heads"],
        window_size=window_size,
        ff_dim=model_config["ff_dim"],
        dropout=model_config["dropout"]
    ).to(device)
    
    batch_size = 2
    
    # Test with different sequence lengths, including those longer than window size
    seq_lengths = [8, window_size, window_size * 2, window_size * 2 + 5]
    
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


def test_sliding_window_with_attention_mask(device, model_config):
    """Test SlidingWindowTransformerModel with a non-trivial attention mask."""
    window_size = 16
    model = SlidingWindowTransformerModel(
        vocab_size=model_config["vocab_size"],
        hidden_dim=model_config["hidden_dim"],
        num_layers=model_config["num_layers"],
        num_heads=model_config["num_heads"],
        window_size=window_size,
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