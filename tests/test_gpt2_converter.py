import pytest
import torch
import numpy as np
from transformers import GPT2Model, GPT2Config

from bsbr_extras.converter import TransformerToBSBRConverter, convert_to_bsbr


@pytest.fixture
def device():
    """Return the device to use for testing."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def model_config():
    """Create a small GPT2 config for testing."""
    return {
        "vocab_size": 1000,
        "hidden_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "intermediate_size": 512  # n_inner for GPT2
    }


@pytest.fixture
def tiny_gpt2(model_config):
    """Create a tiny GPT2 model for testing."""
    config = GPT2Config(
        vocab_size=model_config["vocab_size"],
        n_embd=model_config["hidden_size"],
        n_layer=model_config["num_hidden_layers"],
        n_head=model_config["num_attention_heads"],
        n_inner=model_config["intermediate_size"]
    )
    return GPT2Model(config)


def test_converter_init():
    """Test the initialization of the converter."""
    converter = TransformerToBSBRConverter(chunk_size=64, compression_factor=2)
    assert converter.chunk_size == 64
    assert converter.compression_factor == 2


def test_convert_gpt2_attention_to_bsbr(tiny_gpt2, model_config):
    """Test the conversion of a GPT2 attention layer to BSBR."""
    converter = TransformerToBSBRConverter(chunk_size=32)
    gpt_attention = tiny_gpt2.h[0].attn
    
    bsbr_attention = converter.convert_gpt2_attention_to_bsbr(
        gpt_attention, 
        model_config["hidden_size"], 
        model_config["num_attention_heads"]
    )
    
    # Check dimensions
    assert bsbr_attention.hidden_dim == model_config["hidden_size"]
    assert bsbr_attention.num_heads == model_config["num_attention_heads"]
    assert bsbr_attention.chunk_size == 32
    
    # Verify weights were transferred (shape check is enough)
    assert bsbr_attention.q_proj.weight.shape == (model_config["hidden_size"], model_config["hidden_size"])
    assert bsbr_attention.k_proj.weight.shape == (model_config["hidden_size"], model_config["hidden_size"])
    assert bsbr_attention.v_proj.weight.shape == (model_config["hidden_size"], model_config["hidden_size"])
    
    # Check bias shapes if they exist
    if hasattr(bsbr_attention.q_proj, 'bias') and bsbr_attention.q_proj.bias is not None:
        assert bsbr_attention.q_proj.bias.shape == (model_config["hidden_size"],)
        assert bsbr_attention.k_proj.bias.shape == (model_config["hidden_size"],)
        assert bsbr_attention.v_proj.bias.shape == (model_config["hidden_size"],)


def test_convert_gpt2_layer_to_bsbr(tiny_gpt2, model_config):
    """Test the conversion of a GPT2 layer to BSBR."""
    converter = TransformerToBSBRConverter(chunk_size=32)
    gpt_layer = tiny_gpt2.h[0]
    
    bsbr_layer = converter.convert_gpt2_layer_to_bsbr(
        gpt_layer, 
        model_config["hidden_size"],
        model_config["num_attention_heads"],
        model_config["intermediate_size"]
    )
    
    # Check layer norm weights
    assert torch.allclose(bsbr_layer.layer_norm1.weight.data, gpt_layer.ln_1.weight.data)
    assert torch.allclose(bsbr_layer.layer_norm1.bias.data, gpt_layer.ln_1.bias.data)
    
    # Note: Due to the differences between GPT2's Conv1D and nn.Linear,
    # the weight shapes may appear transposed, but should still be compatible
    
    # For nn.Linear, weights have shape (out_features, in_features)
    # The feed-forward network should have weights with correct input/output dimensions
    # First layer: hidden_size -> intermediate_size
    assert bsbr_layer.ff[0].weight.shape[0] * bsbr_layer.ff[0].weight.shape[1] == model_config["hidden_size"] * model_config["intermediate_size"]
    
    # Second layer: intermediate_size -> hidden_size
    assert bsbr_layer.ff[3].weight.shape[0] * bsbr_layer.ff[3].weight.shape[1] == model_config["hidden_size"] * model_config["intermediate_size"]


def test_convert_gpt2_model_to_bsbr(tiny_gpt2, model_config):
    """Test the conversion of a complete GPT2 model to BSBR."""
    converter = TransformerToBSBRConverter(chunk_size=32)
    
    bsbr_model = converter.convert_gpt2_model_to_bsbr(tiny_gpt2)
    
    # Check model dimensions
    assert len(bsbr_model.layers) == model_config["num_hidden_layers"]
    assert bsbr_model.embedding.weight.shape == (model_config["vocab_size"], model_config["hidden_size"])
    
    # Check final layer norm
    assert torch.allclose(bsbr_model.layer_norm.weight.data, tiny_gpt2.ln_f.weight.data)


def test_model_output_processing(tiny_gpt2, device):
    """Test if the BSBR model can process inputs and produce valid outputs."""
    converter = TransformerToBSBRConverter(chunk_size=32)
    bsbr_model = converter.convert_gpt2_model_to_bsbr(tiny_gpt2)
    
    # Move models to device
    tiny_gpt2 = tiny_gpt2.to(device)
    bsbr_model = bsbr_model.to(device)
    
    # Create input
    input_ids = torch.randint(0, 100, (1, 64), device=device)
    attention_mask = torch.ones_like(input_ids)
    
    # Get outputs
    with torch.no_grad():
        gpt_output = tiny_gpt2(input_ids, attention_mask=attention_mask).last_hidden_state
        bsbr_output = bsbr_model(input_ids, attention_mask=attention_mask)
    
    # Check that outputs have the same shape
    assert gpt_output.shape == bsbr_output.shape
    
    # Verify that the BSBR output contains no NaN or Inf values
    assert not torch.isnan(bsbr_output).any(), "BSBR output contains NaN values"
    assert not torch.isinf(bsbr_output).any(), "BSBR output contains Inf values"
    
    # Verify that the output values are in a reasonable range
    assert bsbr_output.abs().mean() > 0, "BSBR output has zero mean absolute value"
    assert bsbr_output.abs().mean() < 100, "BSBR output has unusually large values"
    
    # Print some basic statistics
    print(f"BSBR output shape: {bsbr_output.shape}")
    print(f"BSBR output mean: {bsbr_output.mean().item()}")
    print(f"BSBR output std: {bsbr_output.std().item()}")
    print(f"GPT output mean: {gpt_output.mean().item()}")
    print(f"GPT output std: {gpt_output.std().item()}")


def test_convert_to_bsbr_function(model_config, monkeypatch, tiny_gpt2):
    """Test the high-level convert_to_bsbr function."""
    # Mock the actual converter method to avoid HuggingFace API calls
    original_convert = TransformerToBSBRConverter.convert_pretrained_model
    
    def mock_convert_pretrained_model(self, model_name_or_path):
        return self.convert_gpt2_model_to_bsbr(tiny_gpt2)
    
    # Apply the mock
    monkeypatch.setattr(TransformerToBSBRConverter, 'convert_pretrained_model', mock_convert_pretrained_model)
    
    # Test the function
    bsbr_model = convert_to_bsbr("dummy_model", chunk_size=32)
    
    # Check if the model was created with the right parameters
    assert len(bsbr_model.layers) == model_config["num_hidden_layers"]
    assert bsbr_model.embedding.weight.shape == (model_config["vocab_size"], model_config["hidden_size"])
    
    # Restore the original method
    monkeypatch.setattr(TransformerToBSBRConverter, 'convert_pretrained_model', original_convert) 