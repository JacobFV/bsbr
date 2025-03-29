import pytest
import torch

@pytest.fixture
def device():
    """Return the device to run tests on (CPU or CUDA if available)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture
def sample_batch(device):
    """Create a sample batch of data for testing."""
    batch_size = 2
    seq_len = 128
    hidden_dim = 64
    
    # Create random input tensors
    input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
    attention_mask = torch.ones(batch_size, seq_len, device=device)
    hidden_states = torch.randn(batch_size, seq_len, hidden_dim, device=device)
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "hidden_states": hidden_states,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "hidden_dim": hidden_dim
    }

@pytest.fixture
def model_config():
    """Default model configuration for testing."""
    return {
        "vocab_size": 1000,
        "hidden_dim": 64,
        "num_heads": 4,
        "num_layers": 2,
        "chunk_size": 32,
        "ff_dim": 128,
        "dropout": 0.1,
        "compression_factor": 2
    } 