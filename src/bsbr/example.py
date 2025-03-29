import torch
from bsbr import BSBRModel

def run_example():
    """
    Example demonstrating how to use the BSBR model.
    """
    # Model configuration
    vocab_size = 10000
    hidden_dim = 512
    num_layers = 4
    num_heads = 8
    chunk_size = 128
    ff_dim = 2048
    
    # Create a sample input
    batch_size = 2
    seq_len = 256
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    # Initialize the model
    model = BSBRModel(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        chunk_size=chunk_size,
        ff_dim=ff_dim,
        dropout=0.1,
        compression_factor=4  # Optional compression
    )
    
    # Forward pass
    with torch.no_grad():
        output = model(input_ids, attention_mask)
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Test a longer sequence to demonstrate chunk handling
    long_seq_len = 512
    long_input_ids = torch.randint(0, vocab_size, (1, long_seq_len))
    long_attention_mask = torch.ones(1, long_seq_len)
    
    with torch.no_grad():
        long_output = model(long_input_ids, long_attention_mask)
    
    print(f"Long input shape: {long_input_ids.shape}")
    print(f"Long output shape: {long_output.shape}")
    
    return model

if __name__ == "__main__":
    model = run_example()
    print("Example completed successfully!") 