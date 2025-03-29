# Quick Start Guide

This guide will help you get started with BSBR quickly. We'll cover basic usage, model configuration, and common patterns.

## Basic Usage

### Creating a BSBR Model

```python
import torch
from bsbr import BSBRModel

# Create a model with default settings
model = BSBRModel(
    vocab_size=10000,      # Size of your vocabulary
    hidden_dim=512,        # Hidden dimension of the model
    num_layers=4,          # Number of transformer layers
    num_heads=8,           # Number of attention heads
    chunk_size=128,        # Size of attention chunks
    ff_dim=2048,           # Feed-forward network dimension
    dropout=0.1,           # Dropout rate
    compression_factor=4   # Optional compression factor
)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
```

### Processing Input

```python
# Create sample input
batch_size = 2
seq_length = 256
input_ids = torch.randint(0, 10000, (batch_size, seq_length))
attention_mask = torch.ones(batch_size, seq_length)

# Move inputs to the same device as the model
input_ids = input_ids.to(device)
attention_mask = attention_mask.to(device)

# Forward pass
outputs = model(input_ids, attention_mask)
```

## Advanced Configuration

### Customizing Attention

```python
from bsbr import BSBRModel, BSBRAttention

# Create a custom attention layer
attention = BSBRAttention(
    hidden_dim=512,
    num_heads=8,
    chunk_size=128,
    compression_factor=4,
    dropout=0.1
)

# Use it in a model
model = BSBRModel(
    vocab_size=10000,
    hidden_dim=512,
    num_layers=4,
    num_heads=8,
    chunk_size=128,
    ff_dim=2048,
    dropout=0.1,
    attention_layer=attention  # Use custom attention
)
```

### Using Different Models

BSBR provides several attention variants for comparison:

```python
from bsbr_extras import (
    LinearTransformer,
    DeltaNet,
    SlidingWindowTransformer,
    HopfieldNetwork,
    GAU
)

# Linear Transformer
linear_model = LinearTransformer(
    vocab_size=10000,
    hidden_dim=512,
    num_layers=4,
    num_heads=8
)

# DeltaNet
deltanet_model = DeltaNet(
    vocab_size=10000,
    hidden_dim=512,
    num_layers=4,
    num_heads=8
)

# Sliding Window Transformer
window_model = SlidingWindowTransformer(
    vocab_size=10000,
    hidden_dim=512,
    num_layers=4,
    num_heads=8,
    window_size=128
)
```

## Training Example

Here's a basic training loop:

```python
import torch.nn as nn
from torch.optim import Adam

# Create model and move to device
model = BSBRModel(
    vocab_size=10000,
    hidden_dim=512,
    num_layers=4,
    num_heads=8
).to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

# Training loop
model.train()
for epoch in range(num_epochs):
    for batch in dataloader:
        input_ids, attention_mask, labels = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs.view(-1, vocab_size), labels.view(-1))

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## Evaluation

BSBR provides tools for evaluating different models:

```python
from evals.compare_models import compare_models

# Compare models across different sequence lengths
results = compare_models(
    models=["BSBR", "Linear", "Hopfield", "GAU"],
    seq_lengths=[64, 128, 256, 512, 1024]
)

# Analyze results
from evals.analyze_results import analyze_results
analysis = analyze_results(results)
```

## Next Steps

1. Explore the [User Guide](user-guide/core-concepts.md) for detailed explanations
2. Check out [Examples](examples/basic-usage.md) for more use cases
3. Read the [API Reference](api/models.md) for complete documentation 