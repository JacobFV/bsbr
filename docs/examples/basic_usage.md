# Basic Usage Examples

This guide demonstrates basic usage of the BSBR model and its variants.

## Basic BSBR Model

```python
import torch
from bsbr import BSBRModel

# Create model
model = BSBRModel(
    vocab_size=10000,
    hidden_dim=512,
    num_layers=4,
    num_heads=8,
    chunk_size=128,
    ff_dim=2048,
    dropout=0.1
)

# Generate sample input
batch_size = 2
seq_length = 256
input_ids = torch.randint(0, 10000, (batch_size, seq_length))
attention_mask = torch.ones(batch_size, seq_length)

# Forward pass
outputs = model(input_ids, attention_mask)
```

## Using Different Model Variants

### Linear Transformer

```python
from bsbr_extras import LinearTransformer

# Create linear transformer
model = LinearTransformer(
    vocab_size=10000,
    hidden_dim=512,
    num_layers=4,
    num_heads=8,
    ff_dim=2048,
    dropout=0.1
)

# Forward pass
outputs = model(input_ids, attention_mask)
```

### DeltaNet

```python
from bsbr_extras import DeltaNet

# Create DeltaNet
model = DeltaNet(
    vocab_size=10000,
    hidden_dim=512,
    num_layers=4,
    num_heads=8,
    ff_dim=2048,
    dropout=0.1
)

# Forward pass
outputs = model(input_ids, attention_mask)
```

### Sliding Window Transformer

```python
from bsbr_extras import SlidingWindowTransformer

# Create sliding window transformer
model = SlidingWindowTransformer(
    vocab_size=10000,
    hidden_dim=512,
    num_layers=4,
    num_heads=8,
    window_size=64,
    ff_dim=2048,
    dropout=0.1
)

# Forward pass
outputs = model(input_ids, attention_mask)
```

## Training Example

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from bsbr import BSBRModel

# Create model
model = BSBRModel(
    vocab_size=10000,
    hidden_dim=512,
    num_layers=4,
    num_heads=8,
    chunk_size=128,
    ff_dim=2048,
    dropout=0.1
)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train(model, dataloader, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for batch in dataloader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
            
            # Forward pass
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

## Evaluation Example

```python
from bsbr.evals import compare_models, analyze_results

# Compare different models
results = compare_models(
    seq_lengths=[64, 128, 256, 512, 1024],
    models=['BSBR', 'Linear', 'DeltaNet', 'SlidingWindow']
)

# Analyze results
analysis = analyze_results(results)
print(analysis)
``` 