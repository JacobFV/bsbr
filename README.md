# BSBR: Block Sparse Attention with Block Retrieval

A PyTorch implementation of Block Sparse Attention with Block Retrieval (BSBR), a novel attention mechanism for efficient processing of long sequences.

## Features

- Efficient processing of long sequences by combining:
  - Standard attention within chunks
  - Block retrieval between chunks
- Configurable chunk size
- Optional state compression
- Memory efficient with linear complexity in sequence length

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/bsbr.git
cd bsbr

# Install the package
pip install -e .
```

## Usage

Here's a simple example of how to use the BSBR model:

```python
import torch
from bsbr import BSBRModel

# Model configuration
model = BSBRModel(
    vocab_size=10000,
    hidden_dim=512,
    num_layers=4,
    num_heads=8,
    chunk_size=128,
    ff_dim=2048,
    dropout=0.1,
    compression_factor=4  # Optional compression
)

# Input data
input_ids = torch.randint(0, 10000, (2, 256))
attention_mask = torch.ones(2, 256)

# Forward pass
outputs = model(input_ids, attention_mask)
```

## Components

- **BSBRAttention**: The core attention mechanism
- **BSBRLayer**: A complete transformer layer with BSBR attention and feed-forward network
- **BSBRModel**: A full model with embedding, multiple BSBR layers, and normalization

## Algorithm

BSBR works by combining two types of attention:

1. **Within-chunk attention**: Standard attention with softmax
   ```
   softmax(QK^T · M_in)V
   ```

2. **Between-chunk attention**: Block retrieval using meta queries and keys
   ```
   Q ⊙ softmax(RH^T · M_out)F
   ```

Where:
- Q, K, V: Query, Key, Value matrices
- R, H: Meta queries and keys for chunk-level attention
- F: State vectors (flattened K^T·V for each chunk)
- M_in: Block diagonal mask
- M_out: Causal mask for chunk-level attention

## License

This project is licensed under the MIT License - see the LICENSE file for details.
