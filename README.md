![](https://github.com/JacobFV/bsbr/bsbr_repo_cover.png)

# BSBR: Block Sparse Attention with Block Retrieval

[![PyPI version](https://badge.fury.io/py/bsbr.svg)](https://badge.fury.io/py/bsbr)
[![Python Version](https://img.shields.io/pypi/pyversions/bsbr.svg)](https://pypi.org/project/bsbr/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://github.com/JacobFV/bsbr/actions/workflows/docs.yml/badge.svg)](https://github.com/JacobFV/bsbr/actions/workflows/docs.yml)
[![Tests](https://github.com/JacobFV/bsbr/actions/workflows/tests.yml/badge.svg)](https://github.com/JacobFV/bsbr/actions/workflows/tests.yml)
[![Coverage Status](https://coveralls.io/repos/github/JacobFV/bsbr/badge.svg?branch=main)](https://coveralls.io/github/JacobFV/bsbr?branch=main)

A PyTorch implementation of Block Sparse Attention with Block Retrieval (BSBR), a novel attention mechanism for efficient processing of long sequences. This implementation is inspired by Shengding Hu's blog post [*Streaming models for efficient long-context reasoning*](https://shengdinghu.github.io/blogs/streaming_model/). [1]

## Features

- Efficient processing of long sequences by combining:
  - Standard attention within chunks
  - Block retrieval between chunks
- Configurable chunk size
- Optional state compression
- Memory efficient with linear complexity in sequence length
- Utilities to convert standard HuggingFace Transformer models (like GPT-2) to BSBR format.

Read our analysis of the BSBR compared against other models [here](https://github.com/JacobFV/bsbr/tree/main/blog.md)

## Implemented Transformer Architectures

The repository includes implementations of several efficient transformer architectures:

1. **BSBR (Block Sparse with Block Retrieval)**: Our core implementation with chunk-based attention and efficient block retrieval
2. **Standard Transformer**: The classic self-attention mechanism with O(n²) complexity
3. **Linear Transformer**: Removes softmax for O(n) complexity using associative property of matrix multiplication
4. **DeltaNet**: Enhanced Linear Transformer with a removal component for better memory management
5. **Sliding Window Transformer**: Restricts attention to a fixed window size for O(n·w) complexity
6. **Hopfield Network**: Memory-based attention inspired by modern Hopfield Networks
7. **GAU (Gated Attention Unit)**: Chunk-based parallel attention with gating mechanisms

## Installation

```bash
# Clone the repository
git clone https://github.com/JacobFV/bsbr.git
cd bsbr

# Install the core BSBR package
pip install -e .

# Install with extra architectures (Standard, Linear, etc.) for evaluations
pip install -e ".[extras]"

# Install with transformer conversion utilities (requires transformers library)
pip install -e ".[transformers]"

# Install everything
pip install -e ".[extras,transformers]"
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

### Core Model (`bsbr` package)

- **BSBRAttention**: The core attention mechanism
- **BSBRLayer**: A complete transformer layer with BSBR attention and feed-forward network
- **BSBRModel**: A full model with embedding, multiple BSBR layers, and normalization

### Additional Architectures (`bsbr_extras` package)

Requires installing with `pip install -e ".[extras]"`.

For evaluation and research purposes, we also include several alternative attention architectures in the `bsbr_extras` package:

- **Standard Transformer**: Classic transformer with full attention (baseline)
- **Linear Transformer**: Linear complexity transformer using a reformulated attention mechanism
- **DeltaNet**: Enhanced linear transformer with a removal component 
- **Sliding Window Transformer**: Efficient attention with a fixed context window
- **Hopfield Network**: Associative memory-based attention for pattern completion
- **GAU**: Gated Attention Unit with chunk-based parallelism

### Transformer Conversion Utilities (`bsbr_transformers` package)

Requires installing with `pip install -e ".[transformers]"`.

This package provides tools to convert pre-trained HuggingFace Transformer models to the BSBR format, allowing you to leverage existing weights with the BSBR architecture.

- **`convert_to_bsbr`**: Function to convert a model (currently supports GPT-2).
- **`TransformerToBSBRConverter`**: Class providing the conversion logic.

Example:
```python
from bsbr_transformers import convert_to_bsbr

# Convert a pre-trained GPT-2 model
bsbr_gpt2_model = convert_to_bsbr("gpt2", chunk_size=128)

# Now you can use bsbr_gpt2_model
```

## Evaluation

The repository includes tools to evaluate and compare different architectures:

```bash
# Run comparison of all models
python evals/compare_models.py --seq_lengths 64 128 256 512 1024

# Compare specific models
python evals/compare_models.py --models BSBR Linear Hopfield GAU

# Analyze results
python evals/analyze_results.py --use_example_data
```

Evaluations include:
- Inference time
- Memory usage
- Parameter counts
- Computational complexity analysis

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

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References

1. Hu, S. (2025). Streaming models for efficient long-context reasoning. arXiv preprint arXiv:2403.xxxxx. https://shengdinghu.github.io/blogs/streaming_model/
