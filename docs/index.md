# BSBR: Block Sparse Attention with Block Retrieval

[![PyPI version](https://badge.fury.io/py/bsbr.svg)](https://badge.fury.io/py/bsbr)
[![Python Version](https://img.shields.io/pypi/pyversions/bsbr.svg)](https://pypi.org/project/bsbr/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://github.com/JacobFV/bsbr/actions/workflows/docs.yml/badge.svg)](https://github.com/JacobFV/bsbr/actions/workflows/docs.yml)
[![Tests](https://github.com/JacobFV/bsbr/actions/workflows/tests.yml/badge.svg)](https://github.com/JacobFV/bsbr/actions/workflows/tests.yml)
[![Coverage Status](https://coveralls.io/repos/github/JacobFV/bsbr/badge.svg?branch=main)](https://coveralls.io/github/JacobFV/bsbr?branch=main)

BSBR (Block Sparse Attention with Block Retrieval) is a novel attention mechanism for efficient processing of long sequences in transformer architectures. It combines standard attention within chunks and block retrieval between chunks to achieve near-linear complexity while maintaining high model expressivity.

## Features

- 🔄 **Efficient Processing**: Near-linear complexity in sequence length
- 🧩 **Chunk-Based Attention**: Standard attention within chunks
- 🔍 **Block Retrieval**: Efficient information retrieval between chunks
- 🎯 **Configurable**: Adjustable chunk size and compression
- 💾 **Memory Efficient**: Optimized memory usage for long sequences

## Quick Start

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

## Installation

```bash
# Install the core package
pip install bsbr

# Install with extras for evaluations and research
pip install "bsbr[extras]"
```

## Documentation

- [Getting Started](getting-started/installation.md)
- [User Guide](user-guide/core-concepts.md)
- [API Reference](api/bsbr.md)
- [Examples](examples/basic_usage.md)
- [Research](research/background.md)

## Research

BSBR is based on research presented in our paper [*BSBR: Block Sparse Attention with Block Retrieval for Efficient Long-Context Reasoning*](research/background.md). The implementation is inspired by Shengding Hu's blog post [*Streaming models for efficient long-context reasoning*](https://shengdinghu.github.io/blogs/streaming_model/).

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](https://github.com/JacobFV/bsbr/blob/main/CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/JacobFV/bsbr/blob/main/LICENSE) file for details. 