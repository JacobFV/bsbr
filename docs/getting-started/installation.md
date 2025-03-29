# Installation

## Requirements

BSBR requires Python 3.12 or higher and PyTorch 2.6.0 or higher. The package is designed to work with modern deep learning frameworks and tools.

## Basic Installation

The simplest way to install BSBR is using pip:

```bash
pip install bsbr
```

## Development Installation

For development or to use the latest features, you can install directly from GitHub:

```bash
# Clone the repository
git clone https://github.com/JacobFV/bsbr.git
cd bsbr

# Install in editable mode
pip install -e .
```

## Optional Dependencies

BSBR provides several optional dependency groups that you can install:

```bash
# Install with all extras (evaluation tools, visualization, etc.)
pip install "bsbr[extras]"

# Install with documentation tools
pip install "bsbr[docs]"

# Install with all optional dependencies
pip install "bsbr[all]"
```

### Available Extras

- `extras`: Evaluation tools, visualization utilities, and research components
- `docs`: Documentation building tools and dependencies
- `all`: All optional dependencies

## GPU Support

BSBR automatically uses GPU acceleration when available. To ensure GPU support:

1. Install CUDA-enabled PyTorch:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

2. Verify GPU availability:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Current device: {torch.cuda.get_device_name(0)}")
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   - Ensure you're using Python 3.12 or higher
   - Check that PyTorch is installed correctly
   - Verify your Python environment is activated

2. **CUDA Errors**
   - Confirm CUDA toolkit is installed
   - Verify PyTorch CUDA version matches your system
   - Check GPU drivers are up to date

3. **Memory Issues**
   - Adjust batch size or sequence length
   - Enable gradient checkpointing
   - Use smaller model configurations

### Getting Help

If you encounter issues:

1. Check the [GitHub Issues](https://github.com/JacobFV/bsbr/issues)
2. Join our [Discord Community](https://discord.gg/bsbr)
3. Create a new issue with:
   - Python version
   - PyTorch version
   - Error message
   - System information

## Next Steps

After installation, you can:

1. Try the [Quick Start Guide](quickstart.md)
2. Explore the [Examples](examples/basic-usage.md)
3. Read the [User Guide](user-guide/core-concepts.md) 