#!/bin/bash
# Setup script for BSBR development environment using uv
# This script creates a virtual environment and installs the project with development dependencies

# Check if uv is installed
if ! command -v uv &> /dev/null
then
    echo "uv is not installed. Installing uv..."
    # Install uv (platform-specific instructions may vary)
    pip install uv
fi

# Create a virtual environment
echo "Creating virtual environment with uv..."
uv venv

# Activate the virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install the project with development dependencies
echo "Installing project with dependencies..."
uv pip install -e .[extras]

# Install additional development packages
echo "Installing development packages..."
uv pip install pytest pytest-cov black

echo "Development environment setup complete!"
echo "To use the environment, run: source .venv/bin/activate"

# Optional: Show how to run the converter example
echo ""
echo "To run the converter example, activate the environment and run:"
echo "python -m bsbr_extras.convert_example --model_name gpt2 --chunk_size 128" 