# Setup script for BSBR development environment using uv
# This script creates a virtual environment and installs the project with development dependencies

# Check if uv is installed
try {
    uv --version
}
catch {
    Write-Host "uv is not installed. Installing uv..."
    # Instructions for installing uv can vary by platform
    # For Windows, we'll use pip
    pip install uv
}

# Create a virtual environment
Write-Host "Creating virtual environment with uv..."
uv venv

# Activate the virtual environment
Write-Host "Activating virtual environment..."
& .\.venv\Scripts\activate

# Install the project with development dependencies
Write-Host "Installing project with dependencies..."
uv pip install -e .[extras]

# Install additional development packages
Write-Host "Installing development packages..."
uv pip install pytest pytest-cov black

Write-Host "Development environment setup complete!"
Write-Host "To use the environment, run: .\.venv\Scripts\activate"

# Optional: Show how to run the converter example
Write-Host "`nTo run the converter example, activate the environment and run:"
Write-Host "python -m bsbr_extras.convert_example --model_name gpt2 --chunk_size 128" 