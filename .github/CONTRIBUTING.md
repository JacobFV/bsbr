# Contributing to BSBR

Thank you for your interest in contributing to BSBR! This document provides guidelines and instructions for contributing.

## Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

## How to Contribute

1. Fork the repository
2. Create a new branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run the tests (`pytest`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## Development Setup

1. Clone your fork:
   ```bash
   git clone https://github.com/your-username/bsbr.git
   cd bsbr
   ```

2. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

3. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Code Style

- We use `black` for code formatting
- We use `isort` for import sorting
- We use `flake8` for linting
- We use `mypy` for type checking

Run the formatters:
```bash
black .
isort .
```

## Testing

- Write tests for new features
- Ensure all tests pass before submitting PR
- Run tests with:
  ```bash
  pytest
  ```

## Documentation

- Update documentation for new features
- Follow the existing documentation style
- Build documentation locally:
  ```bash
  mkdocs serve
  ```

## Pull Request Process

1. Update the README.md with details of changes if needed
2. Update the documentation if needed
3. Add tests for new features
4. Ensure the test suite passes
5. Update the CHANGELOG.md with a note describing your changes

## Release Process

1. Update version in pyproject.toml
2. Update CHANGELOG.md
3. Create a new release on GitHub
4. Tag the release
5. Build and publish to PyPI

## Getting Help

- Open an issue for bugs or feature requests
- Join our [Discord Community](https://discord.gg/bsbr)
- Check the [documentation](https://bsbr.readthedocs.io/)

## License

By contributing, you agree that your contributions will be licensed under the project's MIT License. 