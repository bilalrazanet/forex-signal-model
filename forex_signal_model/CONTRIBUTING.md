# Contributing to Forex Signal Model

Thank you for your interest in contributing to this project! Here's how you can help.

## Ways to Contribute

### 1. **Report Bugs**
- Use the GitHub Issues tab
- Describe the bug clearly with steps to reproduce
- Include your environment details (Python version, OS, dependencies)
- Attach relevant error logs or screenshots

### 2. **Suggest Features**
- Open an issue with the `[FEATURE REQUEST]` prefix
- Explain the use case and expected behavior
- Link relevant research or references if applicable

### 3. **Improve Documentation**
- Fix typos or clarify existing docs
- Add examples or use cases
- Document edge cases or known limitations

### 4. **Submit Code**
- Fork the repository
- Create a feature branch: `git checkout -b feature/your-feature`
- Make focused, atomic commits
- Add tests for new functionality
- Push to your fork and open a Pull Request

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR-USERNAME/forex-signal-model.git
cd forex-signal-model

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/
```

## Code Standards

- **Style**: Follow PEP 8 conventions
- **Type Hints**: Use Python type hints for function signatures
- **Docstrings**: Use Google-style docstrings for functions and classes
- **Testing**: Write tests for new features and bug fixes
- **Commits**: Use clear, descriptive commit messages

## Pull Request Process

1. Update the README.md if your changes introduce new features
2. Ensure all tests pass: `pytest`
3. Keep your PR focused on a single feature or fix
4. Reference related issues in your PR description
5. Be responsive to code review feedback

## Code of Conduct

- Be respectful and inclusive
- Assume good intentions
- Provide constructive feedback
- Keep discussions professional

## Questions?

- Check existing issues and discussions first
- Ask in GitHub Discussions or open a new issue
- Include relevant context and error messages

Thank you for contributing! 🚀
