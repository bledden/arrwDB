# Pre-commit Hooks Setup Guide

Pre-commit hooks automatically check your code before every commit, catching issues early and maintaining code quality.

## Quick Start

```bash
# Install pre-commit
pip install pre-commit

# Install the git hooks
pre-commit install

# (Optional) Run against all files once
pre-commit run --all-files
```

That's it! Hooks will now run automatically on `git commit`.

---

## What Gets Checked

### 1. **General File Checks**
- No large files (> 500KB)
- Files end with newline
- No trailing whitespace
- Valid YAML/JSON/TOML syntax
- No private keys committed
- No merge conflict markers
- No debugger statements (pdb, breakpoint, etc.)
- Consistent line endings (LF)

### 2. **Code Formatting**
- **Black**: Automatic Python code formatting
  - Line length: 100 characters
  - Consistent style across entire codebase

- **isort**: Automatic import sorting
  - Groups: stdlib, third-party, local
  - Compatible with Black

### 3. **Code Quality**
- **flake8**: Linting and style checking
  - Line length: 100
  - Cyclomatic complexity: max 15
  - Includes plugins:
    - flake8-bugbear (find bugs)
    - flake8-comprehensions (better comprehensions)
    - flake8-simplify (suggest simplifications)

### 4. **Type Safety**
- **mypy**: Static type checking
  - Catch type errors before runtime
  - Warn about redundant casts
  - Warn about unused ignores

### 5. **Security**
- **bandit**: Security vulnerability scanner
  - Checks for common security issues
  - SQL injection, hardcoded passwords, etc.
  - Only shows medium+ severity (tests excluded)

### 6. **Documentation**
- **interrogate**: Docstring coverage checker
  - Requires 50% docstring coverage
  - Ignores __init__, magic methods, tests

---

## Usage

### Auto-run on Commit
Hooks run automatically:
```bash
git add file.py
git commit -m "Add feature"
# Hooks run automatically here
```

### Manual Run
Run hooks manually on all files:
```bash
pre-commit run --all-files
```

Run specific hook:
```bash
pre-commit run black --all-files
pre-commit run mypy --all-files
```

Run on specific files:
```bash
pre-commit run --files app/api/main.py
```

### Skip Hooks (Emergency Only)
If you absolutely must skip hooks:
```bash
git commit --no-verify -m "Emergency fix"
```

**⚠️ Use sparingly!** Skipping hooks bypasses quality checks.

---

## Updating Hooks

Update to latest hook versions:
```bash
pre-commit autoupdate
```

This updates `.pre-commit-config.yaml` with latest versions.

---

## CI Integration

Hooks run in CI/CD pipeline automatically:
- Auto-fixes are committed by pre-commit.ci
- Failed checks block PRs
- Runs weekly auto-updates

---

## Fixing Common Issues

### Black Formatting
```bash
# Black modified your code
# Review changes and re-add
git add file.py
git commit
```

### Import Sorting (isort)
```bash
# isort reorganized imports
# Review and re-add
git add file.py
git commit
```

### Flake8 Errors
```bash
# Fix reported issues manually
# Common fixes:
# - Remove unused imports
# - Simplify complex functions
# - Fix line length (use Black first)
```

### Mypy Type Errors
```bash
# Add type hints
def my_function(x: int) -> str:
    return str(x)

# Or use type: ignore for third-party code
import untyped_library  # type: ignore
```

### Bandit Security Issues
```bash
# Fix security vulnerabilities
# Example: Don't use pickle.loads() on untrusted data
# Example: Use parameterized SQL queries
```

---

## Configuration

### Customize `.pre-commit-config.yaml`

```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.11.0
    hooks:
      - id: black
        args: ['--line-length=120']  # Increase line length
```

### Skip Specific Files

```yaml
hooks:
  - id: black
    exclude: ^migrations/  # Don't format migrations
```

### Disable Specific Hooks

Comment out hooks you don't want:
```yaml
# - repo: https://github.com/PyCQA/bandit
#   rev: 1.7.5
#   hooks:
#     - id: bandit
```

---

## Troubleshooting

### Hooks Not Running
```bash
# Reinstall hooks
pre-commit uninstall
pre-commit install
```

### Performance Issues
```bash
# Run hooks in parallel (default)
# Limit to specific files
pre-commit run --files app/*.py
```

### Clean Hook Cache
```bash
# Clear pre-commit cache
pre-commit clean
```

---

## Best Practices

1. **Run hooks early**: `pre-commit run --all-files` before committing large changes
2. **Fix issues incrementally**: Don't accumulate lint/type errors
3. **Don't skip hooks**: They're there to help you
4. **Keep hooks updated**: `pre-commit autoupdate` monthly
5. **Add type hints**: mypy works best with good type annotations

---

## Benefits

✅ **Catch bugs early** - Before they hit CI/CD
✅ **Consistent code style** - Entire team uses same formatting
✅ **Security checks** - Prevent vulnerabilities
✅ **Faster reviews** - Automated checks reduce manual review
✅ **Better documentation** - Docstring coverage enforced
✅ **Type safety** - Static type checking prevents runtime errors

---

## Additional Tools

### Install All Dev Dependencies

```bash
pip install -r requirements-dev.txt
```

This includes:
- pre-commit
- pytest + plugins
- coverage tools
- All hook dependencies

---

## Resources

- [Pre-commit Documentation](https://pre-commit.com/)
- [Black Documentation](https://black.readthedocs.io/)
- [flake8 Documentation](https://flake8.pycqa.org/)
- [mypy Documentation](https://mypy.readthedocs.io/)
- [Bandit Documentation](https://bandit.readthedocs.io/)
