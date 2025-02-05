# Testing Guide

## Overview
This guide explains how to add and run unit tests. All tests are located in the `/tests` directory, which mirrors the structure of the `/kithara` directory. Test files are excluded from the distributed wheel package.

## Test Discovery
Tests are automatically discovered by `unittest` when they meet these criteria:
- Files are named with the pattern `test_*.py`
- Test functions start with `test_`

## Test Categories
We use three categories of tests, controlled by environment variables:

### 1. Light Tests
- Default test category
- Quick to execute (under 3 minutes)
- Sanity checks for basic correctness
- No special decorator needed

### 2. Heavy Tests
- Takes longer than 3 minutes to execute
- Should run before merging to main branch
- Requires following decorator:
    ```python
    @unittest.skipIf(int(os.getenv('RUN_LIGHT_TESTS_ONLY', 0)) == 1, "Heavy Test")
    ```

### 3. Manual Tests
- Tests model/task correctness
- Results unlikely to change
- Run on-demand only
- Requires decorator:
    ```python
    @unittest.skipIf(int(os.getenv('RUN_SKIPPED_TESTS', 0)) != 1, "Manual Test")
    ```

## Choosing Test Categories

Use this decision flow to categorize your tests:

1. **Manual Test** if:
   - Tests numerical correctness, such as checkpoint conversion correctness.
   - Results should be stable across runs
   - Requires significant computation time

2. **Heavy Test** if:
   - Must run before main branch merges
   - Takes >3 minutes to execute
   - Tests critical functionality

3. **Light Test** if:
   - Quick to execute
   - Tests syntax, basic functionality
   - Should run frequently during development

## Environment Variables

| Variable | Purpose | Values |
|----------|----------|---------|
| `RUN_LIGHT_TESTS_ONLY` | Run only light tests | `1` = light tests only, `0` = include heavy tests |
| `RUN_SKIPPED_TESTS` | Include manual tests | `1` = run all tests, `0` = skip manual tests |

## Running Tests

### Quick Development Check (Light Tests Only)
```bash
RUN_LIGHT_TESTS_ONLY=1 python -m unittest
```

### Pre-merge Check (Light + Heavy Tests)
```bash
python -m unittest
```

### Full Test Suite (All Tests)
```bash
RUN_SKIPPED_TESTS=1 python -m unittest
```

## Best Practices

1. **Test Organization**: Group related tests in the same file, following the source code structure

2. **Documentation**: Include docstrings at the top of every test file to describe the purpose the tests and how to run the tests in a standlone manner. E.g.
    ```
    These tests validate that the MaxText implementations produces logits that are
    numerically comparable to the reference HuggingFace implementation. 

    Usage:
        Run script on single host VM: RUN_SKIPPED_TESTS=1 python -m unittest tests/model/maxtext/ckpt_compatibility/test_loading_models.py
    ```

## Contributing Tests

1. Create test file in appropriate `/tests` subdirectory
2. Choose appropriate test category
3. Add necessary decorators
4. Include clear docstrings and comments
5. Verify tests pass 
