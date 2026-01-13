"""Pytest configuration and fixtures."""

import pytest
import sys


def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring a GPU"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to handle GPU tests."""
    import os
    
    # Check if we have GPU support (either real GPU or lavapipe software renderer)
    has_gpu = (
        os.environ.get('HAS_GPU') or 
        os.environ.get('VK_ICD_FILENAMES') or  # lavapipe in CI
        not os.environ.get('CI')  # Local development
    )
    
    if not has_gpu:
        skip_gpu = pytest.mark.skip(reason="No GPU available in CI")
        for item in items:
            if "gpu" in item.keywords or "test_gpu" in str(item.fspath):
                item.add_marker(skip_gpu)

