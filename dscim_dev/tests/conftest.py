"""
Pytest configuration for DSCIM development tests.
"""

import sys
from pathlib import Path

# Add the parent directory to the path so we can import dscim_dev modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Configure pytest
def pytest_configure(config):
    """Configure pytest settings."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    ) 