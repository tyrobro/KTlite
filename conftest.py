"""
conftest.py — make the project root importable from the tests/ subdirectory.
"""
import sys
import os

# Add the project root to sys.path so that `import app` works from tests/
sys.path.insert(0, os.path.dirname(__file__))
