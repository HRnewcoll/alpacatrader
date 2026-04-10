"""Shared test utilities and fixtures."""
from __future__ import annotations

import os
import sys

# Ensure the package root is on the path when running tests directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
