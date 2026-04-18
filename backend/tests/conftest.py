"""
Pytest configuration and stub patches for the backend test suite.

Several modules in the package import heavy optional dependencies (chromadb,
litellm) or reference constants that were removed in the installed version of
smolagents.  This conftest provides lightweight stubs so that tests that do
NOT exercise those libraries can still be collected and run without a full
production dependency install.

Stubs are inserted into sys.modules *before* any test module is imported,
which is exactly when pytest processes conftest.py files.
"""

from __future__ import annotations

import importlib.util
import sys
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Stub out heavy dependencies that may not be installed in the dev venv
# ---------------------------------------------------------------------------

for _mod_name in ["chromadb"]:
    if _mod_name not in sys.modules and importlib.util.find_spec(_mod_name) is None:
        sys.modules[_mod_name] = MagicMock()

# ---------------------------------------------------------------------------
# Patch smolagents for API differences between versions
#
# smolagents 1.24.0 no longer exports TOOL_CALLING_SYSTEM_PROMPT, but the
# existing question_agent.py was written against a version that did.
# Adding the attribute as an empty string lets the module import cleanly.
# ---------------------------------------------------------------------------

import smolagents as _smolagents  # noqa: E402

if not hasattr(_smolagents, "TOOL_CALLING_SYSTEM_PROMPT"):
    _smolagents.TOOL_CALLING_SYSTEM_PROMPT = ""
