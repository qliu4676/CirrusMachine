"""Compatibility entry point for older notebooks; installs this checkout."""
from pathlib import Path
import runpy

globals().update(runpy.run_path(str(Path(__file__).resolve().parent / "scripts/setup_colab.py")))
