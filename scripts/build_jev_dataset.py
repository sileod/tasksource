#!/usr/bin/env python3
"""Compatibility entry point; the Jev builder now lives in jev/build.py."""

from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "jev" / "build.py"),
        run_name="__main__",
    )
