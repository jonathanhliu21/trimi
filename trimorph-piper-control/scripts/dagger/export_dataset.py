#!/usr/bin/env python3
"""Raw DAgger export with optional --trim; export_segments.py is also supported."""
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.dagger.export_segments import main

if __name__ == "__main__":
    sys.exit(main())
