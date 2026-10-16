import sys
from pathlib import Path

# The harness modules live flat in evaluation/ (run as scripts); make them importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
