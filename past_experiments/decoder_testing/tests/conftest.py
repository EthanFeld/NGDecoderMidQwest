import sys
from pathlib import Path

# Make the standalone example/fixture modules importable by spawned workers too.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
