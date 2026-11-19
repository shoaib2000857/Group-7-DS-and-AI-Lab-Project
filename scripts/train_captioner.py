"""Command-line entry point for src.training.captioner."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.training.captioner import main
if __name__ == "__main__":
    main()
