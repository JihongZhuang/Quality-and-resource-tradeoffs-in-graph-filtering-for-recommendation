"""Run complete-user graph-filter scoring and evaluation."""
from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parent / "src/evaluate.py"), run_name="__main__")
