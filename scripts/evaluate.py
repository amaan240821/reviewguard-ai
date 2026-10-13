import sys, logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reviewguard.evaluate import evaluate_all

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    evaluate_all()
