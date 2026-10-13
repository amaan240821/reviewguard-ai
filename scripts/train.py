import sys, logging, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reviewguard.classifier import train_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--finetune", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.finetune:
        parser.error(
            "CPU transformer fine-tuning is optional and not implemented; pretrained feature encoders are used."
        )
    logging.info("Trained: %s", train_model()["algorithm"])
