"""Acquire Ott v1.4 via Kaggle when configured, otherwise author's archive."""

import sys, os, io, zipfile, logging, shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import requests, pandas as pd
from reviewguard.config import ROOT

logging.basicConfig(level=logging.INFO, format="%(message)s")


def main():
    target = ROOT / "data/raw/deceptive-opinion.csv"
    if target.exists():
        logging.info("Dataset already present: %s", target)
        return
    if (
        os.environ.get("KAGGLE_API_TOKEN")
        or (Path.home() / ".kaggle/kaggle.json").exists()
    ):
        try:
            import kagglehub

            folder = Path(
                kagglehub.dataset_download("rtatman/deceptive-opinion-spam-corpus")
            )
            csvs = list(folder.rglob("*.csv"))
            if csvs:
                shutil.copy(csvs[0], target)
                return
        except Exception as e:
            logging.warning("Kaggle unavailable: %s. Trying original archive.", e)
    try:
        response = requests.get("https://myleott.com/op_spam_v1.4.zip", timeout=60)
        response.raise_for_status()
        convert_zip(response.content, target)
    except Exception as e:
        logging.error(
            "Download failed: %s. Download the CSV from https://www.kaggle.com/datasets/rtatman/deceptive-opinion-spam-corpus and place it at %s",
            e,
            target,
        )
        sys.exit(1)


def convert_zip(data, target):
    z = zipfile.ZipFile(io.BytesIO(data))
    rows = []
    for name in sorted(z.namelist()):
        if not name.endswith(".txt") or "/fold" not in name:
            continue
        p = Path(name)
        group = p.parts[2]
        polarity = p.parts[1].split("_")[0]
        rows.append(
            dict(
                deceptive=group.split("_from_")[0],
                hotel=p.stem.split("_")[1],
                polarity=polarity,
                source=group.split("_from_")[1],
                text=z.read(name).decode("utf-8").strip(),
            )
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(target, index=False)
    (target.parent / "CORPUS_LICENSE.txt").write_bytes(z.read("op_spam_v1.4/LICENSE"))
    logging.info("Saved %d original corpus rows", len(rows))


if __name__ == "__main__":
    main()
