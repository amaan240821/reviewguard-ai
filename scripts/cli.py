"""CLI analysis: text or CSV to annotated CSV / JSON / PDF."""

import argparse, sys, json, logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from reviewguard.pipeline import analyze_review, analyze_batch
from reviewguard.report import csv_bytes, pdf_bytes
from reviewguard.config import Config

p = argparse.ArgumentParser()
g = p.add_mutually_exclusive_group(required=True)
g.add_argument("--text")
g.add_argument("--csv", type=Path)
p.add_argument("--text-column", default="text")
p.add_argument("--rating-column")
p.add_argument("--rating", type=float)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--offline", action="store_true")
a = p.parse_args()
config = (
    Config(sentiment_backend="vader", embedding_backend="lsa")
    if a.offline
    else Config()
)
try:
    r = (
        analyze_batch(
            pd.read_csv(a.csv),
            {"text": a.text_column, "rating": a.rating_column},
            config,
        )
        if a.csv
        else analyze_review(a.text, a.rating, config=config)
    )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    if a.output.suffix == ".pdf":
        a.output.write_bytes(pdf_bytes(r))
    elif a.csv and a.output.suffix == ".csv":
        a.output.write_bytes(csv_bytes(r))
    else:
        a.output.write_text(json.dumps(r.to_dict(), indent=2, allow_nan=False))
except (ValueError, RuntimeError, FileNotFoundError) as e:
    p.error(str(e))
