import pandas as pd
from sklearn.model_selection import train_test_split
from .config import ROOT
from .preprocess import normalize


def load_dataset(path=None) -> pd.DataFrame:
    """Normalize the corpus, validate labels, and remove duplicate texts."""
    p = path or ROOT / "data/raw/deceptive-opinion.csv"
    df = pd.read_csv(p)
    df.columns = df.columns.str.strip().str.lower()
    required = {"text", "deceptive", "hotel", "polarity", "source"}
    if not required.issubset(df.columns):
        raise ValueError(f"Dataset needs columns: {sorted(required)}")
    df = df.dropna(subset=["text"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    for c in ["hotel", "polarity", "deceptive", "source"]:
        df[c] = df[c].astype(str).str.strip().str.lower()
    df = df[df.text.map(normalize).str.len() > 0]
    df["key"] = df.text.map(normalize)
    df = df.drop_duplicates("key").drop(columns="key").reset_index(drop=True)
    labels = (
        df.deceptive.astype(str)
        .str.lower()
        .map({"truthful": 0, "deceptive": 1, "0": 0, "1": 1})
    )
    if labels.isna().any():
        raise ValueError("Unknown deception labels.")
    df["label"] = labels.astype(int)
    return df


def split_data(df: pd.DataFrame, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    strata = df.label.astype(str) + "_" + df.polarity
    train, test = train_test_split(
        df, test_size=0.2, stratify=strata, random_state=seed
    )
    return train.copy(), test.copy()


def validate_batch(df: pd.DataFrame, mapping: dict, limit: int = 5000) -> pd.DataFrame:
    if len(df) == 0:
        raise ValueError("The file has no reviews.")
    if len(df) > limit:
        raise ValueError(
            f"Batch limit is {limit} reviews. Split the file into smaller batches."
        )
    col = mapping.get("text")
    if col not in df.columns:
        raise ValueError("Choose a valid review text column.")
    out = df.copy().reset_index(drop=True)
    if out[col].isna().any():
        raise ValueError("Review text contains missing values.")
    out["text"] = out[col].astype(str)
    if (out.text.map(normalize).str.len() == 0).any():
        raise ValueError("Review text contains empty or non-alphabetic rows.")
    if out.text.str.len().gt(20000).any():
        raise ValueError("Each review must be at most 20,000 characters.")
    rating = mapping.get("rating")
    if rating:
        if rating not in df.columns:
            raise ValueError("Choose a valid rating column.")
        out["rating"] = pd.to_numeric(out[rating], errors="coerce")
        if (
            (df[rating].notna() & out.rating.isna()).any()
            or out.rating.dropna().lt(1).any()
            or out.rating.dropna().gt(5).any()
        ):
            raise ValueError(
                "Ratings must be numeric values between 1 and 5, or empty."
            )
    else:
        out["rating"] = float("nan")
    return out
