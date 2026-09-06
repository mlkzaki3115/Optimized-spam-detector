import re
import string
import pandas as pd


def load_and_clean_data(path):
    df = pd.read_csv(path, encoding="latin1")
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    df = df.dropna(axis=1, how="all")
    df = df[["v1", "v2"]].rename(columns={"v1": "label", "v2": "message"})
    df = df.dropna().drop_duplicates()

    df["message"] = (
        df["message"]
        .astype(str)
        .str.lower()
        .str.replace(f"[{re.escape(string.punctuation)}]", "", regex=True)
        .str.replace(r"\d+", "", regex=True)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

    return df

