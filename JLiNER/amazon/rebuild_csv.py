#!/usr/bin/env python3
"""Recreate amazon_labeled.csv from autolabel_checkpoint.json."""

import json
from pathlib import Path

import pandas as pd

BASE_DIR        = Path(__file__).parent
INPUT_CSV       = BASE_DIR / "amazon_mined_shuffled.csv"
CHECKPOINT_FILE = BASE_DIR / "autolabel_checkpoint.json"
OUTPUT_CSV      = BASE_DIR / "amazon_labeled.csv"

LABEL_TO_COLUMN = {
    "minor_anchor":    "minor_children",
    "minor_pronoun":   "minor_children",
    "medical":         "medical_condition",
    "gender_noun":     "gender_indication",
    "gender_bio":      "gender_indication",
    "gender_clothing": "gender_indication",
}
ENTITY_COLUMNS = ["minor_children", "medical_condition", "gender_indication"]


def split_entities(entities: list[dict] | None) -> dict[str, str]:
    if entities is None:
        return {col: None for col in ENTITY_COLUMNS}
    buckets: dict[str, list] = {col: [] for col in ENTITY_COLUMNS}
    for ent in entities:
        col = LABEL_TO_COLUMN.get(ent.get("label", ""))
        if col:
            buckets[col].append(ent)
    return {
        col: json.dumps(spans, ensure_ascii=False) if spans else "[]"
        for col, spans in buckets.items()
    }


def main() -> None:
    with open(CHECKPOINT_FILE, encoding="utf-8") as f:
        checkpoint = json.load(f)

    results: dict = checkpoint["results"]
    df = pd.read_csv(INPUT_CSV)

    split = df["unique_id"].astype(str).map(lambda uid: split_entities(results.get(uid)))
    for col in ENTITY_COLUMNS:
        df[col] = split.map(lambda s, c=col: s[c])

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    labeled = df["unique_id"].astype(str).isin(results).sum()
    print(f"Rebuilt {OUTPUT_CSV} — {labeled}/{len(df)} rows labeled.")


if __name__ == "__main__":
    main()
