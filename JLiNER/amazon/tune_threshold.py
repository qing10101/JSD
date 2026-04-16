import pandas as pd
import torch
import re
import numpy as np
from gliner import GLiNER
from tqdm import tqdm

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
MODEL_PATH = "/Users/scottwang/PycharmProjects/JSD/JLiNER/models/round 3-1"  # Path to your fine-tuned folder
BENCHMARK_CSV = "/Users/scottwang/PycharmProjects/JSD/JLiNER/amazon/bronze_final_cleaned.csv"

LABELS = [
    "reviewer's gender indication",
    "medical condition related",
    "author's minor children related"
]

# Map CSV columns to Model Labels
COL_MAP = {
    "gender_col": "reviewer's gender indication",
    "medical_col": "medical condition related",
    "children_col": "minor children related"
}


# ------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------
def normalize(text):
    """Fuzzy matching normalization (ignores punctuation/case)"""
    if not text or pd.isna(text): return ""
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return " ".join(text.split())


def get_metrics(tp, fp, fn):
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
    return prec, rec, f1


# ------------------------------------------------------------------
# MAIN OPTIMIZER
# ------------------------------------------------------------------
def optimize():
    # 1. Load Model and Data
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"⏳ Loading model onto {device.upper()}...")
    model = GLiNER.from_pretrained(MODEL_PATH).to(device)
    df = pd.read_csv(BENCHMARK_CSV)

    # 2. Pre-compute Predictions at very low threshold
    # This allows us to "simulate" higher thresholds without re-running the model
    print("🚀 Pre-computing all possible entities (this takes a moment)...")
    raw_predictions = []
    ground_truths = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        text = str(row.get('original_text') or row.get('original_sentence') or "")

        # Store model output (using 0.1 to catch everything)
        preds = model.predict_entities(text, LABELS, threshold=0.1)
        raw_predictions.append(preds)

        # Store ground truth (split by semicolon)
        row_gt = {label: [] for label in LABELS}
        for col, label in COL_MAP.items():
            val = str(row.get(col, ""))
            if val.lower() not in ["nan", "", "none"]:
                row_gt[label] = [normalize(x) for x in val.split(";") if x.strip()]
        ground_truths.append(row_gt)

    # 3. Search Loop
    print("\n🔍 Testing thresholds from 0.30 to 0.95...")
    results = []

    thresholds = np.arange(0.3, 1.0, 0.05)

    for t in thresholds:
        total_tp, total_fp, total_fn = 0, 0, 0

        for i in range(len(raw_predictions)):
            # Filter pre-computed predictions by current threshold 't'
            current_preds = [p for p in raw_predictions[i] if p['score'] >= t]
            pred_by_cat = {label: [normalize(p['text']) for p in current_preds if p['label'] == label] for label in
                           LABELS}
            gt_by_cat = ground_truths[i]

            for label in LABELS:
                gts = gt_by_cat[label].copy()
                prs = pred_by_cat[label].copy()

                # Matching logic
                for gt in gts:
                    match = False
                    for pr in prs:
                        if gt in pr or pr in gt:
                            total_tp += 1
                            prs.remove(pr)
                            match = True
                            break
                    if not match:
                        total_fn += 1
                total_fp += len(prs)

        p, r, f1 = get_metrics(total_tp, total_fp, total_fn)
        results.append({"threshold": t, "precision": p, "recall": r, "f1": f1})
        print(f"T: {t:.2f} | Prec: {p:.3f} | Rec: {r:.3f} | F1: {f1:.3f}")

    # 4. Final Report
    res_df = pd.DataFrame(results)
    best = res_df.loc[res_df['f1'].idxmax()]

    print("\n" + "=" * 50)
    print("🏆 OPTIMIZATION COMPLETE")
    print("=" * 50)
    print(f"Best Threshold:  {best['threshold']:.2f}")
    print(f"Max F1-Score:    {best['f1']:.4f}")
    print(f"Final Precision: {best['precision']:.4f}")
    print(f"Final Recall:    {best['recall']:.4f}")
    print("=" * 50)
    print(f"👉 Use threshold={best['threshold']:.2f} in your production script.")


if __name__ == "__main__":
    optimize()