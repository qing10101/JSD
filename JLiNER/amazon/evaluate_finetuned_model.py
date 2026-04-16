import pandas as pd
import torch
import os
import re
from gliner import GLiNER
from tqdm import tqdm
from collections import defaultdict

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
INPUT_FILE = "/Users/scottwang/PycharmProjects/JSD/old_sheet.csv"

# Paths to the models
MODEL_FINETUNED = "/Users/scottwang/PycharmProjects/JSD/JLiNER/models/round Alpha"
MODEL_BASELINE = "numind/NuNER_Zero-span"

ROW_LIMIT = 500
THRESHOLD = 0.5

LABELS = [
    "reviewer's gender indication",
    "medical condition related",
    "minor children related"
]

COL_MAP = {
    "gender_col": "reviewer's gender indication",
    "medical_col": "medical condition related",
    "minor_col": "minor children related"
}


# ------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------
def robust_normalize(text):
    """
    Standardizes text for comparison:
    1. Lowercase
    2. Removes possessive 's (son's -> son)
    3. Removes all punctuation
    4. Collapses whitespace
    """
    if not text or pd.isna(text): return ""
    text = str(text).lower()
    # Remove possessive 's
    text = re.sub(r"'s\b", "", text)
    # Remove punctuation
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return " ".join(text.split())


def get_metrics(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1


# ------------------------------------------------------------------
# EVALUATION LOGIC
# ------------------------------------------------------------------
def evaluate_model(model_path, df):
    print(f"\n⏳ Loading model for evaluation: {model_path}...")
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = GLiNER.from_pretrained(model_path).to(device)

    stats = {label: {"tp": 0, "fp": 0, "fn": 0} for label in LABELS}

    # Use actual min to avoid out of bounds
    num_to_process = min(len(df), ROW_LIMIT)
    subset_df = df.head(num_to_process)

    for _, row in tqdm(subset_df.iterrows(), total=num_to_process, desc=f"Evaluating {os.path.basename(model_path)}"):
        # Handle different potential column names for text
        text = str(row.get('ori_review') or row.get('original_text') or "")
        if not text: continue

        # 1. Get Ground Truth (Normalized)
        gt_by_cat = {}
        for col, label in COL_MAP.items():
            val = str(row.get(col, ""))
            if val.lower() in ["nan", "", "none"]:
                gt_by_cat[label] = []
            else:
                # Normalize every entity in the ground truth
                gt_by_cat[label] = [robust_normalize(x) for x in val.split(";") if x.strip()]

        # 2. Get Model Predictions (Normalized)
        preds = model.predict_entities(text, LABELS, threshold=THRESHOLD)
        pred_by_cat = {label: [] for label in LABELS}
        for p in preds:
            # Normalize every predicted entity
            pred_by_cat[p['label']].append(robust_normalize(p['text']))

        # 3. Compare per category using normalized strings
        for label in LABELS:
            gts = gt_by_cat[label].copy()
            prs = pred_by_cat[label].copy()

            for gt in gts:
                match = False
                for pr in prs:
                    # Check for exact normalized match or substring match
                    if gt == pr or gt in pr or pr in gt:
                        stats[label]["tp"] += 1
                        prs.remove(pr)
                        match = True
                        break
                if not match:
                    stats[label]["fn"] += 1

            # Remaining items in prediction list are False Positives
            stats[label]["fp"] += len(prs)

    return stats


# ------------------------------------------------------------------
# MAIN EXECUTION
# ------------------------------------------------------------------
def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: {INPUT_FILE} not found.")
        return

    df = pd.read_csv(INPUT_FILE)

    # Run evaluations
    baseline_results = evaluate_model(MODEL_BASELINE, df)
    finetuned_results = evaluate_model(MODEL_FINETUNED, df)

    # Compile Final Report
    report = []

    for label in LABELS:
        b_p, b_r, b_f1 = get_metrics(baseline_results[label]["tp"], baseline_results[label]["fp"],
                                     baseline_results[label]["fn"])
        f_p, f_r, f_f1 = get_metrics(finetuned_results[label]["tp"], finetuned_results[label]["fp"],
                                     finetuned_results[label]["fn"])

        report.append({
            "Category": label,
            "Base F1": f"{b_f1:.3f}",
            "Fine-Tuned F1": f"{f_f1:.3f}",
            "Recall Δ": f"{(f_r - b_r):+.2f}",
            "Precision Δ": f"{(f_p - b_p):+.2f}"
        })

    total_b_f1 = sum([float(x["Base F1"]) for x in report]) / len(LABELS)
    total_f_f1 = sum([float(x["Fine-Tuned F1"]) for x in report]) / len(LABELS)

    print("\n" + "=" * 85)
    print(f"🏆 FINAL PERFORMANCE EVALUATION (Threshold: {THRESHOLD})")
    print("=" * 85)
    report_df = pd.DataFrame(report)
    print(report_df.to_string(index=False))
    print("-" * 85)
    print(f"OVERALL SYSTEM SCORE: Baseline F1: {total_b_f1:.3f} | Fine-Tuned F1: {total_f_f1:.3f}")

    if total_b_f1 > 0:
        improvement = ((total_f_f1 - total_b_f1) / total_b_f1) * 100
        print(f"TOTAL PERFORMANCE IMPROVEMENT: {improvement:+.1f}%")

    print("=" * 85)


if __name__ == "__main__":
    main()