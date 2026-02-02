import pandas as pd
import torch
import os
from gliner import GLiNER
from tqdm import tqdm
from collections import defaultdict

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
# The file created by your labeling script
INPUT_FILE = "/Users/scottwang/PycharmProjects/JSD/developing/amazon/test_auto_labeled_new.csv"

# Paths to the models
MODEL_FINETUNED = "gliner_nuner_finetuned"  # Your local folder
MODEL_BASELINE = "numind/NuNER_Zero-span"  # The original

# Evaluation Settings
THRESHOLD = 0.5  # You can experiment with this (0.4 to 0.7)
LABELS = [
    "occupation indication",
    "medical condition related",
    "children/minor related"
]

# Map CSV columns to Model Labels for individual category scoring
COL_MAP = {
    "occupation_col": "occupation indication",
    "medical_col": "medical condition related",
    "children_col": "children/minor related"
}


# ------------------------------------------------------------------
# METRIC CALCULATION LOGIC
# ------------------------------------------------------------------
def get_metrics(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1


def evaluate_model(model_path, df):
    print(f"\n⏳ Loading model for evaluation: {model_path}...")
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = GLiNER.from_pretrained(model_path).to(device)

    # Trackers per category: { label: {tp: 0, fp: 0, fn: 0} }
    stats = {label: {"tp": 0, "fp": 0, "fn": 0} for label in LABELS}

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Analyzing Rows"):
        text = str(row['original_sentence'])

        # 1. Get Ground Truth for this row
        gt_by_cat = {}
        for col, label in COL_MAP.items():
            val = str(row.get(col, ""))
            if val.lower() in ["nan", "", "none"]:
                gt_by_cat[label] = []
            else:
                gt_by_cat[label] = [x.strip().lower() for x in val.split(";") if x.strip()]

        # 2. Get Model Predictions
        preds = model.predict_entities(text, LABELS, threshold=THRESHOLD)
        pred_by_cat = {label: [] for label in LABELS}
        for p in preds:
            pred_by_cat[p['label']].append(p['text'].lower().strip())

        # 3. Compare per category
        for label in LABELS:
            gts = gt_by_cat[label].copy()
            prs = pred_by_cat[label].copy()

            for gt in gts:
                match = False
                for pr in prs:
                    if gt in pr or pr in gt:  # Substring match
                        stats[label]["tp"] += 1
                        prs.remove(pr)
                        match = True
                        break
                if not match:
                    stats[label]["fn"] += 1

            # Remaining predictions are False Positives
            stats[label]["fp"] += len(prs)

    return stats


# ------------------------------------------------------------------
# MAIN EXECUTION
# ------------------------------------------------------------------
def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: {INPUT_FILE} not found. Run the Groq script first.")
        return

    df = pd.read_csv(INPUT_FILE)

    # Run evaluations
    baseline_results = evaluate_model(MODEL_BASELINE, df)
    finetuned_results = evaluate_model(MODEL_FINETUNED, df)

    # Compile Final Report
    report = []

    for label in LABELS:
        # Baseline Metrics
        b_p, b_r, b_f1 = get_metrics(baseline_results[label]["tp"], baseline_results[label]["fp"],
                                     baseline_results[label]["fn"])
        # Finetuned Metrics
        f_p, f_r, f_f1 = get_metrics(finetuned_results[label]["tp"], finetuned_results[label]["fp"],
                                     finetuned_results[label]["fn"])

        report.append({
            "Category": label,
            "Base F1": f"{b_f1:.3f}",
            "Fine-Tuned F1": f"{f_f1:.3f}",
            "Recall Δ": f"{(f_r - b_r):+.2f}",
            "Precision Δ": f"{(f_p - b_p):+.2f}"
        })

    # Summary Row (Global Averages)
    total_b_f1 = sum([float(x["Base F1"]) for x in report]) / len(LABELS)
    total_f_f1 = sum([float(x["Fine-Tuned F1"]) for x in report]) / len(LABELS)

    print("\n" + "=" * 85)
    print(f"🏆 FINAL PERFORMANCE EVALUATION (Threshold: {THRESHOLD})")
    print("=" * 85)
    report_df = pd.DataFrame(report)
    print(report_df.to_string(index=False))
    print("-" * 85)
    print(f"OVERALL SYSTEM SCORE: Baseline F1: {total_b_f1:.3f} | Fine-Tuned F1: {total_f_f1:.3f}")
    improvement = ((total_f_f1 - total_b_f1) / (total_b_f1 if total_b_f1 > 0 else 1)) * 100
    print(f"TOTAL PERFORMANCE IMPROVEMENT: {improvement:+.1f}%")
    print("=" * 85)


if __name__ == "__main__":
    main()