import pandas as pd
import json
import os
import torch
from gliner import GLiNER
from tqdm import tqdm

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
INPUT_FILE = "gold_final_cleaned.csv"
SAMPLE_SIZE = 300  # Number of rows to benchmark
THRESHOLD = 0.35  # Standard confidence threshold

MODELS = {
    "NuNER-Zero": "numind/NuNER_Zero-span",
    "NVIDIA-PII": "nvidia/gliner-PII",
    "GLiNER-X-Large": "knowledgator/gliner-x-large"
}

# The categories Gemma labeled for us
LABELS = [
    "occupation indication",
    "medical condition related",
    "author's minor children related"
]


# ------------------------------------------------------------------
# HELPER: MATCHING LOGIC
# ------------------------------------------------------------------
def calculate_matches(true_entities, pred_entities):
    """
    Compares LLM Ground Truth vs GLiNER Predictions.
    Uses substring matching to handle 'son' vs 'my son'.
    """
    tps = 0
    fps = 0
    fns = 0

    # 1. Prepare Ground Truth list (split semicolons from Gemma's output)
    gt_list = []
    for val in true_entities:
        if val and str(val).lower() != 'nan':
            # Gemma separated multiple items with ;
            gt_list.extend([x.strip().lower() for x in str(val).split(';') if x.strip()])

    # 2. Prepare Prediction list
    p_list = [x['text'].lower().strip() for x in pred_entities]

    # 3. Calculate True Positives & False Negatives
    temp_p = p_list.copy()
    for gt in gt_list:
        found = False
        for p in temp_p:
            # Check if one is a substring of the other
            if gt in p or p in gt:
                tps += 1
                temp_p.remove(p)
                found = True
                break
        if not found:
            fns += 1

    # 4. Remaining items in prediction list are False Positives
    fps = len(temp_p)

    return tps, fps, fns


# ------------------------------------------------------------------
# MAIN EXECUTION
# ------------------------------------------------------------------
def run_benchmark():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: {INPUT_FILE} not found.")
        return

    # Read data and ensure we drop rows where LLM found absolutely nothing
    # (to make the benchmark focused on extraction quality)
    df_raw = pd.read_csv(INPUT_FILE)
    df = df_raw.head(SAMPLE_SIZE)

    print(f"📊 Benchmarking {len(df)} rows across {len(MODELS)} models...")
    print(f"🎯 Categories: {LABELS}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results_table = []

    # Iterate Models
    for model_alias, model_path in MODELS.items():
        print(f"\n⏳ Evaluating {model_alias} ({model_path})...")

        try:
            model = GLiNER.from_pretrained(model_path)
            model.to(device)
            model.eval()
        except Exception as e:
            print(f"❌ Could not load {model_alias}: {e}")
            continue

        total_tp = 0
        total_fp = 0
        total_fn = 0

        # Process rows
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Scanning {model_alias}"):
            text = str(row['original_text'])

            # Ground truth columns from your labeling script
            true_pii = [
                row.get('occupation_col', ''),
                row.get('medical_col', ''),
                row.get('children_col', '')
            ]

            # Model Inference
            with torch.no_grad():
                pred_entities = model.predict_entities(text, LABELS, threshold=THRESHOLD)

            # Match
            tp, fp, fn = calculate_matches(true_pii, pred_entities)
            total_tp += tp
            total_fp += fp
            total_fn += fn

        # Metrics Calculation
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        results_table.append({
            "Model": model_alias,
            "F1-Score": round(f1, 3),
            "Precision": round(precision, 3),
            "Recall": round(recall, 3),
            "Hits (TP)": total_tp,
            "Misses (FN)": total_fn,
            "False Alarms (FP)": total_fp
        })

    # Display Results
    report_df = pd.DataFrame(results_table).sort_values(by="F1-Score", ascending=False)
    print("\n" + "=" * 90)
    print("🏆 GLiNER TRI-MODEL BENCHMARK SUMMARY")
    print("=" * 90)
    print(report_df.to_string(index=False))
    print("=" * 90)
    print("Interpretation:")
    print(" • Higher Recall    -> Better at catching PII (Safer for Privacy)")
    print(" • Higher Precision -> Fewer mistakes (Cleaner data, less manual work)")
    print(" • Ground Truth     -> Gemma 3 12B / Llama 4 Scout Extractions")


if __name__ == "__main__":
    run_benchmark()