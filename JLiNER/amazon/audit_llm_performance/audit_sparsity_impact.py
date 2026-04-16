import pandas as pd
import numpy as np
from tqdm import tqdm

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
HUMAN_FILE = "/Users/scottwang/PycharmProjects/JSD/merged_datasets.csv"
LLM_FILE = "/Users/scottwang/PycharmProjects/JSD/JLiNER/amazon/merged_test_llm.csv"
CATEGORIES = ["gender_col", "medical_col", "minor_col"]


def is_empty(val):
    if not val or pd.isna(val) or str(val).lower() in ['nan', 'none', '', '[]']:
        return True
    return False


def analyze_sparsity():
    df_h = pd.read_csv(HUMAN_FILE)
    df_l = pd.read_csv(LLM_FILE)

    total_rows = len(df_h)

    print(f"📊 Analyzing Sparsity for {total_rows} rows...")

    report = []

    for cat in CATEGORIES:
        # 1. Identify row types
        h_pos_mask = df_h[cat].apply(lambda x: not is_empty(x))
        l_pos_mask = df_l[cat].apply(lambda x: not is_empty(x))

        # True Negatives (TN): Both agree there is NO PII
        tn_count = ((~h_pos_mask) & (~l_pos_mask)).sum()

        # True Positives (TP): Both agree there IS PII (doesn't mean strings match yet)
        tp_indicator_count = (h_pos_mask & l_pos_mask).sum()

        # False Negatives (FN): Human found PII, LLM missed it
        fn_count = (h_pos_mask & (~l_pos_mask)).sum()

        # False Positives (FP): Human said Clean, LLM found PII
        fp_count = ((~h_pos_mask) & l_pos_mask).sum()

        # 2. Calculate Density
        pii_density = h_pos_mask.mean()

        # 3. Calculate "Pure" Categorical Accuracy (Binary)
        # This is the accuracy of just detecting IF there is PII, ignoring the text content
        binary_accuracy = (tn_count + tp_indicator_count) / total_rows

        # 4. Calculate Sparsity Inflation Factor
        # How much of your 0.888 score is just from "agreeing on empty cells"?
        inflation = tn_count / total_rows

        report.append({
            "Category": cat.upper(),
            "Density (PII Presence)": f"{pii_density:.1%}",
            "True Negatives": tn_count,
            "False Negatives": fn_count,
            "False Positives": fp_count,
            "Detection Accuracy": f"{binary_accuracy:.3f}",
            "Sparsity Inflation": f"{inflation:.1%}"
        })

    # Display results
    report_df = pd.DataFrame(report)
    print("\n" + "=" * 70)
    print("🔍 SPARSITY & MASKING AUDIT")
    print("=" * 70)
    print(report_df.to_string(index=False))
    print("-" * 70)

    avg_density = report_df["Density (PII Presence)"].apply(lambda x: float(x.strip('%'))).mean()
    print(f"OVERALL DATASET DENSITY: {avg_density:.1f}%")
    print(f"Meaning: {100 - avg_density:.1f}% of the data contains NO entities.")

    print("\n💡 INTERPRETATION:")
    if avg_density < 15:
        print("⚠️  HIGH SPARSITY: Your previous 0.888 score is likely inflated by True Negatives.")
        print("   The LLM might be 'lazy' and getting rewarded for it.")
    else:
        print("✅ HEALTHY DENSITY: Your scores are representative of actual model performance.")
    print("=" * 70)


if __name__ == "__main__":
    analyze_sparsity()