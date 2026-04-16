import pandas as pd
import re
from difflib import SequenceMatcher
from tqdm import tqdm

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
HUMAN_FILE = "/Users/scottwang/PycharmProjects/JSD/merged_datasets.csv"
LLM_FILE = "/Users/scottwang/PycharmProjects/JSD/JLiNER/amazon/merged_test_llm.csv"
OUTPUT_REPORT = "high_signal_audit_report.csv"

CATEGORIES = ["gender_col", "medical_col", "minor_col"]
STOP_WORDS = {'im', 'i', 'a', 'the', 'is', 'am', 'was', 'my', 'our', 'for', 'and', 'with', 'to'}


# ------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------
def advanced_normalize(text):
    if not text or pd.isna(text) or str(text).lower() in ['nan', 'none', '', '[]']:
        return set()
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    tokens = text.split()
    clean_tokens = []
    for t in tokens:
        if t not in STOP_WORDS:
            if t.endswith('s') and len(t) > 3: t = t[:-1]
            clean_tokens.append(t)
    return set(clean_tokens)


def get_token_similarity(h_str, l_str):
    h_set = advanced_normalize(h_str)
    l_set = advanced_normalize(l_str)
    if not h_set and not l_set: return 1.0
    if not h_set or not l_set: return 0.0
    intersection = h_set.intersection(l_set)
    union = h_set.union(l_set)
    return len(intersection) / len(union)


def calculate_row_metrics(h_val, l_val):
    """
    Calculates Precision and Recall for a single cell.
    Strictly ignores 'nan', 'none', and empty strings.
    """

    # 1. Helper to filter out garbage strings
    def get_clean_list(val):
        if pd.isna(val): return []
        s_val = str(val).strip().lower()
        if s_val in ['nan', 'none', '', '[]', 'null']: return []

        # Split and remove 'nan' fragments from inside a list
        parts = [x.strip() for x in s_val.split(';') if x.strip()]
        return [p for p in parts if p not in ['nan', 'none', 'null']]

    h_list = get_clean_list(h_val)
    l_list = get_clean_list(l_val)

    # CRITICAL: If the HUMAN didn't find anything, this is NOT a high-signal row.
    # Return None so the main loop ignores this row for the 'Real' F1 calculation.
    if not h_list:
        return None

        # RECALL: How many of the human items did the LLM find?
    tp = 0
    for h_ent in h_list:
        # Check against the LLM list
        best_match = 0
        if l_list:
            best_match = max([get_token_similarity(h_ent, l_ent) for l_ent in l_list])

        if best_match >= 0.7:
            tp += 1

    recall = tp / len(h_list)

    # PRECISION: How many of the LLM items were actually valid?
    valid_l = 0
    if not l_list:
        precision = 0.0
    else:
        for l_ent in l_list:
            best_match = max([get_token_similarity(l_ent, h_ent) for h_ent in h_list])
            if best_match >= 0.7:
                valid_l += 1
        precision = valid_l / len(l_list)

    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {"precision": precision, "recall": recall, "f1": f1}


# ------------------------------------------------------------------
# MAIN EXECUTION
# ------------------------------------------------------------------
def main():
    df_h = pd.read_csv(HUMAN_FILE)
    df_l = pd.read_csv(LLM_FILE)

    final_results = []

    print(f"🚀 Filtering for High-Signal Rows (where Human found PII)...")

    category_summary = {cat: {"prec": [], "rec": [], "f1": []} for cat in CATEGORIES}

    for i in tqdm(range(min(len(df_h), len(df_l)))):
        h_row = df_h.iloc[i]
        l_row = df_l.iloc[i]

        row_has_pii = False
        row_report = {"row_index": i, "text": str(h_row['original_text'])[:100]}

        for cat in CATEGORIES:
            metrics = calculate_row_metrics(h_row[cat], l_row[cat])

            if metrics:
                row_has_pii = True
                row_report[f"{cat}_f1"] = metrics['f1']
                row_report[f"{cat}_recall"] = metrics['recall']

                category_summary[cat]["prec"].append(metrics['precision'])
                category_summary[cat]["rec"].append(metrics['recall'])
                category_summary[cat]["f1"].append(metrics['f1'])
            else:
                row_report[f"{cat}_f1"] = "N/A (True Negative)"

        if row_has_pii:
            final_results.append(row_report)

    # Output Results
    print("\n" + "=" * 60)
    print("🔥 HIGH-SIGNAL PERFORMANCE REPORT (NO INFLATION)")
    print("=" * 60)
    print(f"{'Category':<15} | {'Count':<5} | {'Prec':<6} | {'Rec':<6} | {'F1':<6}")
    print("-" * 60)

    global_f1 = []
    for cat in CATEGORIES:
        c = category_summary[cat]
        if c["f1"]:
            avg_p = sum(c["prec"]) / len(c["prec"])
            avg_r = sum(c["rec"]) / len(c["rec"])
            avg_f = sum(c["f1"]) / len(c["f1"])
            global_f1.append(avg_f)
            print(f"{cat.upper():<15} | {len(c['f1']):<5} | {avg_p:.3f} | {avg_r:.3f} | {avg_f:.3f}")
        else:
            print(f"{cat.upper():<15} | 0     | 0.000  | 0.000  | 0.000")

    print("-" * 60)
    print(f"REAL SYSTEM F1 SCORE: {sum(global_f1) / len(global_f1):.3f}")
    print("=" * 60)

    pd.DataFrame(final_results).to_csv(OUTPUT_REPORT, index=False)
    print(f"💾 Disagreement report saved to: {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()