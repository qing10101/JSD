import pandas as pd
import re
from difflib import SequenceMatcher
from tqdm import tqdm

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
HUMAN_FILE = "/Users/scottwang/PycharmProjects/JSD/merged_datasets.csv"
LLM_FILE = "/Users/scottwang/PycharmProjects/JSD/JLiNER/amazon/merged_test_llm.csv"
OUTPUT_REPORT = "similarity_audit_by_index.csv"

# Category columns to compare
CATEGORIES = ["gender_col", "medical_col", "minor_col"]

# The text columns (Change if your headers are different)
TEXT_COL_HUMAN = "original_text"
TEXT_COL_LLM = "original_text"

# Words that don't carry PII meaning
STOP_WORDS = {'im', 'i', 'a', 'the', 'is', 'am', 'was', 'my', 'our', 'for', 'and', 'with', 'to'}


# ------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------
def advanced_normalize(text):
    """Cleans text, removes stop words, handles basic plurals."""
    if not text or pd.isna(text) or str(text).lower() in ['nan', 'none', '']:
        return set()

    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)

    tokens = text.split()
    clean_tokens = []
    for t in tokens:
        if t not in STOP_WORDS:
            # Simple plural removal
            if t.endswith('s') and len(t) > 3:
                t = t[:-1]
            clean_tokens.append(t)
    return set(clean_tokens)


def get_token_similarity(h_str, l_str):
    """Order-independent Jaccard similarity."""
    h_set = advanced_normalize(h_str)
    l_set = advanced_normalize(l_str)

    if not h_set and not l_set: return 1.0
    if not h_set or not l_set: return 0.0

    intersection = h_set.intersection(l_set)
    union = h_set.union(l_set)
    return len(intersection) / len(union)


def compare_entities_fuzzy(h_str, l_str, l_text_raw):
    """
    Handles subsets, order independence, and stop words.
    """
    h_list = [x.strip() for x in str(h_str).split(';') if x.strip()]
    l_list = [x.strip() for x in str(l_str).split(';') if x.strip()]
    l_text_norm_set = advanced_normalize(l_text_raw)

    if not h_list and not l_list:
        return 1.0, 1.0, 0

    # 1. RECALL Direction (Human -> LLM)
    tp_count = 0
    out_of_scope = 0
    recall_total_sim = 0

    for h_ent in h_list:
        h_set = advanced_normalize(h_ent)
        # Check if the text the LLM saw actually contained this PII
        if h_set and not h_set.issubset(l_text_norm_set):
            out_of_scope += 1
            continue

        best_match = max([get_token_similarity(h_ent, l_ent) for l_ent in l_list]) if l_list else 0
        if best_match >= 0.7: tp_count += 1
        recall_total_sim += best_match

    effective_h_count = len(h_list) - out_of_scope
    recall = tp_count / effective_h_count if effective_h_count > 0 else 1.0

    # 2. PRECISION Direction (LLM -> Human)
    precision_total_sim = 0
    for l_ent in l_list:
        best_match = max([get_token_similarity(l_ent, h_ent) for h_ent in h_list]) if h_list else 0
        precision_total_sim += best_match

    precision = precision_total_sim / len(l_list) if l_list else 1.0

    # Overall Similarity (F1 of Precision/Recall)
    if (recall + precision) == 0:
        overall_sim = 0
    else:
        overall_sim = 2 * (precision * recall) / (precision + recall)

    return overall_sim, recall, out_of_scope


# ------------------------------------------------------------------
# MAIN EXECUTION
# ------------------------------------------------------------------
def main():
    print(f"📂 Loading files...")
    df_h = pd.read_csv(HUMAN_FILE)
    df_l = pd.read_csv(LLM_FILE)

    if len(df_h) != len(df_l):
        print(f"⚠️ Warning: Files have different lengths (Human: {len(df_h)}, LLM: {len(df_l)})")
        print(f"   The script will only compare the first {min(len(df_h), len(df_l))} rows.")

    results = []

    print(f"📊 Auditing rows by index...")
    for i in tqdm(range(min(len(df_h), len(df_l)))):
        h_row = df_h.iloc[i]
        l_row = df_l.iloc[i]

        # Safety Check: Do the original texts match?
        h_text = str(h_row[TEXT_COL_HUMAN])
        l_text = str(l_row[TEXT_COL_LLM])

        # Check alignment by looking at a snippet
        if h_text[:30].lower() != l_text[:30].lower() and l_text.lower() not in h_text.lower():
            print(f"\n🚨 ALIGNMENT ERROR at index {i}:")
            print(f"   Human Text: {h_text[:50]}...")
            print(f"   LLM Text:   {l_text[:50]}...")
            # We continue, but this row will likely have 0 similarity

        row_report = {"row_index": i}
        total_row_sim = 0

        for cat in CATEGORIES:
            sim, rec, oos = compare_entities_fuzzy(
                h_row[cat],
                l_row[cat],
                l_text
            )
            row_report[f"{cat}_similarity"] = round(sim, 3)
            row_report[f"{cat}_recall"] = round(rec, 3)
            row_report[f"{cat}_out_of_scope"] = oos
            total_row_sim += sim

        row_report["overall_similarity"] = round(total_row_sim / len(CATEGORIES), 3)
        row_report["llm_text_sample"] = l_text[:100]
        results.append(row_report)

    report_df = pd.DataFrame(results)

    print("\n" + "=" * 50)
    print("📈 FINAL INDEX-ALIGNED SUMMARY")
    print("=" * 50)
    for cat in CATEGORIES:
        print(
            f"{cat.upper():<15} | Sim: {report_df[f'{cat}_similarity'].mean():.3f} | Rec: {report_df[f'{cat}_recall'].mean():.3f}")

    print("-" * 50)
    print(f"Overall Dataset Similarity: {report_df['overall_similarity'].mean():.3f}")
    print("=" * 50)

    report_df.to_csv(OUTPUT_REPORT, index=False)
    print(f"💾 Report saved to: {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()