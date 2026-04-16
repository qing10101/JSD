import pandas as pd
import os

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
SMALL_SET_PATH = "/Users/scottwang/PycharmProjects/JSD/JLiNER/amazon/merged_test.csv"  # Your first file
BIG_SET_PATH = "/Users/scottwang/PycharmProjects/JSD/JLiNER/amazon/merged_test_llm.csv"  # Your second file

# Update these if your column names differ
SMALL_TEXT_COL = "original_text"
BIG_TEXT_COL = "original_text"


def normalize(text):
    """Simple normalization to catch matches despite whitespace/casing diffs."""
    return str(text).strip().lower().replace('\n', ' ')


def main():
    if not os.path.exists(SMALL_SET_PATH) or not os.path.exists(BIG_SET_PATH):
        print("❌ One of the files is missing. Check paths.")
        return

    # 1. Load Data
    print(f"📂 Loading Small Set: {SMALL_SET_PATH}")
    df_small = pd.read_csv(SMALL_SET_PATH)

    print(f"📂 Loading Big Set: {BIG_SET_PATH}")
    df_big = pd.read_csv(BIG_SET_PATH)

    # 2. Extract and Normalize Text
    small_texts = set(df_small[SMALL_TEXT_COL].apply(normalize))
    big_texts = set(df_big[BIG_TEXT_COL].apply(normalize))

    # 3. Find Intersection
    overlap = small_texts.intersection(big_texts)

    # 4. Report
    total_small = len(small_texts)
    total_big = len(big_texts)
    overlap_count = len(overlap)
    unique_to_small = total_small - overlap_count

    print("\n" + "=" * 50)
    print("📊 DATASET OVERLAP REPORT")
    print("=" * 50)
    print(f"Rows in Small Set:   {total_small}")
    print(f"Rows in Big Set:     {total_big}")
    print("-" * 50)
    print(f"🔴 Overlapping Rows:  {overlap_count} (Matches found in train set)")
    print(f"🟢 Clean Benchmark:  {unique_to_small} (New data unseen by model)")
    print("=" * 50)

    if unique_to_small > 0:
        print(f"\n✅ SUCCESS: You have {unique_to_small} rows you can use for evaluation.")

        # Optionally save the 'Clean' rows to a new file
        # clean_df = df_small[~df_small[SMALL_TEXT_COL].apply(normalize).isin(big_texts)]
        # clean_df.to_csv("clean_benchmark_set.csv", index=False)
        # print(f"💾 Saved {len(clean_df)} clean rows to 'clean_benchmark_set.csv'")
    else:
        print("\n❌ WARNING: Your small set is entirely contained within your training data.")
        print("   You cannot use this for evaluation.")


if __name__ == "__main__":
    main()