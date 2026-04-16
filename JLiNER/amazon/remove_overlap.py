import pandas as pd
import os

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
FILE_A = "amazon_to_label.csv"  # The large dataset
FILE_B = "already_processed.csv"  # The dataset you want to remove
OUTPUT_FILE = "amazon_cleaned_fresh.csv"

# Columns that contain the text content
COL_A = "original_text"
COL_B = "original_text"


def remove_overlap():
    if not os.path.exists(FILE_A) or not os.path.exists(FILE_B):
        print("❌ Error: One of the input files not found.")
        return

    print(f"📂 Loading datasets...")
    df_a = pd.read_csv(FILE_A)
    df_b = pd.read_csv(FILE_B)

    # 1. Normalize text for accurate matching
    # We strip and lower to avoid missing matches due to minor formatting diffs
    def norm(s): return str(s).strip().lower()

    df_a['temp_norm'] = df_a[COL_A].apply(norm)
    df_b['temp_norm'] = df_b[COL_B].apply(norm)

    # 2. Perform Anti-Join
    # This keeps rows in df_a where the normalized text is NOT in df_b's normalized text
    mask = ~df_a['temp_norm'].isin(df_b['temp_norm'])
    df_filtered = df_a[mask].copy()

    # 3. Cleanup
    df_filtered = df_filtered.drop(columns=['temp_norm'])

    # 4. Save
    print(f"📉 Original size: {len(df_a)} rows")
    print(f"✂️ Removed: {len(df_a) - len(df_filtered)} overlapping rows")
    print(f"✅ Final size: {len(df_filtered)} rows")

    df_filtered.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"💾 Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    remove_overlap()