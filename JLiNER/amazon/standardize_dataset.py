import pandas as pd
import re
import os

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
INPUT_FILE = "/Users/scottwang/PycharmProjects/JSD/merged_datasets.csv"
OUTPUT_FILE = "merged_cleaned.csv"

TEXT_COLUMN = "original_text"
PII_COLUMNS = ["gender_col", "medical_col", "minor_col"]


def robust_normalize(text):
    if not text or pd.isna(text): return ""
    text = str(text).lower()

    # 1. Specifically remove possessive 's so "son's" becomes "son"
    # The space before 's? handles cases where it's at the end of a word
    text = re.sub(r"'s\b", "", text)

    # 2. Remove all other punctuation
    text = re.sub(r'[^a-z0-9\s]', '', text)

    # 3. Collapse whitespace
    return " ".join(text.split())


def validate_and_extract(potential_fragment, normalized_original, original_text):
    """
    Determines if a fragment is a single entity or a comma-separated list.
    Returns a list of valid entities found in the original text.
    """
    fragment = potential_fragment.strip()
    if not fragment:
        return []

    norm_fragment = robust_normalize(fragment)

    # 1. If the whole fragment (with commas) exists in the text, keep it as one.
    # This protects phrases like "My 5, 3, and 2 year olds"
    if norm_fragment in normalized_original:
        return [fragment]

    # 2. If the whole fragment is NOT found, it might be a comma-separated list.
    # Try splitting by comma.
    sub_entities = fragment.split(',')
    found_entities = []

    for sub in sub_entities:
        sub = sub.strip()
        norm_sub = robust_normalize(sub)
        if norm_sub and norm_sub in normalized_original:
            found_entities.append(sub)

    return found_entities


def process_row(row):
    original_text = str(row[TEXT_COLUMN])
    normalized_original = robust_normalize(original_text)

    for col in PII_COLUMNS:
        if col not in row or pd.isna(row[col]):
            row[col] = ""
            continue

        # First pass: Split by the "Hard" separators (Newlines and Semicolons)
        # We don't split by comma yet because it might be internal to the phrase.
        initial_pieces = re.split(r'[\n;]', str(row[col]))

        final_entities = []
        seen_normalized = set()

        for piece in initial_pieces:
            # Second pass: Smart validation (handles internal vs separator commas)
            validated_list = validate_and_extract(piece, normalized_original, original_text)

            for entity in validated_list:
                norm_e = robust_normalize(entity)
                if norm_e not in seen_normalized:
                    final_entities.append(entity)
                    seen_normalized.add(norm_e)

        # Update column with standard semicolon separation
        row[col] = "; ".join(final_entities)

    return row


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: {INPUT_FILE} not found.")
        return

    print(f"📂 Loading {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE, encoding='utf-8-sig')

    print("🔍 Cleaning separators and removing hallucinations...")
    # Apply the logic row by row
    df = df.apply(process_row, axis=1)

    # Save
    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

    print("-" * 50)
    print(f"✅ SUCCESS!")
    print(f"💾 File saved: {OUTPUT_FILE}")
    print(f"💡 Strategy: Internal commas preserved, separator commas split.")


if __name__ == "__main__":
    main()