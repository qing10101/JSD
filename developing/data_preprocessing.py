import csv
import os
import random
from gliner import GLiNER

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
INPUT_FILE = "full_test.csv"
OUTPUT_FILE = "filtered_dataset_1_3rd.csv"

# LIMITS & THRESHOLDS
MAX_INPUT_ROWS = None  # Set to None to process the whole file
THRESHOLD = 0.25  # Confidence threshold for GLiNER detection

# FILTER LOGIC
# "Drop two-thirds" means "Keep one-third"
# 1/3 = 0.33
KEEP_RATIO_FOR_IRRELEVANT = 0.33

# The 3 specific categories to focus on
LABELS = [
    "occupation indication",
    "medical condition related",
    "children/minor related"
]

# ------------------------------------------------------------------
# SETUP
# ------------------------------------------------------------------
print(f"⏳ Loading GLiNER model (numind/NuNER_Zero-span)...")
model = GLiNER.from_pretrained("numind/NuNER_Zero-span")

print(f"🚀 Starting Filter Process...")
if MAX_INPUT_ROWS:
    print(f"   - Input Limit: Process first {MAX_INPUT_ROWS} rows only.")
print(f"   - Rules: Keep ALL rows containing '{LABELS}'")
print(f"   - Rules: Drop 4/5ths ({1 - KEEP_RATIO_FOR_IRRELEVANT:.0%}) of the rest.")
print("-" * 60)

# ------------------------------------------------------------------
# MAIN LOOP
# ------------------------------------------------------------------
stats = {
    "total_read": 0,
    "kept_pii": 0,
    "kept_random": 0,
    "discarded": 0
}

if not os.path.exists(INPUT_FILE):
    print(f"❌ Error: {INPUT_FILE} not found.")
    exit()

with open(INPUT_FILE, 'r', encoding='utf-8-sig', errors='replace') as infile, \
        open(OUTPUT_FILE, 'w', encoding='utf-8-sig', newline='') as outfile:
    reader = csv.DictReader(infile)

    # Verify input columns
    if 'original_sentence' not in reader.fieldnames:
        print("❌ Error: 'original_sentence' column missing in input CSV.")
        exit()

    # Prepare Output CSV
    fieldnames = reader.fieldnames + ['filter_reason']
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()

    for i, row in enumerate(reader, 1):
        # --- STOP CHECK ---
        if MAX_INPUT_ROWS is not None and i > MAX_INPUT_ROWS:
            print(f"🛑 Limit of {MAX_INPUT_ROWS} input rows reached. Stopping.")
            break

        stats["total_read"] += 1
        text = row.get('original_sentence', '').strip()

        if not text:
            continue

        # 1. Run GLiNER Detection
        entities = model.predict_entities(
            text,
            LABELS,
            threshold=THRESHOLD,
            flat_ner=False
        )

        # 2. Decision Logic
        if entities:
            # Case A: PII Detected -> KEEP IT (100%)
            detected_types = set([e['label'] for e in entities])
            reason = f"Found: {', '.join(detected_types)}"

            row['filter_reason'] = reason
            writer.writerow(row)
            stats["kept_pii"] += 1

            print(f"[{i}] ✅ Kept (PII): {text[:60]}...")

        else:
            # Case B: No PII -> Randomly Sample (Keep 20%, Drop 80%)
            if random.random() < KEEP_RATIO_FOR_IRRELEVANT:
                row['filter_reason'] = "Random Sample (No PII)"
                writer.writerow(row)
                stats["kept_random"] += 1
            else:
                stats["discarded"] += 1

# ------------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------------
print("-" * 60)
print(f"🏁 Filtering Complete.")
print(f"   📂 Output saved to: {OUTPUT_FILE}")
print(f"   -------------------------------")
print(f"   Total Input Rows:     {stats['total_read']}")
print(f"   ✅ Kept (Contains PII): {stats['kept_pii']}")
print(f"   🎲 Kept (Random 20%):   {stats['kept_random']}")
print(f"   🗑️  Discarded (80%):    {stats['discarded']}")
print(f"   -------------------------------")
print(f"   New File Size:        {stats['kept_pii'] + stats['kept_random']} rows")