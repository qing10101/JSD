import json
import csv
import os
from gliner import GLiNER
from tqdm import tqdm

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
INPUT_FILE = "/Users/scottwang/PycharmProjects/JSD/developing/yelp_dataset/yelp_academic_dataset_review.json"
OUTPUT_FILE = "yelp_to_label.csv"

# MODEL SELECTION
# "numind/NuNER_Zero-span" is best for zero-shot mining (finding things it wasn't explicitly trained on).
# If this crashes your Mac (OOM), switch to "urchade/gliner_medium-v2.1"
MODEL_NAME = "numind/NuNER_Zero-span"

# THRESHOLDS & LIMITS
CONFIDENCE_THRESHOLD = 0.25  # Keep low to maximize Recall (let Gemma filter later)
MAX_TOKEN_LENGTH = 300  # Skip reviews longer than this to avoid truncation warnings/errors
MAX_SCAN_LIMIT = 200000  # Safety stop after scanning this many rows if quotas aren't met

# QUOTAS (Targeting ~3,500 total rows)
# We prioritize Rare categories (Medical/Children) over Common ones (Occupation)
TARGET_COUNTS = {
    "medical condition related": 800,
    "children/minor related": 800,
    "occupation indication": 800,
    "no_pii_negative": 1100  # Hard Negatives (Clean rows that might look like PII)
}

LABELS = [
    "occupation indication",
    "medical condition related",
    "children/minor related"
]

# ------------------------------------------------------------------
# SETUP
# ------------------------------------------------------------------
print(f"⏳ Loading Model: {MODEL_NAME}...")
try:
    model = GLiNER.from_pretrained(MODEL_NAME)
    model.eval()  # Set to inference mode
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("Tip: If OOM, try changing MODEL_NAME to 'urchade/gliner_small-v2.1'")
    exit()

# Trackers
current_counts = {k: 0 for k in TARGET_COUNTS.keys()}
total_saved = 0
rows_scanned = 0

print(f"🚀 Starting Mining Operation...")
print(f"   Targets: {json.dumps(TARGET_COUNTS, indent=2)}")
print("-" * 60)

# ------------------------------------------------------------------
# MAIN MINING LOOP
# ------------------------------------------------------------------
if not os.path.exists(INPUT_FILE):
    print(f"❌ Error: {INPUT_FILE} not found. Make sure it is in the same folder.")
    exit()

with open(INPUT_FILE, 'r', encoding='utf-8') as infile, \
        open(OUTPUT_FILE, 'w', encoding='utf-8-sig', newline='') as outfile:
    # We save 'gliner_hint_text' to help the LLM labeler know what to look for
    fieldnames = ['review_id', 'original_text', 'gliner_hint_category', 'gliner_hint_text']
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()

    # Progress bar based on Total Target Rows
    pbar = tqdm(total=sum(TARGET_COUNTS.values()), desc="Filling Quotas")

    for line in infile:
        # 1. STOPPING CONDITIONS
        if rows_scanned > MAX_SCAN_LIMIT:
            print(f"\n⚠️ Reached scan limit of {MAX_SCAN_LIMIT}. Stopping.")
            break

        if all(current_counts[k] >= TARGET_COUNTS[k] for k in TARGET_COUNTS):
            print("\n✅ All quotas filled!")
            break

        rows_scanned += 1

        try:
            data = json.loads(line)
            text = data.get('text', '').strip()
            review_id = data.get('review_id', 'unknown')

            # 2. FILTER: LENGTH CHECK
            # Skip very short reviews (no context) or very long reviews (truncation issues)
            # Simple whitespace tokenization is a good enough proxy for speed
            token_count = len(text.split())
            if token_count < 10 or token_count > MAX_TOKEN_LENGTH:
                continue

            # 3. GLINER INFERENCE
            entities = model.predict_entities(text, LABELS, threshold=CONFIDENCE_THRESHOLD, flat_ner=False)
            detected_labels = set([e['label'] for e in entities])

            # 4. PRIORITY SIEVE (The Bucket Logic)
            # We check buckets in order of rarity. If a bucket is full, we skip to the next.
            selected_category = None

            # Check Medical (Rare)
            if "medical condition related" in detected_labels:
                if current_counts["medical condition related"] < TARGET_COUNTS["medical condition related"]:
                    selected_category = "medical condition related"

            # Check Children (Medium Rare)
            elif "children/minor related" in detected_labels:
                if current_counts["children/minor related"] < TARGET_COUNTS["children/minor related"]:
                    selected_category = "children/minor related"

            # Check Occupation (Common)
            elif "occupation indication" in detected_labels:
                if current_counts["occupation indication"] < TARGET_COUNTS["occupation indication"]:
                    selected_category = "occupation indication"

            # Check Negatives (No PII found)
            elif not detected_labels:
                if current_counts["no_pii_negative"] < TARGET_COUNTS["no_pii_negative"]:
                    selected_category = "no_pii_negative"

            # 5. WRITE TO CSV
            if selected_category:
                # Filter hints to only include the relevant category
                # (e.g. if we selected "medical", only save the medical text hints)
                if selected_category == "no_pii_negative":
                    hints_str = ""
                else:
                    relevant_hints = [e['text'] for e in entities if e['label'] == selected_category]
                    hints_str = "; ".join(relevant_hints)

                writer.writerow({
                    'review_id': review_id,
                    'original_text': text.replace('\n', ' '),  # Cleanup for CSV safety
                    'gliner_hint_category': selected_category,
                    'gliner_hint_text': hints_str
                })

                current_counts[selected_category] += 1
                total_saved += 1
                pbar.update(1)

        except json.JSONDecodeError:
            continue
        except Exception as e:
            # print(f"Skipping error row: {e}")
            continue

    pbar.close()

# ------------------------------------------------------------------
# FINAL REPORT
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print(f"🏁 Mining Complete.")
print(f"   - Scanned: {rows_scanned} reviews")
print(f"   - Saved:   {total_saved} entries to '{OUTPUT_FILE}'")
print("-" * 60)
print("Distribution:")
for cat, count in current_counts.items():
    target = TARGET_COUNTS[cat]
    status = "✅ Full" if count >= target else f"⚠️ {target - count} short"
    print(f"{cat:<25} : {count:>4} / {target} | {status}")
print("=" * 60)
print("👉 Next Step: Run your Gemma 3 script on this CSV to verify/label the columns.")