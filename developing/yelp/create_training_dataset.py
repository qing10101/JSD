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

# Confidence threshold for the "Miner" (Keep it low to maximize Recall)
# We want GLiNER to catch POTENTIAL PII, so Gemma can verify it later.
THRESHOLD = 0.25

# ------------------------------------------------------------------
# QUOTAS (Targeting ~3,500 total rows)
# ------------------------------------------------------------------
# Medical is rare (~3%), so we prioritize keeping it.
# Occupation is common (~15%), so we cap it early.
TARGET_COUNTS = {
    "medical condition related": 800,
    "children/minor related": 800,
    "occupation indication": 800,
    "no_pii_negative": 1100  # Crucial for training the model what NOT to label
}

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
with open(INPUT_FILE, 'r', encoding='utf-8') as infile, \
        open(OUTPUT_FILE, 'w', encoding='utf-8-sig', newline='') as outfile:
    # We save the Review ID so you can trace it back if needed
    fieldnames = ['review_id', 'original_text', 'gliner_hint_category', 'gliner_hint_text']
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()

    pbar = tqdm(total=sum(TARGET_COUNTS.values()), desc="Collecting Rows")

    for line in infile:
        # Safety Break: If we scan too many without filling quotas, stop (avoid infinite loop)
        if rows_scanned > 150000:
            print("\n⚠️ Scanned 150k rows. Stopping to prevent infinite run.")
            break

        # Stop if all buckets are full
        if all(current_counts[k] >= TARGET_COUNTS[k] for k in TARGET_COUNTS):
            print("\n✅ All quotas filled!")
            break

        rows_scanned += 1

        try:
            data = json.loads(line)
            text = data.get('text', '').strip()
            review_id = data.get('review_id', '')

            # Skip very short reviews (usually not enough context for training)
            if len(text) < 50: continue

            # Run Inference
            entities = model.predict_entities(text, LABELS, threshold=THRESHOLD, flat_ner=False)

            detected_labels = set([e['label'] for e in entities])

            # ---------------------------------------------------------
            # PRIORITY LOGIC (The "Sieve")
            # We check rare categories first. If a review has Medical AND Occupation,
            # we count it as Medical because we need those more.
            # ---------------------------------------------------------
            selected_category = None

            if "medical condition related" in detected_labels and current_counts["medical condition related"] < \
                    TARGET_COUNTS["medical condition related"]:
                selected_category = "medical condition related"

            elif "children/minor related" in detected_labels and current_counts["children/minor related"] < \
                    TARGET_COUNTS["children/minor related"]:
                selected_category = "children/minor related"

            elif "occupation indication" in detected_labels and current_counts["occupation indication"] < TARGET_COUNTS[
                "occupation indication"]:
                selected_category = "occupation indication"

            elif not detected_labels and current_counts["no_pii_negative"] < TARGET_COUNTS["no_pii_negative"]:
                selected_category = "no_pii_negative"

            # ---------------------------------------------------------
            # SAVE IF SELECTED
            # ---------------------------------------------------------
            if selected_category:
                # Format hints for your LLM labeler
                hints = [e['text'] for e in entities]

                writer.writerow({
                    'review_id': review_id,
                    'original_text': text.replace('\n', ' '),  # Clean newlines for easier CSV handling
                    'gliner_hint_category': selected_category,
                    'gliner_hint_text': "; ".join(hints) if hints else ""
                })

                current_counts[selected_category] += 1
                total_saved += 1
                pbar.update(1)

        except json.JSONDecodeError:
            continue

    pbar.close()

# ------------------------------------------------------------------
# REPORT
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print(f"🏁 Mining Complete. Scanned {rows_scanned} rows.")
print(f"📂 Saved {total_saved} entries to '{OUTPUT_FILE}'")
print("-" * 60)
for cat, count in current_counts.items():
    status = "✅ Full" if count >= TARGET_COUNTS[cat] else "⚠️ Under"
    print(f"{cat:<25} : {count:>4} / {TARGET_COUNTS[cat]} ({status})")
print("=" * 60)