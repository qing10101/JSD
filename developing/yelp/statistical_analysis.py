import json
import itertools
from collections import Counter
from gliner import GLiNER
from tqdm import tqdm

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
INPUT_FILE = "/Users/scottwang/PycharmProjects/JSD/developing/yelp_dataset/yelp_academic_dataset_review.json"
LIMIT = 500
THRESHOLD = 0.3  # Confidence threshold

# The 3 specific categories
LABELS = [
    "occupation indication",
    "medical condition related",
    "children/minor related"
]

# ------------------------------------------------------------------
# 1. LOAD MODEL
# ------------------------------------------------------------------
print(f"⏳ Loading GLiNER model (numind/NuNER_Zero-span)...")
# Using the powerful zero-shot model to get a baseline reading on Yelp data
model = GLiNER.from_pretrained("numind/NuNER_Zero-span")

# ------------------------------------------------------------------
# 2. PROCESSING LOOP
# ------------------------------------------------------------------
stats = Counter()
reviews_with_any_pii = 0

print(f"🚀 Analyzing first {LIMIT} reviews from {INPUT_FILE}...")
print("-" * 60)

try:
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        # Use islice to efficiently read only top N lines
        for line in tqdm(itertools.islice(f, LIMIT), total=LIMIT):
            try:
                data = json.loads(line)
                text = data.get('text', '')  # Yelp stores the review body in 'text'

                if not text: continue

                # Predict
                entities = model.predict_entities(
                    text,
                    LABELS,
                    threshold=THRESHOLD,
                    flat_ner=False
                )

                if entities:
                    reviews_with_any_pii += 1

                    # Track which specific categories were found
                    # Use set() so we don't double count if a review has 2 medical terms
                    found_labels = set([e['label'] for e in entities])
                    stats.update(found_labels)

            except json.JSONDecodeError:
                continue

except FileNotFoundError:
    print(f"❌ Error: {INPUT_FILE} not found.")
    exit()

# ------------------------------------------------------------------
# 3. REPORT
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print(f"📊 YELP DATASET ANALYSIS (Sample: {LIMIT} reviews)")
print("=" * 60)

print(f"Total Reviews Scanned:      {LIMIT}")
print(f"Reviews containing PII:     {reviews_with_any_pii} ({reviews_with_any_pii / LIMIT:.1%})")
print("-" * 60)
print("Breakdown by Category:")

for label in LABELS:
    count = stats[label]
    pct = (count / LIMIT) * 100
    bar = "█" * int(pct * 2)  # Scale bar for visibility
    print(f"{label:<25} : {count:>4} ({pct:5.1f}%) | {bar}")

print("=" * 60)
print("💡 Interpretation:")
if stats["occupation indication"] > stats["medical condition related"]:
    print("   Expect lots of Waiters/Bartenders/Managers (Occupation).")
else:
    print("   Expect mentions of food poisoning or allergies (Medical).")