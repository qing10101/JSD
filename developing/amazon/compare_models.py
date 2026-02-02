import csv
import os
from gliner import GLiNER

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
INPUT_CSV = "test.csv"
ROWS_TO_COMPARE = 20  # How many rows to check

# MODEL A: The Generalist (State-of-the-art Zero-shot)
# You can also use "urchade/gliner_medium-v2.1" here
BASELINE_MODEL_NAME = "numind/NuNER_Zero-span"

# MODEL B: The Specialist (Your Fine-Tuned Model)
# Point this to the unzipped folder you just downloaded
CUSTOM_MODEL_PATH = "gliner_finetuned_colab"

# The labels you care about
LABELS = [
    "occupation indication",
    "medical condition related",
    "children/minor related"
]

# ------------------------------------------------------------------
# 1. LOAD BOTH MODELS
# ------------------------------------------------------------------
print(f"⏳ Loading Baseline: {BASELINE_MODEL_NAME}...")
baseline_model = GLiNER.from_pretrained(BASELINE_MODEL_NAME)
baseline_model.eval()  # Set to eval mode

print(f"⏳ Loading Custom: {CUSTOM_MODEL_PATH}...")
if not os.path.exists(CUSTOM_MODEL_PATH):
    print(f"❌ Error: Could not find folder '{CUSTOM_MODEL_PATH}'. Did you unzip it?")
    exit()
custom_model = GLiNER.from_pretrained(CUSTOM_MODEL_PATH)
custom_model.eval()

print("🚀 Starting Head-to-Head Comparison...")
print("=" * 80)

# ------------------------------------------------------------------
# 2. RUN COMPARISON
# ------------------------------------------------------------------
with open(INPUT_CSV, 'r', encoding='utf-8-sig', errors='replace') as f:
    reader = csv.DictReader(f)

    # Helper to check if text column exists
    if 'original_sentence' not in reader.fieldnames:
        print("❌ Error: 'original_sentence' column missing.")
        exit()

    for i, row in enumerate(reader):
        if i >= ROWS_TO_COMPARE:
            break

        text = row.get('original_sentence', '').strip()
        if not text: continue

        # Run Inference
        # We use a slightly lower threshold for the baseline to give it a fair chance
        base_ents = baseline_model.predict_entities(text, LABELS, threshold=0.3)

        # We use the standard threshold for your custom model
        custom_ents = custom_model.predict_entities(text, LABELS, threshold=0.85)

        # Skip rows where BOTH are empty (to save screen space)
        if not base_ents and not custom_ents:
            continue

        print(f"\n📝 Row {i + 1}: \"{text[:100]}...\"")
        print("-" * 80)

        # FORMAT OUTPUT SIDE-BY-SIDE

        # 1. Baseline Output
        print(f"🔵 BASELINE ({BASELINE_MODEL_NAME}):")
        if not base_ents:
            print("   (No PII detected)")
        for e in base_ents:
            print(f"   • {e['text']} -> [{e['label']}] ({e['score']:.2f})")

        # 2. Custom Output
        print(f"🟢 CUSTOM   (Fine-Tuned):")
        if not custom_ents:
            print("   (No PII detected)")
        for e in custom_ents:
            # simple check: did custom find something baseline missed?
            is_new = e['text'] not in [x['text'] for x in base_ents]
            marker = "✨ NEW!" if is_new else ""
            print(f"   • {e['text']} -> [{e['label']}] ({e['score']:.2f}) {marker}")

        print("=" * 80)