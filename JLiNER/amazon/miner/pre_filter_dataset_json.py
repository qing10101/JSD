import json
import csv
import re
import os
from tqdm import tqdm

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
INPUT_FILE = "/Users/scottwang/PycharmProjects/JSD/JLiNER/amazon/Unknown.jsonl"
OUTPUT_FILE = "amazon_mined_balanced.csv"

# SET YOUR QUOTAS HERE
QUOTAS = {
    "gender_indication": 3000,
    "medical_condition": 3000,
    "minor_children": 3000,
    "no_pii_negative": 0  # Hard negatives for training
}

# --- REGEX RULES ---
PET_EXCLUSION = r'(dog|cat|puppy|pupper|kitten|pet|fur\.baby|paw)'
CHILDREN_REGEX = re.compile(
    rf'\b(son|daughter|nephew|niece|grandchild|toddler|infant|baby|kids?|children|grade|school)\b', re.I)
MEDICAL_REGEX = re.compile(
    r'\b(diagnosed|symptoms|chronic|prescription|dosage|treatment|recovery|allergy|disease|doctor|physician|medication|pain|ailment|illness|side\.effects)\b',
    re.I)
GENDER_REGEX = re.compile(r'\b(husband|wife|boyfriend|girlfriend|gentleman|lady|as.a.(woman|man|girl|boy|female|male)|myself.as)\b',
                          re.I)


def get_category(text):
    text_lower = text.lower()

    # 1. Check Children (with Pet Exclusion)
    if not re.search(PET_EXCLUSION, text_lower):
        if CHILDREN_REGEX.search(text_lower):
            return "minor_children"

    # 2. Check Medical
    if MEDICAL_REGEX.search(text_lower):
        return "medical_condition"

    # 3. Check Occupation
    if GENDER_REGEX.search(text_lower):
        return "gender_indication"

    return "no_pii_negative"


# ------------------------------------------------------------------
# MAIN MINING
# ------------------------------------------------------------------
def mine():
    print(f"🚀 Mining with Quotas: {QUOTAS}")

    current_counts = {k: 0 for k in QUOTAS.keys()}
    total_needed = sum(QUOTAS.values())

    with open(INPUT_FILE, 'r', encoding='utf-8') as infile, \
            open(OUTPUT_FILE, 'w', encoding='utf-8-sig', newline='') as outfile:

        writer = csv.DictWriter(outfile, fieldnames=['unique_id', 'original_text', 'category'])
        writer.writeheader()

        pbar = tqdm(total=total_needed, desc="Filling Quotas")

        for line in infile:
            if all(current_counts[k] >= QUOTAS[k] for k in QUOTAS):
                break

            try:
                data = json.loads(line)
                text = data.get('text', '').strip()
                if len(text.split()) < 10: continue

                cat = get_category(text)

                # Only keep if we haven't hit the quota for this specific category
                if current_counts[cat] < QUOTAS[cat]:
                    writer.writerow({
                        'unique_id': f"{data.get('user_id')}_{data.get('asin')}",
                        'original_text': text.replace('\n', ' '),
                        'category': cat
                    })
                    current_counts[cat] += 1
                    pbar.update(1)
            except:
                continue
        pbar.close()

    print("\n🏁 Mining Complete.")
    print(f"Stats: {current_counts}")


if __name__ == "__main__":
    mine()