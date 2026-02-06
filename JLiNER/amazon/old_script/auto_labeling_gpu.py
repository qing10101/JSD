import csv
import json
import ollama
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
INPUT_FILE = "../gold.csv"
OUTPUT_FILE = "amazon_labeled_final_v4.csv"
MODEL_NAME = "gemma3:27b"
WORKERS = 8  # A100 is a beast; use 8 workers

# ------------------------------------------------------------------
# ALIGNED SYSTEM PROMPT (Matches Groq v4)
# ------------------------------------------------------------------
SYSTEM_PROMPT = """You are a High-Precision PII Auditor. Output strictly valid JSON.
Extract exact substrings from the review for these 3 categories.

CATEGORIES:
1. "occupation_col": Professional roles or work tasks (e.g., "nurse", "editing"). 
   - EXCLUDE: Hobbies, generic staff (the waiter), objects (laptop).
2. "medical_col": Specific diseases or chronic conditions (e.g., "diabetes", "IBS"). 
   - EXCLUDE: General pain, temporary sickness.
3. "children_col": The AUTHOR'S OWN minor children ONLY.
   - Use label: "author's minor children related".
   - CRITERIA: Look for possessive context like "my son", "our daughter".
   - IGNORE: Hypothetical children, adult children, or generic "kids".

Return Format: {"occupation_col": "", "medical_col": "", "children_col": ""}"""

# ------------------------------------------------------------------
# DATA SANITIZER (Null-Safe)
# ------------------------------------------------------------------
CHILD_KEYWORDS = ["toddler", "son", "daughter", "kid", "baby", "grandson", "granddaughter", "niece", "nephew", "child"]


def fix_data(data):
    # Ensure all keys exist and are strings (prevents NoneType errors)
    occ = str(data.get("occupation_col") or "")
    med = str(data.get("medical_col") or "")
    child = str(data.get("children_col") or "")

    # Logic fix: Ensure children aren't listed as occupations
    for kw in CHILD_KEYWORDS:
        if kw in occ.lower():
            if "therapist" not in occ.lower() and "teacher" not in occ.lower() and "care" not in occ.lower():
                child = f"{child}; {occ}" if child else occ
                occ = ""
                break
    return {"occupation_col": occ, "medical_col": med, "children_col": child}


# ------------------------------------------------------------------
# WORKER FUNCTION
# ------------------------------------------------------------------
def process_row(row):
    text = row.get('original_text') or row.get('original_sentence') or ""

    if not text: return row

    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': f"Review: {text}"}
            ],
            format='json',
            options={'temperature': 0, 'num_ctx': 2048}
        )

        raw_json = json.loads(response['message']['content'])
        clean_data = fix_data(raw_json)

        row.update(clean_data)
        row['model_used'] = 'colab_a100_gemma3_v4_final'
        return row
    except Exception as e:
        # Keep original if it fails (unlikely on A100)
        return row


# ------------------------------------------------------------------
# MAIN EXECUTION
# ------------------------------------------------------------------
if __name__ == "__main__":
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Missing {INPUT_FILE}. Upload it to the sidebar!")
    else:
        with open(INPUT_FILE, 'r', encoding='utf-8-sig') as f:
            all_rows = list(csv.DictReader(f))

        # Only process what Groq skipped
        to_fix = [r for r in all_rows if r.get('model_used') == "SAFETY_FILTER_SKIP"]

        print(f"🚀 A100 processing {len(to_fix)} safety skips...")
        print(f"🎯 Target Label: 'author's minor children related'")

        fixed_map = {}
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=WORKERS) as executor:
            futures = {executor.submit(process_row, r): r for r in to_fix}
            for i, future in enumerate(as_completed(futures), 1):
                res = future.result()
                fixed_map[res['unique_id']] = res
                if i % 50 == 0:
                    elapsed = time.time() - start_time
                    print(f"⚡ {i}/{len(to_fix)} rows done ({i / elapsed:.1f} rows/sec)")

        # Merge and Save
        print("💾 Saving merged dataset...")
        with open(OUTPUT_FILE, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
            writer.writeheader()
            for row in all_rows:
                writer.writerow(fixed_map.get(row['unique_id'], row))

        print(f"\n🏁 DONE! Download '{OUTPUT_FILE}'.")
        print(f"Total entries in final set: {len(all_rows)}")