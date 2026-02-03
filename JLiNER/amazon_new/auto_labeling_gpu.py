import csv
import json
import ollama
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# ------------------------------------------------------------------
# CONFIGURATION (Tuned for A100)
# ------------------------------------------------------------------
INPUT_FILE = "amazon_labeled_groq_v2.csv"
OUTPUT_FILE = "amazon_labeled_complete.csv"
MODEL_NAME = "gemma3:12b"
WORKERS = 8  # Matches OLLAMA_NUM_PARALLEL

# ------------------------------------------------------------------
# PROMPT & SANITIZER (Logic preserved from previous versions)
# ------------------------------------------------------------------
SYSTEM_PROMPT = """You are a Data Cleaning Bot. Output strictly JSON.
Task: Extract substrings based on the Review and Hint.
Rules:
1. "occupation_col": Jobs/Professions ONLY. NO family, NO hobbies.
2. "medical_col": Diseases/Conditions ONLY.
3. "children_col": Author's minor children ONLY.
Output Format: {"occupation_col": "string", "medical_col": "string", "children_col": "string"}"""

CHILD_KEYWORDS = ["toddler", "son", "daughter", "kid", "baby", "grandson", "granddaughter", "niece", "nephew", "child"]

def fix_data(data):
    # Null-safe extraction
    occ = str(data.get("occupation_col") or "")
    med = str(data.get("medical_col") or "")
    child = str(data.get("children_col") or "")

    # Logic fix for "Toddler = Job"
    for kw in CHILD_KEYWORDS:
        if kw in occ.lower():
            if "provider" not in occ.lower() and "teacher" not in occ.lower():
                child = f"{child}; {occ}" if child else occ
                occ = ""
                break
    return {"occupation_col": occ, "medical_col": med, "children_col": child}

# ------------------------------------------------------------------
# WORKER
# ------------------------------------------------------------------
def process_row(row):
    text = row.get('original_text', '').strip()
    hint_text = row.get('gliner_hint_text', '')
    hint_cat = row.get('gliner_hint_category', '')

    if not text: return row

    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': f"Review: {text}\nHint ({hint_cat}): {hint_text}"}
            ],
            format='json',
            options={'temperature': 0, 'num_ctx': 2048}
        )

        raw_json = json.loads(response['message']['content'])
        clean_data = fix_data(raw_json)

        row.update(clean_data)
        row['model_used'] = 'colab_a100_gemma3_12b'
        return row
    except Exception as e:
        return row # Return original if it fails

# ------------------------------------------------------------------
# EXECUTION
# ------------------------------------------------------------------
if __name__ == "__main__":
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Missing {INPUT_FILE}")
    else:
        with open(INPUT_FILE, 'r', encoding='utf-8-sig') as f:
            all_rows = list(csv.DictReader(f))

        to_fix = [r for r in all_rows if r.get('model_used') == "SAFETY_FILTER_SKIP"]
        print(f"🚀 A100 processing {len(to_fix)} rows with {WORKERS} threads...")

        fixed_map = {}
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=WORKERS) as executor:
            futures = {executor.submit(process_row, r): r for r in to_fix}
            for i, future in enumerate(as_completed(futures), 1):
                res = future.result()
                fixed_map[res['unique_id']] = res
                if i % 50 == 0:
                    elapsed = time.time() - start_time
                    print(f"⚡ {i}/{len(to_fix)} rows done ({i/elapsed:.1f} rows/sec)")

        # Save and Merge
        with open(OUTPUT_FILE, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
            writer.writeheader()
            for row in all_rows:
                writer.writerow(fixed_map.get(row['unique_id'], row))

        print(f"\n🏁 Finished in {time.time()-start_time:.1f} seconds.")
        print(f"📄 Saved: {OUTPUT_FILE}")