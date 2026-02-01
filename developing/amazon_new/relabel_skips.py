import csv
import json
import ollama
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
INPUT_FILE = "amazon_labeled_groq_v2.csv"
OUTPUT_FILE = "amazon_labeled_complete.csv"
MODEL_NAME = "gemma3n:e2b"
WORKERS = 4  # Your M3 Max can handle this easily

# The flag to look for
TARGET_FLAG = "SAFETY_FILTER_SKIP"

# ------------------------------------------------------------------
# 1. SAME PROMPT & SANITIZER (For Consistency)
# ------------------------------------------------------------------
SYSTEM_PROMPT = """You are a Data Cleaning Bot. Output strictly JSON.
Task: Extract substrings based on the Review and Hint.

Rules:
1. "occupation_col": Jobs/Professions ONLY. NO family, NO hobbies.
2. "medical_col": Diseases/Conditions ONLY.
3. "children_col": Author's minor children ONLY.

Output Format:
{
  "occupation_col": "string",
  "medical_col": "string",
  "children_col": "string"
}
"""

CHILD_KEYWORDS = ["toddler", "son", "daughter", "kid", "baby", "grandson", "granddaughter", "niece", "nephew", "child"]


def fix_hallucinations(data):
    """
    Python logic to fix common model mistakes before saving.
    Handles 'null' values from LLM gracefully.
    """
    # Use 'or ""' to catch cases where the key exists but value is None/null
    occ_raw = data.get("occupation_col") or ""
    occ = occ_raw.lower()

    child = data.get("children_col") or ""

    # Fix: If Occupation contains a child word, move it to Children
    for kw in CHILD_KEYWORDS:
        if kw in occ:
            # Check if it's actually a job like "child care provider"
            if "provider" not in occ and "teacher" not in occ:
                # Move it
                if child:
                    # Append if child column already has data
                    if data["occupation_col"]:
                        child = f"{child}; {data['occupation_col']}"
                else:
                    child = data["occupation_col"]

                data["occupation_col"] = ""  # Clear the wrong column
                data["children_col"] = child
                break

    # Final cleanup to ensure everything is a string for the CSV writer
    data["occupation_col"] = data.get("occupation_col") or ""
    data["medical_col"] = data.get("medical_col") or ""
    data["children_col"] = data.get("children_col") or ""

    return data


# ------------------------------------------------------------------
# 2. LOCAL WORKER FUNCTION
# ------------------------------------------------------------------
def process_local_row(row):
    text = row.get('original_text', '').strip()
    hint_text = row.get('gliner_hint_text', '')
    hint_cat = row.get('gliner_hint_category', '')

    user_content = f"Review: {text}\nHint ({hint_cat}): {hint_text}"

    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': user_content}
            ],
            format='json',
            options={'temperature': 0, 'num_ctx': 2048}
        )

        content = response['message']['content']
        data = json.loads(content)

        # Apply Sanitizer
        data = fix_hallucinations(data)

        # Update Row
        row['occupation_col'] = data.get('occupation_col', '')
        row['medical_col'] = data.get('medical_col', '')
        row['children_col'] = data.get('children_col', '')
        row['model_used'] = 'local_gemma3_cleanup'

        return row

    except Exception as e:
        print(f"⚠️ Local Error on {row.get('unique_id')}: {e}")
        # Return row as is (still marked skipped) if local fails too
        return row


# ------------------------------------------------------------------
# 3. MAIN EXECUTION
# ------------------------------------------------------------------
def main():
    print(f"🚀 Starting Local Cleanup with {MODEL_NAME}")

    # 1. Read all rows
    all_rows = []
    rows_to_fix = []

    try:
        with open(INPUT_FILE, 'r', encoding='utf-8-sig', errors='replace') as f:
            reader = csv.DictReader(f)
            all_rows = list(reader)
            fieldnames = reader.fieldnames
    except FileNotFoundError:
        print("Input file not found.")
        return

    # 2. Identify Safety Skips
    for row in all_rows:
        if row.get('model_used') == TARGET_FLAG:
            rows_to_fix.append(row)

    print(f"   - Total Rows: {len(all_rows)}")
    print(f"   - Rows to Fix: {len(rows_to_fix)}")
    print("-" * 60)

    if not rows_to_fix:
        print("✅ No safety skips found! You are good to go.")
        return

    # 3. Process the "Bad" rows in parallel
    fixed_rows_map = {}
    completed = 0
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        future_to_row = {executor.submit(process_local_row, r): r for r in rows_to_fix}

        for future in as_completed(future_to_row):
            result_row = future.result()
            unique_id = result_row['unique_id']
            fixed_rows_map[unique_id] = result_row  # Store for merging

            completed += 1
            if completed % 10 == 0:
                elapsed = time.time() - start_time
                rate = completed / elapsed
                print(f"   ... Fixed {completed}/{len(rows_to_fix)} ({rate:.1f} rows/sec)")

    # 4. Merge and Write Final File
    print("-" * 60)
    print("💾 Merging and Saving...")

    with open(OUTPUT_FILE, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in all_rows:
            uid = row['unique_id']
            # If this row was fixed, write the fixed version. Otherwise, write original.
            if uid in fixed_rows_map:
                writer.writerow(fixed_rows_map[uid])
            else:
                writer.writerow(row)

    print(f"✅ Final Dataset Ready: '{OUTPUT_FILE}'")


if __name__ == "__main__":
    main()