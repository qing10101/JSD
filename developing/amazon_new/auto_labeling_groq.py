import os
import csv
import json
import asyncio
import time
from groq import AsyncGroq, RateLimitError, BadRequestError

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
INPUT_FILE = "amazon_to_label.csv"
OUTPUT_FILE = "amazon_labeled_groq_v2.csv"
API_KEY = ""  # REPLACE THIS

# Removed Qwen (it was causing JSON errors). Added Llama-3.3-70b.
MODEL_POOL = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "moonshotai/kimi-k2-instruct",
    "meta-llama/llama-4-scout-17b-16e-instruct"
]

MAX_CONCURRENT_REQUESTS = 12

# ------------------------------------------------------------------
# 1. STRICTER PROMPT
# ------------------------------------------------------------------
SYSTEM_PROMPT = """You are a Data Cleaning Bot. Output strictly JSON.
Task: Extract substrings based on the Review and Hint.

Rules:
1. "occupation_col": Jobs/Professions ONLY (e.g. Nurse, Teacher, Driver). 
   - NEVER put family members (son, toddler, wife) here.
   - NEVER put hobbies (gamer, runner) here.
2. "medical_col": Diseases/Conditions ONLY (e.g. Diabetes, Asthma).
3. "children_col": Author's minor children ONLY (e.g. My son, My toddler).

Output Format:
{
  "occupation_col": "string",
  "medical_col": "string",
  "children_col": "string"
}
"""

# ------------------------------------------------------------------
# 2. SANITIZER FUNCTION (Fixes Hallucinations)
# ------------------------------------------------------------------
CHILD_KEYWORDS = ["toddler", "son", "daughter", "kid", "baby", "grandson", "granddaughter", "niece", "nephew", "child"]


def fix_hallucinations(data):
    """
    Python logic to fix common model mistakes before saving.
    """
    occ = data.get("occupation_col", "").lower()
    med = data.get("medical_col", "")
    child = data.get("children_col", "")

    # Fix: If Occupation contains a child word, move it to Children
    for kw in CHILD_KEYWORDS:
        if kw in occ:
            # Check if it's actually a job like "child care provider"
            if "provider" not in occ and "teacher" not in occ:
                # Move it
                if child:
                    child += "; " + data["occupation_col"]
                else:
                    child = data["occupation_col"]
                data["occupation_col"] = ""  # Clear the wrong column
                data["children_col"] = child
                break

    return data


# ------------------------------------------------------------------
# MAIN LOGIC
# ------------------------------------------------------------------
client = AsyncGroq(api_key=os.environ.get("GROQ_API_KEY", API_KEY))


async def process_row(row, semaphore):
    text = row.get('original_text', '').strip()
    hint_text = row.get('gliner_hint_text', '')
    hint_cat = row.get('gliner_hint_category', '')

    # Initialize defaults
    row['occupation_col'] = ''
    row['medical_col'] = ''
    row['children_col'] = ''
    row['model_used'] = ''

    if not text: return row

    user_content = f"Review: {text}\nHint ({hint_cat}): {hint_text}"

    async with semaphore:
        for model in MODEL_POOL:
            try:
                chat_completion = await client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_content}
                    ],
                    model=model,
                    temperature=0,
                    response_format={"type": "json_object"},
                    max_tokens=256,
                )

                content = chat_completion.choices[0].message.content
                data = json.loads(content)

                # Apply Sanitizer
                data = fix_hallucinations(data)

                row['occupation_col'] = data.get('occupation_col', '')
                row['medical_col'] = data.get('medical_col', '')
                row['children_col'] = data.get('children_col', '')
                row['model_used'] = model
                return row

            except RateLimitError:
                continue  # Rotate model
            except BadRequestError:
                # Likely Safety Filter or Context Length
                continue  # Rotate model
            except Exception:
                continue  # Rotate model

    # If we get here, ALL models failed (likely Safety Filter)
    row['model_used'] = 'SAFETY_FILTER_SKIP'
    return row


async def main():
    print(f"🚀 Starting Groq Labeling v2")

    rows = []
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8-sig', errors='replace') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except FileNotFoundError:
        print("Input file not found.")
        return

    # Resume Logic
    processed_ids = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for r in reader:
                processed_ids.add(r['unique_id'])
        print(f"🔄 Resuming... {len(processed_ids)} rows already done.")

    rows_to_do = [r for r in rows if r['unique_id'] not in processed_ids]

    if not rows_to_do:
        print("✅ Nothing to do!")
        return

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    tasks = [process_row(row, semaphore) for row in rows_to_do]

    # Write output
    with open(OUTPUT_FILE, 'a', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['unique_id', 'original_text', 'gliner_hint_category', 'gliner_hint_text',
                                               'occupation_col', 'medical_col', 'children_col', 'model_used'])
        if len(processed_ids) == 0:
            writer.writeheader()

        completed = 0
        start_time = time.time()

        for future in asyncio.as_completed(tasks):
            result = await future
            writer.writerow(result)
            completed += 1

            if completed % 20 == 0:
                elapsed = time.time() - start_time
                rate = completed / elapsed
                print(f"   ... {completed}/{len(rows_to_do)} done ({rate:.1f} rows/sec)")

    print(f"✅ Finished.")


if __name__ == "__main__":
    asyncio.run(main())