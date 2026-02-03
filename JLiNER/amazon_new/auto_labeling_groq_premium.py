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
OUTPUT_FILE = "amazon_gold_labeled_v4.csv"
API_KEY = "YOUR_GROQ_API_KEY"

# THE POOL
MODEL_POOL = [
    "openai/gpt-oss-120b",
    "moonshotai/kimi-k2-instruct-0905",
    "moonshotai/kimi-k2-instruct",
    "openai/gpt-oss-20b"
]

MAX_CONCURRENT_REQUESTS = 12

# ------------------------------------------------------------------
# SYSTEM PROMPT (Semantic Labeling)
# ------------------------------------------------------------------
SYSTEM_PROMPT = """You are a strict PII Auditor. Output strictly valid JSON.
Extract exact substrings for 3 specific categories.

CATEGORIES:
1. "occupation_col": Professional roles/tasks (e.g., "nurse", "grading"). NO hobbies, NO generic staff.
2. "medical_col": Diagnosed diseases or chronic conditions. NO general pain.
3. "children_col": The AUTHOR'S OWN minor children ONLY.
   Label: "author's minor children related".
   Criteria: Look for "my son", "our daughter", etc. IGNORE hypothetical or adult children.

Return: {"occupation_col": "", "medical_col": "", "children_col": ""}"""

# ------------------------------------------------------------------
# LOGIC
# ------------------------------------------------------------------
client = AsyncGroq(api_key=API_KEY)


async def process_row(row, semaphore):
    text = row.get('original_text') or row.get('original_sentence') or ""
    row.update({'occupation_col': '', 'medical_col': '', 'children_col': '', 'model_used': ''})

    if not text: return row

    async with semaphore:
        for model in MODEL_POOL:
            try:
                response = await client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"Review: {text}"}
                    ],
                    model=model,
                    temperature=0,
                    response_format={"type": "json_object"}
                )

                data = json.loads(response.choices[0].message.content)
                row.update({
                    'occupation_col': data.get('occupation_col') or '',
                    'medical_col': data.get('medical_col') or '',
                    'children_col': data.get('children_col') or '',
                    'model_used': model
                })
                return row

            except RateLimitError:
                # transient limit (RPM/TPM) -> Try next model
                continue

            except BadRequestError as e:
                err_msg = str(e).lower()
                # CRITICAL FIX: If it's a safety filter, STOP immediately.
                # Do NOT try other models. Save the RPD limit.
                if "safety" in err_msg or "content" in err_msg or "policy" in err_msg:
                    row['model_used'] = 'SAFETY_FILTER_SKIP'
                    return row

                # If it's just a context length or random error, try next
                continue

            except Exception:
                continue

    row['model_used'] = 'LIMIT_EXHAUSTED_OR_FAILED'
    return row


# ------------------------------------------------------------------
# MAIN LOOP (With Resume)
# ------------------------------------------------------------------
async def main():
    print(f"🚀 Starting Optimized Groq Labeling (v4)")

    # ... (Same loading and resume logic as before) ...
    rows = []
    with open(INPUT_FILE, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    processed_ids = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for r in reader: processed_ids.add(r.get('unique_id') or r.get('index'))
        print(f"🔄 Resuming... {len(processed_ids)} done.")

    rows_to_do = [r for r in rows if (r.get('unique_id') or r.get('index')) not in processed_ids]
    if not rows_to_do: return

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    tasks = [process_row(row, semaphore) for row in rows_to_do]

    fieldnames = list(rows[0].keys())
    for col in ['occupation_col', 'medical_col', 'children_col', 'model_used']:
        if col not in fieldnames: fieldnames.append(col)

    with open(OUTPUT_FILE, 'a', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if len(processed_ids) == 0: writer.writeheader()

        completed = 0
        start_time = time.time()
        for future in asyncio.as_completed(tasks):
            result = await future
            writer.writerow(result)
            completed += 1
            if completed % 25 == 0:
                elapsed = time.time() - start_time
                print(f"✅ {completed}/{len(rows_to_do)} done ({completed / elapsed:.1f} rows/sec)")

    print(f"🏁 Process Complete.")


if __name__ == "__main__":
    asyncio.run(main())