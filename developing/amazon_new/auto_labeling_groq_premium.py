import os
import csv
import json
import asyncio
import time
from groq import AsyncGroq, RateLimitError, BadRequestError

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
INPUT_FILE = "amazon_to_label.csv"  # or "test_auto_labeled_new.csv"
OUTPUT_FILE = "amazon_gold_standard_labels.csv"
API_KEY = "YOUR_GROQ_API_KEY"

# THE HEAVYWEIGHT POOL (Aggregate RPD: 4,000)
MODEL_POOL = [
    "openai/gpt-oss-120b",  # Precision Leader (1k RPD)
    "moonshotai/kimi-k2-instruct-0905",  # Logic Leader (1k RPD)
    "llama-3.3-70b-versatile",  # Context Leader (1k RPD)
    "meta-llama/llama-4-scout-17b-16e-instruct"  # Backup Leader (1k RPD)
]

# Using 12 concurrent requests to maximize throughput across the pool
MAX_CONCURRENT_REQUESTS = 12

# ------------------------------------------------------------------
# STRICT PROMPT (Optimized for 120B Reasoning)
# ------------------------------------------------------------------
SYSTEM_PROMPT = """You are a Professional Privacy Auditor. Output strictly valid JSON.
Task: Extract exact substrings for the following categories:

1. "occupation_col": Professional roles or work tasks (e.g., "nurse", "grading papers"). 
   - EXCLUDE: Hobbies (runner), Generic staff (the waiter), Objects (laptop).
2. "medical_col": Specific diseases or chronic conditions (e.g., "diabetes", "IBS"). 
   - EXCLUDE: General pain, temporary sickness.
3. "children_col": The author's OWN minor children (e.g., "my toddler", "son's homework").

LOGIC: If a clue relates to an anonymous person, ignore it. Focus only on the author and their family.
Return Format: {"occupation_col": "", "medical_col": "", "children_col": ""}"""

# ------------------------------------------------------------------
# SANITIZER & NULL-SAFETY
# ------------------------------------------------------------------
CHILD_KEYWORDS = ["toddler", "son", "daughter", "kid", "baby", "grandson", "granddaughter", "niece", "nephew", "child"]


def clean_and_fix(data):
    # 1. Null Safety (Fixes the NoneType .lower() error)
    occ = str(data.get("occupation_col") or "")
    med = str(data.get("medical_col") or "")
    child = str(data.get("children_col") or "")

    # 2. Logic Correction (Move children out of occupation)
    for kw in CHILD_KEYWORDS:
        if kw in occ.lower():
            if "provider" not in occ.lower() and "teacher" not in occ.lower():
                child = f"{child}; {occ}" if child else occ
                occ = ""
                break
    return {"occupation_col": occ, "medical_col": med, "children_col": child}


# ------------------------------------------------------------------
# WORKER LOGIC
# ------------------------------------------------------------------
client = AsyncGroq(api_key=API_KEY)


async def process_row(row, semaphore):
    # Auto-detect column name (handles both your file formats)
    text = row.get('original_text') or row.get('original_sentence') or ""
    text = text.strip()

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

                raw_data = json.loads(response.choices[0].message.content)
                clean_data = clean_and_fix(raw_data)

                row.update(clean_data)
                row['model_used'] = model
                return row

            except (RateLimitError, BadRequestError):
                # If RPM/TPM limit hit, skip to the next model in the pool
                continue
            except Exception:
                continue

    row['model_used'] = 'FAILED_ALL_MODELS'
    return row


# ------------------------------------------------------------------
# MAIN LOOP (With Resume Logic)
# ------------------------------------------------------------------
async def main():
    print(f"🚀 Starting High-Precision Labeling")

    if not os.path.exists(INPUT_FILE):
        print(f"❌ Input file {INPUT_FILE} not found.")
        return

    rows = []
    with open(INPUT_FILE, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    processed_ids = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for r in reader: processed_ids.add(r.get('unique_id') or r.get('index'))
        print(f"🔄 Resuming... {len(processed_ids)} already done.")

    rows_to_do = [r for r in rows if (r.get('unique_id') or r.get('index')) not in processed_ids]
    if not rows_to_do:
        print("✅ Nothing to do!")
        return

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    tasks = [process_row(row, semaphore) for row in rows_to_do]

    # Setup Output Columns
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
            if completed % 20 == 0:
                elapsed = time.time() - start_time
                print(f"✅ {completed}/{len(rows_to_do)} labeled ({completed / elapsed:.1f} rows/sec)")

    print(f"🏁 Finished. Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())