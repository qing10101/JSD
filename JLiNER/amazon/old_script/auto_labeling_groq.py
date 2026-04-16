import os
import csv
import json
import asyncio
import time
from groq import AsyncGroq, RateLimitError, BadRequestError

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
INPUT_FILE = "/Users/scottwang/PycharmProjects/JSD/JLiNER/amazon/merged_test.csv"
OUTPUT_FILE = "merged_test_llm.csv"
API_KEY = "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" # API Key goes here

# THE POOL
MODEL_POOL = [
    "openai/gpt-oss-120b",
    "moonshotai/kimi-k2-instruct-0905",
    "moonshotai/kimi-k2-instruct",
    "openai/gpt-oss-20b"
]

MAX_CONCURRENT_REQUESTS = 12

# ------------------------------------------------------------------
# SYSTEM PROMPT (Strict Inference-Based Labeling)
# ------------------------------------------------------------------
SYSTEM_PROMPT = """You are a High-Precision PII Auditor. Output strictly valid JSON.
Your goal is to extract EXACT substrings that provide a direct inference for the following categories:

CATEGORIES:
1. "gender_col": Substrings that allow a direct inference of the AUTHOR'S (the reviewer's) gender.
   - Examples: "As a mom", "Being a guy", "My husband bought this for me".
   - IGNORE: Gender of people mentioned in the text who are not the author.

2. "medical_col": Substrings that provide a direct inference to a real person's medical condition.
   - Includes: Diagnosed diseases, chronic conditions, symptoms, or specific medical treatments.
   - Examples: "my diabetes", "inhaler for my asthma", "post-surgery recovery".
   - IGNORE: General, non-specific pain (e.g., "headache", "my back hurts") unless linked to a chronic condition.

3. "minor_col": Substrings that provide a direct inference to a REAL child (not a fictional character) under the age of 18.
   - Examples: "my toddler", "son's 5th birthday", "my 3rd grader", "picking kids up from daycare".
   - CRITICAL: Exclude characters from books, movies, or games. 
   - CRITICAL: Exclude references where the "child" could be an adult (e.g., "visiting my son" is NOT a minor inference unless age/context is provided).

INSTRUCTIONS:
- Extract the EXACT substring from the text.
- If multiple distinct inferences exist in one category, separate them with a semicolon (;).
- If no direct inference exists for a category, return an empty string "".
- Output Format: {"gender_col": "", "medical_col": "", "minor_col": ""}"""

# ------------------------------------------------------------------
# LOGIC
# ------------------------------------------------------------------
client = AsyncGroq(api_key=API_KEY)


async def process_row(row, semaphore):
    text = row.get('original_text') or row.get('original_sentence') or ""
    row.update({'gender_col': '', 'medical_col': '', 'minor_col': '', 'model_used': ''})

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
                    'gender_col': data.get('gender_col') or '',
                    'medical_col': data.get('medical_col') or '',
                    'minor_col': data.get('minor_col') or '',
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
    for col in ['gender_col', 'medical_col', 'minor_col', 'model_used']:
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