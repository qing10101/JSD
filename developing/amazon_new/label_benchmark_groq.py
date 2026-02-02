import os
import csv
import json
import asyncio
import time
from groq import AsyncGroq, RateLimitError, BadRequestError

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
INPUT_FILE = "test_auto_labeled_new.csv"  # Your small dataset
OUTPUT_FILE = "gold_benchmark_set.csv"  # The final evaluation file
API_KEY = "YOUR_GROQ_API_KEY"  # Or set GROQ_API_KEY env var

# FOR THE BENCHMARK: We use the smartest models first for maximum accuracy
MODEL_POOL = [
    "llama-3.3-70b-versatile",  # Primary: Best at complex context
    "moonshotai/kimi-k2-instruct-0905",  # Backup smart model
    "meta-llama/llama-4-scout-17b-16e-instruct",  # Secondary Backup
    "llama-3.1-8b-instant"  # Last resort
]

MAX_CONCURRENT_REQUESTS = 5  # 70B models have lower RPM limits (usually 30)

# ------------------------------------------------------------------
# SYSTEM PROMPT (Strict Gold Standard)
# ------------------------------------------------------------------
SYSTEM_PROMPT = """You are a professional PII Auditor. Your goal is to create a 'Gold Standard' dataset.
Task: Extract exact substrings from the review for these 3 categories.

Categories:
1. "occupation_col": Jobs, professional roles, or specific work tasks (e.g., "nurse", "grading papers"). 
   - IGNORE hobbies (runner, gamer).
   - IGNORE generic service staff (the waiter, the manager) unless it's the author's job.
2. "medical_col": Specific diseases, chronic conditions, or medical devices (e.g., "diabetes", "inhaler"). 
   - IGNORE general pain, headaches, or physical attributes (small ears).
3. "children_col": The author's OWN minor children (e.g., "my son", "my 3rd grader"). 
   - IGNORE hypothetical children or adult children.

Output strictly JSON:
{
  "occupation_col": "string",
  "medical_col": "string",
  "children_col": "string"
}"""

# ------------------------------------------------------------------
# SANITIZER & LOGIC
# ------------------------------------------------------------------
client = AsyncGroq(api_key=os.environ.get("GROQ_API_KEY", API_KEY))


def clean_val(val):
    return str(val or "").strip()


async def process_row(row, semaphore):
    # Use 'original_sentence' for your small dataset
    text = row.get('original_sentence', '').strip()

    row['occupation_col'] = ''
    row['medical_col'] = ''
    row['children_col'] = ''
    row['gold_model'] = ''

    if not text: return row

    async with semaphore:
        for model in MODEL_POOL:
            try:
                chat_completion = await client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"Review Text: {text}"}
                    ],
                    model=model,
                    temperature=0,
                    response_format={"type": "json_object"},
                    max_tokens=512,
                )

                content = chat_completion.choices[0].message.content
                data = json.loads(content)

                # Update with clean strings
                row['occupation_col'] = clean_val(data.get('occupation_col'))
                row['medical_col'] = clean_val(data.get('medical_col'))
                row['children_col'] = clean_val(data.get('children_col'))
                row['gold_model'] = model
                return row

            except (RateLimitError, BadRequestError):
                continue
            except Exception as e:
                print(f"Error: {e}")
                continue

    row['gold_model'] = 'FAILED'
    return row


async def main():
    print(f"🚀 Starting Gold Standard Labeling for Benchmark")

    rows = []
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8-sig', errors='replace') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except FileNotFoundError:
        print(f"❌ {INPUT_FILE} not found.")
        return

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    tasks = [process_row(row, semaphore) for row in rows]

    # Define output fields (preserve original columns + new labels)
    fieldnames = list(rows[0].keys())
    for col in ['occupation_col', 'medical_col', 'children_col', 'gold_model']:
        if col not in fieldnames: fieldnames.append(col)

    print(f"📂 Processing {len(rows)} rows...")
    start_time = time.time()

    with open(OUTPUT_FILE, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        count = 0
        for future in asyncio.as_completed(tasks):
            result = await future
            writer.writerow(result)
            count += 1
            if count % 10 == 0:
                print(f"   ... {count}/{len(rows)} gold labels created")

    print(f"\n✅ Done! Gold Standard set saved to: {OUTPUT_FILE}")
    print(f"⏱️  Time taken: {time.time() - start_time:.1f} seconds")


if __name__ == "__main__":
    asyncio.run(main())