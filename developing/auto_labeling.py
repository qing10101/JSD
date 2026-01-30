import csv
import json
import ollama
import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
INPUT_FILE = "/Users/scottwang/PycharmProjects/JSD/developing/filtered_dataset_1_5th_2.csv"
OUTPUT_FILE = "test_auto_labeled_new.csv"
MODEL_NAME = "gemma3:12b"
ROWS_TO_PROCESS = 750  # Adjust as needed
WORKERS = 4  # Matches your OLLAMA_NUM_PARALLEL setting

# ------------------------------------------------------------------
# SYSTEM PROMPT
# ------------------------------------------------------------------
system_prompt = """
You are a data labeling assistant. Your job is to extract EXACT substrings from product reviews that match specific privacy categories.

Categories:
1. occupation_col: Jobs, specific work tasks (e.g. "grading papers"), workplace objects (e.g. "my badge").
2. medical_col: Diseases, chronic conditions, prescriptions, medical devices. (Ignore general pain/headaches).
3. children_col: Mentions of the author's OWN children/minors. (Ignore "my son" if adult context is clear).

Rules:
- EXTRACT EXACT SUBSTRINGS only. Do not paraphrase.
- Separate multiple entities with a semicolon (;).
- If nothing matches, return an empty string.

Output strictly valid JSON:
{
  "occupation_col": "string",
  "medical_col": "string",
  "children_col": "string"
}
"""


def process_row(row):
    """
    Worker function: Takes a CSV row, sends text to LLM, returns row with new columns filled.
    """
    text = row.get('original_sentence', '').strip()

    # Defaults
    row['occupation_col'] = ''
    row['medical_col'] = ''
    row['children_col'] = ''

    if not text:
        return row

    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': f"Review: \"{text}\""}
            ],
            format='json',
            options={'temperature': 0, 'num_ctx': 2048}
        )
        labels = json.loads(response['message']['content'])

        row['occupation_col'] = labels.get('occupation_col', '')
        row['medical_col'] = labels.get('medical_col', '')
        row['children_col'] = labels.get('children_col', '')

        return row
    except Exception as e:
        print(f"⚠️ Error processing row: {e}")
        return row


def main():
    print(f"🚀 Starting Parallel Auto-Labeling with {WORKERS} workers...")

    # 1. Read the Input Data
    rows = []
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8-sig', errors='replace') as infile:
            reader = csv.DictReader(infile)
            if 'original_sentence' not in reader.fieldnames:
                print("❌ Error: 'original_sentence' column missing.")
                return

            # Read only the amount we need
            for i, row in enumerate(reader):
                if i >= ROWS_TO_PROCESS:
                    break
                rows.append(row)
    except FileNotFoundError:
        print(f"❌ Error: {INPUT_FILE} not found.")
        return

    print(f"📂 Loaded {len(rows)} rows to process.")

    # 2. Setup Output
    fieldnames = reader.fieldnames + ['occupation_col', 'medical_col', 'children_col']

    write_lock = Lock()
    start_time = time.time()
    completed_count = 0

    with open(OUTPUT_FILE, 'w', encoding='utf-8-sig', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        # 3. Parallel Execution
        with ThreadPoolExecutor(max_workers=WORKERS) as executor:
            # Submit all jobs
            future_to_row = {executor.submit(process_row, row): row for row in rows}

            for future in as_completed(future_to_row):
                result_row = future.result()

                # Thread-safe writing
                with write_lock:
                    writer.writerow(result_row)
                    completed_count += 1

                    # Progress log
                    if completed_count % 5 == 0:
                        print(f"   ... Processed {completed_count}/{len(rows)}")

    duration = time.time() - start_time
    print("-" * 50)
    print(f"✅ Done! Processed {completed_count} rows in {duration:.2f} seconds.")
    print(f"📂 Output saved to: {OUTPUT_FILE}")
    print("👉 Next Step: Open the CSV, verify the labels manually, then run the GLiNER training script.")


if __name__ == "__main__":
    main()