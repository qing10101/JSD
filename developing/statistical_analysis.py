import csv
import json
import ollama
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from threading import Lock

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
INPUT_FILE = "test.csv"
MODEL_NAME = "llama3.1:8b"  # or "mistral"
WORKERS = 4  # Number of parallel threads

# The categories we want to track
CATEGORIES = [
    "Occupation_Related",
    "Medical_Condition",
    "Children_Minor",
    "Location_Specific",
    "None"
]


# ------------------------------------------------------------------
# 1. THE LLM CLASSIFIER
# ------------------------------------------------------------------
def classify_text(text):
    """
    Asks Ollama to tag the text with relevant categories.
    """
    system_prompt = f"""
    You are a data classifier. Analyze the text and identify if it contains specific types of personal information.

    Categories to check for:
    1. Occupation_Related: Mentions jobs, shifts, salary, workplace, tasks.
    2. Medical_Condition: Mentions diseases, symptoms, meds, doctors, therapy.
    3. Children_Minor: Mentions sons, daughters, kids, schools, grades.
    4. Location_Specific: Mentions specific cities, landmarks, highways.

    Output strictly valid JSON with a single key "categories" containing a list of matching categories. 
    If none match, return ["None"].
    Example: {{ "categories": ["Occupation_Related", "Children_Minor"] }}
    """

    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': f"Text: \"{text}\""}
            ],
            format='json',
            options={'temperature': 0}
        )
        result = json.loads(response['message']['content'])
        return result.get('categories', ['None'])
    except Exception as e:
        return []


# ------------------------------------------------------------------
# 2. MAIN EXECUTION
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Check category distribution using Ollama.")
    parser.add_argument("--n", type=int, default=50, help="Number of lines to process.")
    args = parser.parse_args()

    print(f"📂 Reading first {args.n} lines from '{INPUT_FILE}'...")

    rows_to_process = []

    # Read the CSV
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8-sig', errors='replace') as f:
            reader = csv.DictReader(f)

            # Handle potential missing column
            if 'original_sentence' not in reader.fieldnames:
                print("❌ Error: 'original_sentence' column not found.")
                print(f"   Found: {reader.fieldnames}")
                return

            for i, row in enumerate(reader):
                if i >= args.n:
                    break

                text = row.get('original_sentence', '').strip()
                if text:
                    rows_to_process.append(text)
    except FileNotFoundError:
        print(f"❌ Error: {INPUT_FILE} not found.")
        return

    print(f"🚀 Classifying {len(rows_to_process)} items with {MODEL_NAME}...")
    print("-" * 60)

    # Parallel Processing
    stats = Counter()
    processed_count = 0
    lock = Lock()

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        future_to_text = {executor.submit(classify_text, text): text for text in rows_to_process}

        for future in as_completed(future_to_text):
            categories = future.result()

            with lock:
                stats.update(categories)
                processed_count += 1

                # Simple progress indicator
                if processed_count % 10 == 0:
                    print(f"   ... processed {processed_count}/{len(rows_to_process)}")

    # ------------------------------------------------------------------
    # 3. REPORT
    # ------------------------------------------------------------------
    print("-" * 60)
    print(f"📊 DISTRIBUTION REPORT (Based on sample of {processed_count} rows)")
    print("-" * 60)

    total_tags = sum(stats.values())

    # Sort by most common
    for category, count in stats.most_common():
        # Calculate percentage relative to number of rows processed
        # (Note: Percentages can add up to >100% because one text can have multiple tags)
        percentage = (count / processed_count) * 100
        bar = "█" * int(percentage / 5)
        print(f"{category:<20} : {count:>3} ({percentage:5.1f}%) | {bar}")

    print("-" * 60)
    print("💡 Use this data to decide which labels need more training examples.")


if __name__ == "__main__":
    main()