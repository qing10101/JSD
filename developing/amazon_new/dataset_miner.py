import json
import csv
import os
import multiprocessing as mp
from gliner import GLiNER
from tqdm import tqdm

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
INPUT_FILE = "/Users/scottwang/PycharmProjects/JSD/developing/amazon_new/Unknown.jsonl"
OUTPUT_FILE = "amazon_to_label.csv"

# CRITICAL: Use the SMALL model for multiprocessing to save RAM
MODEL_NAME = "urchade/gliner_small-v2.1"

# WORKER CONFIG
# Set this to the number of CPU cores you have (e.g., 4, 8, or 10)
NUM_WORKERS = max(1, os.cpu_count() - 2)
BATCH_SIZE = 50  # Number of reviews sent to a worker at once

# THRESHOLDS
CONFIDENCE_THRESHOLD = 0.25
MAX_TOKEN_LENGTH = 300
MAX_SCAN_LIMIT = 2000000  # Scan up to 2 million rows if needed

# QUOTAS
TARGET_COUNTS = {
    "medical condition related": 800,
    "children/minor related": 800,
    "occupation indication": 800,
    "no_pii_negative": 1100
}

LABELS = [
    "occupation indication",
    "medical condition related",
    "children/minor related"
]

# ------------------------------------------------------------------
# WORKER FUNCTIONS (Must be at top level for multiprocessing)
# ------------------------------------------------------------------
# Global variable for the worker process to hold the model
worker_model = None


def init_worker():
    """Initializes the model once per process."""
    global worker_model
    # Suppress warnings in workers to keep console clean
    import logging
    logging.getLogger("transformers").setLevel(logging.ERROR)

    try:
        worker_model = GLiNER.from_pretrained(MODEL_NAME)
        worker_model.eval()
    except Exception as e:
        print(f"Worker failed to load model: {e}")


def process_batch(batch_lines):
    """
    Receives a list of JSON strings. Returns a list of results.
    """
    global worker_model
    results = []

    for line in batch_lines:
        try:
            data = json.loads(line)
            text = data.get('text', '').strip()

            # Simple unique ID generation
            user_id = data.get('user_id', 'u')
            asin = data.get('asin', 'a')
            unique_id = f"{user_id}_{asin}"

            # Filter Length
            token_count = len(text.split())
            if token_count < 10 or token_count > MAX_TOKEN_LENGTH:
                continue

            # Inference
            entities = worker_model.predict_entities(
                text, LABELS, threshold=CONFIDENCE_THRESHOLD, flat_ner=False
            )

            # Pack result
            detected_labels = [e['label'] for e in entities]

            # We return the data needed for the Main Process to make decisions
            results.append({
                "unique_id": unique_id,
                "text": text,
                "labels": set(detected_labels),
                "entities": entities  # Needed for hints
            })

        except:
            continue

    return results


# ------------------------------------------------------------------
# MAIN PROCESS
# ------------------------------------------------------------------
def main():
    print(f"🚀 Starting Multiprocessing Miner")
    print(f"   - Model: {MODEL_NAME}")
    print(f"   - Workers: {NUM_WORKERS}")
    print(f"   - Batch Size: {BATCH_SIZE}")
    print("-" * 60)

    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: {INPUT_FILE} not found.")
        return

    # Trackers
    current_counts = {k: 0 for k in TARGET_COUNTS.keys()}
    total_saved = 0
    rows_scanned = 0

    # Open Output File
    with open(OUTPUT_FILE, 'w', encoding='utf-8-sig', newline='') as outfile:
        fieldnames = ['unique_id', 'original_text', 'gliner_hint_category', 'gliner_hint_text']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        # Initialize Pool
        # "spawn" is safer for PyTorch on Mac/Linux than "fork"
        ctx = mp.get_context("spawn")

        with ctx.Pool(processes=NUM_WORKERS, initializer=init_worker) as pool:

            # Generator to read file in chunks
            def file_reader():
                batch = []
                with open(INPUT_FILE, 'r', encoding='utf-8') as f:
                    for line in f:
                        batch.append(line)
                        if len(batch) >= BATCH_SIZE:
                            yield batch
                            batch = []
                    if batch: yield batch

            # Progress Bar
            pbar = tqdm(total=sum(TARGET_COUNTS.values()), desc="Filling Quotas")

            # Process batches as they complete
            for batch_results in pool.imap_unordered(process_batch, file_reader()):

                # Check Global Stop limit
                rows_scanned += len(batch_results)  # Approximate
                if rows_scanned > MAX_SCAN_LIMIT:
                    print("\n🛑 Scan limit reached.")
                    pool.terminate()
                    break

                # Check Quotas
                if all(current_counts[k] >= TARGET_COUNTS[k] for k in TARGET_COUNTS):
                    print("\n✅ All quotas filled!")
                    pool.terminate()
                    break

                # Process the results from the worker
                for res in batch_results:
                    labels_found = res['labels']

                    selected_category = None

                    # PRIORITY LOGIC
                    # 1. Medical
                    if "medical condition related" in labels_found:
                        if current_counts["medical condition related"] < TARGET_COUNTS["medical condition related"]:
                            selected_category = "medical condition related"

                    # 2. Children
                    elif "children/minor related" in labels_found:
                        if current_counts["children/minor related"] < TARGET_COUNTS["children/minor related"]:
                            selected_category = "children/minor related"

                    # 3. Occupation
                    elif "occupation indication" in labels_found:
                        if current_counts["occupation indication"] < TARGET_COUNTS["occupation indication"]:
                            selected_category = "occupation indication"

                    # 4. Negatives
                    elif not labels_found:
                        if current_counts["no_pii_negative"] < TARGET_COUNTS["no_pii_negative"]:
                            selected_category = "no_pii_negative"

                    # SAVE
                    if selected_category:
                        if selected_category == "no_pii_negative":
                            hints_str = ""
                        else:
                            # Extract hints specifically for the chosen category
                            hints = [e['text'] for e in res['entities'] if e['label'] == selected_category]
                            hints_str = "; ".join(hints)

                        writer.writerow({
                            'unique_id': res['unique_id'],
                            'original_text': res['text'].replace('\n', ' '),
                            'gliner_hint_category': selected_category,
                            'gliner_hint_text': hints_str
                        })

                        current_counts[selected_category] += 1
                        total_saved += 1
                        pbar.update(1)

            pbar.close()

    # Final Report
    print("\n" + "=" * 60)
    print(f"🏁 Mining Complete.")
    print(f"   - Saved: {total_saved} entries")
    print("-" * 60)
    for cat, count in current_counts.items():
        status = "✅ Full" if count >= TARGET_COUNTS[cat] else f"⚠️ {TARGET_COUNTS[cat] - count} short"
        print(f"{cat:<25} : {count:>4} / {TARGET_COUNTS[cat]} | {status}")
    print("=" * 60)


if __name__ == "__main__":
    main()