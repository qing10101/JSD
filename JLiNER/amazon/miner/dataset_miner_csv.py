import csv
import os
import json
import multiprocessing as mp
from gliner import GLiNER
from tqdm import tqdm

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
INPUT_FILE = "/Users/scottwang/PycharmProjects/JSD/JLiNER/reserve_data/test.csv"  # Replace with your actual filename
OUTPUT_FILE = "mined_queries_to_label.csv"

# Using Small model for high-speed, low-RAM multi-processing
MODEL_NAME = "urchade/gliner_small-v2.1"

NUM_WORKERS = max(1, os.cpu_count() - 2)
BATCH_SIZE = 50

# QUOTAS (Targeting 3,500 total)
TARGET_COUNTS = {
    "medical condition related": 800,
    "author's minor children related": 800,
    "occupation indication": 800,
    "no_pii_negative": 1100
}

LABELS = list(TARGET_COUNTS.keys())[:3]  # The first 3 labels for GLiNER

# ------------------------------------------------------------------
# WORKER LOGIC
# ------------------------------------------------------------------
worker_model = None


def init_worker():
    global worker_model
    import logging
    logging.getLogger("transformers").setLevel(logging.ERROR)
    worker_model = GLiNER.from_pretrained(MODEL_NAME)
    worker_model.eval()


def process_batch(batch_rows):
    global worker_model
    results = []

    for row in batch_rows:
        text = row.get('ori_review', '').strip()
        unique_id = row.get('qid', 'unknown')

        if len(text.split()) < 5: continue

        entities = worker_model.predict_entities(
            text, LABELS, threshold=0.25, flat_ner=False
        )

        results.append({
            "unique_id": unique_id,
            "text": text,
            "labels": set([e['label'] for e in entities]),
            "entities": entities
        })
    return results


# ------------------------------------------------------------------
# MAIN PROCESS
# ------------------------------------------------------------------
def main():
    print(f"🚀 Starting Miner on CSV: {INPUT_FILE}")

    current_counts = {k: 0 for k in TARGET_COUNTS.keys()}
    total_saved = 0

    with open(OUTPUT_FILE, 'w', encoding='utf-8-sig', newline='') as outfile:
        fieldnames = ['unique_id', 'original_text', 'gliner_hint_category', 'gliner_hint_text']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=NUM_WORKERS, initializer=init_worker) as pool:

            def csv_reader():
                batch = []
                with open(INPUT_FILE, 'r', encoding='utf-8-sig', errors='replace') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        batch.append(row)
                        if len(batch) >= BATCH_SIZE:
                            yield batch
                            batch = []
                    if batch: yield batch

            pbar = tqdm(total=sum(TARGET_COUNTS.values()), desc="Filling Quotas")

            for batch_results in pool.imap_unordered(process_batch, csv_reader()):
                if all(current_counts[k] >= TARGET_COUNTS[k] for k in TARGET_COUNTS):
                    pool.terminate()
                    break

                for res in batch_results:
                    labels_found = res['labels']
                    selected_category = None

                    # QUOTA LOGIC
                    if "medical condition related" in labels_found and current_counts["medical condition related"] < \
                            TARGET_COUNTS["medical condition related"]:
                        selected_category = "medical condition related"
                    elif "author's minor children related" in labels_found and current_counts[
                        "author's minor children related"] < TARGET_COUNTS["author's minor children related"]:
                        selected_category = "author's minor children related"
                    elif "occupation indication" in labels_found and current_counts["occupation indication"] < \
                            TARGET_COUNTS["occupation indication"]:
                        selected_category = "occupation indication"
                    elif not labels_found and current_counts["no_pii_negative"] < TARGET_COUNTS["no_pii_negative"]:
                        selected_category = "no_pii_negative"

                    if selected_category:
                        hints = [e['text'] for e in res['entities'] if e['label'] == selected_category]
                        writer.writerow({
                            'unique_id': res['unique_id'],
                            'original_text': res['text'].replace('\n', ' ').strip(),
                            'gliner_hint_category': selected_category,
                            'gliner_hint_text': "; ".join(hints)
                        })
                        current_counts[selected_category] += 1
                        total_saved += 1
                        pbar.update(1)

            pbar.close()

    print("\n" + "=" * 60)
    print(f"🏁 Mining Complete. Saved: {total_saved} entries")
    for cat, count in current_counts.items():
        print(f"{cat:<35} : {count:>4} / {TARGET_COUNTS[cat]}")


if __name__ == "__main__":
    main()