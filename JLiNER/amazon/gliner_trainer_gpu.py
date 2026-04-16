import os
import random
import pandas as pd
import torch
from torch.utils.data import DataLoader
from gliner import GLiNER
import re
from tqdm import tqdm
from collections import defaultdict

# --- IMPORT FIX ---
try:
    from gliner.data_processing.collator import SpanDataCollator as MyCollator
except ImportError:
    from gliner.data_processing.collator import DataCollator as MyCollator

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
INPUT_CSV = "merged_cleaned.csv"
OUTPUT_DIR = "round alpha"
MODEL_NAME = "numind/NuNER_Zero-span"
BATCH_SIZE = 4
ACCUMULATION_STEPS = 2
LEARNING_RATE = 5e-6
EPOCHS = 3

COLUMN_MAPPING = {
    "gender_col": "reviewer's gender indication",
    "medical_col": "medical condition related",
    "minor_col": "minor children related"
}


# ------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------
def robust_normalize(text):
    if not text or pd.isna(text): return ""
    text = str(text).lower()

    # 1. Specifically remove possessive 's so "son's" becomes "son"
    # The space before 's? handles cases where it's at the end of a word
    text = re.sub(r"'s\b", "", text)

    # 2. Remove all other punctuation
    text = re.sub(r'[^a-z0-9\s]', '', text)

    # 3. Collapse whitespace
    return " ".join(text.split())


# ------------------------------------------------------------------
# METRIC CALCULATION LOGIC
# ------------------------------------------------------------------
def calculate_f1_metrics(model, eval_data, device):
    model.eval()
    stats = {label: {"tp": 0, "fp": 0, "fn": 0} for label in COLUMN_MAPPING.values()}
    labels = list(COLUMN_MAPPING.values())

    for item in eval_data:
        text = " ".join(item["tokenized_text"])
        gt_by_cat = defaultdict(list)
        for start, end, label in item["ner"]:
            phrase = " ".join(item["tokenized_text"][start:end + 1])
            gt_by_cat[label].append(robust_normalize(phrase))

        with torch.no_grad():
            preds = model.predict_entities(text, labels, threshold=0.5)

        pred_by_cat = defaultdict(list)
        for p in preds:
            pred_by_cat[p['label']].append(robust_normalize(p['text']))

        for label in labels:
            gts = gt_by_cat[label].copy()
            prs = pred_by_cat[label].copy()
            for gt in gts:
                match = False
                for pr in prs:
                    if gt in pr or pr in gt:
                        stats[label]["tp"] += 1
                        prs.remove(pr)
                        match = True
                        break
                if not match:
                    stats[label]["fn"] += 1
            stats[label]["fp"] += len(prs)

    results = {}
    for label in labels:
        tp, fp, fn = stats[label]["tp"], stats[label]["fp"], stats[label]["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
        results[label] = {"f1": f1, "accuracy": accuracy, "tp": tp, "fp": fp, "fn": fn}
    return results


# ------------------------------------------------------------------
# DATA PREPARATION (Includes Neural Diagnostic & Match All Logic)
# ------------------------------------------------------------------
def prepare_data_neural_safe(csv_path, model):
    df = pd.read_csv(csv_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    clean_data = []
    print(f"📂 Diagnostic Check: Testing {len(df)} rows...")
    stats = {"kept": 0, "skipped": 0}

    for _, row in tqdm(df.iterrows(), total=len(df)):
        text = str(row.get('original_text', '')).strip()
        if not text or len(text.split()) < 10:
            stats["skipped"] += 1
            continue

        tokens = text.split()
        entities = []
        claimed_indices = set()  # Keeps track of tokens already assigned to an entity

        # 1. Collect all phrases and sort by length DESC (Longest first)
        all_phrases = []
        for csv_col, label_name in COLUMN_MAPPING.items():
            if csv_col not in df.columns: continue
            val = str(row[csv_col])
            if val.lower() in ['nan', '', 'none']: continue
            phrases = [p.strip() for p in val.split(';') if p.strip()]
            for p in phrases:
                all_phrases.append((p, label_name))

        # Sort by phrase length (longest first)
        all_phrases.sort(key=lambda x: len(x[0].split()), reverse=True)

        # 2. Match phrases
        for phrase, label_name in all_phrases:
            target_norm = robust_normalize(phrase)
            phrase_len = len(phrase.split())

            # Find ALL occurrences in the text
            for i in range(len(tokens) - phrase_len + 1):
                window = tokens[i: i + phrase_len]
                if target_norm == robust_normalize(" ".join(window)):

                    # 3. Check if tokens are already "claimed" by a longer entity
                    current_span = set(range(i, i + phrase_len))
                    if not claimed_indices.intersection(current_span):
                        entities.append([i, i + phrase_len - 1, label_name])
                        claimed_indices.update(current_span)

        # 4. Final Neural Check
        try:
            with torch.no_grad():
                _ = model.predict_entities(text, labels=["test"], threshold=0.5)
            clean_data.append({"tokenized_text": tokens, "ner": entities})
            stats["kept"] += 1
        except:
            continue

    print(f"✅ Prepared {stats['kept']} samples. Skipped {stats['skipped']} rows.")
    return clean_data


# ------------------------------------------------------------------
# EVALUATION FUNCTION (Loss Only)
# ------------------------------------------------------------------
def run_evaluation(model, dataloader, device):
    model.eval()
    total_eval_loss = 0
    valid_batches = 0
    data_iter = iter(dataloader)
    for _ in range(len(dataloader)):
        try:
            batch = next(data_iter)
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    outputs = model(**batch)
                    total_eval_loss += outputs.loss.item()
                    valid_batches += 1
        except:
            continue
    return total_eval_loss / valid_batches if valid_batches > 0 else 0


# ------------------------------------------------------------------
# MAIN TRAINING LOOP
# ------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Model
    model = GLiNER.from_pretrained(MODEL_NAME)

    # 2. Prepare Data & Split
    all_data = prepare_data_neural_safe(INPUT_CSV, model)
    random.seed(42)
    random.shuffle(all_data)

    split_idx = int(len(all_data) * 0.9)
    train_data = all_data[:split_idx]
    eval_data = all_data[split_idx:]

    # 3. Setup Loaders
    data_processor = model.data_processor if hasattr(model, "data_processor") else model
    collator = MyCollator(model.config, data_processor=data_processor, prepare_labels=True)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator)
    eval_loader = DataLoader(eval_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator)

    # 4. Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    best_eval_loss = float('inf')

    print(f"\n🚀 Training on {len(train_data)} samples | Evaluating on {len(eval_data)} samples")
    print("-" * 60)

    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0
        train_batches = 0
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{EPOCHS} [Train]")

        data_iter = iter(train_loader)
        for i in range(len(train_loader)):
            try:
                batch = next(data_iter)
            except:
                continue
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            optimizer.zero_grad()
            try:
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    outputs = model(**batch)
                    loss = outputs.loss
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
                train_batches += 1
                pbar.update(1)
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            except:
                optimizer.zero_grad()
                continue

        # --- EVALUATION PHASE ---
        avg_train_loss = total_train_loss / train_batches if train_batches > 0 else 0
        avg_eval_loss = run_evaluation(model, eval_loader, device)

        print(f"\n🧪 Calculating Category Metrics for Epoch {epoch + 1}...")
        cat_metrics = calculate_f1_metrics(model, eval_data, device)

        print(f"\n📊 EPOCH {epoch + 1} SUMMARY:")
        print(f"   Avg Train Loss: {avg_train_loss:.4f}")
        print(f"   Avg Eval Loss:  {avg_eval_loss:.4f}")
        print("\n   Category Performance:")
        print(f"   {'Category':<35} | {'F1':<6} | {'Acc':<6} | {'TP/FP/FN'}")
        print(f"   {'-' * 35}-|--------|--------|----------")

        for cat, m in cat_metrics.items():
            print(f"   {cat:<35} | {m['f1']:.3f}  | {m['accuracy']:.3f}  | {m['tp']}/{m['fp']}/{m['fn']}")

        if avg_eval_loss < best_eval_loss:
            best_eval_loss = avg_eval_loss
            print(f"\n🌟 NEW BEST MODEL FOUND! Saving to '{OUTPUT_DIR}'...")
            model.save_pretrained(OUTPUT_DIR)
        else:
            print(f"\n⚠️ Eval loss did not improve (Best: {best_eval_loss:.4f}). Skipping save.")
        print("-" * 60)

    print(f"\n🎉 Training Complete. The best version (Loss: {best_eval_loss:.4f}) is in '{OUTPUT_DIR}'")


if __name__ == "__main__":
    main()