import os
import random
import pandas as pd
import torch
from torch.utils.data import DataLoader
from gliner import GLiNER
import re
from tqdm import tqdm

# --- IMPORT FIX ---
try:
    from gliner.data_processing.collator import SpanDataCollator as MyCollator
except ImportError:
    from gliner.data_processing.collator import DataCollator as MyCollator

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
INPUT_CSV = "gold_final_cleaned.csv"
OUTPUT_DIR = "gliner_nuner_final_model"
MODEL_NAME = "numind/NuNER_Zero-span"
BATCH_SIZE = 4           # Small batch to minimize data loss per skip
ACCUMULATION_STEPS = 2   # Update weights every 2 batches (4x2 = 8 effective batch size)
LEARNING_RATE = 5e-6
EPOCHS = 3

COLUMN_MAPPING = {
    "occupation_col": "occupation indication",
    "medical_col": "medical condition related",
    "children_col": "author's minor children related"
}


# ------------------------------------------------------------------
# DATA PREPARATION (Includes Neural Diagnostic)
# ------------------------------------------------------------------
def robust_normalize(text):
    if not text or pd.isna(text): return ""
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return " ".join(text.split())


def prepare_data_neural_safe(csv_path, model):
    df = pd.read_csv(csv_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    clean_data = []
    print(f"📂 Diagnostic Check: Testing {len(df)} rows...")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        text = str(row.get('original_text', '')).strip()
        if not text or len(text.split()) < 10: continue

        tokens = text.split()
        entities = []
        for csv_col, label_name in COLUMN_MAPPING.items():
            if csv_col not in df.columns: continue
            val = str(row[csv_col])
            if val.lower() in ['nan', '', 'none']: continue
            phrases = [p.strip() for p in val.split(';') if p.strip()]
            for phrase in phrases:
                target_norm = robust_normalize(phrase)
                for i in range(len(tokens) - len(phrase.split()) + 1):
                    if target_norm == robust_normalize(" ".join(tokens[i: i + len(phrase.split())])):
                        entities.append([i, i + len(phrase.split()) - 1, label_name])
                        break

        try:
            with torch.no_grad():
                _ = model.predict_entities(text, labels=["test"], threshold=0.5)
            clean_data.append({"tokenized_text": tokens, "ner": entities})
        except:
            continue

    return clean_data


# ------------------------------------------------------------------
# EVALUATION FUNCTION
# ------------------------------------------------------------------
def run_evaluation(model, dataloader, device):
    model.eval()
    total_eval_loss = 0
    valid_batches = 0

    # We use iter to handle potential collator crashes in eval set too
    data_iter = iter(dataloader)

    for _ in range(len(dataloader)):
        try:
            batch = next(data_iter)
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    outputs = model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        words_mask=batch['words_mask'],
                        text_lengths=batch['text_lengths'],
                        span_idx=batch['span_idx'],
                        span_mask=batch['span_mask'],
                        labels=batch['labels']
                    )
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

    # 2. Prepare Data & Split (90/10)
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

    print(f"\n🚀 Training on {len(train_data)} samples | Evaluating on {len(eval_data)} samples")
    print("-" * 60)

    # Inside the Epoch Loop:
    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0
        train_batches = 0

        data_iter = iter(train_loader)
        optimizer.zero_grad()  # Initialize zero grad at start of epoch

        for i in range(len(train_loader)):
            try:
                batch = next(data_iter)
            except:
                continue

            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            try:
                # We DON'T call optimizer.zero_grad() here anymore
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    outputs = model(**batch)
                    # Divide loss by accumulation steps to average the gradient
                    loss = outputs.loss / ACCUMULATION_STEPS

                loss.backward()

                # ONLY UPDATE WEIGHTS every 'ACCUMULATION_STEPS'
                if (i + 1) % ACCUMULATION_STEPS == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                total_train_loss += outputs.loss.item()
                train_batches += 1

            except Exception as e:
                # If batch fails, clear the accumulated gradients to prevent "poisoning" the next batch
                optimizer.zero_grad()
                continue

        # --- EVALUATION PHASE ---
        avg_train_loss = total_train_loss / train_batches if train_batches > 0 else 0
        avg_eval_loss = run_evaluation(model, eval_loader, device)

        print(f"\n📊 EPOCH {epoch + 1} SUMMARY:")
        print(f"   Avg Train Loss: {avg_train_loss:.4f}")
        print(f"   Avg Eval Loss:  {avg_eval_loss:.4f}")
        print("-" * 60)

    # 5. Save
    model.save_pretrained(OUTPUT_DIR)
    print(f"🎉 Training Complete. Model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()