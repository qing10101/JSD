# 1. ENSURE STABLE LIBRARIES (Run this in a cell before the script if on Colab)
# !pip install gliner==0.2.24 transformers==4.40.1 accelerate==0.30.0 sentencepiece==0.2.0

import os
import random
import pandas as pd
import torch
import re
from gliner import GLiNER
from gliner.training import Trainer, TrainingArguments

# --- ROBUST COLLATOR IMPORT ---
try:
    from gliner.data_processing.collator import SpanDataCollator as MyCollator
except ImportError:
    from gliner.data_processing.collator import DataCollator as MyCollator

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
INPUT_CSV = "gold_final_cleaned.csv"  # Your standardized file
OUTPUT_DIR = "gliner_nuner_gold_v4"
MODEL_NAME = "numind/NuNER_Zero-span"

# Final mapping to semantic labels
COLUMN_MAPPING = {
    "occupation_col": "occupation indication",
    "medical_col": "medical condition related",
    "children_col": "author's minor children related"
}


# ------------------------------------------------------------------
# HELPER: ROBUST NORMALIZATION
# ------------------------------------------------------------------
def robust_normalize(text):
    if not text or pd.isna(text): return ""
    text = str(text).lower()
    # Remove all non-alphanumeric characters except spaces
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return " ".join(text.split())


# ------------------------------------------------------------------
# DATA PREPARATION
# ------------------------------------------------------------------
def prepare_data(csv_path, model):
    df = pd.read_csv(csv_path)
    training_data = []

    # Use the model's internal processor to validate sequence dimensions
    processor = model.data_processor

    print(f"📂 Processing {len(df)} standardized rows...")
    stats = {"kept": 0, "skipped": 0, "entities": 0, "failed_align": 0}

    for _, row in df.iterrows():
        text = str(row.get('original_text', '')).strip()

        # Skip rows that are too short for NuNER span-pooling
        if not text or len(text.split()) < 5:
            stats["skipped"] += 1
            continue

        tokens = text.split()
        entities = []

        for csv_col, label_name in COLUMN_MAPPING.items():
            if csv_col not in df.columns: continue

            # Since the cleaner standardized everything, we trust the semicolon
            cell_val = str(row[csv_col])
            if cell_val.lower() in ['nan', '', 'none']: continue

            phrases = [p.strip() for p in cell_val.split(';') if p.strip()]

            for phrase in phrases:
                phrase_words = phrase.split()
                if not phrase_words: continue

                # Use Token-Window Matching (ignores punctuation diffs)
                target_norm = robust_normalize(phrase)
                len_phrase = len(phrase_words)

                match_found = False
                for i in range(len(tokens) - len_phrase + 1):
                    window = tokens[i: i + len_phrase]
                    window_norm = robust_normalize(" ".join(window))

                    if target_norm == window_norm:
                        entities.append([i, i + len_phrase - 1, label_name])
                        stats["entities"] += 1
                        match_found = True
                        break

                if not match_found:
                    stats["failed_align"] += 1

        # FINAL GUARD: Ensure the model's CNN layer can handle the sequence shape
        try:
            sample = [{"tokenized_text": tokens, "ner": entities}]
            processor.collate_raw_batch(sample)
            training_data.append({"tokenized_text": tokens, "ner": entities})
            stats["kept"] += 1
        except:
            stats["skipped"] += 1

    print(f"✅ Prepared {stats['kept']} samples (Skipped {stats['skipped']}).")
    print(f"📊 Total Entities: {stats['entities']} (Failed Align: {stats['failed_align']})")
    return training_data


# ------------------------------------------------------------------
# MAIN FUNCTION
# ------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Initialize GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Using Device: {device.upper()}")

    # 2. Load Model FIRST (NuNER needs latest GLiNER lib, but stable Transformers)
    print(f"⏳ Loading Base Model: {MODEL_NAME}...")
    model = GLiNER.from_pretrained(MODEL_NAME)
    model.to(device)

    # 3. Prepare Data (Passing the model instance for validation)
    raw_data = prepare_data(INPUT_CSV, model)

    # Shuffle and Split
    random.seed(42)
    random.shuffle(raw_data)
    split_idx = int(len(raw_data) * 0.9)
    train_set = raw_data[:split_idx]
    eval_set = raw_data[split_idx:]

    print(f"📈 Split: {len(train_set)} Train | {len(eval_set)} Eval")

    # 4. Setup Collator
    data_processor = model.data_processor if hasattr(model, "data_processor") else model
    data_collator = MyCollator(model.config, data_processor=data_processor, prepare_labels=True)

    # 5. Training Arguments (OPTIMIZED FOR A100)
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=5e-6,  # Very low for fine-tuning stability
        num_train_epochs=5,
        weight_decay=0.1,

        # Hardware Stability
        bf16=True,  # Use BF16 for A100 (Native, no Scaler crash)
        fp16=False,  # Disable FP16 to avoid AssertionError
        gradient_checkpointing=False,  # Disable to avoid dimension errors

        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,

        # Logging & Strategy
        eval_strategy="steps",
        eval_steps=50,
        save_steps=50,
        logging_steps=20,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="loss",

        remove_unused_columns=False,
        report_to="none"
    )

    # 6. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_set,
        eval_dataset=eval_set,
        data_collator=data_collator
    )

    # 7. Start Training
    print("🏋️ Starting Training...")
    trainer.train()

    # 8. Save
    print(f"💾 Saving final model to '{OUTPUT_DIR}'...")
    model.save_pretrained(OUTPUT_DIR)
    print("🎉 Done! Model is ready for evaluation.")