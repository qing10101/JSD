import os
import random
import torch
import pandas as pd
from gliner import GLiNER
from gliner.training import Trainer, TrainingArguments
from gliner.data_processing.collator import SpanDataCollator

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
INPUT_CSV = "complete.csv"  # Your new final file
OUTPUT_DIR = "gliner_nuner_finetuned"
MODEL_NAME = "numind/NuNER_Zero-span"  # The winner from your benchmark

COLUMN_MAPPING = {
    "occupation_col": "occupation indication",
    "medical_col": "medical condition related",
    "children_col": "children/minor related"
}


# ------------------------------------------------------------------
# DATA PREPARATION
# ------------------------------------------------------------------
def prepare_data(csv_path):
    df = pd.read_csv(csv_path)
    training_data = []

    print(f"📂 Processing {len(df)} rows...")

    for _, row in df.iterrows():
        # UPDATE: CSV uses 'original_text' now
        text = str(row.get('original_text', ''))
        if not text or text.lower() == 'nan': continue

        tokens = text.split()
        if not tokens: continue

        entities = []
        for csv_col, label_name in COLUMN_MAPPING.items():
            if csv_col not in df.columns: continue

            cell_value = str(row[csv_col])
            if cell_value.lower() in ['nan', '', 'none']: continue

            phrases = [p.strip() for p in cell_value.split(';') if p.strip()]

            for phrase in phrases:
                phrase_words = phrase.split()
                len_phrase = len(phrase_words)

                for i in range(len(tokens) - len_phrase + 1):
                    window = tokens[i: i + len_phrase]
                    window_clean = " ".join(window).lower().replace(",", "").replace(".", "")
                    phrase_clean = " ".join(phrase_words).lower().replace(",", "").replace(".", "")

                    if window_clean == phrase_clean:
                        entities.append([i, i + len_phrase - 1, label_name])
                        break

                        # We include the row if it has entities OR if it's one of your 1/3rd negatives (no entities)
        training_data.append({"tokenized_text": tokens, "ner": entities})

    print(f"✅ Prepared {len(training_data)} training samples.")
    return training_data


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Prepare Data
    raw_data = prepare_data(INPUT_CSV)
    random.seed(42)
    random.shuffle(raw_data)

    split_idx = int(len(raw_data) * 0.9)
    train_set = raw_data[:split_idx]
    eval_set = raw_data[split_idx:]

    # 2. Load Model
    print(f"⏳ Loading Base Model: {MODEL_NAME}...")
    model = GLiNER.from_pretrained(MODEL_NAME)

    # 3. Setup Collator
    if hasattr(model, "data_processor"):
        data_processor = model.data_processor
    else:
        data_processor = model
    data_collator = SpanDataCollator(model.config, data_processor=data_processor, prepare_labels=True)

    # 4. Training Arguments (Tuned for A100 & 3,500 rows)
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=5e-6,  # Lower learning rate for fine-tuning
        num_train_epochs=6,  # 3 is enough for this dataset size
        weight_decay=0.1,

        # A100 OPTIMIZATIONS
        per_device_train_batch_size=16,  # Higher batch size for A100
        per_device_eval_batch_size=16,
        fp16=True,  # Use mixed precision for speed

        # Logging & Saving
        eval_strategy="steps",
        eval_steps=100,
        save_steps=100,
        logging_steps=20,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="loss",

        remove_unused_columns=False,
        report_to="none"
    )

    # 5. Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_set,
        eval_dataset=eval_set,
        data_collator=data_collator
    )

    print("🚀 Starting Training on A100...")
    trainer.train()

    # 6. Final Evaluation
    print("\n📊 Final Evaluation Results:")
    metrics = trainer.evaluate()
    print(metrics)

    # 7. Save
    print(f"💾 Saving model to '{OUTPUT_DIR}'...")
    model.save_pretrained(OUTPUT_DIR)
    print("✅ Training Complete!")