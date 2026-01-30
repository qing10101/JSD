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
INPUT_CSV = "test_auto_labeled_new.csv"
OUTPUT_DIR = "gliner_finetuned_colab"
MODEL_NAME = "numind/NuNER_Zero-span"

COLUMN_MAPPING = {
    "occupation_col": "occupation indication",
    "medical_col": "medical condition related",
    "children_col": "children/minor related"
}


# ------------------------------------------------------------------
# DATA PREPARATION
# ------------------------------------------------------------------
def prepare_data(csv_path):
    if not os.path.exists(csv_path):
        print(f"❌ Error: {csv_path} not found. Please upload it to Colab files.")
        return []

    df = pd.read_csv(csv_path)
    training_data = []

    print(f"📂 Reading {len(df)} rows...")

    for _, row in df.iterrows():
        # Ensure text is a string
        text = str(row.get('original_sentence', ''))
        if not text or text.lower() == 'nan': continue

        # Simple whitespace tokenization
        tokens = text.split()
        if not tokens: continue

        entities = []
        for csv_col, label_name in COLUMN_MAPPING.items():
            if csv_col not in df.columns: continue

            cell_value = str(row[csv_col])
            if cell_value.lower() in ['nan', '', 'none']: continue

            # Split multiple entries by semicolon
            phrases = [p.strip() for p in cell_value.split(';') if p.strip()]

            for phrase in phrases:
                phrase_words = phrase.split()
                len_phrase = len(phrase_words)

                # Find the phrase in the token list
                for i in range(len(tokens) - len_phrase + 1):
                    window = tokens[i: i + len_phrase]
                    # Normalize for matching
                    window_clean = " ".join(window).lower().replace(",", "").replace(".", "")
                    phrase_clean = " ".join(phrase_words).lower().replace(",", "").replace(".", "")

                    if window_clean == phrase_clean:
                        entities.append([i, i + len_phrase - 1, label_name])
                        break

                        # Only add rows that actually have entities to learn
        if entities:
            training_data.append({"tokenized_text": tokens, "ner": entities})

    print(f"✅ Prepared {len(training_data)} valid training samples.")
    return training_data


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Prepare Data
    raw_data = prepare_data(INPUT_CSV)
    if not raw_data:
        print("❌ Script stopped. No data.")
        exit()

    # 2. Split Data (Simple List Slice)
    random.seed(42)
    random.shuffle(raw_data)
    split_idx = int(len(raw_data) * 0.9)
    train_set = raw_data[:split_idx]
    eval_set = raw_data[split_idx:]

    print(f"📊 Train: {len(train_set)} | Eval: {len(eval_set)}")

    # 3. Load Model
    print(f"⏳ Loading Model: {MODEL_NAME}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   - Using Device: {device.upper()}")

    model = GLiNER.from_pretrained(MODEL_NAME)
    model.to(device)

    # 4. Setup Collator
    if hasattr(model, "data_processor"):
        data_processor = model.data_processor
    else:
        data_processor = model

    data_collator = SpanDataCollator(model.config, data_processor=data_processor, prepare_labels=True)

    # 5. Training Config (Optimized for Colab GPU)
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=5e-6,
        num_train_epochs=10,
        per_device_train_batch_size=8,  # T4 GPU can handle batch size 8 easily
        per_device_eval_batch_size=8,
        weight_decay=0.1,
        eval_strategy="steps",
        save_steps=50,
        eval_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="none",
        fp16=True,  # ENABLE MIXED PRECISION (Crucial for Colab speed/memory)
        remove_unused_columns=False
    )

    # 6. Train
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_set,
        eval_dataset=eval_set,
        data_collator=data_collator
    )

    print("🚀 Starting Training...")
    trainer.train()

    print(f"💾 Saving model to '{OUTPUT_DIR}'...")
    model.save_pretrained(OUTPUT_DIR)
    print("✅ Done!")
    print("\n🔍 SANITY CHECK (Visual Inspection)")
    # Load the best model we just trained
    model = GLiNER.from_pretrained(OUTPUT_DIR)
    model.to("cuda")

    test_sentences = [
        "I had to leave my shift early to take my son to the doctor for his asthma.",
        "This job is killing my back."
    ]

    for text in test_sentences:
        print(f"\nText: {text}")
        entities = model.predict_entities(text, list(COLUMN_MAPPING.values()), threshold=0.3)
        for e in entities:
            print(f"  👉 Found: '{e['text']}' -> {e['label']} ({e['score']:.1%})")