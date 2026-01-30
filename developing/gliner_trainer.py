import random
import pandas as pd
from gliner import GLiNER
from gliner.training import Trainer, TrainingArguments
# 1. IMPORT THE CORRECT COLLATOR
from gliner.data_processing.collator import SpanDataCollator

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
INPUT_CSV = "test_auto_labeled_new.csv"
OUTPUT_DIR = "gliner_finetuned"

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
    print(f"📂 Reading {len(df)} rows...")
    success = 0

    for _, row in df.iterrows():
        text = str(row['original_sentence'])
        tokens = text.split()
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

        if entities:
            training_data.append({
                "tokenized_text": tokens,
                "ner": entities
            })
            success += 1

    print(f"✅ Prepared {success} valid training samples.")
    return training_data


# ------------------------------------------------------------------
# MAIN EXECUTION
# ------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Prepare Data
    raw_data = prepare_data(INPUT_CSV)

    if len(raw_data) == 0:
        print("❌ No data found.")
        exit()

    # 2. Split Data
    random.seed(42)
    random.shuffle(raw_data)
    split_idx = int(len(raw_data) * 0.9)
    train_set = raw_data[:split_idx]
    eval_set = raw_data[split_idx:]

    print(f"📊 Data Split: {len(train_set)} Training | {len(eval_set)} Evaluation")

    # 3. Load Model
    print("⏳ Loading Model...")
    model = GLiNER.from_pretrained("numind/NuNER_Zero-span")

    # 4. INITIALIZE DATA COLLATOR
    if hasattr(model, "data_processor"):
        data_processor = model.data_processor
    else:
        data_processor = model

    # FIX: Use SpanDataCollator instead of DataCollator
    data_collator = SpanDataCollator(model.config, data_processor=data_processor, prepare_labels=True)

    # 5. Configure Training
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=1e-5,
        num_train_epochs=5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        eval_strategy="steps",
        save_steps=100,
        eval_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="none",
        remove_unused_columns=False,
    )

    # 6. Train
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_set,
        eval_dataset=eval_set,
        data_collator=data_collator,
    )

    print("🚀 Starting Training...")
    trainer.train()

    print(f"💾 Saving model to '{OUTPUT_DIR}'...")
    model.save_pretrained(OUTPUT_DIR)
    print("✅ Done!")