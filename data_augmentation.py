import pandas as pd
import json
import time
import os
import csv
from groq import Groq

# --- CONFIGURATION ---
# Replace with your actual Groq API Key
GROQ_API_KEY = "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"

# Models for Round-Robin (Verify these IDs in your Groq playground)
MODELS = [
    "llama-3.3-70b-versatile",
    "moonshotai/kimi-k2-instruct-0905",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b"
]

OUTPUT_FILE = "test_augmented_live.csv"
client = Groq(api_key=GROQ_API_KEY)


def get_variations_with_labels(text, category, start_index):
    """
    Tries models in order. Skips if 3 models in a row fail (Safety Filter).
    """
    failures = 0
    for i in range(len(MODELS)):
        if failures >= 3:
            return "SKIP"

        model_id = MODELS[(start_index + i) % len(MODELS)]

        prompt = f"""
        Text: "{text}"
        Category: {category}

        Task: Generate 6 variations of the text. Keep meaning identical.
        For each variation, extract these entities if present:
        - medical: any medical conditions/terms
        - minor: any mentions of children/minors
        - gender: any gender-specific terms

        Return ONLY a JSON object:
        {{
          "variations": [
            {{"text": "...", "medical": "...", "minor": "...", "gender": "..."}},
            ...
          ]
        }}
        """

        try:
            completion = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.7,
                timeout=15
            )
            data = json.loads(completion.choices[0].message.content)
            return data.get("variations", [])
        except Exception as e:
            failures += 1
            print(f"   ⚠️ {model_id} failed ({failures}/3): {str(e)[:50]}")
            time.sleep(2)  # Cool down

    return None


# --- MAIN EXECUTION ---
df = pd.read_csv('test.csv')
fieldnames = df.columns.tolist()

# Initialize file and write header if it's a new run
file_exists = os.path.isfile(OUTPUT_FILE)
with open(OUTPUT_FILE, 'a', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if not file_exists:
        writer.writeheader()

    print(f"🚀 Starting Augmentation. Saving directly to {OUTPUT_FILE}...")

    for i, row in df.iterrows():
        # 1. Write the Original Row immediately
        writer.writerow(row.to_dict())
        f.flush()  # Forces write to disk

        # 2. Get 6 Variations
        print(f"[{i + 1}/{len(df)}] Processing: {row['unique_id']}...")
        results = get_variations_with_labels(row['original_text'], row['gliner_hint_category'], i)

        if results == "SKIP":
            print(f"   🛑 Safety Filter triggered. Skipping variations for this row.")
            continue

        if results:
            for idx, item in enumerate(results):
                new_row = row.copy()
                new_row['unique_id'] = f"{row['unique_id']}_v{idx + 1}"
                new_row['original_text'] = item.get('text')
                new_row['medical_col'] = item.get('medical')
                new_row['minor_col'] = item.get('minor')
                new_row['gender_col'] = item.get('gender')

                # 3. Save variation on-the-fly
                writer.writerow(new_row.to_dict())

            f.flush()  # Ensure variations are saved before next iteration
            print(f"   ✅ Saved 6 variations.")

        # 4. Stay within Free Tier limits (RPM)
        time.sleep(1.5)

print(f"\n🎉 Done! You can inspect '{OUTPUT_FILE}' even while the script is running.")