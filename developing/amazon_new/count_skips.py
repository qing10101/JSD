import csv
import os
from collections import Counter

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
INPUT_FILE = "amazon_labeled_groq_v2.csv"
SAFETY_FLAG = "SAFETY_FILTER_SKIP"


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: {INPUT_FILE} not found.")
        return

    print(f"📂 Scanning '{INPUT_FILE}'...")

    stats = Counter()
    safety_entries = []
    total_rows = 0

    with open(INPUT_FILE, 'r', encoding='utf-8-sig', errors='replace') as f:
        reader = csv.DictReader(f)

        if 'model_used' not in reader.fieldnames:
            print("❌ Error: Column 'model_used' missing in CSV.")
            return

        for row in reader:
            total_rows += 1
            model = row.get('model_used', 'Unknown')
            stats[model] += 1

            if model == SAFETY_FLAG:
                safety_entries.append(row)

    # ------------------------------------------------------------------
    # REPORT
    # ------------------------------------------------------------------
    print("\n" + "=" * 50)
    print(f"📊 PRODUCTION REPORT")
    print("=" * 50)
    print(f"Total Rows Processed: {total_rows}")
    print("-" * 50)

    # 1. Model Usage Breakdown
    print("Model Workload Distribution:")
    for model, count in stats.most_common():
        pct = (count / total_rows) * 100
        print(f"   • {model:<40} : {count:>4} ({pct:5.1f}%)")

    # 2. Safety Filter Analysis
    skipped_count = stats[SAFETY_FLAG]
    print("-" * 50)
    print(f"🚫 SAFETY FILTERED ROWS: {skipped_count}")

    if skipped_count > 0:
        print("\n📝 Examples of text that triggered filters:")
        print("-" * 50)
        # Show first 5 examples
        for i, row in enumerate(safety_entries[:5]):
            text = row.get('original_text', '')[:100] + "..."
            uid = row.get('unique_id', 'unknown')
            print(f"   [{i + 1}] ID: {uid}")
            print(f"       Text: \"{text}\"")
            print("")

    print("=" * 50)


if __name__ == "__main__":
    main()