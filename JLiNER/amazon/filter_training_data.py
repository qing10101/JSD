import pandas as pd

# Load your last merged file
df = pd.read_csv("gold.csv")

# 1. Create the Training Set (Only successful labels)
# Filter out the skips and any rows where all models failed
train_df = df[~df['model_used'].isin(['SAFETY_FILTER_SKIP', 'FAILED_ALL_MODELS', 'LIMIT_EXHAUSTED_OR_FAILED'])]

# 2. Save for the A100 training script
train_df.to_csv("gold_final.csv", index=False)

print(f"📊 Training Set cleaned!")
print(f"   - Original: {len(df)}")
print(f"   - Successfully Labeled: {len(train_df)}")
print(f"   - Removed (for Eval/Discard): {len(df) - len(train_df)}")