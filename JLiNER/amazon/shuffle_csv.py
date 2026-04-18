import pandas as pd
from pathlib import Path

input_path = Path(__file__).parent / "miner" / "amazon_mined_balanced.csv"
output_path = Path(__file__).parent / "amazon_mined_shuffled.csv"

df = pd.read_csv(input_path)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv(output_path, index=False)

print(f"Shuffled {len(df)} rows -> {output_path}")
