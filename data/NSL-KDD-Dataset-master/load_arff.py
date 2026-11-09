import pandas as pd
from io import StringIO

file_path = "KDDTest+.arff"   # 🔹 Change here for test dataset

# Read ARFF file as text (skip bad header)
with open(file_path, "r") as f:
    lines = f.readlines()

# Skip lines starting with @ (metadata)
data_lines = [line for line in lines if not line.strip().startswith("@")]

# Convert into DataFrame
df = pd.read_csv(StringIO("".join(data_lines)), header=None)

print("✅ Dataset Loaded as CSV-like")
print("Rows:", df.shape[0], " Columns:", df.shape[1])
print(df.head())

# 🔹 Save to CSV for testing
df.to_csv("KDDTest.csv", index=False)
print("💾 Dataset saved as KDDTest.csv")
