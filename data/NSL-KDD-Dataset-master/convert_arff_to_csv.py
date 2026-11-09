from scipy.io import arff
import pandas as pd

# Load ARFF file
data = arff.loadarff("KDDTrain+.arff")  # make sure this matches your file name

# Convert to DataFrame
df = pd.DataFrame(data[0])

# Decode byte strings to normal strings
for col in df.select_dtypes([object]):
    df[col] = df[col].str.decode('utf-8')

# Save as CSV
df.to_csv("KDDTrain+.csv", index=False)
print("✅ ARFF file converted to CSV successfully!")
