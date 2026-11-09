import pandas as pd
import joblib

# ----------------------------
# Load trained artifacts
# ----------------------------
model = joblib.load("model.pkl")               # Trained model
encoders = joblib.load("encoders.pkl")        # Dictionary of LabelEncoders for categorical features
feature_columns = joblib.load("feature_columns.pkl")  # List of feature column names used during training

# ----------------------------
# Load input CSV
# ----------------------------
input_file = "large_test_input_realistic.csv"  # Replace with your file if different
df = pd.read_csv(input_file)

# ----------------------------
# Encode categorical columns
# ----------------------------
for col, encoder in encoders.items():
    if col in df.columns:
        df[col] = encoder.transform(df[col].astype(str))

# ----------------------------
# Ensure correct column order
# ----------------------------
df = df[feature_columns]

# ----------------------------
# Make predictions
# ----------------------------
predictions = model.predict(df)

# ----------------------------
# Save or print results
# ----------------------------
df['prediction'] = predictions
df.to_csv("predictions_output.csv", index=False)
print("Predictions saved to predictions_output.csv")
print(df[['prediction']].head(10))  # Print first 10 predictions for quick check
