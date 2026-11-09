import pandas as pd

# Load your original test data
test_df = pd.read_csv("test_input.csv")

# Load the predictions
predictions_df = pd.read_csv("predictions_output.csv")

# Add the predictions as a new column to your test data
test_df['prediction'] = predictions_df['prediction']

# Save the combined result to a new CSV
test_df.to_csv("test_with_predictions.csv", index=False)

# Optional: display the first 10 rows
print(test_df.head(10))
