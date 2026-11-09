# predict.py
import argparse
import os
import pandas as pd
import joblib
from utils import make_binary_label_series

MODEL_DIR = "outputs/models"

def predict(model_path, preprocessor_path, label_encoder_path, input_csv, output_csv):
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    le = joblib.load(label_encoder_path)

    df = pd.read_csv(input_csv)
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip()

    X = df
    X_trans = preprocessor.transform(X)
    preds_encoded = model.predict(X_trans)
    preds_names = le.inverse_transform(preds_encoded)
    df['prediction'] = preds_names
    df.to_csv(output_csv, index=False)
    print("Saved predictions to", output_csv)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.path.join(MODEL_DIR, "best_model.joblib"))
    parser.add_argument("--preprocessor", default=os.path.join(MODEL_DIR, "preprocessor.joblib"))
    parser.add_argument("--label_encoder", default=os.path.join(MODEL_DIR, "label_encoder.joblib"))
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="outputs/predictions/predictions.csv")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    predict(args.model, args.preprocessor, args.label_encoder, args.input, args.output)
