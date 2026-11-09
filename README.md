# IDS Project (Beginner-friendly)

This project contains a simple Network Intrusion Detection System (IDS) pipeline:
- Download / load dataset (KDDCup99 or NSL-KDD CSV)
- Preprocess (one-hot encode categorical, scale numeric)
- Train models (Decision Tree, KNN, optionally XGBoost & LightGBM)
- Save models and preprocessor
- Predict on new CSV
- Optional Streamlit UI

See files:
- utils.py
- train_ids.py
- predict.py
- app.py

How to run:
1. Create venv and activate
2. pip install -r requirements.txt
3. python train_ids.py
4. python predict.py --input new_data.csv --output outputs/predictions/preds.csv
5. streamlit run app.py (optional)
