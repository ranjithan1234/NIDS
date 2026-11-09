import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# ---------------------------
# Step 1: Load dataset
# ---------------------------
print("📂 Loading dataset...")
df = pd.read_csv("KDDTrain+.txt", header=None)

# NSL-KDD dataset column names
columns = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes",
    "land","wrong_fragment","urgent","hot","num_failed_logins","logged_in",
    "num_compromised","root_shell","su_attempted","num_root","num_file_creations",
    "num_shells","num_access_files","num_outbound_cmds","is_host_login",
    "is_guest_login","count","srv_count","serror_rate","srv_serror_rate",
    "rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate",
    "srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate",
    "dst_host_serror_rate","dst_host_srv_serror_rate","dst_host_rerror_rate",
    "dst_host_srv_rerror_rate","label"
]

df.columns = columns

# ---------------------------
# Step 2: Simplify labels
# ---------------------------
print("🔄 Processing labels...")
df["label"] = df["label"].apply(lambda x: "normal" if x == "normal" else "anomaly")

# ---------------------------
# Step 3: Encode categorical features
# ---------------------------
print("🔠 Encoding categorical features...")
categorical_cols = ["protocol_type", "service", "flag"]
encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# ---------------------------
# Step 4: Split data
# ---------------------------
X = df.drop("label", axis=1)
y = df["label"]

feature_columns = X.columns.tolist()

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------
# Step 5: Train model
# ---------------------------
print("🤖 Training RandomForest model...")
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)

# ---------------------------
# Step 6: Evaluate model
# ---------------------------
y_pred = clf.predict(X_val)
print("✅ Validation Accuracy:", accuracy_score(y_val, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_val, y_pred))

# ---------------------------
# Step 7: Save artifacts
# ---------------------------
print("💾 Saving model artifacts...")
joblib.dump(clf, "model.pkl")
joblib.dump(encoders, "encoders.pkl")
joblib.dump(feature_columns, "feature_columns.pkl")

print("\n🎉 Training complete! Artifacts saved:")
print(" - model.pkl")
print(" - encoders.pkl")
print(" - feature_columns.pkl")
