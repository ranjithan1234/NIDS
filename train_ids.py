# train_ids.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# -----------------------------
# Load Dataset
# -----------------------------
dataset_path = "data/NSL-KDD-Dataset-master/KDDTrain+.txt"
columns = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'class'
]

print("✅ Loading dataset...")
df = pd.read_csv(dataset_path, names=columns)
print(f"Columns: {list(df.columns)}")
print(f"Number of samples: {len(df)}")

# -----------------------------
# Convert numeric columns to float
# -----------------------------
numeric_features = [
    'duration', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment',
    'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
    'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds',
    'is_host_login', 'is_guest_login', 'count', 'srv_count',
    'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
    'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
    'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'
]

for col in numeric_features:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# -----------------------------
# Categorical columns
# -----------------------------
categorical_features = ['protocol_type', 'service', 'flag']

# -----------------------------
# Target
# -----------------------------
label_col = 'class'
X = df.drop(columns=[label_col])
y = df[label_col].apply(lambda x: 0 if x == 'normal' else 1)  # 0=Normal, 1=Anomaly

# -----------------------------
# Preprocessing
# -----------------------------
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# -----------------------------
# Pipeline
# -----------------------------
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# -----------------------------
# Split dataset
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# Train model
# -----------------------------
print("✅ Training model...")
pipeline.fit(X_train, y_train)

# -----------------------------
# Evaluate
# -----------------------------
test_acc = pipeline.score(X_test, y_test)
print(f"✅ Test Accuracy: {test_acc:.4f}")

# -----------------------------
# Save model
# -----------------------------
output_dir = "model_output"
os.makedirs(output_dir, exist_ok=True)
model_path = os.path.join(output_dir, "model.joblib")
joblib.dump(pipeline, model_path)
print(f"✅ Model saved at: {model_path}")
