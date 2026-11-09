import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# -----------------------
# 1. Load Test Dataset
# -----------------------
df_test = pd.read_csv("KDDTest.csv", header=None)

X_test = df_test.iloc[:, :-1]
y_test = df_test.iloc[:, -1]

# -----------------------
# 2. Encode categorical features
# -----------------------
cat_cols = [1, 2, 3]  # protocol_type, service, flag
for col in cat_cols:
    le = LabelEncoder()
    X_test[col] = le.fit_transform(X_test[col].astype(str))

# -----------------------
# 3. Map attack labels -> 5 categories
# -----------------------
attack_category = {
    "normal": "Normal",

    # DoS Attacks
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS",
    "smurf": "DoS", "teardrop": "DoS", "mailbomb": "DoS", "apache2": "DoS",
    "processtable": "DoS", "udpstorm": "DoS",

    # Probe Attacks
    "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe",
    "satan": "Probe", "mscan": "Probe", "saint": "Probe",

    # R2L Attacks
    "guess_passwd": "R2L", "ftp_write": "R2L", "imap": "R2L",
    "phf": "R2L", "multihop": "R2L", "warezmaster": "R2L",
    "warezclient": "R2L", "spy": "R2L", "xlock": "R2L", "xsnoop": "R2L",
    "snmpguess": "R2L", "snmpgetattack": "R2L", "httptunnel": "R2L",
    "sendmail": "R2L", "named": "R2L",

    # U2R Attacks
    "buffer_overflow": "U2R", "loadmodule": "U2R", "rootkit": "U2R",
    "perl": "U2R", "sqlattack": "U2R", "xterm": "U2R", "ps": "U2R"
}

# Convert y_test to categories
y_test = y_test.apply(lambda x: attack_category.get(str(x).lower(), "Other"))

# -----------------------
# 4. Load trained model
# -----------------------
clf = joblib.load("trained_model.pkl")

# -----------------------
# 5. Predict on test data
# -----------------------
y_pred = clf.predict(X_test)

# Convert predictions to categories too
y_pred = pd.Series(y_pred).apply(lambda x: attack_category.get(str(x).lower(), "Other"))

# -----------------------
# 6. Evaluation Report
# -----------------------
print("✅ Accuracy on Test Set:", accuracy_score(y_test, y_pred))
print("\nClassification Report (5 categories):\n")
print(classification_report(
    y_test,
    y_pred,
    labels=["Normal", "DoS", "Probe", "R2L", "U2R"]
))
