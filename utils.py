# utils.py
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def load_kddcup99(as_frame=True):
    """
    Download & return the KDDCup99 dataset as a pandas DataFrame.
    """
    from sklearn.datasets import fetch_kddcup99
    bunch = fetch_kddcup99(as_frame=as_frame, subset=None, download_if_missing=True)
    if as_frame and hasattr(bunch, 'frame'):
        df = bunch.frame.copy()
        if 'target' in df.columns:
            df = df.rename(columns={'target': 'label'})
    else:
        data = pd.DataFrame(bunch.data)
        data['label'] = bunch.target
        df = data
    return df

def load_csv(path):
    """Load dataset from a CSV file."""
    df = pd.read_csv(path)
    return df

def make_binary_label_series(y_series):
    """
    Convert label strings (e.g., 'normal.', 'smurf.', b'normal.') to 'normal' or 'attack'.
    """
    s = y_series.astype(str).str.lower()
    return s.apply(lambda v: 'normal' if 'normal' in v else 'attack')

def build_preprocessor(df, target_col):
    """
    Build a ColumnTransformer that standard scales numeric columns
    and one-hot encodes categorical columns.
    """
    df2 = df.copy()
    for c in df2.columns:
        if not np.issubdtype(df2[c].dtype, np.number):
            df2[c] = df2[c].astype(str).str.strip()

    feature_cols = [c for c in df2.columns if c != target_col]
    num_cols = [c for c in feature_cols if np.issubdtype(df2[c].dtype, np.number)]
    cat_cols = [c for c in feature_cols if c not in num_cols]

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)  # updated for scikit-learn 1.2+

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_cols),
            ('cat', categorical_transformer, cat_cols),
        ],
        remainder='drop'
    )
    return preprocessor, num_cols, cat_cols

def preprocess_df(df, preprocessor=None, fit=False, target_col=None):
    """
    Preprocess the dataframe.
    Auto-detects target column if not specified ('label' or 'class').
    Returns X (features), y (binary labels), fitted preprocessor, label encoder.
    """
    df2 = df.copy()

    # Auto-detect target column
    if target_col is None:
        if 'label' in df2.columns:
            target_col = 'label'
        elif 'class' in df2.columns:
            target_col = 'class'
        else:
            raise KeyError("Could not find a target column ('label' or 'class') in the dataframe.")

    df2[target_col] = make_binary_label_series(df2[target_col])

    if preprocessor is None:
        preprocessor, _, _ = build_preprocessor(df2, target_col=target_col)
        fit = True

    X = df2.drop(columns=[target_col])
    if fit:
        X_trans = preprocessor.fit_transform(X)
    else:
        X_trans = preprocessor.transform(X)

    le = LabelEncoder()
    y = le.fit_transform(df2[target_col]) if fit else le.transform(df2[target_col])

    return X_trans, y, preprocessor, le
