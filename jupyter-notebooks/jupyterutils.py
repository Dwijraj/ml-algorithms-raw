import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sympy as sp
import os
import sys
from sklearn.preprocessing import StandardScaler

# Add project root to sys.path for clean imports
sys.path.append(os.path.abspath(".."))
__all__ = ['sp', 'np' , 'sns' , 'plt' , 'pd']

def ensure_column_vector(arr):
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr

def train_test_split(X, y, training_percent = 0.85):
    df = pd.DataFrame(X)
    df['label'] = y
    df = df.dropna()  # or use df.fillna() with strategy
    df = df.drop_duplicates()
    scaler = StandardScaler()
    df.iloc[:, :-1] = scaler.fit_transform(df.iloc[:, :-1])  # all but label
    df = df[df['label'].notna()]
    X_cleaned = df.drop('label', axis=1).values
    y_cleaned = df['label'].values
    samples = X_cleaned.shape[0]

    training_numbers = int(training_percent*samples)

    training_features = ensure_column_vector(np.array(X_cleaned[:training_numbers]))
    training_labels = np.array(y_cleaned[:training_numbers])

    testing_attributes=ensure_column_vector(np.array(X_cleaned[training_numbers:]))
    testing_labels=np.array(y_cleaned[training_numbers:])

    return training_features, training_labels, testing_attributes, testing_labels
