import pandas as pd
import numpy as np
from preprocessing import load_preprocess
from features import extract_mfcc_features

df = pd.read_csv("Data/dataset_index_splits.csv")

X = []
y = []
splits = []

for _, row in df.iterrows():

    signal, sr = load_preprocess(row["path"])
    feat = extract_mfcc_features(signal, sr)

    X.append(feat)
    y.append(row["label"])
    splits.append(row["split"])

X = np.array(X)
y = np.array(y)
splits = np.array(splits)

np.save("Data/X_mfcc.npy", X)
np.save("Data/y.npy", y)
np.save("Data/splits.npy", splits)

print("Feature matrix shape:", X.shape)