import pandas as pd
import librosa
import numpy as np

df = pd.read_csv("Data/dataset_index_splits.csv")

# sample some files
sample_df = df.sample(20, random_state=42)

srs = []
durations = []

for path in sample_df["path"]:
    y, sr = librosa.load(path, sr=None)
    srs.append(sr)
    durations.append(len(y) / sr)

print("Unique sampling rates:", sorted(set(srs)))
print("Durations (seconds):", np.round(durations, 3))
print("Min duration:", np.min(durations))
print("Max duration:", np.max(durations))
print("Mean duration:", np.mean(durations))