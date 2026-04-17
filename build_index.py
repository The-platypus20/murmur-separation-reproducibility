import os
import pandas as pd

ROOT = "Data"
AUDIO_DIR = os.path.join(ROOT, "training_data")
META_CSV = os.path.join(ROOT, "training_data.csv")

meta = pd.read_csv(META_CSV)

# keep usable labels
meta = meta[meta["Murmur"].isin(["Absent", "Present"])].copy()
meta["label"] = meta["Murmur"].map({"Absent": 0, "Present": 1})

rows = []
for f in os.listdir(AUDIO_DIR):
    if f.endswith(".wav"):
        stem = f[:-4]                 # e.g., 2530_AV
        patient, valve = stem.split("_", 1)
        patient = int(patient)

        rec = meta.loc[meta["Patient ID"] == patient]
        if len(rec) == 0:
            continue

        rows.append({
            "patient_id": patient,
            "valve": valve,
            "path": os.path.join(AUDIO_DIR, f),
            "label": int(rec["label"].values[0])
        })

df = pd.DataFrame(rows).sort_values(["patient_id","valve"])
out = os.path.join(ROOT, "dataset_index.csv")
df.to_csv(out, index=False)

print("Index saved:", out)
print("Total recordings:", len(df))
print("Unique patients:", df.patient_id.nunique())
print(df.head())