import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("Data/dataset_index.csv")

# unique patients
patients = df["patient_id"].unique()

# first split: train+val vs test
trainval_patients, test_patients = train_test_split(
    patients,
    test_size=0.2,
    random_state=42
)

# second split: train vs val
train_patients, val_patients = train_test_split(
    trainval_patients,
    test_size=0.125,   # 0.125 x 0.8 = 0.10 total
    random_state=42
)

# assign split
df["split"] = "train"
df.loc[df["patient_id"].isin(val_patients), "split"] = "val"
df.loc[df["patient_id"].isin(test_patients), "split"] = "test"

# save
df.to_csv("Data/dataset_index_splits.csv", index=False)

print(df["split"].value_counts())
print()
print("Unique patients in each split:")
print("Train:", df[df["split"] == "train"]["patient_id"].nunique())
print("Val  :", df[df["split"] == "val"]["patient_id"].nunique())
print("Test :", df[df["split"] == "test"]["patient_id"].nunique())

print()
print("Label counts by split:")
print(pd.crosstab(df["split"], df["label"]))

print()
print("Label proportions by split:")
print(pd.crosstab(df["split"], df["label"], normalize="index"))