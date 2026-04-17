import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

X = np.load("Data/X_mfcc.npy")
y = np.load("Data/y.npy")
splits = np.load("Data/splits.npy")

X_train = X[splits == "train"]
y_train = y[splits == "train"]

X_val = X[splits == "val"]
y_val = y[splits == "val"]

X_test = X[splits == "test"]
y_test = y[splits == "test"]

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

val_pred = model.predict(X_val)
test_pred = model.predict(X_test)

print("Validation F1:", f1_score(y_val, val_pred))
print("Test F1:", f1_score(y_test, test_pred))

print("\nTest classification report:")
print(classification_report(y_test, test_pred))