import librosa
import numpy as np

from ssa import (
    ssa_reconstruct_components,
    cssa_zcr_select_auto,
    cssa_kurtosis_select_topk,
    signal_correlation
)

# load audio
path = "Data/training_data/2530_AV.wav"
signal, sr = librosa.load(path, sr=None)

# use first 5 seconds
signal = signal[:5 * sr]

# SSA
components, singular_values = ssa_reconstruct_components(
    signal,
    L=200,
    n_components=10
)

# ----------------
# ZCR branch
# ----------------
_, zcr_values, threshold, heart_zcr = cssa_zcr_select_auto(
    components,
    percentile=40
)

murmur_zcr = signal - heart_zcr

# ----------------
# Kurtosis branch
# ----------------
_, kurt_values, heart_kurt = cssa_kurtosis_select_topk(
    components,
    top_k=4
)

murmur_kurt = signal - heart_kurt

# ----------------
# Correlation comparison
# ----------------
corr_zcr = signal_correlation(heart_zcr, murmur_zcr)
corr_kurt = signal_correlation(heart_kurt, murmur_kurt)

print("\nZCR correlation:", corr_zcr)
print("Kurtosis correlation:", corr_kurt)

if abs(corr_zcr) < abs(corr_kurt):
    print("Best method: ZCR")
    best_heart = heart_zcr
    best_murmur = murmur_zcr
else:
    print("Best method: Kurtosis")
    best_heart = heart_kurt
    best_murmur = murmur_kurt