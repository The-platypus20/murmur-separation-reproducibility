import librosa
import numpy as np
import matplotlib.pyplot as plt
from ssa import ssa_reconstruct_components, cssa_zcr_select_auto

path = "Data/training_data/2530_AV.wav"

signal, sr = librosa.load(path, sr=None)
signal = signal[:5 * sr]

components, singular_values = ssa_reconstruct_components(
    signal,
    L=200,
    n_components=10
)

selected_indices, zcr_values, threshold, heart_estimate = cssa_zcr_select_auto(
    components,
    percentile=40
)

murmur_estimate = signal - heart_estimate

print("ZCR values:")
for i, z in enumerate(zcr_values):
    print(f"Component {i}: ZCR = {z:.6f}")

print("\nThreshold:", threshold)
print("Selected component indices:", selected_indices)
print("Heart estimate mean abs amplitude:", np.mean(np.abs(heart_estimate)))
print("Murmur estimate mean abs amplitude:", np.mean(np.abs(murmur_estimate)))

t = np.arange(len(signal)) / sr

plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(t, signal)
plt.title("Original signal")

plt.subplot(3, 1, 2)
plt.plot(t, heart_estimate)
plt.title("Heart estimate (CSSA-ZCR)")

plt.subplot(3, 1, 3)
plt.plot(t, murmur_estimate)
plt.title("Residual / murmur estimate")

plt.tight_layout()
plt.show()