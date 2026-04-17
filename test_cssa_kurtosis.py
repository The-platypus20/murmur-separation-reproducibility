import librosa
import numpy as np
import matplotlib.pyplot as plt
from ssa import ssa_reconstruct_components, cssa_kurtosis_select_topk

path = "Data/training_data/2530_AV.wav"

signal, sr = librosa.load(path, sr=None)
signal = signal[:5 * sr]

components, singular_values = ssa_reconstruct_components(
    signal,
    L=200,
    n_components=10
)

selected_indices, kurt_values, heart_estimate = cssa_kurtosis_select_topk(
    components,
    top_k=4
)

murmur_estimate = signal - heart_estimate

print("Kurtosis values:")
for i, k in enumerate(kurt_values):
    print(f"Component {i}: Kurtosis = {k:.6f}")

print("\nSelected component indices:", selected_indices)
print("Heart estimate mean abs amplitude:", np.mean(np.abs(heart_estimate)))
print("Murmur estimate mean abs amplitude:", np.mean(np.abs(murmur_estimate)))

t = np.arange(len(signal)) / sr

plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(t, signal)
plt.title("Original signal")

plt.subplot(3, 1, 2)
plt.plot(t, heart_estimate)
plt.title("Heart estimate (CSSA-Kurtosis)")

plt.subplot(3, 1, 3)
plt.plot(t, murmur_estimate)
plt.title("Residual / murmur estimate")

plt.tight_layout()
plt.show()