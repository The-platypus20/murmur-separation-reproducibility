import librosa
from ssa import ssa_reconstruct_components

path = "Data/training_data/2530_AV.wav"

signal, sr = librosa.load(path, sr=None)

# use only first 5 seconds
signal = signal[:5 * sr]

signals, singular_values = ssa_reconstruct_components(
    signal,
    L=200,
    n_components=10
)

print("Number of reconstructed SSA components:", len(signals))
print("Length of first reconstructed signal:", len(signals[0]))
print("First 10 singular values:", singular_values[:10])