import librosa
import numpy as np

def load_preprocess(path):
    y, sr = librosa.load(path, sr=None)

    # normalize amplitude
    y = y / (np.max(np.abs(y)) + 1e-8)

    return y, sr