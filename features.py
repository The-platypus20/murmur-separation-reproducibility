import librosa
import numpy as np

def extract_mfcc_features(y, sr, n_mfcc=20):

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=n_mfcc
    )

    features = np.concatenate([
        mfcc.mean(axis=1),
        mfcc.std(axis=1)
    ])

    return features