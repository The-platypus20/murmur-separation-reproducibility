import numpy as np

def diagonal_averaging(X):
    L, K = X.shape
    N = L + K - 1

    reconstructed = np.zeros(N, dtype=np.float32)
    counts = np.zeros(N, dtype=np.float32)

    for i in range(L):
        for j in range(K):
            reconstructed[i + j] += X[i, j]
            counts[i + j] += 1

    return reconstructed / counts


def ssa_reconstruct_components(signal, L, n_components=None):
    """
    Perform SSA and directly reconstruct 1D components
    without storing all large elementary matrices.
    """
    signal = np.asarray(signal, dtype=np.float32)

    N = len(signal)
    K = N - L + 1

    # trajectory matrix
    X = np.column_stack([signal[i:i+L] for i in range(K)]).astype(np.float32)

    # SVD
    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    if n_components is None:
        n_components = len(S)
    else:
        n_components = min(n_components, len(S))

    reconstructed_signals = []

    for i in range(n_components):
        Xi = S[i] * np.outer(U[:, i], Vt[i]).astype(np.float32)
        ri = diagonal_averaging(Xi)
        reconstructed_signals.append(ri)
        del Xi

    return reconstructed_signals, S


"""
Add ZCR functions
"""
import numpy as np

def zero_crossing_rate(signal):
    signal = np.asarray(signal)
    return np.mean(signal[:-1] * signal[1:] < 0)


def cssa_zcr_select(components, threshold):
    """
    Select SSA reconstructed components whose ZCR is <= threshold.
    Returns:
        selected_indices
        zcr_values
        reconstructed_signal
    """
    zcr_values = [zero_crossing_rate(c) for c in components]

    selected_indices = [
        i for i, z in enumerate(zcr_values)
        if z <= threshold
    ]

    if len(selected_indices) == 0:
        reconstructed = np.zeros_like(components[0])
    else:
        reconstructed = np.sum([components[i] for i in selected_indices], axis=0)

    return selected_indices, zcr_values, reconstructed

def cssa_zcr_select_auto(components, percentile=40):
    import numpy as np

    zcr_values = np.array([zero_crossing_rate(c) for c in components])
    threshold = np.percentile(zcr_values, percentile)

    selected_indices = [
        i for i, z in enumerate(zcr_values)
        if z <= threshold
    ]

    if len(selected_indices) == 0:
        reconstructed = np.zeros_like(components[0])
    else:
        reconstructed = np.sum([components[i] for i in selected_indices], axis=0)

    return selected_indices, zcr_values, threshold, reconstructed

"""
Implement CSSA with kurtosis
"""
def kurtosis_value(signal):
    import numpy as np

    x = np.asarray(signal)
    mu = np.mean(x)
    sigma = np.std(x) + 1e-8

    return np.mean(((x - mu) / sigma) ** 4)


def cssa_kurtosis_select_topk(components, top_k=4):
    import numpy as np

    kurt_values = np.array([kurtosis_value(c) for c in components])

    selected_indices = np.argsort(kurt_values)[-top_k:]
    selected_indices = sorted(selected_indices.tolist())

    reconstructed = np.sum([components[i] for i in selected_indices], axis=0)

    return selected_indices, kurt_values, reconstructed

"""
Correlation parrison
"""
def signal_correlation(x, y):
    import numpy as np

    x = np.asarray(x)
    y = np.asarray(y)

    if np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return 0.0

    return np.corrcoef(x, y)[0, 1]