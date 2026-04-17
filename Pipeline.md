# murmur-separation-reproducibility
## Pipeline: Murmur Separation & Classification

### Dataset
CirCor DigiScope Phonocardiogram Dataset (PhysioNet CinC 2022) — 218 WAV files from 63 patients across 4 auscultation locations (AV, PV, TV, MV), sampled at 4000 Hz. Labels extracted from training_data.csv: **Holosystolic** vs **Non-holosystolic** (Early-systolic and Mid-systolic merged due to limited samples).

---

### Stage 1: CSSA Separation

Decompose the heart sound signal using **Singular Spectrum Analysis (SSA)**:
- Build trajectory matrix with window length L=100
- Perform SVD and reconstruct components via anti-diagonal averaging
- Apply two CSSA constraints:
  - **ZCR constraint** — keep components with Zero-Crossing Rate ≤ threshold (low ZCR = smooth, periodic = normal heart sound)
  - **Kurtosis constraint** — select top-k components with highest kurtosis (high kurtosis = peaky distribution = normal heart sound)
- Compute **Pearson correlation** between separated normal sound and murmur residual
- Keep the method with the **lower correlation**

**Results:** Kurtosis selected in ~60% of files. Mean correlation ≈ 0.13. Best separations at AV/PV locations (as low as 0.001).

---

### Stage 2: Wavelet Transform Refinement

- Apply **Discrete Wavelet Transform (DWT)** to Stage 1 output to remove residual murmur
- Threshold and reconstruct signal at multiple frequency scales
- **Final murmur = original signal − refined normal heart sound**

---

### Feature Extraction

Three input conditions evaluated:

| Input | Description |
|-------|-------------|
| **Original WAV** | Full 10-second raw recording |
| **Separated Murmur** | Full 10-second CSSA+DWT output |
| **TSV Systole Only** | Only systole regions (label=2) from PhysioNet TSV segmentation files |

For each input: extract **40 MFCC coefficients** → take mean across all frames → 1 × 40 feature vector per file.

---

### Classification

- **SVM** (RBF kernel, class_weight=balanced)
- **Random Forest** (100 estimators, class_weight=balanced)
- Evaluation: **5-fold stratified cross-validation**, 159 samples from 47 patients

---

### Results

| | Original WAV | Separated Murmur | TSV Systole Only |
|--|:-----------:|:---------------:|:---------------:|
| **SVM Accuracy** | 55.97% | 52.82% | **62.48%** |
| **SVM F1** | 49.65% | 46.05% | **62.15%** |
| **SVM Sensitivity** | 38.48% | 35.20% | **58.09%** |
| **SVM Specificity** | 79.56% | 76.48% | **68.35%** |
| **RF Accuracy** | **64.17%** | 58.49% | 58.44% |
| **RF F1** | **71.66%** | 67.94% | 66.70% |
| **RF Sensitivity** | **79.18%** | 76.96% | 73.68% |
| **RF Specificity** | 44.18% | 33.74% | 39.34% |

---

### Key Findings

- **SVM + TSV Systole Only** achieved the best SVM accuracy (62.48%) — focusing on systole regions removes irrelevant audio and improves classification
- **Random Forest + Original WAV** achieved the best overall accuracy (64.17%) — RF benefits from full signal context
- Separated murmur did **not** consistently outperform original WAV — attributed to **limited dataset size** (only 63/1,568 patients downloaded)

---

### Limitation
Only 63 out of 1,568 patients were available locally. With the full dataset, more conclusive comparison between input conditions would be achievable. Future work should also explore CNN and Wav2Vec 2.0 classification on the full dataset.
