# Murmur Separation and Classification Reproduction

This repository reproduces the **two-stage murmur separation and classification pipeline** proposed in the paper *“Murmur Separation and Classification from Heart Sound Using Constrained Singular Spectrum Analysis and Wavelet Transform.”*
The goal of this project is **method reproduction**, not exact numerical replication of the reported results.

---

# Project Goal

The purpose of this project is to rebuild the full murmur separation and classification pipeline described in the paper and evaluate the core scientific claim:

> Separating murmurs from heart sounds before classification improves classification performance.

The experiment compares classification performance between **raw heart sound signals** and **separated murmur signals**.

---

# Pipeline Overview

The proposed system applies a **two-stage murmur separation process** before performing classification.

```
Heart Sound WAV
│
▼
Preprocessing
│
▼
Stage 1: CSSA separation
├─ CSSA using ZCR constraint
└─ CSSA using Kurtosis constraint
│
▼
Choose better separation
│
▼
Stage 2: Wavelet Transform refinement
│
▼
Separated Murmur
│
▼
Feature extraction
│
▼
Classifier
│
▼
Evaluation
```

A baseline pipeline without murmur separation is also evaluated.

```
Heart Sound WAV
│
▼
Feature extraction
│
▼
Classifier
│
▼
Evaluation
```

The results from both pipelines are compared.

---

# Stage 1: CSSA Separation

Constrained Singular Spectrum Analysis (CSSA) is used to decompose the heart sound signal into components and reconstruct the **normal heart sound**.

Two reconstruction strategies are tested:

### ZCR Constraint

Components with **low zero-crossing rate (ZCR)** are selected because normal heart sounds tend to contain lower-frequency components than murmurs.

### Kurtosis Constraint

Components are selected based on **kurtosis maximization**, optimized using a genetic algorithm.
This approach attempts to capture impulsive characteristics present in murmur signals.

---

# Separation Selection

Both CSSA methods produce reconstructed heart sounds.

The residual murmur is computed as:

```
murmur_residual = original_signal − reconstructed_heart_sound
```

The correlation between the reconstructed heart sound and the murmur residual is calculated.
The method producing the **lower correlation** is selected as the better separation.

---

# Stage 2: Wavelet Transform Refinement

After CSSA separation, residual murmur components may remain in the signal.

A **Discrete Wavelet Transform (DWT)** is applied to refine the heart sound reconstruction and remove remaining murmur artifacts.

The final murmur signal is obtained by subtracting the refined heart sound from the original signal.

---

# Feature Extraction

Different feature representations are used depending on the classifier.

### MFCC

Mel-Frequency Cepstral Coefficients (MFCC) are extracted for classical machine learning models.

### Mel Spectrogram

For convolutional neural networks, the audio is converted into a **Mel spectrogram**, resized to **224×224**, and expanded to **3 channels**.

### Raw Waveform

Transfer learning models operate directly on the **raw waveform**.

---

# Classification Models

Four classifiers are evaluated in this project.

### Support Vector Machine (SVM)

A classical machine learning model trained using MFCC features.

### Random Forest (RF)

An ensemble learning model also trained using MFCC features.

### Convolutional Neural Network (CNN)

A deep learning model trained on Mel spectrogram images.

### Wav2Vec 2.0

A transfer learning model applied directly to the raw audio waveform.

---

# Evaluation Strategy

Two experimental setups are compared:

**Baseline**

```
Heart Sound → Feature Extraction → Classifier
```

**Proposed Pipeline**

```
Heart Sound → Murmur Separation → Feature Extraction → Classifier
```

The objective is to determine whether **murmur separation improves classification accuracy**.

---

# Reproducibility Warning

The original paper does not fully specify several critical implementation details, including:

* dataset composition for additional classes collected from open sources
* train / validation / test split
* sampling rate and resampling policy
* SSA window length
* ZCR threshold
* kurtosis optimization and genetic algorithm parameters
* wavelet type and decomposition level
* MFCC extraction settings
* CNN architecture and training parameters
* Wav2Vec fine-tuning configuration

Because of these missing details, **exact replication of the reported accuracy is not expected**.

---

# Expected Outcome

The original paper reports slightly higher classification performance when using separated murmurs compared to raw heart sounds.

This project evaluates whether the same **performance trend** can be observed when reproducing the method.

The key comparison tested in this repository is:

```
Accuracy(separated murmur) > Accuracy(raw heart sound)
```

---

# Repository Structure

```
murmur-separation-reproduction

data/
    raw_audio/

preprocessing/

cssa/
    ssa_decomposition.py
    zcr_constraint.py
    kurtosis_ga.py

wavelet/
    wavelet_refinement.py

features/
    mfcc.py
    mel_spectrogram.py

models/
    svm.py
    random_forest.py
    cnn.py
    wav2vec.py

evaluation/

README.md
```

---

# Summary

This project reconstructs the murmur separation and classification pipeline and evaluates whether separating murmurs from heart sounds improves classification performance.

The focus is on **reproducing the methodology and validating the core scientific claim of the paper**, rather than matching the exact numerical results.
