# 🧠 Parkinson's Disease Detection using XGBoost

Detecting Parkinson's Disease from **voice measurements** using machine learning.  
Built with Python, XGBoost, and scikit-learn. 

---

## 📌 Overview

Parkinson's Disease affects the vocal muscles, causing measurable abnormalities in speech.
This project uses **22 voice biomarker features** (like jitter, shimmer, and HNR) from the
UCI Parkinson's Dataset to train a classifier that predicts whether a patient has Parkinson's.

---

## How It Works

Parkinson's degrades the laryngeal (voice box) muscles. Scientists can detect this from audio:

| Feature | What It Measures | In Parkinson's |
|---------|-----------------|----------------|
| **Jitter** | Cycle-to-cycle pitch variation | ↑ Higher |
| **Shimmer** | Cycle-to-cycle volume variation | ↑ Higher |
| **HNR** | Harmonics-to-Noise Ratio | ↓ Lower |
| **PPE** | Pitch Period Entropy | ↑ Higher |
| **RPDE** | Recurrence Period Density Entropy | ↑ Higher |

These 22 features are scaled and fed into an XGBoost model, which learns the pattern
that separates Parkinson's patients (147 samples) from healthy controls (48 samples).

---

## Results

```
Accuracy : ~94–97%

              precision    recall  f1-score
Healthy           0.90      0.90      0.90
Parkinson's       0.97      0.97      0.97
```

> **Sensitivity** (Parkinson's correctly detected): ~96%  
> **Specificity** (Healthy correctly cleared): ~90%

---

## 📚 References

- [Parkinson's Dataset](https://www.kaggle.com/datasets/vikasukani/parkinsons-disease-data-set) — 2020
- [DataFlair — Detecting Parkinson's Disease](https://data-flair.training/blogs/python-machine-learning-project-detecting-parkinson-disease/)

---

