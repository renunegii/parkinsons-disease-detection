# 🧬 Parkinson's Disease Detection — ML Classification Study

> *Can a person's voice tell us if they have Parkinson's? Turns out, it can.*

---

## 📌 Overview

Parkinson's Disease is a neurological condition that affects movement — but before the tremors become visible, it quietly changes the way a person speaks. Tiny variations in vocal frequency, pitch stability, and noise ratios that no human ear can catch start showing up in the data.

This project takes those vocal measurements, runs them through several machine learning models, and tries to answer one question: **can we predict whether someone has Parkinson's Disease based on their voice alone?**

The answer, after a lot of cleaning, balancing, and comparing — is yes, with over **94% accuracy**.

---

## 📂 Dataset

| Detail | Info |
|---|---|
| **Source** | UCI Machine Learning Repository |
| **Dataset** | Parkinsons Disease Dataset |
| **Rows** | 195 recordings |
| **Features** | 22 vocal biomarker columns |
| **Target** | `status` — 1 (Parkinson's) / 0 (Healthy) |

The dataset contains voice recordings from real patients. Each row is one recording, and each column is a different way of measuring the voice — things like average pitch, jitter (small pitch jumps), shimmer (volume variation), and noise-to-harmonics ratio.

---

## 🔍 What This Project Does — Step by Step

### 1. Data Preprocessing
Before anything else, the data needed to be cleaned and made ready.

- Dropped the `name` column — it had no predictive value and would confuse the model
- Checked for missing values and duplicate rows — the dataset was clean on both counts
- Converted the `status` column from integer to boolean for memory efficiency

### 2. Exploratory Data Analysis (EDA)
This is where the data started telling its story.

- **Class imbalance check** — 147 patients had Parkinson's, only 48 were healthy. That's a heavily skewed dataset
- **Correlation heatmap** — many features were highly correlated with each other, which is a red flag for model confusion
- **Box plots** — revealed that patients with lower values of `HNR`, `MDVP:Fo(Hz)`, `MDVP:Fhi(Hz)`, and `MDVP:Flo(Hz)` were more likely to have Parkinson's

### 3. Handling Class Imbalance — SMOTE
A model trained on imbalanced data will just learn to always predict the majority class. To fix this, **SMOTE (Synthetic Minority Oversampling Technique)** was used to synthetically generate new samples for the healthy class, bringing the dataset into balance before training.

### 4. Feature Scaling
All features were scaled to a range of **-1 to 1** using `MinMaxScaler` so that no single feature could dominate the model just because of its numerical size.

### 5. PCA — Tested But Rejected
Principal Component Analysis was applied to see if reducing the number of features would help. It didn't — in fact, it hurt every model. The reason is simple: PCA works best on large, high-dimensional datasets prone to overfitting. With only 195 rows, removing features meant losing meaningful information. PCA was dropped and the full feature set was kept.

---

## 🤖 Models Trained & Compared

Seven classification models were trained and evaluated — both with and without PCA:

| Model | Notes |
|---|---|
| Decision Tree | Tuned with GridSearchCV |
| Random Forest | Tuned with GridSearchCV |
| Logistic Regression | Standard linear classifier |
| SVM (RBF Kernel) | Non-linear support vector machine |
| Naive Bayes | Probabilistic baseline |
| KNN | Distance-based classifier |
| **XGBoost** | ✅ Best performing model |

Each model was evaluated using:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score
- Confusion Matrix

> **Winner: XGBoost** — achieved the highest accuracy with the best balance across all metrics. For a medical use case, recall (catching actual Parkinson's cases) was treated as more important than raw accuracy — false negatives are far more costly than false positives.

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-189B3E?style=flat-square)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat-square&logo=python&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=flat-square&logo=python&logoColor=white)
![Google Colab](https://img.shields.io/badge/Google_Colab-F9AB00?style=flat-square&logo=googlecolab&logoColor=black)
