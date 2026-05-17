# B.U.S.T. — Bogus Text Uncovering System

## Overview

**B.U.S.T. (Bogus Text Uncovering System)** is a machine learning–based fake news detection framework designed to classify textual news content as either **FAKE** or **REAL**.

The project combines:

- Multi-source dataset aggregation
- Natural Language Processing (NLP)
- Text normalization and lemmatization
- TF-IDF feature engineering
- Multiple supervised machine learning classifiers
- Automated evaluation and model comparison
- Exportable production-ready inference artifacts

The notebook implements a complete end-to-end pipeline:

1. Dataset ingestion
2. Dataset cleaning
3. Label normalization
4. NLP preprocessing
5. Feature extraction
6. Model training
7. Evaluation
8. Model comparison
9. Model export
10. Reusable inference pipeline

---

# Project Goals

The primary objective of B.U.S.T. is to build a robust and reusable fake news classification system capable of:

- Detecting misinformation from textual content
- Combining multiple benchmark datasets into one unified corpus
- Standardizing heterogeneous labels across datasets
- Producing deployable machine learning artifacts
- Supporting future production APIs or web applications

---

# System Architecture

```text
Raw Datasets
     │
     ▼
Dataset Cleaning + Label Normalization
     │
     ▼
Text Cleaning
     │
     ▼
spaCy + NLTK NLP Preprocessing
     │
     ▼
TF-IDF Vectorization
     │
     ▼
Model Training
(Logistic Regression / SVM / Naive Bayes)
     │
     ▼
Evaluation + Model Comparison
     │
     ▼
Best Model Selection
     │
     ▼
Artifact Export (.pkl)
     │
     ▼
Inference Pipeline
```

---

# Datasets Used

The project merges multiple fake news datasets into a single consolidated training corpus.

## 1. LIAR Dataset (https://www.kaggle.com/datasets/csmalarkodi/liar-fake-news-dataset)

The LIAR dataset contains short political statements labeled for factual accuracy.

## 2. ISOT Fake News Dataset (https://www.kaggle.com/datasets/rahulogoel/isot-fake-news-dataset)

The ISOT dataset contains a combination of Real and Fake news articles

## 3. FakeNewsNet Dataset (https://www.kaggle.com/datasets/mdepak/fakenewsnet)

The FakeNewsNet dataset includes metadata-rich fake and real news articles.

# Final Unified Dataset

All cleaned datasets are merged into:

```python
df_total_combined
```

The merged dataframe is:

- Randomly shuffled
- Reset indexed
- Standardized into a unified schema

### Expected Core Columns

- `statement`
- `label`

---

# NLP Pipeline

The NLP pipeline combines:

- Regex cleaning
- NLTK preprocessing
- spaCy tokenization
- Lemmatization
- Stopword filtering

---

# Text Cleaning Process

The `clean_text()` function performs several preprocessing operations.

## Operations Performed

### 1. Lowercasing

```python
text = text.lower()
```

### 2. URL Removal

Removes:

- http links
- https links
- www links

### 3. HTML Tag Removal

HTML markup is stripped using regex.

### 4. Non-Alphabetic Character Removal

Only alphabetic characters are retained.

### 5. Number Removal

Numeric values are removed.

### 6. Whitespace Normalization

Extra spaces are collapsed.

---

# NLP Preprocessing

The `preprocess_text()` function applies linguistic normalization.

## spaCy Processing

The system loads:

```python
en_core_web_sm
```

If unavailable, it falls back to:

```python
spacy.blank("en")
```

## Stopword Removal

English stopwords are removed using NLTK.

## Lemmatization

Two-stage lemmatization is used:

1. spaCy lemma extraction
2. NLTK WordNet lemmatization

## Token Filtering

Tokens are removed if:

- Empty
- Stopwords
- Shorter than 3 characters

---

# Feature Engineering

The project uses vector-based text representations.

## TF-IDF Vectorizer

Primary vectorization strategy:

```python
TfidfVectorizer
```

### Configuration

| Parameter | Value |
|---|---|
| max_features | 50000 |
| ngram_range | (1, 2) |
| stop_words | english |
| sublinear_tf | True |
| min_df | 2 |
| max_df | 0.95 |

## Alternative Count Vectorizer

The notebook also supports:

```python
CountVectorizer
```

Though TF-IDF is used in the actual training pipeline.

---

# Machine Learning Models

The system evaluates multiple supervised learning algorithms.

## 1. Logistic Regression

Configuration:

```python
LogisticRegression(
    max_iter=500,
    random_state=42,
    n_jobs=-1
)
```

### Characteristics

- Strong baseline for text classification
- Efficient on sparse TF-IDF vectors
- Interpretable coefficients
- Fast inference

---

## 2. Linear SVM

Implemented using:

```python
CalibratedClassifierCV(LinearSVC())
```

### Purpose of Calibration

`LinearSVC` does not natively support probability estimation.

`CalibratedClassifierCV` enables:

- Probability predictions
- ROC-AUC computation
- Confidence scoring

### Characteristics

- Excellent performance on sparse text data
- Strong margin-based classification
- Often competitive for NLP tasks

---

## 3. Multinomial Naive Bayes

Configuration:

```python
MultinomialNB()
```

### Characteristics

- Extremely fast
- Efficient for bag-of-words problems
- Strong baseline for document classification

---

# Imported but Unused Models

The notebook imports several advanced models that are currently not integrated into the active training loop.

## Imported Models

- RandomForestClassifier
- XGBClassifier
- LGBMClassifier
- CatBoostClassifier

## Potential Future Expansion

The architecture is already partially prepared for:

- Gradient boosting
- Ensemble learning
- Tree-based classification
- Hybrid model benchmarking

---

# Hyperparameter Tuning

The notebook includes a generalized tuning framework.

## Logistic Regression Tuning

Uses:

```python
GridSearchCV
```

### Search Parameters

| Parameter | Values |
|---|---|
| C | [0.1, 1, 5] |
| solver | ["liblinear"] |

## Random Forest Tuning

Uses:

```python
RandomizedSearchCV
```

### Search Parameters

| Parameter | Values |
|---|---|
| n_estimators | [100, 200] |
| max_depth | [10, 20, None] |

### Important Note

Although RandomForest tuning exists in the tuning function, RandomForest is not currently included in `get_models()`.

---

# Training Pipeline

The main orchestration function is:

```python
train_pipeline(df_total_combined)
```

## Pipeline Stages

### 1. Data Validation

Checks:

- Empty dataframe validation
- Text column existence
- Label column existence

### 2. Data Cleaning

- Drop null values
- Remove duplicates
- Remove short samples

### 3. Label Normalization

Maps all labels into binary:

| Label | Meaning |
|---|---|
| 0 | FAKE |
| 1 | REAL |

### 4. NLP Preprocessing

Applies:

- Cleaning
- Tokenization
- Stopword removal
- Lemmatization

### 5. Train/Test Split

Configuration:

```python
train_test_split(
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)
```

### 6. Vectorization

Transforms text into sparse TF-IDF vectors.

### 7. Model Training

Each model is trained independently.

### 8. Evaluation

Performance metrics are generated.

### 9. Best Model Selection

The best-performing model is selected based on:

```python
F1 score
```

### 10. Artifact Export

The best model and supporting components are serialized.

---

# Evaluation Metrics

The project computes several classification metrics.

## Metrics Used

| Metric | Purpose |
|---|---|
| Accuracy | Overall correctness |
| Precision | False positive control |
| Recall | False negative control |
| F1 Score | Balanced classification performance |
| ROC-AUC | Probability discrimination quality |

## Classification Report

Generated using:

```python
classification_report()
```

## Confusion Matrix

Generated using:

```python
confusion_matrix()
```

Visualized using seaborn heatmaps.

---

# Visualization System

The notebook automatically generates:

- Dataset distribution charts
- Model comparison charts
- Confusion matrices

## Output Directory

```python
plots/
```

## Generated Files

Examples:

- `model_comparison.png`
- `LogisticRegression_confusion_matrix.png`
- `LinearSVM_confusion_matrix.png`
- `NaiveBayes_confusion_matrix.png`

---

# Model Export System

The system exports reusable inference artifacts.

## Saved Artifacts

| File | Purpose |
|---|---|
| fake_news_model.pkl | Best trained classifier |
| tfidf_vectorizer.pkl | TF-IDF vectorizer |
| label_encoder.pkl | Label encoder |

## Serialization Library

```python
joblib
```

---

# Inference Pipeline

The notebook includes a reusable prediction function:

```python
predict_news(text)
```

## Inference Steps

1. Validate input length
2. Load serialized artifacts
3. Preprocess text
4. Vectorize input
5. Predict class
6. Compute probabilities
7. Return structured result

## Output Format

```python
{
    "prediction": "REAL",
    "confidence": 0.9821,
    "probabilities": {
        "FAKE": 0.0179,
        "REAL": 0.9821
    }
}
```

---

# Error Handling

The notebook includes several defensive programming mechanisms.

## Covered Failures

- Missing spaCy model
- NLP preprocessing failures
- Empty input text
- Empty dataframe
- Missing columns
- Model tuning failures
- Individual model training failures

## Logging

Uses Python's built-in logging module.

### Logging Format

```python
%(asctime)s | %(levelname)s | %(message)s
```

---

# Reproducibility

The project uses:

```python
RANDOM_STATE = 42
```

Applied to:

- NumPy
- Python random
- Train/test splitting
- Model initialization

---

# External Dependencies

## Core Libraries

- pandas
- numpy
- scikit-learn
- nltk
- spacy
- matplotlib
- seaborn
- joblib

## Advanced ML Libraries

- xgboost
- lightgbm
- catboost

---

# Required NLP Downloads

The notebook downloads:

```python
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("omw-1.4")
```

---

# Suggested Installation

## Create Virtual Environment

```bash
python -m venv venv
```

## Activate Environment

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

## Install Dependencies

```bash
pip install pandas numpy scikit-learn nltk spacy matplotlib seaborn joblib xgboost lightgbm catboost
```

## Download spaCy Model

```bash
python -m spacy download en_core_web_sm
```

---

# Running the Notebook

## Main Training Execution

```python
best_model, vectorizer, label_encoder, results = train_pipeline(df_total_combined)
```

## Verify Exported Files

```python
from pathlib import Path

print(Path("fake_news_model.pkl").exists())
print(Path("tfidf_vectorizer.pkl").exists())
print(Path("label_encoder.pkl").exists())
```

---

# Production Readiness

The project already contains several production-oriented features.

## Included

- Modular architecture
- Artifact persistence
- Reusable inference pipeline
- Probability outputs
- Logging system
- Defensive validation
- Automatic best-model selection

## Missing for Full Production Deployment

- REST API layer
- Dockerization
- CI/CD pipeline
- Batch inference service
- Model versioning
- Monitoring
- Drift detection
- Unit tests
- Security hardening

---

# Limitations

## 1. Classical ML Only

The project currently uses classical machine learning models instead of transformer architectures such as:

- BERT
- RoBERTa
- DistilBERT
- DeBERTa

## 2. English-Only Pipeline

The preprocessing pipeline is designed specifically for English text.

## 3. No Deep Semantic Reasoning

TF-IDF models rely heavily on lexical patterns rather than contextual understanding.

## 4. Potential Dataset Bias

Merged datasets may contain:

- Political bias
- Source imbalance
- Temporal bias
- Topic imbalance

---

# Future Improvements

## Model Improvements

- Add transformer models
- Ensemble stacking
- Cross-validation benchmarking
- Feature selection
- Semantic embeddings

## Engineering Improvements

- FastAPI deployment
- Streamlit frontend
- Docker support
- GPU acceleration
- MLflow integration
- Experiment tracking

## Data Improvements

- More balanced datasets
- Multilingual support
- Real-time news ingestion
- Social media integration

---

# Suggested API Design

Example request structure:

```json
{
  "text": "Breaking news article content here"
}
```

Example response:

```json
{
  "prediction": "FAKE",
  "confidence": 0.93
}
```

---

# Repository Structure

Suggested repository structure for scaling the project:

```text
BUST/
│
├── data/
├── notebooks/
├── plots/
├── models/
├── api/
├── src/
│   ├── preprocessing.py
│   ├── training.py
│   ├── inference.py
│   └── evaluation.py
│
├── fake_news_model.pkl
├── tfidf_vectorizer.pkl
├── label_encoder.pkl
├── requirements.txt
├── README.md
└── LICENSE
```

---

# Technical Summary

| Component | Technology |
|---|---|
| Language | Python |
| NLP | NLTK + spaCy |
| Vectorization | TF-IDF |
| ML Framework | scikit-learn |
| Visualization | matplotlib + seaborn |
| Serialization | joblib |
| Models | Logistic Regression, Linear SVM, Naive Bayes |
| Dataset Strategy | Multi-source merged corpus |
| Classification Type | Binary classification |
| Output Classes | FAKE / REAL |

---

# Conclusion

B.U.S.T. is a structured and extensible fake news detection framework that combines:

- Multi-dataset aggregation
- NLP preprocessing
- Sparse vector representations
- Classical machine learning
- Automated evaluation
- Production-oriented serialization

The system is suitable as:

- An NLP portfolio project
- A machine learning baseline system
- A misinformation detection prototype
- A foundation for future deep learning upgrades

While the current implementation focuses on classical NLP pipelines, the architecture is modular enough to evolve into a transformer-based misinformation detection platform.

