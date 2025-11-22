# FakeNewsDetection-TFIDF-SVC

A concise and high‑performance benchmark comparing classical machine learning models with transformer-based approaches for clickbait and fake‑news headline classification.

## Overview

This project demonstrates that a well‑optimized **TF‑IDF + LinearSVC** pipeline can achieve **94.45%** accuracy—competitive with or better than many transformer baselines—while training quickly on CPU. The repository provides a clean, modular starting point for text‑classification tasks.

## Features

* Full preprocessing pipeline (HTML removal, cleaning, lemmatization, stopwords)
* Multiple feature extractors: BoW, TF‑IDF, Word2Vec
* Comparison of 6 different classifiers including Random Forest, SVM, Naive Bayes, and BERT.
* Transformer Tokenization for Deep Learning models.
* Optimization using GridSearchCV and RandomizedSearchCV for ML models.

## Key Results

| Representation | Best Model          | Accuracy   |
| -------------- | ------------------- | ---------- |
| Bag‑of‑Words   | Logistic Regression | 94%        |
| Bag‑of‑Words   | MultinomialNB       | 93.85%     |
| Word2Vec Avg.  | XGBoost             | 92.55%     |
| TF‑IDF         | Logistic Regression | **94.45%** |
| TF‑IDF         | LinearSVC           | 94.31%     |
| TF‑IDF         | MultinomialNB       | 93.87%     |

**Top Performer:** TF‑IDF (1–3 n‑grams) with Logistic Regression.

## Features

* Full preprocessing pipeline (HTML removal, cleaning, lemmatization, stopwords)
* Multiple feature extractors: BoW, TF‑IDF, Word2Vec
* Comparison of 7 different classifiers including Random Forest, SVM, Naive Bayes, and BERT.
* Transformer Tokenization for Deep Learning models.
* Optimization using GridSearchCV and RandomizedSearchCV for ML models.

## Installation

```bash
git clone https://github.com/igirishkumar/FakeNewsDetection-TFIDF-SVC.git
cd FakeNewsDetection-TFIDF-SVC
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Citation

```bibtex
@software{kumar2025clickbait,
  author = {Girish Kumar},
  title = {FakeNewsDetection-TFIDF-SVC: High-Accuracy Clickbait Detection with Classical ML},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/igirishkumar/FakeNewsDetection-TFIDF-SVC}
}
```

