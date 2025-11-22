# FakeNewsDetection-TFIDF-SVC

A concise and high‑performance benchmark comparing classical machine learning models with transformer-based approaches for clickbait and fake‑news headline classification.

## Overview

This project offers an efficient solution for clickbait and fake-news headline detection, comparing classical ML methods (Word2Vec, BOW, TF-IDF) with fine-tuned BERT transformers. It includes a complete text-processing pipeline, feature extraction, reproducible experiments, and ready-to-use training and inference scripts.

## Features

* Full preprocessing pipeline (HTML removal, cleaning, lemmatization, stopwords)
* Multiple feature extractors: BoW, TF‑IDF, Word2Vec
* Comparison of 6 different classifiers including Random Forest, SVM, Naive Bayes, and BERT.
* Transformer Tokenization for Deep Learning models.
* Optimization using GridSearchCV and RandomizedSearchCV for ML models.

## Key Results

| Rank | Model / Representation                  | Validation Accuracy |
|------|-----------------------------------------|---------------------|
| 1    | **BERT / RoBERTa (fine-tuned)**         | **98.62%**          |
| 2    | **TF-IDF + Logistic Regression**        | **94.45%**          |
| 3    | Bag-of-Words + Logistic Regression      | 94.00%              |
| 4    | TF-IDF + LinearSVC                      | 94.31%              |
| 5    | TF-IDF + MultinomialNB                  | 93.87%              |
| 6    | Bag-of-Words + MultinomialNB            | 93.85%              |
| 7    | Word2Vec (averaged) + XGBoost           | 92.55%              |

**Best Overall**: BERT / RoBERTa → **98.62%**  
**Best Classical Model**: TF-IDF (1–3 n‑grams) + Logistic Regression → **94.45%**  

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

