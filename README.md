# FakeNewsDetection-TFIDF-SVC

<div align="center">
  
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-orange)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Hugging%20Face-Transformers-lightgrey)](https://huggingface.co/docs/transformers/index)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/igirishkumar/FakeNewsDetection-TFIDF-SVC/blob/main/main.ipynb)
[![GitHub stars](https://img.shields.io/github/stars/igirishkumar/FakeNewsDetection-TFIDF-SVC?style=social)](https://github.com/igirishkumar/FakeNewsDetection-TFIDF-SVC/stargazers)

</div>

A concise and high‑performance benchmark comparing classical machine learning models with transformer-based approaches for clickbait and fake‑news headline classification.

## Project Overview

This project offers an efficient solution for clickbait and fake-news headline detection, comparing classical ML methods (Word2Vec, BOW, TF-IDF) with fine-tuned BERT transformers. It includes a complete text-processing pipeline, feature extraction, reproducible experiments, and ready-to-use training and inference scripts.

## Features

* Full preprocessing pipeline (HTML removal, cleaning, lemmatization, stopwords)
* Multiple feature extractors: BoW, TF‑IDF, Word2Vec
* Comparison of 6 different classifiers including Random Forest, SVM, Naive Bayes, and BERT.
* Transformer Tokenization for Deep Learning models.
* Optimization using GridSearchCV and RandomizedSearchCV for ML models.

## Results & Model Comparison

| Rank | Model / Representation                  | Validation Accuracy |
|------|-----------------------------------------|---------------------|
| 1    | **BERT / RoBERTa (fine-tuned)**         | **98.62%**          |
| 2    | **TF-IDF + Logistic Regression**        | **94.45%**          |
| 3    | Bag-of-Words + Logistic Regression      | 94.00%              |
| 4    | TF-IDF + LinearSVC                      | 94.31%              |
| 5    | TF-IDF + MultinomialNB                  | 93.87%              |
| 6    | Bag-of-Words + MultinomialNB            | 93.85%              |
| 7    | Word2Vec (averaged) + XGBoost           | 92.55%              |

> **Best Overall**: BERT / RoBERTa → **98.62%**  
> **Best Classical Model**: TF-IDF (1–3 n‑grams) + Logistic Regression → **94.45%**

## Tech Stack & Tools

| Category              | Tools / Libraries Used                             |
|-----------------------|----------------------------------------------------|
| Framework / ML        | scikit-learn, XGBoost                               |
| Deep Learning         | PyTorch, Hugging Face Transformers (BERT/RoBERTa) |
| Text Preprocessing    | NLTK, regex, BeautifulSoup4                         |
| Feature Extraction    | TF-IDF, Bag-of-Words, Word2Vec                      |
| Tokenization          | Hugging Face Tokenizers                              |
| Model Optimization    | GridSearchCV, RandomizedSearchCV                    |
| Evaluation Metrics    | Accuracy, F1-Score, Confusion Matrix                |
| Visualization         | Matplotlib, Seaborn                                  |
| Environment           | Local CPU / GPU, Google Colab                        |
| Version Control       | Git & GitHub                                        |


## Installation

```bash
git clone https://github.com/igirishkumar/FakeNewsDetection-TFIDF-SVC.git
cd FakeNewsDetection-TFIDF-SVC
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```
## Authors

- **Girish Kumar**  
  GitHub: [@igirishkumar](https://github.com/igirishkumar)  
  Role: Model architecture, classical ML pipeline, transformer fine-tuning,  
        text preprocessing, feature engineering, results visualization, README & documentation

