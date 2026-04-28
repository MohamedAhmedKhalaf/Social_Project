# BankFive Sentiment Analysis Project

This repository contains an end-to-end sentiment analysis workflow for BankFive customer reviews, from data collection to model comparison and API serving.

## What this project does

- Collects BankFive reviews from Trustpilot (Task 1).
- Cleans and preprocesses text with optional NLP steps (Task 2).
- Builds labeled data with Gemini API and runs baseline lexical + ML models (Task 3).
- Optimizes stronger classifiers using TF-IDF weighted GloVe features, saves best model, and exports error analysis (Task 4).
- Compares the optimized local model against a SOTA transformer (RoBERTa) and creates plots (Task 5).
- Serves the optimized model through a FastAPI app and a simple web UI.

## Project layout

- task1.ipynb: Data collection and initial dataset checks.
- notes_for_task2.ipynb: Design notes and rationale for preprocessing choices.
- preprocessing.py: CLI preprocessing pipeline.
- task3_api_full_data.py: Gemini-based annotation to build full_dataset.csv.
- task3.py: Baseline lexical + classical ML experiments.
- task4_optimization.py: Hyperparameter tuning and optimized pipeline export.
- task5.ipynb: Optimized model vs SOTA comparison.
- app/main.py: FastAPI backend for sentiment prediction.
- templates/index.html: Frontend page for manual sentiment testing.

## Prerequisites

- Python 3.10+ (recommended: 3.11).
- Internet access for:
  - Package installation.
  - NLTK resource downloads (auto-run by scripts).
  - Gemini API requests (Task 3 annotation step).
  - Hugging Face model download (Task 5).
- GloVe embeddings file: glove.6B.100d.txt.

## 1. Create and activate a virtual environment

### Windows PowerShell

```powershell
cd I:\Social\Social_Project
python -m venv venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\venv\Scripts\Activate.ps1
```

### macOS/Linux

```bash
cd /path/to/Social_Project
python3 -m venv venv
source venv/bin/activate
```

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

## 3. Download GloVe (required)

Download from Kaggle:
https://www.kaggle.com/datasets/danielwillgeorge/glove6b100dtxt

After download:

1. Extract the archive.
2. Copy glove.6B.100d.txt into the project root (same folder as task3.py and task4_optimization.py).

Expected final path:

- ./glove.6B.100d.txt

## 4. Run tasks in order

## Task 1: Collect BankFive reviews

File: task1.ipynb

What it does:

- Scrapes Trustpilot pages for BankFive reviews.
- Keeps source, rating, and combined title + review text.
- Removes duplicate text records.
- Exports bankfive_sentiment_data.csv.

How to run:

- Open task1.ipynb in Jupyter or VS Code Notebook.
- Run all cells in order.
- You can change total_pages in scrape_bankfive_reviews(total_pages=7).

Main output:

- bankfive_sentiment_data.csv

## Task 2: Preprocess and enrich text

Primary file: preprocessing.py
Supporting notes: notes_for_task2.ipynb

What it does (optional flags):

- Noise cleanup (URLs, Trustpilot artifacts).
- Subject tagging (Loans, Customer Service, Digital Banking, General).
- Lowercasing.
- Spelling correction via SymSpell.
- Lemmatization via TextBlob.

Example command (recommended full pass):

```bash
python preprocessing.py --input bankfive_sentiment_data.csv --output refined_bank_data.csv --remove_noise --tag_subjects --lower --fix_spelling --lemmatize
```

Main output:

- refined_bank_data.csv

Note:

- Your repo already includes additional cleaned datasets used later:
  - full_balanced_basic_clean.csv
  - full_lemmatized_clean.csv
  - refined_data_full.csv

## Task 3 (Part 1): Build labeled dataset with Gemini API

File: task3_api_full_data.py

Important before run:

- Open task3_api_full_data.py and set your API key:
  - API_KEY = "YOUR_API_KEY_HERE"
- Ensure input file refined_bank_data.csv exists.

What it does:

- Calls Gemini twice per text (two prompt styles).
- Stores ann_1 and ann_2 labels.
- Computes Fleiss Kappa agreement.
- Sets ground_truth from ann_1.
- Exports full_dataset.csv.

Run:

```bash
python task3_api_full_data.py
```

Main output:

- full_dataset.csv

Project note:

- full_dataset.csv is the output from this Task 3 API labeling stage.

## Task 3 (Part 2): Baseline lexical + ML experiments

File: task3.py

Required inputs:

- full_dataset.csv (from Task 3 API stage)
- full_balanced_basic_clean.csv
- full_lemmatized_clean.csv
- positive-words.txt
- negative-words.txt
- glove.6B.100d.txt

What it does:

- Lexical sentiment models:
  - SentiWordNet
  - Bing Liu dictionary (custom negation handling)
- Representations:
  - Bag of Words
  - GloVe 100d averages
- ML models:
  - Naive Bayes
  - Decision Tree
- Exports full predictions and detailed metrics.

Run:

```bash
python task3.py
```

Main outputs:

- full_dataset_with_predictions.csv
- detailed_model_metrics.csv

## Task 4: Optimization and deployment pipeline

File: task4_optimization.py

Required inputs:

- full_dataset.csv
- glove.6B.100d.txt

What it does:

- Builds TF-IDF weighted GloVe features.
- Hyperparameter tuning with GridSearchCV for:
  - RandomForest
  - SVM
  - LogisticRegression
- Picks best model by weighted F1 CV score.
- Evaluates on test set.
- Exports confusion matrix, error analysis, and serialized model pipeline.

Run:

```bash
python task4_optimization.py
```

Main outputs:

- sentiment_pipeline.pkl
- confusion_matrix.png
- error_analysis.csv

## FastAPI app (related to Task 4)

Backend file: app/main.py
Frontend file: templates/index.html

Why this is Task 4-related:

- The API loads sentiment_pipeline.pkl generated in Task 4.

Start backend:

```bash
uvicorn app.main:app --reload
```

Expected endpoint:

- POST http://127.0.0.1:8000/predict

Example request:

```json
{
  "text": "The teller was very helpful and kind."
}
```

Quick test with curl:

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"text\":\"Great service and fast support\"}"
```

UI usage:

- Open templates/index.html in your browser.
- Enter review text.
- It calls the local FastAPI endpoint.

## Task 5: Optimized model vs SOTA comparison

File: task5.ipynb

Required inputs:

- full_dataset.csv
- sentiment_pipeline.pkl

What it does:

- Loads optimized local model artifacts.
- Loads SOTA model from Hugging Face:
  - cardiffnlp/twitter-roberta-base-sentiment-latest
- Compares metrics (Accuracy, Precision, Recall, F1, ROC-AUC).
- Computes per-category breakdown.
- Generates visualizations and final dataset with both predictions.

How to run:

- Open task5.ipynb.
- Run all cells in order.

Main outputs:

- sota_vs_svm_metrics.csv
- category_breakdown.csv
- plot_overall_comparison.png
- plot_category_breakdown.png
- plot_confusion_matrices.png
- final_full_dataset_with_SOTA.csv

## Suggested full execution order

1. Task 1 notebook.
2. Task 2 preprocessing script.
3. Task 3 API labeling script (creates full_dataset.csv).
4. Task 3 baseline experiments script.
5. Task 4 optimization script (creates sentiment_pipeline.pkl).
6. FastAPI app startup and test.
7. Task 5 notebook for SOTA comparison.

## Common issues and fixes

- Missing glove.6B.100d.txt:
  - Download from the Kaggle link and place it in project root.
- API key error in Task 3:
  - Update API_KEY in task3_api_full_data.py.
- Slow Task 3 API labeling:
  - The script intentionally sleeps between requests for rate limits.
- Hugging Face download errors in Task 5:
  - Check internet access and retry.
- Module import errors:
  - Confirm virtual environment is activated and dependencies were installed.

## Team handoff checklist

Before sharing with classmates/friends, make sure these are true:

- They create and activate a venv first.
- They run pip install -r requirements.txt.
- They place glove.6B.100d.txt in project root.
- They set API_KEY in task3_api_full_data.py before Task 3 API run.
- They run tasks in the recommended sequence above.
