# Spam Email Classification (UCI Spambase)

Machine Learning, Assignment 2 | M.Tech (AIML), BITS Pilani WILP

> **Fill in every `<<...>>` placeholder before you submit, then delete this line.**

## a. Problem statement

Unsolicited email wastes time and can carry scams or malware. The goal of this project is to classify an email as **spam (1)** or **not spam (0)** from numeric features describing its text, such as how often certain words and characters appear and how long runs of capital letters are. Five classification algorithms are implemented on the same dataset, compared on six evaluation metrics, and demonstrated in an interactive Streamlit app.

In spam filtering the two kinds of error are not equally costly: a false positive (a real email sent to the spam folder) is usually worse than a false negative (a spam email reaching the inbox), so Precision is read alongside Recall and F1 in the observations below.

## b. Dataset description

| Item | Detail |
|---|---|
| Name | Spambase |
| Source | UCI Machine Learning Repository (https://archive.ics.uci.edu/dataset/94/spambase) |
| Task | Binary classification (1 = spam, 0 = not spam) |
| Instances | 4,601 raw rows; 4,210 after removing 391 duplicate rows |
| Features | 57 numeric features (minimum required: 12) |
| Target | `is_spam` (about 39% spam in the raw data) |
| Missing values | None |

The 57 features are:

- **48 word-frequency features** (`word_freq_*`): percentage of words in the email matching a given word, e.g. `free`, `money`, `remove`, `george`.
- **6 character-frequency features** (`char_freq_*`): percentage of characters that are `;`, `(`, `[`, `!`, `$`, `#`.
- **3 capital-letter features**: average length, longest length, and total length of uninterrupted runs of capital letters.

Preprocessing: duplicate rows removed; stratified 80/20 train/test split (`random_state=2026`); `StandardScaler` fitted on the training data inside each model's pipeline (needed by Logistic Regression, KNN and SVM because the capital-run features are on a much larger scale than the rest).

## c. GitHub repository

<<paste your GitHub repository link>>

```
spam-classifier/
|-- app.py                       Streamlit app
|-- spam_classification.ipynb    training and evaluation notebook
|-- spambase.data                raw dataset (UCI)
|-- requirements.txt
|-- README.md
|-- test_data.csv                held-out test split (features + is_spam)
|-- results.csv                  metrics table produced by the notebook
`-- model/                       one saved pipeline (scaler + model) per classifier
```

**Live app:** <<paste your Streamlit Community Cloud link>>

### Run it locally

```bash
pip install -r requirements.txt
jupyter notebook spam_classification.ipynb   # run all cells to regenerate model/ and test_data.csv
streamlit run app.py
```

## d. Models used

All five models are trained on the same training split and evaluated on the same held-out test split (842 rows; 3,368 training rows). Each saved model is a scikit-learn `Pipeline` of `StandardScaler` plus the classifier. Precision, Recall and F1 are for the spam class; AUC is computed from predicted probabilities.

### Comparison table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9097 | 0.9627 | 0.9088 | 0.8601 | 0.8838 | 0.8109 |
| Decision Tree | 0.9014 | 0.8970 | 0.8776 | 0.8750 | 0.8763 | 0.7944 |
| KNN | 0.8931 | 0.9457 | 0.8844 | 0.8423 | 0.8628 | 0.7760 |
| Naive Bayes | 0.8088 | 0.9376 | 0.6866 | 0.9583 | 0.8000 | 0.6584 |
| Support Vector Machine | 0.9287 | 0.9712 | 0.9367 | 0.8810 | 0.9080 | 0.8510 |

### Observations on model performance

<<Write 2 to 3 sentences per model based on YOUR numbers: compare Precision against Recall, say where the model is strong or weak and why that makes sense for the algorithm.>>

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | <<...>> |
| Decision Tree | <<...>> |
| KNN | <<...>> |
| Naive Bayes | <<...>> |
| Support Vector Machine | <<...>> |

**Overall winner for this dataset:** <<model name, with the metrics that justify it>>

## Streamlit app features

- Upload a test CSV (it must contain the 57 feature columns and the `is_spam` target); the bundled `test_data.csv` is used if nothing is uploaded.
- Dropdown to choose one of the five models.
- Accuracy, AUC, Precision, Recall, F1 and MCC for the selected model.
- Confusion matrix and classification report.
- A comparison tab with all five models side by side (table, bar chart, ROC curves) and a tab listing the rows the selected model got wrong.

Models are loaded from `model/` and are never retrained on the uploaded data.
