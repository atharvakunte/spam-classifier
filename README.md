# Spam Email Classification (UCI Spambase)

Machine Learning, Assignment 2 | M.Tech (AIML), BITS Pilani WILP

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

https://github.com/atharvakunte/spam-classifier

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

**Live app:** https://spam-classifier-wilp.streamlit.app

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

Running the deployed Streamlit app on `test_data.csv` reproduces this table exactly, which confirms the saved models load and score correctly.

### Observations on model performance

The test set has 842 emails (336 spam, 506 not spam). FP = legitimate emails wrongly flagged as spam; FN = spam that slipped through.

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Second best overall (Accuracy 0.9097, AUC 0.9627, MCC 0.8109). Precision (0.9088) is higher than Recall (0.8601), with 29 FP and 47 FN. A linear boundary on scaled features works well here because many word and character frequencies shift steadily between spam and non-spam. |
| Decision Tree | Accuracy 0.9014, but the lowest AUC (0.8970). Precision (0.8776) and Recall (0.8750) are balanced, with 41 FP and 42 FN. A single unpruned tree gives coarse probabilities (each leaf is nearly pure), so its ROC curve is blocky, and it likely overfits the training data. |
| KNN | Lowest Recall of all models (0.8423, 53 FN) and the weakest F1 (0.8628) and MCC (0.7760) apart from Naive Bayes. With 57 sparse, mostly-zero features, distances between emails are less informative, so neighbours are less reliable. Its AUC (0.9457) is still decent. |
| Naive Bayes | Highest Recall (0.9583, only 14 FN) but the lowest Precision (0.6866) and Accuracy (0.8088), with 147 FP, so it flags many legitimate emails as spam. The independence assumption does not hold (word frequencies are correlated) and Gaussian curves fit skewed, zero-heavy features poorly. Its AUC (0.9376) is high, which suggests the ranking is good but the default 0.5 threshold is badly placed. |
| Support Vector Machine | Best on Accuracy (0.9287), AUC (0.9712), Precision (0.9367), F1 (0.9080) and MCC (0.8510), with the fewest false positives (20) and 40 FN. The RBF kernel captures non-linear interactions between features that Logistic Regression cannot, and the scaled inputs suit it well. |

**Overall winner for this dataset:** **Support Vector Machine.** It leads on five of the six metrics (Accuracy, AUC, Precision, F1, MCC) and has the fewest false positives, which matters most in spam filtering because a lost legitimate email costs more than a spam email reaching the inbox. Naive Bayes wins only on Recall, at a large cost in Precision.

## Streamlit app features

- Upload a test CSV (it must contain the 57 feature columns and the `is_spam` target). Nothing is scored until a file is uploaded; a sidebar button downloads `test_data.csv` to use as the sample file.
- Dropdown to choose one of the five models.
- Accuracy, AUC, Precision, Recall, F1 and MCC for the selected model.
- Confusion matrix and classification report.
- A comparison tab with all five models side by side (table, bar chart, ROC curves) and a tab listing the rows the selected model got wrong.

Models are loaded from `model/` and are never retrained on the uploaded data.