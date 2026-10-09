# Phishing Email Detection with Machine Learning

My undergraduate dissertation project for the BSc Computer Systems (Cyber Security) at Nottingham Trent University (2023 to 2024). The goal was to classify emails as phishing or legitimate from their text, and to compare how well different machine learning models do the job.

Python handles cleaning and feature extraction. WEKA handles feature selection and model comparison.

## Result

Five classifiers were compared, including Random Forest, SVM and J48. The best model reached **91.16% accuracy**.

## Dataset

`Nazario_5.csv` holds 3,065 real emails, roughly balanced between phishing and legitimate, drawn from the Nazario phishing corpus and the SpamAssassin corpus. Each row has the sender, receiver, date, subject, body, extracted URLs and a label.

## How it works

`preprocess.py` turns raw emails into model-ready features:

1. Cleans the subject and body: strips HTML tags, URLs, punctuation, numbers and extra whitespace, and lowercases the text.
2. Joins subject and body into one text field and removes English stopwords (NLTK).
3. Counts the URLs in each email and keeps the count as an extra feature, since links are a common phishing signal.
4. Converts the text into TF-IDF features, keeping the 1,000 most informative terms (scikit-learn).
5. Encodes the labels and saves two outputs: `tfidf_vectorized_emails.csv` (the feature matrix) and `PhishingEmail_pre.csv` (the cleaned emails).

In WEKA, the feature matrix was reduced with correlation-based feature selection (CfsSubsetEval with BestFirst search) and saved as `PhishingEmail.arff`, which was then used to train and evaluate the classifiers.

## Files

| File | What it is |
|---|---|
| `preprocess.py` | Cleaning and TF-IDF feature extraction |
| `Nazario_5.csv` | Raw email dataset |
| `tfidf_vectorized_emails.csv` | TF-IDF features with URL count and label (generated) |
| `PhishingEmail_pre.csv` | Cleaned emails with labels (generated) |
| `PhishingEmail.arff` | Feature-selected dataset used for model training in WEKA |

## Running it

```bash
pip install -r requirements.txt
python preprocess.py
```

The script reads `Nazario_5.csv` from the same folder and writes the two generated CSV files next to it.

## Tools

Python, pandas, NLTK, scikit-learn, WEKA
