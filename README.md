# Phishing Email Detection

This repository contains a Scikit-learn phishing email classifier that uses:
- Email textual content (TF-IDF)
- URL count feature
- Phishing keyword count feature

## Setup

```bash
pip install -r requirements.txt
```

## Run training + evaluation

```bash
python phishing_detector.py --data data/emails.csv
```

The script prints:
- Accuracy
- Confusion matrix

## Classify a new email

```bash
python phishing_detector.py --data data/emails.csv --predict "Urgent: verify your account at http://fake-login.com"
```

Output label is either `Phishing` or `Safe`.
