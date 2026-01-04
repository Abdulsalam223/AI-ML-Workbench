# Email Spam/Ham Classifier

A machine learning-based email classification system that automatically identifies spam and legitimate (ham) emails using Natural Language Processing techniques.

## 📊 Overview

This project uses machine learning algorithms to classify emails as either spam or ham (legitimate emails) with high accuracy. The model is trained on a comprehensive dataset of labeled emails and can predict new incoming messages in real-time.

## 🎯 Features

- **Binary Classification**: Accurately classifies emails as Spam or Ham
- **Text Preprocessing**: Advanced NLP techniques for cleaning and preparing email text
- **Multiple ML Models**: Comparison of different algorithms for best performance
- **Real-time Prediction**: Fast classification of new emails
- **Performance Metrics**: Detailed accuracy, precision, recall, and F1-score analysis

## 📁 Dataset

**Dataset File**: `mail_data.csv`  
**Total Emails**: 5,572  
**Features**: 2 columns (Category, Message)  
**Labels**: Spam / Ham

### Dataset Structure
```
Category    Message
ham         Go until jurong point, crazy.. Available only...
ham         Ok lar... Joking wif u oni...
spam        Free entry in 2 a wkly comp to win FA Cup fina...
ham         U dun say so early hor... U c already then say...
ham         Nah I don't think he goes to usf, he lives aro...
```

### Dataset Split
- **Training Set**: 4,457 emails (80%)
- **Test Set**: 1,115 emails (20%)

## 🔍 Key Features Used

- **Text Features**: Email body content, subject lines
- **Word Frequency**: TF-IDF (Term Frequency-Inverse Document Frequency)
- **N-grams**: Unigrams, bigrams for context understanding
- **Special Characters**: Detection of spam patterns (excessive punctuation, caps)

## 🤖 Model Performance

### Multinomial Naive Bayes Model

**Training Accuracy**: 98.07%  
**Test Accuracy**: 97.31%

### Confusion Matrix Analysis

```
              Predicted
              Spam  Ham
Actual Spam   125   30
       Ham      0   960
```

**Performance Breakdown**:
- **Total Test Samples**: 1,115 emails
- **Spam Emails**: 155 (125 correctly identified, 30 missed)
- **Ham Emails**: 960 (all correctly identified)

### Key Metrics

| Metric      | Value   | Meaning |
|-------------|---------|---------|
| **Accuracy**    | 97.31%  | Overall correct predictions |
| **Spam Recall** | 80.65%  | Catches 80.65% of actual spam |
| **Ham Recall**  | 100.00% | Never misses legitimate emails (no false positives!) |
| **Spam Precision** | 100.00% | When marked as spam, it's always spam |

### Why This Model is Excellent

✅ **Perfect Ham Detection**: 0 legitimate emails marked as spam (no false positives)  
✅ **High Accuracy**: 97.31% correct classifications  
✅ **Fast Training**: Quick to train and predict  
✅ **Good Spam Catch Rate**: Identifies 80.65% of spam emails  

**Trade-off**: Some spam (30 emails) gets through to avoid blocking legitimate emails - a safe approach for email filtering!

## ⚙️ Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/email-spam-classifier.git
cd email-spam-classifier
```

2. **Install required packages**
```bash
pip install -r requirements.txt
```

**Required packages**:
```
pandas
numpy
scikit-learn
nltk
matplotlib
seaborn
```

3. **Download NLTK data** (if using)
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

## 🚀 Quick Start Guide

### Using the Pre-trained Model

**You need both files to make predictions:**
- `spam_classifier_model.pkl` (232.99 KB)
- `tfidf_vectorizer.pkl` (146.86 KB)

### Load and Predict in 3 Steps

**Step 1: Load the model and vectorizer**
```python
import joblib

model = joblib.load('spam_classifier_model.pkl')
feature_extraction = joblib.load('tfidf_vectorizer.pkl')
```

**Step 2: Prepare your email text**
```python
email = "Congratulations! You've won a free prize!"
email_features = feature_extraction.transform([email])
```

**Step 3: Get prediction**
```python
prediction = model.predict(email_features)[0]

if prediction == 0:
    print("🚫 SPAM")
else:
    print("✅ HAM")
```

### Interactive Prediction

```python
# Test with your own emails
user_input = input("Enter a message: ")
user_input_features = feature_extraction.transform([user_input])
prediction = model.predict(user_input_features)[0]

result = "SPAM" if prediction == 0 else "HAM"
print(f"Classification: {result}")
```

### Batch Prediction

```python
emails = [
    "Meeting tomorrow at 10am",
    "WINNER! Claim your prize now!!!",
    "Can you review this document?",
    "Free money! Click here immediately!"
]

for email in emails:
    features = feature_extraction.transform([email])
    pred = model.predict(features)[0]
    label = "SPAM" if pred == 0 else "HAM"
    print(f"{label}: {email}")
```

## 🚀 How to Run

### 1. Train the Model
```bash
python train_model.py
```
This will train all models and save the best performer.

### 2. Make Predictions

**Load the trained model:**
```python
import joblib

# Load the trained model and vectorizer
model = joblib.load('spam_classifier_model.pkl')
feature_extraction = joblib.load('tfidf_vectorizer.pkl')
```

**Single email prediction:**
```python
# Test with user input
user_input = input("Enter a message: ")
user_input_features = feature_extraction.transform([user_input])
prediction = model.predict(user_input_features)

if prediction[0] == 0:
    print("🚫 SPAM")
else:
    print("✅ HAM (Legitimate)")
```

**Example:**
```python
# Spam example
email = "Congratulations! You've won $1000. Click here to claim now!"
email_features = feature_extraction.transform([email])
prediction = model.predict(email_features)[0]
# Output: SPAM

# Ham example
email = "Hi, let's meet for coffee tomorrow at 3pm."
email_features = feature_extraction.transform([email])
prediction = model.predict(email_features)[0]
# Output: HAM
```

### 3. Batch Predictions
```python
import pandas as pd

# Load multiple emails
emails_list = [
    "Win a free iPhone now!",
    "Meeting scheduled for tomorrow at 10am",
    "Claim your prize money today!!!",
    "Can you review this document?"
]

# Transform and predict
emails_features = feature_extraction.transform(emails_list)
predictions = model.predict(emails_features)

# Display results
for email, pred in zip(emails_list, predictions):
    label = "SPAM" if pred == 0 else "HAM"
    print(f"{label}: {email}")
```

## 📂 Project Structure

```
email-spam-classifier/
│
├── data/
│   └── mail_data.csv                 # Dataset (5,572 emails)
│
├── models/
│   ├── spam_classifier_model.pkl     # Trained Naive Bayes model
│   └── tfidf_vectorizer.pkl          # TF-IDF feature extractor
│
├── src/
│   ├── train_model.py                # Model training script
│   ├── predict.py                    # Prediction script
│   └── preprocess.py                 # Text preprocessing
│
├── notebooks/
│   └── spam_classifier.ipynb         # Jupyter notebook
│
├── output/
│   └── confusion_matrix.png          # Visualization
│
├── requirements.txt                  # Dependencies
└── README.md                         # This file
```

## 💻 Usage Example

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

# Load pre-trained model and vectorizer
model = joblib.load('models/spam_classifier_model.pkl')
feature_extraction = joblib.load('models/tfidf_vectorizer.pkl')

# Classify an email
email = "Congratulations! You've won a free prize. Click here now!"
email_features = feature_extraction.transform([email])
prediction = model.predict(email_features)[0]

if prediction == 0:
    print("🚫 SPAM - This email is likely spam")
else:
    print("✅ HAM - This is a legitimate email")

# Output: 🚫 SPAM - This email is likely spam
```

### Real-World Example
```python
# Test multiple emails
emails = [
    "Hey, are we still meeting for lunch?",
    "WINNER! You have been selected. Call now!!!",
    "Please review the attached document for tomorrow's meeting",
    "Claim your $1000 prize money today! Limited time offer!"
]

for email in emails:
    email_features = feature_extraction.transform([email])
    prediction = model.predict(email_features)[0]
    label = "SPAM" if prediction == 0 else "HAM"
    print(f"{label}: {email[:50]}...")
```

## 🔍 Text Preprocessing Steps

1. **Lowercasing**: Convert all text to lowercase
2. **Remove Special Characters**: Clean punctuation and symbols
3. **Tokenization**: Split text into individual words
4. **Remove Stop Words**: Filter common words (the, is, at, etc.)
5. **Stemming/Lemmatization**: Reduce words to root form
6. **Vectorization**: Convert text to numerical features (TF-IDF)

## 📈 Visualizations

The project includes:
- Confusion Matrix
- ROC Curve and AUC Score
- Word Clouds (Spam vs Ham)
- Feature Importance
- Model Performance Comparison

## 💡 Key Insights

1. **Common Spam Indicators**:
   - Excessive use of words like "free", "win", "click", "urgent"
   - Multiple exclamation marks and capital letters
   - Suspicious links and phone numbers

2. **Model Performance**:
   - High precision minimizes false spam flags
   - Good recall ensures most spam is caught
   - Balanced F1-score for real-world deployment

3. **Best Practices**:
   - Regularly retrain with new spam patterns
   - Monitor false positive rate (legitimate emails marked as spam)
   - Update feature extraction as spam tactics evolve

## 📦 Pre-trained Model

Two pre-trained files are included for immediate use:

**Files**:
- `spam_classifier_model.pkl` (232.99 KB) - Trained Multinomial Naive Bayes model
- `tfidf_vectorizer.pkl` (146.86 KB) - Text-to-feature converter

**Performance**:
- Training Accuracy: 98.07%
- Test Accuracy: 97.31%
- Zero false positives on legitimate emails

**Quick Start**:
```python
import joblib

# Load both files
model = joblib.load('spam_classifier_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Classify
email = "Your text here"
features = vectorizer.transform([email])
prediction = model.predict(features)[0]
print("SPAM" if prediction == 0 else "HAM")
```

**Note**: Both files are required for predictions.

## 🛠️ Tech Stack

- **Python**: Core programming language
- **Scikit-learn**: Machine learning algorithms
- **NLTK/SpaCy**: Natural Language Processing
- **Pandas**: Data manipulation
- **Matplotlib/Seaborn**: Visualization

## 🎯 Future Improvements

- [ ] Deep learning models (LSTM, BERT)
- [ ] Multi-class classification (spam categories)
- [ ] Email header analysis
- [ ] Real-time API deployment
- [ ] Mobile app integration

## 📝 License

This project is available for educational and commercial use.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

**Note**: This classifier is designed to assist in email filtering but should be used as part of a comprehensive email security strategy.

*Protect your inbox from unwanted emails with machine learning!* 🛡️
