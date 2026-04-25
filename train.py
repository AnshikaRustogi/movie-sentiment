import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from preprocess import clean_text

# Load data
df = pd.read_csv('../data/IMDB Dataset.csv')

# Convert label to 0/1
df['label'] = (df['sentiment'] == 'positive').astype(int)

# Clean text
df['clean'] = df['review'].apply(clean_text)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['clean'], df['label'], test_size=0.2, random_state=42
)

# TF-IDF
vectorizer = TfidfVectorizer(max_features=10000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Predict
preds = model.predict(X_test_vec)

# Accuracy
acc = accuracy_score(y_test, preds)
print("Accuracy:", acc)

# Save model
joblib.dump(model, 'models/model.pkl')
joblib.dump(vectorizer, 'models/vectorizer.pkl')

print("Model saved successfully")