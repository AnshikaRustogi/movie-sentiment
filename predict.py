import joblib
from preprocess import clean_text

# Load model and vectorizer
model = joblib.load('models/model.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')

def predict(text):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    result = model.predict(vec)[0]

    if result == 1:
        return "positive"
    else:
        return "negative"

# Test
review = input("Enter movie review: ")
print("Sentiment:", predict(review))