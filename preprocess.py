import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# load stopwords
stop_words = set(stopwords.words('english'))

# keep important words like 'not'
stop_words = stop_words - {'not', 'no', 'never'}

lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', ' ', text)          # remove HTML
    text = re.sub(r'[^a-z\s]', ' ', text)       # remove symbols
    text = re.sub(r'\s+', ' ', text).strip()    # remove extra spaces

    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]

    return " ".join(words)