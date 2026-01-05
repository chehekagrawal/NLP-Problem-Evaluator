import re
import nltk
from nltk.corpus import stopwords

# Ensure stopwords are downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

STOP_WORDS = set(stopwords.words('english'))

def clean_text(text):
    """
    Cleans raw text by converting to lowercase, removing special chars,
    and removing stopwords.
    """
    text = text.lower()
    # Remove special chars but keep spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Remove stopwords
    text = " ".join([w for w in text.split() if w not in STOP_WORDS])
    return text