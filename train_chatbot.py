import json
import pickle
import random
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB

# 📚 NLTK setup
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
stop_words = set(nltk.corpus.stopwords.words('english'))

# 🧹 Preprocess text
def preprocess(text):
    tokens = nltk.word_tokenize(text.lower())
    lemmatized = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha()]
    filtered = [word for word in lemmatized if word not in stop_words]
    return ' '.join(filtered)

# 📂 Load intents
with open('intents.json') as file:
    data = json.load(file)

# 🏷️ Prepare training data
texts = []
labels = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        texts.append(preprocess(pattern))
        labels.append(intent['tag'])

# 🔠 Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# 🧠 Vectorize text
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# 🧪 Train model
model = MultinomialNB()
model.fit(X, encoded_labels)

# 💾 Save model
with open('chatbot_model.pkl', 'wb') as f:
    pickle.dump((model, vectorizer, label_encoder), f)

print("✅ Model trained and saved as chatbot_model.pkl")