import json
import pickle
import random
import requests
import nltk
import re
from datetime import datetime
from nltk.stem import WordNetLemmatizer

# 📚 NLTK setup
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

lemmatizer = WordNetLemmatizer()
stop_words = set(nltk.corpus.stopwords.words('english'))

# 📦 Load resources
with open('intents.json') as file:
    intents = json.load(file)

with open('chatbot_model.pkl', 'rb') as f:
    model, vectorizer, label_encoder = pickle.load(f)

# 🧹 Preprocess input
def preprocess(text):
    from nltk.tokenize import wordpunct_tokenize
    tokens = wordpunct_tokenize(text.lower())
    lemmatized = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha()]
    filtered = [word for word in lemmatized if word not in stop_words]
    return ' '.join(filtered)

# 🔮 Predict intent
def predict_class(sentence):
    bow = vectorizer.transform([preprocess(sentence)])
    probs = model.predict_proba(bow)[0]
    ERROR_THRESHOLD = 0.5
    results = [(i, p) for i, p in enumerate(probs) if p > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)

    if not results:
        return [{'intent': 'fallback', 'probability': '1.0'}]
    return [{'intent': label_encoder.classes_[results[0][0]], 'probability': str(results[0][1])}]

# 🧠 DuckDuckGo fallback
def duckduckgo_search(query):
    url = f"https://api.duckduckgo.com/?q={query}&format=json"
    try:
        response = requests.get(url).json()
        abstract = response.get("AbstractText")
        related = response.get("RelatedTopics", [])
        if abstract:
            return abstract
        elif related:
            return related[0].get("Text", "I found something, but it's not very detailed.")
        else:
            return None
    except:
        return None

# 📚 Wikipedia fallback
def wikipedia_summary(query):
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{query.replace(' ', '_')}"
    try:
        response = requests.get(url).json()
        return response.get("extract")
    except:
        return None

# ➕ Math detection and solving
def is_math_query(text):
    return bool(re.search(r'[\d\+\-\*/\^=]', text))

def solve_math(text):
    try:
        expression = re.sub(r'[^\d\+\-\*/\.\(\)]', '', text)
        result = eval(expression)
        return f"The answer is {result}"
    except Exception as e:
        return f"Sorry, I couldn't solve that. Error: {e}"

# 🌦️ Weather API
def get_weather(city):
    api_key = "your_openweather_api_key"  # Replace with your actual key
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    try:
        response = requests.get(url).json()
        temp = response["main"]["temp"]
        desc = response["weather"][0]["description"]
        return f"The weather in {city} is {desc} with a temperature of {temp}°C."
    except:
        return "I couldn't fetch the weather right now."

# 🕒 Time response
def get_time(city):
    now = datetime.now()
    return f"The current time in {city} is {now.strftime('%I:%M %p')} (local time)."

# 💱 Currency exchange
def get_exchange_rate(from_currency="USD", to_currency="PHP"):
    url = f"https://open.er-api.com/v6/latest/{from_currency}"
    try:
        response = requests.get(url).json()
        rate = response["rates"][to_currency]
        return f"1 {from_currency} = {rate} {to_currency}"
    except:
        return "Exchange rate data is unavailable."

# 💬 Get response
pending_topic = None

def get_response(intents_list, user_input):
    global pending_topic
    tag = intents_list[0]['intent']

    # Handle pending location follow-up
    if pending_topic:
        if pending_topic == "weather":
            pending_topic = None
            return get_weather(user_input)
        elif pending_topic == "time":
            pending_topic = None
            return get_time(user_input)
        elif pending_topic == "currency":
            pending_topic = None
            return get_exchange_rate("USD", user_input.upper())
        else:
            pending_topic = None
            return "Thanks! But I wasn't sure what info you wanted."

    # Detect location-based queries
    if "weather" in user_input.lower():
        pending_topic = "weather"
        return "Sure! Which city are you asking about?"

    if "time" in user_input.lower():
        pending_topic = "time"
        return "Got it. Which city or country?"

    if "exchange rate" in user_input.lower() or "currency" in user_input.lower():
        pending_topic = "currency"
        return "Which currency do you want to convert to? (e.g., PHP, EUR)"

    # Math
    if tag == 'math' or is_math_query(user_input):
        return solve_math(user_input)

    # Fallback
    if tag == 'fallback':
        duck_result = duckduckgo_search(user_input)
        if duck_result:
            return duck_result
        wiki_result = wikipedia_summary(user_input)
        if wiki_result:
            return wiki_result
        return "I'm not sure about that, but you can try searching online for more details."

    # Intent match
    for intent in intents['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])

# 🧠 Run chatbot
print("🤖 Chatbot is ready! Type 'exit' to quit.")

while True:
    message = input("You: ")
    if message.lower() in ["exit", "quit"]:
        print("Bot: Goodbye!")
        break
    intents_list = predict_class(message)
    response = get_response(intents_list, message)
    print(f"Bot ({intents_list[0]['intent']}): {response}")