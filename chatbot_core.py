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
intents = {
    "intents": [
        {
            "tag": "greeting",
            "patterns": ["Hey", "Hi there", "Hello", "Good to see you", "Good to see you again", "Yo", "What's up"],
            "responses": ["Hey! What can I help you with today?", "Hi there! Ready when you are.", "Hello! Ask me anything."]
        },
        {
            "tag": "goodbye",
            "patterns": ["Bye", "Catch you later", "I'm done", "See you", "Talk to you soon"],
            "responses": ["Take care!", "See you next time!", "Goodbye for now!"]
        },
        {
            "tag": "thanks",
            "patterns": ["Thanks", "Appreciate it", "Thank you", "Much obliged", "You're a lifesaver"],
            "responses": ["You're very welcome!", "Glad I could help!", "Anytime!"]
        },
        {
            "tag": "weather",
            "patterns": ["What's the weather like?", "Tell me the weather", "Is it sunny in Cebu?", "Weather update"],
            "responses": ["Sure! Which city are you asking about?"]
        },
        {
            "tag": "time",
            "patterns": ["What time is it in Tokyo?", "Tell me the time", "Current time in London", "Time now"],
            "responses": ["Got it. Which city or country?"]
        },
        {
            "tag": "currency",
            "patterns": ["Convert USD to JPY", "What's the exchange rate for EUR?", "How much is 1 dollar in yen?", "Currency info"],
            "responses": ["Which currency do you want to convert to? (e.g., PHP, EUR)"]
        },
        {
            "tag": "math",
            "patterns": ["Can you solve 8 times 7?", "What's 15 divided by 3?", "I need help with math", "Calculate 12 plus 9"],
            "responses": ["Let me do the math for you...", "Crunching the numbers..."]
        },
        {
            "tag": "fallback",
            "patterns": [],
            "responses": ["I'm not sure about that, but I can look it up for you.", "Let me check online sources and get back to you.", "That one's tricky — give me a second to search."]
        }
    ]
}
# 🧹 Preprocess input
def preprocess(text):
    from nltk.tokenize import wordpunct_tokenize
    tokens = wordpunct_tokenize(text.lower())
    lemmatized = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha()]
    filtered = [word for word in lemmatized if word not in stop_words]
    return ' '.join(filtered)

# 🔍 Rule-based intent matcher
def predict_class(sentence):
    sentence_tokens = set(preprocess(sentence).split())
    best_match = None
    best_score = 0

    for intent in intents["intents"]:
        for pattern in intent["patterns"]:
            pattern_tokens = set(preprocess(pattern).split())
            score = len(sentence_tokens & pattern_tokens)
            if score > best_score:
                best_score = score
                best_match = intent["tag"]

    if best_match and best_score > 0:
        return [{'intent': best_match, 'probability': str(best_score)}]
    else:
        return [{'intent': 'fallback', 'probability': '1.0'}]


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