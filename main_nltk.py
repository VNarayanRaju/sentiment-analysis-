import string
from collections import Counter
import pandas as pd
import random
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Load the emotion dictionary from the emotions.txt file
emotion_dict = {}
with open('emotions.txt', 'r') as file:
    for line in file:
        clear_line = line.replace("\n", '').replace(",", '').replace("'", '').strip()
        word, emotion = clear_line.split(':')
        emotion_dict[word] = emotion

# Load the stop words
stop_words = set(stopwords.words('english'))


def clean_text(text):
    lower_case = text.lower()
    cleaned_text = lower_case.translate(str.maketrans('', '', string.punctuation))
    tokenized_words = word_tokenize(cleaned_text, "english")
    final_words = [word for word in tokenized_words if word not in stop_words]
    lemma_words = [WordNetLemmatizer().lemmatize(word) for word in final_words]
    return lemma_words


def analyze_emotion(text):
    lemma_words = clean_text(text)
    emotion_list = [emotion_dict[word] for word in lemma_words if word in emotion_dict]
    emotion_count = Counter(emotion_list)
    return emotion_count


def sentiment_analyze(text):
    score = SentimentIntensityAnalyzer().polarity_scores(text)
    if score['neg'] > score['pos']:
        return "Negative Sentiment"
    elif score['neg'] < score['pos']:
        return "Positive Sentiment"
    else:
        return "Neutral Sentiment"


def load_responses(csv_file):
    df = pd.read_csv(csv_file)
    return df


def get_response(emotion, df):
    responses = df[df['sentiment'] == emotion]['response'].tolist()
    if responses:
        return random.choice(responses)
    return "I'm not sure how to respond to that."


# Chatbot function
def chat(csv_file):
    df = load_responses(csv_file)
    print("Hello! I'm your chatbot. How can I help you today?")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "bye", "quit"]:
            print("Chatbot: Goodbye!")
            break

        emotion_count = analyze_emotion(user_input)
        sentiment = sentiment_analyze(user_input)

        if emotion_count:
            emotion = emotion_count.most_common(1)[0][0]
            response = get_response(emotion, df)
        else:
            response = "I couldn't detect a specific emotion."

        print(f" Sentiment: {sentiment})")


if __name__ == "__main__":
    chat('Sentiment.csv')
