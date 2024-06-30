import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.sentiment.vader import SentimentIntensityAnalyzer


import json
import pandas as pd

def load_json(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            # Read the file as lines
            lines = file.readlines()
            json_str = "[" + ",".join(lines) + "]"
            # Load JSON from the string
            data = json.loads(json_str)
        return data
    except json.JSONDecodeError as e:
        print(f"JSONDecodeError: {e}")
        return None
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

# Example usage
file_path = 'Puffin.json'
data = load_json(file_path)



# Function to preprocess conversations and take text
def preprocess_conversations(data):
    texts = []
    for session in data:
        session_text = ""
        if "conversations" in session:
            for conv in session["conversations"]:
                if conv["from"] == "human":
                    session_text += conv["value"] + " "
            texts.append(session_text.strip())
    return texts

# Preprocess data and extract text
texts = preprocess_conversations(data)

# Vectorization
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(texts)

# K-means clustering
k = 3  # Number of clusters 
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(X)

# Sentiment Analysis using VaderSentiment
sid = SentimentIntensityAnalyzer()

# Function to determine sentiment
def get_sentiment_score(text):
    scores = sid.polarity_scores(text)
    if scores['compound'] >= 0.05:
        return 'positive'
    elif scores['compound'] <= -0.05:
        return 'negative'
    else:
        return 'neutral'

# Streamlit UI
st.title('GPT-4 Conversations Analysis')

# Display counts by topic
st.header('Counts by Topic')
topic_counts = pd.Series(clusters).value_counts().reset_index()
topic_counts.columns = ['Topic', 'Count']
st.write(topic_counts)

# Display counts by sentiment
st.header('Counts by Sentiment')
sentiments = []
for session_text in texts:
    sentiment = get_sentiment_score(session_text)
    sentiments.append(sentiment)
sentiment_counts = pd.Series(sentiments).value_counts().reset_index()
sentiment_counts.columns = ['Sentiment', 'Count']
st.write(sentiment_counts)

# Display conversations with topic and sentiment
st.header('Sessions')
sessions_data = []
for i, session in enumerate(data):
    session_text = preprocess_conversations([session])[0]
    topic = clusters[i]
    sentiment = get_sentiment_score(session_text)
    sessions_data.append({
        'Conversation No': i + 1,
        'Topic': f'Topic {topic}',
        'Sentiment': sentiment
    })

sessions_df = pd.DataFrame(sessions_data)
st.dataframe(sessions_df)
