import pandas as pd
import numpy as np
import re
import html
import json
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import classification_report, accuracy_score, f1_score
import joblib
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# --- CONFIGURATION ---
DATA_PATH = '../data/drug_reviews.csv'
MODEL_DIR = '../models/'

# Ensure NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

def load_and_prep_data(filepath):
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    df.dropna(subset=['review'], inplace=True)
    # Binary Classification: Drop Neutral (5-6)
    df = df[ (df['rating'] <= 4) | (df['rating'] >= 7) ]
    df['sentiment'] = df['rating'].apply(lambda x: 1 if x >= 7 else 0)
    return df

def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    # --- NEW: Add custom stop words to handle the "leaks" ---
    custom_stops = {'im', 'ive', 'dont', 'didnt', 'wont', 'cant', 'drug', 'pill', 'taking'}
    stop_words.update(custom_stops)
    # --------------------------------------------------------

    text = html.unescape(text).lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    
    tokens = text.split()
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(cleaned_tokens)

def generate_wordclouds(df):
    print("Generating Word Clouds...")
    # Positive
    pos_text = " ".join(df[df['sentiment'] == 1]['cleaned_review'])
    wc_pos = WordCloud(width=800, height=400, background_color='white', colormap='Greens').generate(pos_text)
    wc_pos.to_file(MODEL_DIR + 'wordcloud_pos.png')
    
    # Negative
    neg_text = " ".join(df[df['sentiment'] == 0]['cleaned_review'])
    wc_neg = WordCloud(width=800, height=400, background_color='white', colormap='Reds').generate(neg_text)
    wc_neg.to_file(MODEL_DIR + 'wordcloud_neg.png')

def extract_and_save_topics(text_series, n_topics=3):
    print("Extracting Topics...")
    count_vect = CountVectorizer(max_df=0.9, min_df=2, stop_words='english')
    dtm = count_vect.fit_transform(text_series)
    
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(dtm)
    
    feature_names = count_vect.get_feature_names_out()
    topics_dict = {}
    for index, topic in enumerate(lda.components_):
        top_words = [feature_names[i] for i in topic.argsort()[-10:]]
        topics_dict[f"Topic {index+1}"] = top_words
        
    # Save to JSON
    with open(MODEL_DIR + 'topics.json', 'w') as f:
        json.dump(topics_dict, f)

def main():
    # 1. Load
    try:
        df = load_and_prep_data(DATA_PATH)
    except FileNotFoundError:
        print("Error: File not found.")
        return

    # 2. Preprocess
    print("Preprocessing text (Running on full dataset)...")
    # For the final run, we use the FULL dataset (or a larger sample)
    # If this is too slow on your PC, uncomment the next line:
    # df = df.sample(15000, random_state=42) 
    df['cleaned_review'] = df['review'].apply(preprocess_text)

    # 3. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned_review'], df['sentiment'], test_size=0.2, random_state=42
    )

    # 4. Vectorize & Train
    print("Training Model...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    model = LogisticRegression(solver='liblinear')
    model.fit(X_train_tfidf, y_train)

    # 5. Evaluate & Save Metrics
    print("Evaluating...")
    X_test_tfidf = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_tfidf)
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred, average='weighted'),
        "dataset_size": len(df),
        "pos_count": int(df['sentiment'].sum()),
        "neg_count": int(len(df) - df['sentiment'].sum())
    }
    with open(MODEL_DIR + 'metrics.json', 'w') as f:
        json.dump(metrics, f)

    # 6. Generate Visuals (Word Clouds & Topics)
    generate_wordclouds(df)
    
    negative_reviews = df[df['sentiment'] == 0]['cleaned_review']
    extract_and_save_topics(negative_reviews)

    # 7. Save Models
    joblib.dump(model, MODEL_DIR + 'sentiment_model.pkl')
    joblib.dump(vectorizer, MODEL_DIR + 'vectorizer.pkl')
    print("All steps complete. Artifacts saved in models/")

if __name__ == "__main__":
    main()