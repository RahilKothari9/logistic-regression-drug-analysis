import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# --- CONFIGURATION ---
DATA_PATH = '../data/drug_reviews.csv' # Ensure this matches your file name
MODEL_DIR = '../models/'

# Download NLTK resources (run once)
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

def load_and_prep_data(filepath):
    print(f"Loading data from {filepath}...")
    # Assuming the UCI format which usually has tabs or commas
    # If your csv fails, try delimiter='\t'
    df = pd.read_csv(filepath) 
    
    # 1. Handle Missing Values
    df.dropna(subset=['review'], inplace=True) # Adjust column name if needed
    
    # 2. Create Target Variable (Sentiment)
    # Rating 1-4: Negative (0), 5-6: Neutral (dropped for binary, or keep as 1), 7-10: Positive (2)
    # For simplicity, let's do a Binary Split (Positive vs Negative) and drop Neutral to sharpen the model
    df = df[ (df['rating'] <= 4) | (df['rating'] >= 7) ]
    df['sentiment'] = df['rating'].apply(lambda x: 1 if x >= 7 else 0) # 1 = Positive, 0 = Negative
    
    print(f"Data Loaded. Shape: {df.shape}")
    return df

def preprocess_text(text):
    # Initialize Lemmatizer
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove HTML tags & Punctuation using Regex
    text = re.sub(r'<.*?>', '', text) # Remove HTML
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
    
    # 3. Tokenize & Remove Stopwords & Lemmatize
    tokens = text.split()
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    return " ".join(cleaned_tokens)

def extract_topics(text_series, n_topics=3):
    """
    Uses LDA to find topics. 
    Note: We use CountVectorizer for LDA as it works better with raw counts than TF-IDF.
    """
    print("Extracting Topics...")
    count_vect = CountVectorizer(max_df=0.9, min_df=2, stop_words='english')
    dtm = count_vect.fit_transform(text_series)
    
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(dtm)
    
    # Display top words per topic
    feature_names = count_vect.get_feature_names_out()
    topics = {}
    for index, topic in enumerate(lda.components_):
        top_words = [feature_names[i] for i in topic.argsort()[-10:]]
        topics[f"Topic {index+1}"] = top_words
        print(f"Topic {index+1}: {', '.join(top_words)}")
    
    return topics

def main():
    # 1. Load Data
    try:
        df = load_and_prep_data(DATA_PATH)
    except FileNotFoundError:
        print("ERROR: Data file not found. Please put 'drug_reviews.csv' in the data/ folder.")
        return

    # 2. Clean Text (This can take time on large datasets!)
    print("Preprocessing text (this might take a minute)...")
    # Taking a sample of 10k rows for speed during development. 
    # COMMENT OUT the next line to use the full dataset.
    df = df.sample(10000, random_state=42) 
    
    df['cleaned_review'] = df['review'].apply(preprocess_text)

    # 3. Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned_review'], df['sentiment'], test_size=0.2, random_state=42
    )

    # 4. Vectorization (TF-IDF)
    print("Vectorizing text...")
    vectorizer = TfidfVectorizer(max_features=5000) # Limit features to keep it fast
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # 5. Train Model (Logistic Regression)
    print("Training Logistic Regression Model...")
    model = LogisticRegression(solver='liblinear') # liblinear is good for small/med datasets
    model.fit(X_train_tfidf, y_train)

    # 6. Evaluation
    print("\n--- EVALUATION RESULTS ---")
    y_pred = model.predict(X_test_tfidf)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # 7. Topic Extraction (LDA) on Negative Reviews
    print("\n--- TOP TOPICS IN NEGATIVE REVIEWS ---")
    negative_reviews = df[df['sentiment'] == 0]['cleaned_review']
    extract_topics(negative_reviews, n_topics=3)

    # 8. Save Artifacts
    print("\nSaving model and vectorizer...")
    joblib.dump(model, MODEL_DIR + 'sentiment_model.pkl')
    joblib.dump(vectorizer, MODEL_DIR + 'vectorizer.pkl')
    print("Done!")

if __name__ == "__main__":
    main()