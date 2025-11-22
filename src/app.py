import streamlit as st
import joblib
import re
import html
import json
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --- CONFIGURATION ---
st.set_page_config(page_title="Drug Review Analyzer", page_icon="💊", layout="wide")

# --- DATA LOADING ---
@st.cache_resource
def load_artifacts():
    model = joblib.load('../models/sentiment_model.pkl')
    vectorizer = joblib.load('../models/vectorizer.pkl')
    with open('../models/metrics.json', 'r') as f:
        metrics = json.load(f)
    with open('../models/topics.json', 'r') as f:
        topics = json.load(f)
    return model, vectorizer, metrics, topics

def preprocess_text(text):
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
        nltk.download('wordnet')
    
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    text = html.unescape(text).lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(cleaned_tokens)

# Load everything
try:
    model, vectorizer, metrics, topics = load_artifacts()
except FileNotFoundError:
    st.error("Artifacts not found. Please run main_pipeline.py first.")
    st.stop()

# --- UI LAYOUT ---
st.title("💊 Drug Review Analysis & Prediction")

# Create Tabs
tab1, tab2, tab3 = st.tabs(["🔍 Predict Sentiment", "📊 Performance & Metrics", "☁️ Topics & Visuals"])

# TAB 1: PREDICTION (The User Interface)
with tab1:
    st.subheader("Test the Model")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        user_input = st.text_area("Enter a drug review:", height=150, placeholder="Type here... e.g., 'This pill made me feel dizzy but cured my pain.'")
        if st.button("Analyze", type="primary"):
            if user_input.strip():
                cleaned_text = preprocess_text(user_input)
                vec_text = vectorizer.transform([cleaned_text])
                prediction = model.predict(vec_text)[0]
                proba = model.predict_proba(vec_text)[0]
                
                if prediction == 1:
                    st.success(f"**Positive Sentiment** (Confidence: {proba[1]:.1%})")
                else:
                    st.error(f"**Negative Sentiment** (Confidence: {proba[0]:.1%})")
                
                with st.expander("See how the computer sees your text"):
                    st.write(cleaned_text)
            else:
                st.warning("Please enter text.")

    with col2:
        st.info("ℹ️ **How it works:**\nThe model cleans your text, removes 'stop words', and uses Logistic Regression to calculate the probability of the review being positive.")

# TAB 2: METRICS (The Proof)
with tab2:
    st.subheader("Model Performance Evaluation")
    
    # 1. Key Metrics Row
    m1, m2, m3 = st.columns(3)
    m1.metric("Model Accuracy", f"{metrics['accuracy']:.1%}", help="Percentage of correct predictions on Test Set")
    m2.metric("F1 Score", f"{metrics['f1_score']:.3f}", help="Balance between Precision and Recall")
    m3.metric("Training Data Size", f"{metrics['dataset_size']:,} Reviews")
    
    st.divider()
    
    # 2. Distribution Chart (Pie Chart)
    st.markdown("### 🥧 Sentiment Distribution in Dataset")
    fig, ax = plt.subplots(figsize=(6, 4))
    sizes = [metrics['pos_count'], metrics['neg_count']]
    labels = ['Positive Reviews', 'Negative Reviews']
    colors = ['#66b3ff', '#ff9999']
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    ax.axis('equal') 
    st.pyplot(fig)

# TAB 3: VISUALIZATIONS (The Insights)
with tab3:
    st.subheader("What are people talking about?")
    
    # 1. Word Clouds
    st.markdown("### ☁️ Most Frequent Words")
    wc_col1, wc_col2 = st.columns(2)
    
    with wc_col1:
        st.markdown("**Positive Reviews** (Benefits)")
        st.image('../models/wordcloud_pos.png', caption="Common words in Positive reviews")
        
    with wc_col2:
        st.markdown("**Negative Reviews** (Side Effects)")
        st.image('../models/wordcloud_neg.png', caption="Common words in Negative reviews")
        
    st.divider()
    
    # 2. LDA Topics
    st.markdown("### 🔍 Key Topics (Extracted via LDA)")
    st.caption("We analyzed the Negative reviews to find common complaints. Here are the discovered topics:")
    
    topic_cols = st.columns(3)
    for i, (topic_name, words) in enumerate(topics.items()):
        # Cycle through columns
        col = topic_cols[i % 3]
        with col:
            st.markdown(f"**{topic_name}**")
            # Create a nice bullet list
            for word in words[:8]: # Show top 8 words
                st.markdown(f"- {word}")