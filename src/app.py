import streamlit as st
import joblib
import re
import html
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --- SETUP ---
# Load the trained models
# We use st.cache_resource so it doesn't reload the model on every click (faster)
@st.cache_resource
def load_artifacts():
    model = joblib.load('../models/sentiment_model.pkl')
    vectorizer = joblib.load('../models/vectorizer.pkl')
    return model, vectorizer

# We need the same cleaning function here to clean user input
def preprocess_text(text):
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
        nltk.download('wordnet')

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    text = html.unescape(text)
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(cleaned_tokens)

# --- UI LAYOUT ---
st.set_page_config(page_title="Drug Review Analyzer", page_icon="💊")

st.title("💊 Drug Review Sentiment Analyzer")
st.markdown("Type a drug review below to analyze if the sentiment is **Positive** or **Negative**.")

# Load models
try:
    model, vectorizer = load_artifacts()
    st.success("System Ready: Models Loaded Successfully")
except FileNotFoundError:
    st.error("Error: Models not found. Please run 'main_pipeline.py' first.")
    st.stop()

# Input Area
user_input = st.text_area("Enter Review:", height=150, placeholder="Example: This medicine saved my life but gave me a slight headache...")

if st.button("Analyze Sentiment"):
    if user_input.strip():
        # 1. Clean
        cleaned_text = preprocess_text(user_input)
        
        # 2. Vectorize
        vec_text = vectorizer.transform([cleaned_text])
        
        # 3. Predict
        prediction = model.predict(vec_text)[0]
        proba = model.predict_proba(vec_text)[0]
        
        # 4. Display Results
        st.divider()
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction == 1:
                st.success("## 😊 Positive Review")
                st.metric("Confidence", f"{proba[1]:.2%}")
            else:
                st.error("## 😞 Negative Review")
                st.metric("Confidence", f"{proba[0]:.2%}")
        
        with col2:
            st.info("Processed Text (What the model saw):")
            st.caption(cleaned_text)
            
    else:
        st.warning("Please enter some text first.")

# Sidebar for Project Info
with st.sidebar:
    st.header("About Project")
    st.write("This tool uses Logistic Regression with TF-IDF vectorization to classify drug reviews.")
    st.write("---")
    st.write("**Accuracy:** 82.3%")
    st.write("**Macro F1:** 0.74")