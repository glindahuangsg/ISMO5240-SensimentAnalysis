import streamlit as st
from transformers import pipeline

# Title and description
st.title("Sentiment Analysis App")
st.write("Analyze the sentiment of your text using Hugging Face Transformers.")

# Cache the model loading so it doesn't reload on every interaction
@st.cache_resource
def load_sentiment_pipeline():
    return pipeline("sentiment-analysis")

sentiment_pipeline = load_sentiment_pipeline()

# User input text area
text_input = st.text_area(
    "Enter Text:",
    value="Deep Learning (DL) represents a highly promising approach to developing applications in Artificial Intelligence (AI).",
    height=150,
)

# Analyze button
if st.button("Analyze Sentiment"):
    if text_input.strip():
        result = sentiment_pipeline(text_input)
        label = result[0]["label"]
        score = result[0]["score"]

        # Display results
        st.write("---")
        st.subheader("Result")
        
        if label == "POSITIVE":
            st.success(f"**Sentiment:** {label} | **Score:** {score:.4f}")
        else:
            st.error(f"**Sentiment:** {label} | **Score:** {score:.4f}")
    else:
        st.warning("Please enter some text to analyze.")
