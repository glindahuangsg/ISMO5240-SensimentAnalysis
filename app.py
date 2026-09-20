import streamlit as st
from transformers import pipeline

# Page Configuration with a fun icon
st.set_page_config(
    page_title="Magic Feeling Finder!",
    page_icon="🌈",
    layout="centered"
)

# Custom CSS for bright, child-friendly styling
st.markdown("""
    <style>
    /* Main Background Gradient */
    .stApp {
        background: linear-gradient(135deg, #FFDEE9 0%, #B5FFFC 100%);
    }

    /* Primary Container Card */
    .block-container {
        background-color: rgba(255, 255, 255, 0.85);
        padding: 2.5rem;
        border-radius: 25px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        margin-top: 2rem;
    }

    /* Playful Title Styling */
    h1 {
        color: #FF3B00;
        font-family: 'Comic Sans MS', 'Chalkboard SE', sans-serif;
        text-align: center;
        font-size: 2.8rem;
    }

    /* Subheadings */
    p, label {
        font-family: 'Comic Sans MS', 'Chalkboard SE', sans-serif;
        font-size: 1.1rem;
    }

    /* Custom Input Box */
    .stTextArea textarea {
        border: 3px solid #FF8E53 !important;
        border-radius: 15px !important;
        font-size: 1.2rem !important;
        background-color: #FFFDF0 !important;
    }

    /* Magic Button Styling */
    div.stButton > button {
        background: linear-gradient(45deg, #FF6B6B, #FF8E53);
        color: white !important;
        font-size: 1.5rem !important;
        font-weight: bold;
        border-radius: 50px !important;
        border: none !important;
        padding: 0.8rem 2rem !important;
        width: 100%;
        box-shadow: 0 6px 15px rgba(255, 107, 107, 0.4);
        transition: transform 0.2s ease;
    }

    div.stButton > button:hover {
        transform: scale(1.03);
    }

    /* Fun Result Cards */
    .happy-box {
        background-color: #D4EDDA;
        border: 3px solid #28A745;
        border-radius: 20px;
        padding: 1.5rem;
        text-align: center;
        color: #155724;
        font-family: 'Comic Sans MS', sans-serif;
    }

    .sad-box {
        background-color: #FFF3CD;
        border: 3px solid #FFC107;
        border-radius: 20px;
        padding: 1.5rem;
        text-align: center;
        color: #856404;
        font-family: 'Comic Sans MS', sans-serif;
    }
    </style>
""", unsafe_allow_html=True)

# Cache model loading with a friendly spinner message
@st.cache_resource
def load_sentiment_pipeline():
    return pipeline("sentiment-analysis")

with st.spinner("🧙‍♂️ Awakening the Magic Feeling Wizard..."):
    sentiment_pipeline = load_sentiment_pipeline()

# Header Section
st.title("🌈 Magic Feeling Finder! ✨")
st.write("### Type or paste your story below to find out its mood power! 🪄")

# Input text area
text_input = st.text_area(
    "What's on your mind today?",
    value="I love going to the park with my dog and eating delicious ice cream!",
    height=140
)

# Magic Analyze Button
if st.button("✨ Check My Feeling! ✨"):
    if text_input.strip():
        result = sentiment_pipeline(text_input)
        label = result[0]["label"]
        score = result[0]["score"]
        percent = int(score * 100)

        st.write("")  # Spacing

        if label == "POSITIVE":
            st.markdown(f"""
                <div class="happy-box">
                    <h1>🥳 SUPER HAPPY! 🎉</h1>
                    <h3>This text is full of sunshine and good vibes! ☀️</h3>
                    <p style="font-size: 1.3rem;"><b>Happiness Meter:</b> {percent}% Pure Magic!</p>
                </div>
            """, unsafe_allow_html=True)
            st.balloons()
        else:
            st.markdown(f"""
                <div class="sad-box">
                    <h1>💙 A LITTLE BLUE OR SAD 🌧️</h1>
                    <h3>This text feels a bit sleepy, upset, or worried.</h3>
                    <p style="font-size: 1.3rem;"><b>Blue Meter:</b> {percent}% Feeling Strength</p>
                </div>
            """, unsafe_allow_html=True)
            st.snow()
    else:
        st.warning("🎈 Oopsie! Please type a sentence first before clicking the magic button!")
