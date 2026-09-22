import streamlit as st
from PIL import Image
from transformers import pipeline

# 1. Page Configuration
def configure_page() -> None:
    """Sets up page title, icon, and layout."""
    st.set_page_config(
        page_title="Age Classification using ViT",
        page_icon="👤",
        layout="centered"
    )

# 2. Model Loading
@st.cache_resource
def load_age_classifier(model_name: str = "nateraw/vit-age-classifier"):
    """Loads and caches the Hugging Face age classification pipeline."""
    return pipeline("image-classification", model=model_name)

# 3. Age Prediction Helper
def classify_age(classifier, image: Image.Image):
    """Runs prediction on the image and returns sorted results."""
    predictions = classifier(image)
    return sorted(predictions, key=lambda x: x['score'], reverse=True)

# 4. Main Application Entry Point
def main() -> None:
    configure_page()

    st.title("👤 Age Classification using ViT")
    st.write("Upload a photo of a person to predict their age group.")

    MODEL_NAME = "nateraw/vit-age-classifier"

    with st.spinner("Loading age classification model..."):
        age_classifier = load_age_classifier(model_name=MODEL_NAME)

    # Allow users to upload an image instead of hardcoding a local file path
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png", "webp"]
    )

    if uploaded_file is not None:
        try:
            # Load image safely from the uploaded file buffer
            image = Image.open(uploaded_file).convert("RGB")

            col1, col2 = st.columns([1, 1])

            with col1:
                st.image(image, caption="Uploaded Image", use_container_width=True)

            with col2:
                st.subheader("Classification Result")
                if st.button("Predict Age"):
                    with st.spinner("Analyzing image..."):
                        age_predictions = classify_age(age_classifier, image)
                        top_prediction = age_predictions[0]

                        # Display top result
                        st.success(
                            f"**Predicted Age Range:** {top_prediction['label']}\n\n"
                            f"**Confidence:** {top_prediction['score']:.2%}"
                        )

                        # Display all predictions in an expander
                        with st.expander("See full breakdown"):
                            for pred in age_predictions:
                                st.write(f"- **{pred['label']}**: {pred['score']:.2%}")

        except Exception as e:
            st.error(f"Error processing image: {e}")

if __name__ == "__main__":
    main()
