import streamlit as st
from PIL import Image
from transformers import pipeline

# 1. Page Configuration
def configure_page() -> None:
    """Sets up page details like title, icon, and layout."""
    st.set_page_config(
        page_title="Image Caption Generator",
        page_icon="🖼️",
        layout="centered"
    )

# 2. Model Loading
@st.cache_resource
def load_caption_pipeline(model_name: str = "Salesforce/blip-image-captioning-base"):
    """Loads and caches the Hugging Face image-to-text pipeline."""
    return pipeline("image-to-text", model=model_name)

# 3. Image Captioning Logic
def generate_caption(caption_pipeline, image: Image.Image) -> str:
    """Generates a brief text description for the given PIL Image."""
    # max_new_tokens limits length for speed and concise output
    result = caption_pipeline(image, generate_kwargs={"max_new_tokens": 50})
    return result[0]["generated_text"]

# 4. Main Application Entry Point
def main() -> None:
    configure_page()

    st.title("🖼️ Smart Image Captioner")
    st.write("Upload an image to generate a brief, automated description.")

    # Model Specification
    MODEL_NAME = "Salesforce/blip-image-captioning-base"

    with st.spinner("Loading model into memory..."):
        caption_pipeline = load_caption_pipeline(model_name=MODEL_NAME)

    # File Uploader Widget
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png", "webp"]
    )

    if uploaded_file is not None:
        try:
            # Open image using PIL
            image = Image.open(uploaded_file).convert("RGB")

            # Layout into two columns
            col1, col2 = st.columns([1, 1])

            with col1:
                st.image(image, caption="Uploaded Image", use_container_width=True)

            with col2:
                st.subheader("Description")
                if st.button("Generate Caption"):
                    with st.spinner("Analyzing image..."):
                        caption = generate_caption(caption_pipeline, image)
                        st.success(f"**Caption:** {caption}")

        except Exception as e:
            st.error(f"Error processing image: {e}")

if __name__ == "__main__":
    main()
