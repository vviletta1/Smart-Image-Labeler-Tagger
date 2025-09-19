import streamlit as st
from PIL import Image
from transformers import pipeline

st.set_page_config(page_title="Smart Image Labeler", layout="centered")
st.title("🖼️ Smart Image Labeler/Tagger")

# Load Hugging Face CLIP pipeline (this downloads a model the first time)
@st.cache_resource
def get_pipe():
    return pipeline("zero-shot-image-classification", model="openai/clip-vit-base-patch16")

pipe = get_pipe()

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    st.markdown("**Enter candidate labels (comma separated):**")
    default_labels = "dog, cat, person, computer, phone, car, tree, coffee, child, food, animal, plant, book, smile"
    candidate_labels = st.text_input("Labels:", value=default_labels)
    labels = [x.strip() for x in candidate_labels.split(",") if x.strip()]

    if st.button("Label Image"):
        with st.spinner("Analyzing image..."):
            results = pipe(img, labels)
        st.success("Done! Here are your labels:")
        for res in results:
            st.write(f"**{res['label']}** — confidence: `{res['score']:.2%}`")
else:
    st.info("Upload an image to begin.")

st.caption("Powered by 🤗 Hugging Face CLIP + Streamlit | Built by Violetta 💡")
