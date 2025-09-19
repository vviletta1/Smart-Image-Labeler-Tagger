import streamlit as st
from PIL import Image
from transformers import pipeline
import pandas as pd

# --- Streamlit Page Config ---
st.set_page_config(page_title="Smart Image Labeler", layout="centered")

st.markdown("""
    <style>
    .label-card {
        background: #f7faff;
        border-radius: 14px;
        padding: 18px;
        margin: 12px 0;
        box-shadow: 0 3px 12px #eef2f4;
        border: 1px solid #dde2ee;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .label-title {font-size: 1.1rem; color: #1d3557; font-weight: 700;}
    .score-bar {height: 10px; border-radius: 6px; background: linear-gradient(90deg, #4fc3f7, #00bfae);}
    .thumb {font-size:1.5em; margin: 0 8px;}
    </style>
""", unsafe_allow_html=True)

st.title("🖼️ Smart Image Labeler/Tagger")

# --- Model loader
@st.cache_resource
def get_pipe():
    return pipeline("zero-shot-image-classification", model="openai/clip-vit-base-patch16")
pipe = get_pipe()

# --- Category presets
category_presets = {
    "General": "dog, cat, person, computer, phone, car, tree, coffee, child, food, animal, plant, book, smile",
    "Nature": "tree, flower, mountain, sky, bird, river, plant, animal, leaf, sunset, ocean",
    "Pets": "dog, cat, puppy, kitten, rabbit, hamster, animal, fur, cute, play, sleep",
    "Food": "pizza, burger, salad, cake, fruit, vegetable, dessert, coffee, tea, juice, bread",
    "Tech": "phone, computer, laptop, screen, keyboard, code, gadget, electronics, robot, AI, device",
    "Fashion": "dress, shirt, shoe, bag, style, model, glasses, jewelry, beauty, fabric",
}

preset = st.selectbox("Choose a category preset (or write your own):", list(category_presets.keys()) + ["Custom"])
if preset == "Custom":
    user_desc = st.text_input("Describe your image or possible tags (comma separated):", "")
    candidate_labels = [x.strip() for x in user_desc.split(",") if x.strip()]
else:
    candidate_labels = [x.strip() for x in category_presets[preset].split(",") if x.strip()]
    st.info("Tip: Change preset to 'Custom' to enter your own label ideas!")

# --- File uploader
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)
    if st.button("Label Image"):
        with st.spinner("Analyzing image..."):
            results = pipe(img, candidate_labels)
        # --- Only keep results above 30% confidence
        top_results = [res for res in results if res['score'] > 0.3]
        if not top_results:
            st.error("No strong matches found. Try adding better/different candidate labels!")
        else:
            st.success("Found these tags (confidence > 30%):")
            # --- Store feedback in session
            if "feedback" not in st.session_state:
                st.session_state["feedback"] = {}
            for res in top_results:
                # --- Label card with thumbs feedback
                label = res['label']
                score = res['score']
                st.markdown(f"""
                    <div class='label-card'>
                        <span class='label-title'>{label} <span style="opacity:0.65;">({score:.1%})</span></span>
                        <div>
                            <button class='thumb' onclick="window.parent.postMessage({{thumb:'up',label:'{label}'}}, '*')">👍</button>
                            <button class='thumb' onclick="window.parent.postMessage({{thumb:'down',label:'{label}'}}, '*')">👎</button>
                        </div>
                    </div>
                    <div class='score-bar' style='width:{int(score*100)}%; background: linear-gradient(90deg,#00bfae,#4fc3f7); margin-bottom:14px;'></div>
                """, unsafe_allow_html=True)
                # Simple up/down counter
                feedback_key = f"{label}_fb"
                up = st.session_state["feedback"].get(feedback_key+"_up", 0)
                down = st.session_state["feedback"].get(feedback_key+"_down", 0)
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f"👍 {up}", key=f"{label}_up"):
                        st.session_state["feedback"][feedback_key+"_up"] = up + 1
                with col2:
                    if st.button(f"👎 {down}", key=f"{label}_down"):
                        st.session_state["feedback"][feedback_key+"_down"] = down + 1
            # --- Download results as CSV
            df = pd.DataFrame(top_results)
            st.download_button("Download Labels as CSV", df.to_csv(index=False), "labels.csv")
else:
    st.info("Upload an image to begin!")

st.caption("Powered by 🤗 Hugging Face CLIP + Streamlit | Built by Violetta 💡")
