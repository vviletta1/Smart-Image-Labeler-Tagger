import streamlit as st
from PIL import Image
from transformers import pipeline

st.set_page_config(page_title="Smart Image Labeler", layout="centered")

# ---- Force Full Blue Gradient BG + Modern Card UI ----
st.markdown("""
    <style>
    html, body, [data-testid="stAppViewContainer"], [data-testid="stAppViewContainer"] > .main {
        background: linear-gradient(120deg, #2965e8 0%, #67b6f4 100%) !important;
        min-height: 100vh !important;
    }
    .main > div:first-child {padding-top: 0 !important;}
    .custom-card {
        background: #fff; border-radius: 22px;
        box-shadow: 0 8px 40px #2965e844, 0 2px 12px #2965e844;
        padding: 48px 34px 40px 34px;
        margin: 48px auto 48px auto; max-width: 480px; min-height: 430px;
        display: flex; flex-direction: column; align-items: center;
    }
    .upload-icon {
        font-size: 62px;
        color: #1d4ed8;
        background: linear-gradient(135deg, #3b82f6 25%, #6366f1 95%);
        border-radius: 50%; width: 92px; height: 92px; display:flex; align-items:center; justify-content:center;
        margin-bottom: 20px;
        box-shadow: 0 2px 14px #60a5fa33;
    }
    .result-tags {display:flex; flex-wrap:wrap; gap:12px; justify-content:center; margin-top:28px;}
    .tag-bubble {
        background: linear-gradient(90deg,#1d4ed8,#60a5fa);
        color: #fff; border-radius: 20px; padding: 11px 22px;
        font-size: 1.07em; font-weight: 700; box-shadow: 0 2px 12px #1d4ed822;
        border: none; letter-spacing: .03em;
    }
    .footer-note {color:#fff9; margin-top:38px; font-size:1.01em; text-align:center;}
    .stButton>button {background:#2563eb !important; color:#fff !important; border-radius:10px !important; font-weight:600;}
    label, .stSelectbox label, .stTextInput label {font-size:1.1em !important;}
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="custom-card">', unsafe_allow_html=True)
st.markdown('<div class="upload-icon">⬆️</div>', unsafe_allow_html=True)
st.markdown("<h2 style='color:#1d4ed8; margin-top:0; margin-bottom:6px; font-size:2.08em;'>Smart Image Tagger</h2>", unsafe_allow_html=True)
st.markdown("<div style='color:#222c; font-size:1.13em; margin-bottom:24px;'>Upload an image to see what AI finds. Enter your own tags or use a preset for best results.</div>", unsafe_allow_html=True)

# --- Candidate Labels ---
presets = {
    "General": "dog, cat, person, car, food, animal, smile, phone, computer",
    "Nature": "tree, sky, flower, animal, plant, leaf, mountain, sunset",
    "Food": "pizza, cake, salad, burger, fruit, coffee, bread, juice",
    "Tech": "phone, computer, laptop, code, screen, device, robot",
    "Fashion": "dress, shoe, bag, model, glasses, jewelry, style"
}
preset = st.selectbox("Preset tags:", list(presets.keys()) + ["Custom"])
if preset == "Custom":
    user_tags = st.text_input("Enter tags (comma separated):", "")
    candidate_labels = [x.strip() for x in user_tags.split(",") if x.strip()]
else:
    candidate_labels = [x.strip() for x in presets[preset].split(",") if x.strip()]

# --- Upload Area ---
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

st.markdown('</div>', unsafe_allow_html=True)  # End of card

# --- Results Section ---
if uploaded_file:
    st.image(Image.open(uploaded_file), use_column_width=True, caption="Uploaded Image")
    if st.button("Analyze Image", key="analyze", help="Click to tag this image.", type="primary"):
        with st.spinner("Analyzing..."):
            pipe = pipeline("zero-shot-image-classification", model="openai/clip-vit-base-patch16")
            results = pipe(Image.open(uploaded_file), candidate_labels)
        # Filter to only tags with >30% confidence
        tag_results = [r for r in results if r["score"] > 0.3]
        if tag_results:
            st.success("AI found these tags!")
            tag_html = "<div class='result-tags'>" + "".join([
                f"<span class='tag-bubble'>{r['label']} ({r['score']:.0%})</span>" for r in tag_results
            ]) + "</div>"
            st.markdown(tag_html, unsafe_allow_html=True)
        else:
            st.warning("No confident tags found. Try a different preset or add more relevant tags.")

st.markdown('<div class="footer-note">Built by Violetta 💙 | Powered by 🤗 Hugging Face CLIP & Streamlit</div>', unsafe_allow_html=True)




