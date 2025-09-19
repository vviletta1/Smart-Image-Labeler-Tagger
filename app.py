import streamlit as st
from PIL import Image
from transformers import pipeline

st.set_page_config(page_title="Smart Image Labeler", layout="centered")

# ---- Custom CSS for Modern Blue Card UI ----
st.markdown("""
    <style>
    body, .main { background: linear-gradient(135deg, #4895ef 0%, #4361ee 100%) !important; }
    .upload-card {
        max-width: 440px; margin: 36px auto 0 auto; background: #fff;
        border-radius: 24px; box-shadow: 0 8px 36px #185ec622;
        padding: 42px 32px 32px 32px; text-align: center;
    }
    .upload-icon { font-size: 52px; color: #4895ef; margin-bottom: 16px; }
    .action-btn button {background: #4361ee !important; color:#fff !important; font-weight:600; font-size:1.2em;}
    .result-tags {display:flex; flex-wrap:wrap; gap:8px; justify-content:center; margin-top:28px;}
    .tag-bubble {
        background: linear-gradient(90deg,#4895ef,#4cc9f0);
        color: #fff; border-radius: 20px; padding: 8px 18px;
        font-size: 1.02em; margin: 2px 3px 6px 3px; font-weight: 600; box-shadow: 0 2px 8px #4895ef33;
        border: none;
    }
    .footer-note {color:#fff9; margin-top:40px; font-size:0.96em; text-align:center;}
    </style>
""", unsafe_allow_html=True)

st.markdown('<div style="height:36px"></div>', unsafe_allow_html=True)  # Top space

# --- Card Container ---
st.markdown('<div class="upload-card">', unsafe_allow_html=True)

st.markdown('<div class="upload-icon">⬆️</div>', unsafe_allow_html=True)
st.markdown("<h2 style='color:#185ec6; margin-top:0;'>Try Smart Image Tagging!</h2>", unsafe_allow_html=True)
st.markdown("<p style='color:#222b; margin-top:-8px; font-size:1.05em;'>Upload a picture and see what AI finds. Enter your own tags or use a preset for best results.</p>", unsafe_allow_html=True)

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
else:
    st.markdown('<div style="height:38px"></div>', unsafe_allow_html=True)

st.markdown('<div class="footer-note">Built by Violetta 💙 | Powered by 🤗 Hugging Face CLIP & Streamlit</div>', unsafe_allow_html=True)


