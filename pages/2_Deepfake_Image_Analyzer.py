# pages/2_Deepfake_Image_Analyzer.py

import streamlit as st
from PIL import Image

from utils.image_analysis_utils import analyze_image
from utils.visual_utils import plot_funnel_with_categories

st.set_page_config(page_title="Deepfake Analyzer", layout="wide")
st.title("🧠 Deepfake Image Analyzer")

uploaded = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
if not uploaded:
    st.info("Please upload an image.")
    st.stop()

img = Image.open(uploaded).convert("RGB")
st.image(img, caption="Uploaded Image", use_container_width=True)

if st.button("🔍 Analyze"):
    with st.spinner("Running analysis..."):
        res = analyze_image(img, model_path="models/vit_deepfake_detector_augmented.pth")

    # Classification result
    lbl = res["label"]
    conf = res["confidence"] * 100
    if lbl == "REAL":
        st.success(f"✅ REAL IMAGE (Confidence: {conf:.1f}%)")
    else:
        st.error(f"❗ DEEPFAKE (Confidence: {conf:.1f}%)")

    # Reverse Traceability
    if lbl == "DEEPFAKE" and res["reverse_scores"]:
        st.markdown("---")
        st.subheader("🔁 Reverse Traceability Analysis")
        fig = plot_funnel_with_categories(res["reverse_scores"])
        st.pyplot(fig)
