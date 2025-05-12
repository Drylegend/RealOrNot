import streamlit as st

# Page settings
st.set_page_config(
    page_title="Fake News & Deepfake Detection Platform",
    page_icon="🛡️",
    layout="wide"
)

# Title
st.title("🛡️ AI Fake News & Deepfake Detection Platform")

# Intro text
st.markdown("""
Welcome to the **Fake News & Deepfake Detection Platform** powered by AI.

🔍 Use the sidebar to navigate:
- 📰 **Fake News Classifier** — Upload or type text to check if it's fake.
- 🖼️ **Deepfake Image Analyzer** — Analyze images for signs of AI generation.
- 🛰️ **Real-Time Feed Monitor** — Monitor Reddit feeds for misinformation.
""")

