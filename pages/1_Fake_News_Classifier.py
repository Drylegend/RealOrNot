import streamlit as st
import pandas as pd
from streamlit_folium import st_folium

from utils.text_analysis_utils import analyze_article

st.set_page_config(
    page_title="Fake News Classifier",
    page_icon="📰",
    layout="wide"
)
st.title("📰 AI Fake News Text Classifier")

# Session state flags
if "done" not in st.session_state:
    st.session_state["done"] = False
if "show_map" not in st.session_state:
    st.session_state["show_map"] = False

# Input
option = st.radio("Select input type:", ("Paste Text", "Upload File"))
text = ""
if option == "Paste Text":
    text = st.text_area("Paste news content here:", height=200)
else:
    up = st.file_uploader("Upload a .txt or .csv file", type=["txt", "csv"])
    if up:
        if up.name.endswith(".txt"):
            text = up.read().decode("utf-8")
        else:
            df = pd.read_csv(up)
            text = str(df.iloc[0, 0])

# Run analysis
if st.button("🔍 Classify"):
    if not text.strip():
        st.warning("Please enter or upload some text.")
    else:
        key = st.secrets.get("OPENCAGE_KEY", "")
        result = analyze_article(text, key)
        st.session_state["result"] = result
        st.session_state["done"] = True

# Display results
if st.session_state["done"]:
    res = st.session_state["result"]

    # Prediction
    pr = res["prediction"]
    st.subheader("🧠 Prediction")
    st.success(f"{pr['label']}  (Confidence: {pr['conf_pct']:.1f}%)")

    # Trust score
    st.markdown("---")
    st.subheader("📊 Trust Scorecard")
    if pr["pred"] == 0:
        st.warning(
            f"🔎 Adjusted Trust Score: **{res['adjusted_score']} / 100**\n\n"
            f"• Content-based: {res['trust']['Trust Score']}  \n"
            f"• Model Fake Confidence: {pr['conf_prob']:.2%}"
        )
    else:
        st.success(
            f"🔎 Adjusted Trust Score: **{res['adjusted_score']} / 100**\n\n"
            f"• Content-based: {res['trust']['Trust Score']}  \n"
            f"• Model Real Confidence: {pr['conf_prob']:.2%}"
        )
    st.markdown("**Breakdown:**")
    st.write({
        "Bias Keyword Hits": res["trust"]["Bias Keyword Hits"],
        "Toxic Word Hits": res["trust"]["Toxic Word Hits"],
        "Sentiment Polarity": res["trust"]["Sentiment Polarity"],
        "Subjectivity": res["trust"]["Subjectivity"]
    })

    # Explainable AI
    st.markdown("---")
    st.subheader("🧠 Explainable AI (XAI) Preview")
    st.components.v1.html(res["explanation_html"], height=500, scrolling=True)

    # Heatmap
    st.markdown("---")
    st.subheader("🌍 Misinformation Location Heatmap")
    show = st.checkbox("Show Location Heatmap", key="show_map")
    if show:
        if not res["locations"]:
            st.info("No location entities found.")
        else:
            st.write(f"Found locations: {res['locations']}")
            st_folium(res["heatmap"], width=700, height=400)
