# AI-Powered Fake News & Deepfake Detection Platform

> A Streamlit-based web app to detect fake news in text and deepfake images — with real-time Reddit monitoring, location heatmaps, explainable AI, and reverse traceability.

---

## 🚀 Features

1. **📰 Fake News Classifier**  
   - Text‐based detection using a trained TF–IDF + PassiveAggressive/ViT-Small model  
   - Trust scorecard: bias keywords, toxicity, sentiment  
   - Explainable AI (LIME/SHAP) output  

2. **🖼️ Deepfake Image Analyzer**  
   - Vision Transformer (ViT-Small) model for image deepfake detection  
   - Zero-shot reverse traceability via CLIP (DALL·E, MidJourney, Stable Diffusion, etc.)  
   - Funnel‐style confidence visualization  

3. **📡 Real-Time Feed Monitor**  
   - Streams recent posts from Reddit  
   - Classifies headlines as fake/real with live confidence scores  

4. **🌍 Location Heatmap (Bonus)**  
   - Extracts location entities from articles  
   - Plots them on an interactive Folium map  

---

## 📁 Repository Structure
```

├── app.py ← Landing page
├── requirements.txt ← Python dependencies
├── .streamlit/
│ └── secrets.toml ← API keys (Reddit, OpenCage, etc.)
├── models/
│ ├── fake_news_model.pkl
│ ├── vectorizer.pkl
│ └── vit_deepfake_detector.pth
├── pages/ ← Streamlit multi‐page apps
│ ├── 1_📰_Fake_News_Classifier.py
│ ├── 2_🧠_Deepfake_Image_Analyzer.py
│ └── 3_📡_Reddit_Feed_Monitor.py
└── utils/ ← Helper modules (SOLID)
├── image_utils.py ← Preprocessing & model loading
├── image_analysis_utils.py ← Inference & reverse trace logic
├── reddit_utils.py ← Reddit news fetcher
├── visual_utils.py ← Funnel‐chart visualization
├── heatmap_utils.py ← Fake news heatmap
├── text_analysis_utils.py ← Analysis text
├── text_utls.py ← Model loading for text analysis
├── trust_utils.py ← Trust scorecard
└── xai_utils.py ← Explanation why the news is fake
```

---

## 📦 Installation

1. **Clone the repo**  
   ```bash
   git clone https://github.com/<your-username>/<repo-name>.git
   cd <repo-name>
   ```
2. **Create & activate a virtualenv (recommended)**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```
3. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```
4. **Add secrets**
    Create a file at .streamlit/secrets.toml:
    ```bash
    # .streamlit/secrets.toml
    REDDIT_CLIENT_ID="..."
    REDDIT_CLIENT_SECRET="..."
    REDDIT_USER_AGENT="..."
    OPENCAGE_KEY="..."
    ```
 Usage
Run the Streamlit app from the project root:

```bash
streamlit run app.py
```
Use the sidebar to navigate between modules.

Landing page introduces the platform.

Fake News: Paste text or upload .txt/.csv.

Deepfake: Upload images (.jpg, .png) and click “Analyze”.

Reddit Feed: Select subreddits and stream live classification.

📊 **Model Training (Optional)**
If you retrain or fine-tune models:

Fake News: Train a TF–IDF & PassiveAggressiveClassifier, then

```
joblib.dump(model, 'models/fake_news_model.pkl')
joblib.dump(vectorizer, 'models/vectorizer.pkl')
```

Deepfake: In Colab, train ViT-Small and save:
```
torch.save(model.state_dict(), 'models/vit_deepfake_detector.pth')
```
🎯 Future Improvements
🧠 Integrate LIME/SHAP explainers into image analyzer.

📁 Bulk upload & scheduled monitoring.

🔔 Slack/Telegram alerts for high-confidence fake content.

📈 Trend analytics dashboard.
