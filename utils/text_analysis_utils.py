# utils/text_analysis_utils.py

from utils.text_utils import load_text_model, predict_text
from utils.trust_utils import score_article
from utils.xai_utils import explain_fake_news
from utils.heatmap_utils import build_heatmap
from PIL import Image  # for type hints

def analyze_article(text: str, opencage_key: str) -> dict:
    """
    Runs the full Fake News + XAI + Heatmap pipeline.
    Returns a dict:
    {
      "prediction": { ... }     # from predict_text()
      "trust": { ... }          # from score_article()
      "adjusted_score": float,
      "explanation_html": str,
      "heatmap": folium.Map,
      "locations": list[str]
    }
    """
    # 1) Model & prediction
    model, vectorizer = load_text_model()
    pred_res = predict_text(text, model, vectorizer)

    # 2) Trust score
    trust = score_article(text)
    # adjust trust by model confidence
    cs = trust["Trust Score"]
    cp = pred_res["conf_prob"]
    if pred_res["pred"] == 0:  # fake
        adj = round(cs * (1 - cp), 1)
    else:
        adj = round(cs * (1 + cp) / 2, 1)

    # 3) XAI explanation
    expl = explain_fake_news(text)
    html = expl.as_html()
    wrapped = (
        f'<div style="background-color:white; padding:16px; '
        f'border-radius:8px;">{html}</div>'
    )

    # 4) Heatmap
    heatmap, locs = build_heatmap(text, opencage_key)

    return {
        "prediction": pred_res,
        "trust": trust,
        "adjusted_score": adj,
        "explanation_html": wrapped,
        "heatmap": heatmap,
        "locations": locs
    }
