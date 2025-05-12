# utils/visual_utils.py

import matplotlib.pyplot as plt

def plot_funnel_with_categories(all_scores):
    """
    Generates a funnel-style horizontal bar chart for reverse traceability scores,
    using a single red color for all bars.

    Parameters:
    - all_scores: List of dicts with keys 'label' (str) and 'score' (float in [0,1]).

    Returns:
    - A Matplotlib Figure object.
    """
    labels = [item["label"] for item in all_scores]
    scores = [item["score"] * 100 for item in all_scores]  # to percentage
    max_score = max(scores) if scores else 100

    fig, ax = plt.subplots(figsize=(8, 2 + len(scores) * 0.5))

    for i, (label, score) in enumerate(zip(labels, scores)):
        left = (max_score - score) / 2
        ax.barh(label, score, left=left, color="#FF6B6B", edgecolor='black', height=0.6)
        ax.text(left + score/2, i, f"{score:.1f}%", va='center', ha='center',
                fontsize=10, fontweight='bold', color='white')

    ax.set_xlim(0, max_score)
    ax.set_xlabel("Confidence (%)", fontsize=12)
    ax.set_title("🔁 Reverse Traceability Funnel", fontsize=14, weight='bold')
    ax.invert_yaxis()
    ax.grid(False)
    ax.set_facecolor('#f9f9f9')
    fig.patch.set_facecolor('#f9f9f9')
    ax.tick_params(left=False, bottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Optional single-entry legend (can be removed if not needed)
    legend_patch = plt.Line2D([0], [0], marker='s', color='w',
                              label='Source Confidence',
                              markerfacecolor='#FF6B6B', markersize=10)
    ax.legend(handles=[legend_patch], loc='lower right')

    plt.tight_layout()
    return fig
