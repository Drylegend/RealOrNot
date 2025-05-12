# pages/3_📡_Reddit_Feed_Monitor.py

import streamlit as st
import praw
import time
from utils.reddit_utils import classify_reddit_title

st.set_page_config(page_title="Reddit Feed Monitor", page_icon="📡", layout="wide")
st.title("📡 Real-Time Fake News Feed Monitor (Reddit)")

# Connect to Reddit API securely
reddit = praw.Reddit(
    client_id=st.secrets["REDDIT_CLIENT_ID"],
    client_secret=st.secrets["REDDIT_CLIENT_SECRET"],
    user_agent=st.secrets["REDDIT_USER_AGENT"]
)

# UI controls
subreddits = st.multiselect("Select Subreddits", ["worldnews", "news", "politics", "technology"], default=["news"])
limit = st.slider("Number of posts per subreddit", 5, 50, 10)
st.markdown("---")

# Start Monitoring
if st.button("🛰️ Start Monitoring"):
    with st.spinner("Fetching and analyzing posts..."):
        for subreddit_name in subreddits:
            st.subheader(f"r/{subreddit_name}")
            subreddit = reddit.subreddit(subreddit_name)
            posts = subreddit.new(limit=limit)

            for post in posts:
                title = post.title
                label, conf = classify_reddit_title(title)
                color = "red" if label == "Fake" else "green"

                st.markdown(
                    f"<div style='margin-bottom:6px;'>"
                    f"<span style='color:{color}; font-weight:bold;'>[{label} • {conf:.1f}%]</span> {title}"
                    f"</div>",
                    unsafe_allow_html=True
                )
                time.sleep(0.2)  # Simulate streaming effect
