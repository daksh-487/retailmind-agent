"""
app.py — Streamlit UI for the RetailMind Product Intelligence Agent.
"""

import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RetailMind — StyleCraft Intelligence",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Main background */
.stApp { background-color: #0f0f1a; color: #e8e8f0; }

/* Sidebar */
section[data-testid="stSidebar"] { background: #16162a; border-right: 1px solid #2d2d4a; }

/* Chat messages */
.user-bubble {
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    color: white;
    padding: 12px 16px;
    border-radius: 18px 18px 4px 18px;
    margin: 8px 0 8px 20%;
    font-size: 0.95rem;
    line-height: 1.5;
}
.assistant-bubble {
    background: #1e1e35;
    border: 1px solid #2d2d4a;
    color: #e8e8f0;
    padding: 14px 18px;
    border-radius: 18px 18px 18px 4px;
    margin: 8px 20% 8px 0;
    font-size: 0.95rem;
    line-height: 1.6;
}

/* Metric cards */
.metric-card {
    background: #1e1e35;
    border: 1px solid #2d2d4a;
    border-radius: 12px;
    padding: 14px;
    text-align: center;
    margin-bottom: 8px;
}
.metric-value { font-size: 1.6rem; font-weight: 700; color: #a78bfa; }
.metric-label { font-size: 0.75rem; color: #9090b0; margin-top: 2px; }

/* Briefing box */
.briefing-box {
    background: #12122a;
    border: 1px solid #3d3d6a;
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 16px;
}

/* Input styling */
.stTextInput > div > div > input {
    background: #1e1e35 !important;
    color: #e8e8f0 !important;
    border: 1px solid #3d3d6a !important;
    border-radius: 10px !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 8px 20px;
    font-weight: 600;
}
.stButton > button:hover { opacity: 0.9; }

/* Status badges */
.badge-critical { background: #7f1d1d; color: #fca5a5; padding: 2px 8px; border-radius: 6px; font-size: 0.8rem; }
.badge-low { background: #78350f; color: #fcd34d; padding: 2px 8px; border-radius: 6px; font-size: 0.8rem; }
.badge-healthy { background: #14532d; color: #86efac; padding: 2px 8px; border-radius: 6px; font-size: 0.8rem; }

h1, h2, h3 { color: #c4b5fd; }
hr { border-color: #2d2d4a; }
</style>
""", unsafe_allow_html=True)

# ── Load data for sidebar metrics ─────────────────────────────────────────────
@st.cache_data
def load_catalog_summary():
    df = pd.read_csv("retailmind_products.csv")
    df["gross_margin"] = (df["price"] - df["cost"]) / df["price"] * 100
    df["days_to_stockout"] = df.apply(
        lambda r: r["stock_quantity"] / r["avg_daily_sales"] if r["avg_daily_sales"] > 0 else 999,
        axis=1,
    )
    total_skus = len(df)
    critical_skus = int((df["days_to_stockout"] < 7).sum())
    avg_margin = round(df["gross_margin"].mean(), 1)
    avg_rating = round(df["avg_rating"].mean(), 2)
    return total_skus, critical_skus, avg_margin, avg_rating, df

total_skus, critical_skus, avg_margin, avg_rating, products_df = load_catalog_summary()

# ── Session state initialisation ──────────────────────────────────────────────
if "agent" not in st.session_state:
    from agent import RetailMindAgent, generate_daily_briefing
    st.session_state.agent = RetailMindAgent()
    st.session_state.chat_history = []  # list of (role, content)
    st.session_state.briefing = None
    st.session_state.briefing_generated = False

# Generate daily briefing once on startup
if not st.session_state.briefing_generated:
    with st.spinner("🔍 Generating Daily Briefing..."):
        try:
            from agent import generate_daily_briefing
            st.session_state.briefing = generate_daily_briefing(st.session_state.agent)
            st.session_state.briefing_generated = True
        except Exception as e:
            st.session_state.briefing = f"⚠️ Could not generate briefing: {e}"
            st.session_state.briefing_generated = True

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛍️ RetailMind")
    st.markdown("**StyleCraft Catalog Intelligence**")
    st.markdown("---")

    # Category filter
    st.markdown("### 📂 Category Filter")
    selected_category = st.selectbox(
        "Scope your queries",
        ["All Categories", "Tops", "Dresses", "Bottoms", "Outerwear", "Accessories"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    # Catalog Summary metrics
    st.markdown("### 📊 Catalog Summary")

    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{total_skus}</div>
        <div class="metric-label">Total SKUs</div>
    </div>
    <div class="metric-card">
        <div class="metric-value" style="color:#f87171;">{critical_skus}</div>
        <div class="metric-label">Critical Stock SKUs</div>
    </div>
    <div class="metric-card">
        <div class="metric-value">{avg_margin}%</div>
        <div class="metric-label">Avg Catalog Margin</div>
    </div>
    <div class="metric-card">
        <div class="metric-value">{avg_rating}★</div>
        <div class="metric-label">Avg Catalog Rating</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Clear chat button
    if st.button("🔄 Clear Chat & Refresh Briefing"):
        st.session_state.agent.reset()
        st.session_state.chat_history = []
        st.session_state.briefing_generated = False
        st.session_state.briefing = None
        st.rerun()

    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.75rem; color:#6060a0;'>RetailMind Analytics • Set-B Exam</div>",
        unsafe_allow_html=True,
    )

# ── Main content ──────────────────────────────────────────────────────────────
st.markdown("# 🛍️ StyleCraft Product Intelligence Agent")
st.markdown(
    f"*Currently scoped to: **{selected_category}***" if selected_category != "All Categories"
    else "*Showing all categories*"
)

# Daily Briefing
if st.session_state.briefing:
    with st.expander("📋 Daily Briefing (click to expand/collapse)", expanded=True):
        st.markdown(
            f'<div class="briefing-box">{st.session_state.briefing}</div>',
            unsafe_allow_html=True,
        )

st.markdown("---")
st.markdown("### 💬 Chat with your Catalog")

# ── Chat history display ──────────────────────────────────────────────────────
chat_container = st.container()

with chat_container:
    for role, content in st.session_state.chat_history:
        if role == "user":
            st.markdown(
                f'<div class="user-bubble">👤 {content}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="assistant-bubble">🤖 {content}</div>',
                unsafe_allow_html=True,
            )

# ── Chat input ────────────────────────────────────────────────────────────────
with st.form("chat_form", clear_on_submit=True):
    cols = st.columns([5, 1])
    with cols[0]:
        user_input = st.text_input(
            "Ask anything about the StyleCraft catalog...",
            placeholder="e.g. Which dresses are low on stock? What's the margin on winter jackets?",
            label_visibility="collapsed",
        )
    with cols[1]:
        submitted = st.form_submit_button("Send ➤")

if submitted and user_input.strip():
    # Inject category context into query if filtered
    augmented_input = user_input
    if selected_category != "All Categories":
        augmented_input = f"[Category filter: {selected_category}] {user_input}"

    # Add user message to history
    st.session_state.chat_history.append(("user", user_input))

    # Get agent response
    with st.spinner("🧠 Thinking..."):
        try:
            response = st.session_state.agent.chat(augmented_input)
        except Exception as e:
            response = f"⚠️ Error: {str(e)}\n\nPlease check your OPENAI_API_KEY in the .env file."

    st.session_state.chat_history.append(("assistant", response))
    st.rerun()

# ── Suggested prompts ─────────────────────────────────────────────────────────
if not st.session_state.chat_history:
    st.markdown("#### 💡 Try asking:")
    prompt_cols = st.columns(3)
    suggestions = [
        "Which products are about to stock out?",
        "What's the gross margin on the Velvet Party Dress?",
        "Show me the worst-rated product and why customers are unhappy",
        "Give me a performance overview of the Dresses category",
        "Which Tops have a margin below 20%?",
        "What are customers saying about SC027?",
    ]
    for i, suggestion in enumerate(suggestions):
        with prompt_cols[i % 3]:
            if st.button(suggestion, key=f"sugg_{i}"):
                st.session_state.chat_history.append(("user", suggestion))
                with st.spinner("🧠 Thinking..."):
                    try:
                        response = st.session_state.agent.chat(suggestion)
                    except Exception as e:
                        response = f"⚠️ Error: {str(e)}"
                st.session_state.chat_history.append(("assistant", response))
                st.rerun()
