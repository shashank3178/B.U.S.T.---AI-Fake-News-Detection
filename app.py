import re
import os
import time
import random
import joblib
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="B.U.S.T. — Fake News Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# PATHS & MODEL LOADING
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH        = os.path.join(BASE_DIR, "models", "fake_news_model.pkl")
VECTORIZER_PATH   = os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl")
LABEL_ENCODER_PATH = os.path.join(BASE_DIR, "models", "label_encoder.pkl")

# =====================================================
# CACHE MODEL ASSETS
# =====================================================
@st.cache_resource
def load_assets():
    model         = joblib.load(MODEL_PATH)
    vectorizer    = joblib.load(VECTORIZER_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    return model, vectorizer, label_encoder

model, vectorizer, label_encoder = load_assets()

# =====================================================
# SESSION STATE
# =====================================================
if "history" not in st.session_state:
    st.session_state.history = []

# =====================================================
# PREMIUM DARK THEME — CUSTOM CSS
# =====================================================
st.markdown(
    """
    <style>
    /* ── FONT IMPORT ─────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;0,9..40,800;1,9..40,400&family=DM+Mono:wght@400;500&display=swap');

    /* ── DESIGN TOKENS ───────────────────────────── */
    :root {
        --bg-base:        #080c14;
        --bg-surface:     #0e1421;
        --bg-elevated:    #141b2d;
        --border-subtle:  rgba(255,255,255,0.06);
        --border-default: rgba(255,255,255,0.10);
        --border-accent:  rgba(255,255,255,0.18);
        --text-primary:   #f0f4ff;
        --text-secondary: #8b98b8;
        --text-muted:     #505a72;
        --accent-blue:    #3b82f6;
        --accent-indigo:  #6366f1;
        --accent-green:   #10b981;
        --accent-red:     #ef4444;
        --accent-amber:   #f59e0b;
        --glow-blue:      rgba(59,130,246,0.20);
        --glow-indigo:    rgba(99,102,241,0.15);
        --radius-sm:      10px;
        --radius-md:      16px;
        --radius-lg:      22px;
        --radius-xl:      28px;
        --shadow-card:    0 4px 24px rgba(0,0,0,0.45), 0 1px 0 rgba(255,255,255,0.04) inset;
        --shadow-glow:    0 0 40px rgba(59,130,246,0.12);
    }

    /* ── GLOBAL RESET ────────────────────────────── */
    html, body, [class*="css"],
    .stApp, .main, .block-container {
        font-family: 'DM Sans', sans-serif !important;
        -webkit-font-smoothing: antialiased;
    }

    /* ── APP BACKGROUND ──────────────────────────── */
    .stApp {
        background:
            radial-gradient(ellipse 70% 50% at 5% 0%,   rgba(59,130,246,0.10) 0%, transparent 55%),
            radial-gradient(ellipse 55% 45% at 95% 100%, rgba(99,102,241,0.10) 0%, transparent 55%),
            var(--bg-base) !important;
        color: var(--text-primary) !important;
    }

    /* ── BLOCK CONTAINER PADDING ─────────────────── */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 3rem  !important;
        max-width: 1400px !important;
    }

    /* ── SCROLLBAR ───────────────────────────────── */
    ::-webkit-scrollbar              { width: 5px; height: 5px; }
    ::-webkit-scrollbar-track        { background: transparent; }
    ::-webkit-scrollbar-thumb        { background: rgba(255,255,255,0.10); border-radius: 99px; }
    ::-webkit-scrollbar-thumb:hover  { background: rgba(255,255,255,0.20); }

    /* ════════════════════════════════════════════════
       SIDEBAR
       ════════════════════════════════════════════════ */
    section[data-testid="stSidebar"] {
        background: var(--bg-surface) !important;
        border-right: 1px solid var(--border-subtle) !important;
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 1.5rem !important;
    }

    /* sidebar brand */
    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 0 0.2rem 1rem;
        border-bottom: 1px solid var(--border-subtle);
        margin-bottom: 1.2rem;
    }
    .sidebar-brand-icon {
        font-size: 1.6rem;
        line-height: 1;
    }
    .sidebar-brand-text {
        font-size: 1.25rem;
        font-weight: 800;
        color: var(--text-primary);
        letter-spacing: -0.4px;
    }
    .sidebar-brand-sub {
        font-size: 0.7rem;
        color: var(--text-secondary);
        letter-spacing: 0.5px;
        text-transform: uppercase;
        font-weight: 500;
    }

    /* sidebar section headers */
    .sidebar-section-label {
        font-size: 0.68rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        color: var(--text-muted);
        padding: 0.9rem 0 0.4rem;
    }

    /* sidebar system chip list */
    .chip-list {
        display: flex;
        flex-direction: column;
        gap: 6px;
        padding-bottom: 1rem;
        border-bottom: 1px solid var(--border-subtle);
    }
    .chip-item {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 0.8rem;
        color: var(--text-secondary);
        padding: 6px 10px;
        border-radius: var(--radius-sm);
        background: rgba(255,255,255,0.03);
        border: 1px solid var(--border-subtle);
    }
    .chip-dot {
        width: 6px; height: 6px;
        border-radius: 50%;
        background: var(--accent-blue);
        flex-shrink: 0;
        box-shadow: 0 0 6px var(--accent-blue);
    }

    /* sidebar tip box */
    .tip-box {
        background: rgba(59,130,246,0.08);
        border: 1px solid rgba(59,130,246,0.20);
        border-radius: var(--radius-md);
        padding: 0.8rem 1rem;
        font-size: 0.8rem;
        color: #93b4f0;
        line-height: 1.6;
        margin-bottom: 1rem;
    }

    /* sidebar status badge */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        padding: 6px 12px;
        border-radius: 99px;
        background: rgba(16,185,129,0.12);
        border: 1px solid rgba(16,185,129,0.28);
        color: #34d399;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .status-dot {
        width: 7px; height: 7px;
        border-radius: 50%;
        background: #34d399;
        animation: pulse-green 2s infinite;
    }
    @keyframes pulse-green {
        0%,100% { opacity:1; box-shadow: 0 0 0 0 rgba(52,211,153,0.4); }
        50%      { opacity:.85; box-shadow: 0 0 0 6px rgba(52,211,153,0); }
    }

    /* ── Hide streamlit's default sidebar markdown headings ─ */
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 { display: none !important; }
    section[data-testid="stSidebar"] .stCaption { display: none !important; }
    section[data-testid="stSidebar"] hr { display: none !important; }
    section[data-testid="stSidebar"] .stInfo { display: none !important; }
    section[data-testid="stSidebar"] .stSuccess { display: none !important; }

    /* ════════════════════════════════════════════════
       HERO SECTION
       ════════════════════════════════════════════════ */
    .hero-box {
        position: relative;
        padding: 2.8rem 3rem;
        border-radius: var(--radius-xl);
        background: var(--bg-elevated);
        border: 1px solid var(--border-default);
        margin-bottom: 1.8rem;
        overflow: hidden;
        box-shadow: var(--shadow-card);
    }
    .hero-box::before {
        content: "";
        position: absolute;
        width: 380px; height: 380px;
        background: radial-gradient(circle, rgba(59,130,246,0.18), transparent 70%);
        top: -150px; right: -120px;
        pointer-events: none;
    }
    .hero-box::after {
        content: "";
        position: absolute;
        width: 220px; height: 220px;
        background: radial-gradient(circle, rgba(99,102,241,0.14), transparent 70%);
        bottom: -80px; left: 20%;
        pointer-events: none;
    }
    .hero-eyebrow {
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: var(--accent-blue);
        margin-bottom: 0.5rem;
    }
    .hero-title {
        font-size: 3.4rem;
        font-weight: 800;
        line-height: 1.05;
        letter-spacing: -1.5px;
        color: var(--text-primary);
        margin: 0 0 0.6rem;
    }
    .hero-title span {
        background: linear-gradient(120deg, #60a5fa, #818cf8 60%, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .hero-subtitle {
        font-size: 1rem;
        color: var(--text-secondary);
        max-width: 520px;
        line-height: 1.6;
        font-weight: 400;
    }
    .hero-badges {
        display: flex;
        gap: 8px;
        margin-top: 1.4rem;
        flex-wrap: wrap;
    }
    .hero-badge {
        padding: 4px 12px;
        border-radius: 99px;
        border: 1px solid var(--border-default);
        font-size: 0.72rem;
        font-weight: 600;
        color: var(--text-secondary);
        background: rgba(255,255,255,0.04);
        letter-spacing: 0.2px;
    }

    /* ════════════════════════════════════════════════
       GLASS CARD
       ════════════════════════════════════════════════ */
    .glass-card {
        background: var(--bg-elevated);
        border: 1px solid var(--border-default);
        border-radius: var(--radius-lg);
        padding: 1.6rem;
        margin-bottom: 1.2rem;
        box-shadow: var(--shadow-card);
        transition: border-color 0.2s ease;
    }
    .glass-card:hover {
        border-color: var(--border-accent);
    }
    .glass-card-title {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 1.4px;
        text-transform: uppercase;
        color: var(--text-muted);
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .glass-card-title::after {
        content: "";
        flex: 1;
        height: 1px;
        background: var(--border-subtle);
    }

    /* ════════════════════════════════════════════════
       TEXT AREA
       ════════════════════════════════════════════════ */
    .stTextArea label { display: none !important; }

    .stTextArea textarea {
        background: rgba(255,255,255,0.96) !important;
        color: #0d1526 !important;
        border-radius: var(--radius-md) !important;
        border: 1.5px solid rgba(99,102,241,0.30) !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.96rem !important;
        line-height: 1.75 !important;
        padding: 1rem 1.1rem !important;
        min-height: 240px !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
        resize: vertical !important;
    }
    .stTextArea textarea:focus {
        border-color: var(--accent-indigo) !important;
        box-shadow: 0 0 0 3px rgba(99,102,241,0.18) !important;
        outline: none !important;
    }
    .stTextArea textarea::placeholder { color: #9098b0 !important; }

    /* ════════════════════════════════════════════════
       BUTTONS
       ════════════════════════════════════════════════ */
    .stButton > button {
        width: 100% !important;
        border: none !important;
        border-radius: var(--radius-md) !important;
        padding: 0.78rem 1.2rem !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.9rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.2px !important;
        color: #fff !important;
        background: linear-gradient(135deg, #3b82f6 0%, #6366f1 100%) !important;
        transition: opacity 0.2s ease, transform 0.15s ease, box-shadow 0.2s ease !important;
        box-shadow: 0 4px 14px rgba(99,102,241,0.28) !important;
    }
    .stButton > button:hover {
        opacity: 0.92 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 8px 24px rgba(99,102,241,0.38) !important;
    }
    .stButton > button:active {
        transform: translateY(0) !important;
        opacity: 1 !important;
    }

    /* secondary button — Clear Session (2nd stButton in a row) */
    .stButton + .stButton > button {
        background: transparent !important;
        border: 1.5px solid var(--border-default) !important;
        color: var(--text-secondary) !important;
        box-shadow: none !important;
    }
    .stButton + .stButton > button:hover {
        border-color: var(--border-accent) !important;
        color: var(--text-primary) !important;
        background: rgba(255,255,255,0.04) !important;
        box-shadow: none !important;
    }

    /* ════════════════════════════════════════════════
       METRIC CARDS
       ════════════════════════════════════════════════ */
    div[data-testid="metric-container"] {
        background: var(--bg-elevated) !important;
        border: 1px solid var(--border-default) !important;
        border-radius: var(--radius-md) !important;
        padding: 1rem 1.1rem !important;
        box-shadow: var(--shadow-card) !important;
    }
    div[data-testid="metric-container"] label {
        font-size: 0.68rem !important;
        font-weight: 700 !important;
        letter-spacing: 1px !important;
        text-transform: uppercase !important;
        color: var(--text-muted) !important;
    }
    div[data-testid="metric-container"] [data-testid="stMetricValue"] {
        font-size: 1.55rem !important;
        font-weight: 800 !important;
        color: var(--text-primary) !important;
        letter-spacing: -0.5px !important;
        line-height: 1.2 !important;
    }

    /* ════════════════════════════════════════════════
       PREDICTION RESULT CARDS
       ════════════════════════════════════════════════ */
    .prediction-card {
        position: relative;
        padding: 1.8rem 2rem;
        border-radius: var(--radius-lg);
        display: flex;
        align-items: center;
        gap: 1.2rem;
        margin: 1.2rem 0;
        overflow: hidden;
    }
    .prediction-card::before {
        content: "";
        position: absolute;
        inset: 0;
        border-radius: inherit;
        pointer-events: none;
    }
    .pred-icon {
        font-size: 2.4rem;
        line-height: 1;
        flex-shrink: 0;
    }
    .pred-label {
        font-size: 1.5rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        line-height: 1.1;
    }
    .pred-sub {
        font-size: 0.82rem;
        opacity: 0.8;
        margin-top: 3px;
        font-weight: 500;
    }

    .fake-card {
        background: rgba(239,68,68,0.08);
        border: 1px solid rgba(239,68,68,0.35);
        color: #fca5a5;
    }
    .fake-card .pred-label { color: #f87171; }
    .fake-card::before {
        background: radial-gradient(ellipse 80% 80% at 0% 50%, rgba(239,68,68,0.12), transparent);
    }

    .real-card {
        background: rgba(16,185,129,0.07);
        border: 1px solid rgba(16,185,129,0.30);
        color: #6ee7b7;
    }
    .real-card .pred-label { color: #34d399; }
    .real-card::before {
        background: radial-gradient(ellipse 80% 80% at 0% 50%, rgba(16,185,129,0.10), transparent);
    }

    /* ════════════════════════════════════════════════
       PROGRESS BAR
       ════════════════════════════════════════════════ */
    .stProgress > div > div {
        background: linear-gradient(90deg, #3b82f6, #6366f1) !important;
        border-radius: 99px !important;
        height: 6px !important;
    }
    .stProgress > div {
        background: rgba(255,255,255,0.06) !important;
        border-radius: 99px !important;
        height: 6px !important;
    }

    /* ════════════════════════════════════════════════
       EXPANDER
       ════════════════════════════════════════════════ */
    .streamlit-expanderHeader {
        background: var(--bg-elevated) !important;
        border: 1px solid var(--border-default) !important;
        border-radius: var(--radius-md) !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        color: var(--text-secondary) !important;
        padding: 0.8rem 1rem !important;
        transition: border-color 0.2s ease !important;
    }
    .streamlit-expanderHeader:hover {
        border-color: var(--border-accent) !important;
        color: var(--text-primary) !important;
    }
    .streamlit-expanderContent {
        background: var(--bg-elevated) !important;
        border: 1px solid var(--border-default) !important;
        border-top: none !important;
        border-radius: 0 0 var(--radius-md) var(--radius-md) !important;
        padding: 1.2rem !important;
    }

    /* ════════════════════════════════════════════════
       CODE BLOCK
       ════════════════════════════════════════════════ */
    .stCode, code {
        font-family: 'DM Mono', monospace !important;
        font-size: 0.82rem !important;
        border-radius: var(--radius-sm) !important;
    }
    [data-testid="stCode"] {
        background: rgba(0,0,0,0.40) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: var(--radius-md) !important;
    }

    /* ════════════════════════════════════════════════
       DATAFRAME / TABLE
       ════════════════════════════════════════════════ */
    [data-testid="stDataFrame"] {
        border-radius: var(--radius-md) !important;
        border: 1px solid var(--border-default) !important;
        overflow: hidden !important;
    }
    .dvn-scroller {
        border-radius: var(--radius-md) !important;
    }

    /* ════════════════════════════════════════════════
       SPINNER
       ════════════════════════════════════════════════ */
    [data-testid="stSpinner"] {
        color: var(--accent-indigo) !important;
    }

    /* ════════════════════════════════════════════════
       DIVIDER
       ════════════════════════════════════════════════ */
    hr {
        border: none !important;
        border-top: 1px solid var(--border-subtle) !important;
        margin: 1rem 0 !important;
    }

    /* ════════════════════════════════════════════════
       SECTION HEADERS (inline)
       ════════════════════════════════════════════════ */
    .section-label {
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: var(--text-muted);
        margin: 1.6rem 0 0.8rem;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .section-label::after {
        content: "";
        flex: 1;
        height: 1px;
        background: var(--border-subtle);
    }

    /* ════════════════════════════════════════════════
       STAT PILL (sidebar metric replacement)
       ════════════════════════════════════════════════ */
    .stat-pill {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.7rem 1rem;
        border-radius: var(--radius-md);
        background: rgba(255,255,255,0.03);
        border: 1px solid var(--border-subtle);
        margin-top: 0.6rem;
    }
    .stat-pill-label {
        font-size: 0.78rem;
        color: var(--text-secondary);
        font-weight: 500;
    }
    .stat-pill-value {
        font-size: 0.95rem;
        font-weight: 800;
        color: var(--text-primary);
        letter-spacing: -0.2px;
    }

    /* ════════════════════════════════════════════════
       HISTORY SECTION HEADER
       ════════════════════════════════════════════════ */
    .history-header {
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: var(--text-muted);
        margin: 1.8rem 0 0.8rem;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .history-header::after {
        content: "";
        flex: 1;
        height: 1px;
        background: var(--border-subtle);
    }

    /* ════════════════════════════════════════════════
       FOOTER
       ════════════════════════════════════════════════ */
    .footer {
        text-align: center;
        padding: 2.5rem 0 1rem;
        border-top: 1px solid var(--border-subtle);
        margin-top: 2rem;
        color: var(--text-muted);
        font-size: 0.78rem;
        line-height: 1.8;
        letter-spacing: 0.2px;
    }
    .footer strong {
        color: var(--text-secondary);
        font-weight: 600;
    }

    /* ════════════════════════════════════════════════
       WARNING / ALERTS
       ════════════════════════════════════════════════ */
    .stAlert {
        border-radius: var(--radius-md) !important;
        border-width: 1px !important;
        font-size: 0.88rem !important;
    }

    /* ════════════════════════════════════════════════
       HIDE STREAMLIT BRANDING
       ════════════════════════════════════════════════ */
    #MainMenu, footer, header { visibility: hidden !important; }

    </style>
    """,
    unsafe_allow_html=True
)

# =====================================================
# TEXT PREPROCESSING  (unchanged)
# =====================================================
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# =====================================================
# PREDICTION FUNCTION  (unchanged)
# =====================================================
def predict_news(text):
    processed    = preprocess_text(text)
    vectorized   = vectorizer.transform([processed])
    prediction   = model.predict(vectorized)[0]
    probabilities = model.predict_proba(vectorized)[0]

    fake_prob = float(probabilities[0])
    real_prob = float(probabilities[1])
    label     = "REAL" if prediction == 1 else "FAKE"
    confidence = real_prob if label == "REAL" else fake_prob

    return {
        "prediction": label,
        "confidence": confidence,
        "fake_prob":  fake_prob,
        "real_prob":  real_prob,
        "cleaned_text": processed
    }

# =====================================================
# RISK LEVEL  (unchanged)
# =====================================================
def get_risk_level(fake_probability):
    if fake_probability >= 0.85:
        return "🔴 Extreme Risk"
    elif fake_probability >= 0.65:
        return "🟠 High Risk"
    elif fake_probability >= 0.45:
        return "🟡 Moderate Risk"
    return "🟢 Low Risk"

# =====================================================
# SIDEBAR — REDESIGNED
# =====================================================
with st.sidebar:

    # ── Brand ─────────────────────────────────────
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-brand-icon">🛡️</div>
            <div>
                <div class="sidebar-brand-text">B.U.S.T.</div>
                <div class="sidebar-brand-sub">Bogus Uncovering System</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ── System Overview chips ──────────────────────
    st.markdown('<div class="sidebar-section-label">System Components</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="chip-list">
            <div class="chip-item"><div class="chip-dot"></div> TF-IDF Vectorization</div>
            <div class="chip-item"><div class="chip-dot"></div> Scikit-Learn Classifier</div>
            <div class="chip-item"><div class="chip-dot"></div> Probability Inference</div>
            <div class="chip-item"><div class="chip-dot"></div> Real / Fake Detection</div>
            <div class="chip-item"><div class="chip-dot"></div> NLP Cleaning Pipeline</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ── Detection tips ────────────────────────────
    st.markdown('<div class="sidebar-section-label">Detection Tip</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="tip-box">
            Suspicious news often features emotional language, excessive
            capitalisation, conspiracy framing, or unverifiable claims.
        </div>
        """,
        unsafe_allow_html=True
    )

    # ── Model status ──────────────────────────────
    st.markdown('<div class="sidebar-section-label">Model Status</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="status-badge"><div class="status-dot"></div> Model Loaded</div>',
        unsafe_allow_html=True
    )

    # ── Session stat ──────────────────────────────
    st.markdown(
        f"""
        <div class="stat-pill">
            <span class="stat-pill-label">Session Analyses</span>
            <span class="stat-pill-value">{len(st.session_state.history)}</span>
        </div>
        """,
        unsafe_allow_html=True
    )

# =====================================================
# HERO SECTION
# =====================================================
st.markdown(
    """
    <div class="hero-box">
        <div class="hero-eyebrow">Bogus Text Uncovering System</div>
        <div class="hero-title">🛡️ <span>B.U.S.T.</span></div>
        <div class="hero-subtitle">
            Linguistic credibility analysis and fake news detection
            powered by TF-IDF vectorisation and ML classification.
        </div>
        <div class="hero-badges">
            <span class="hero-badge">NLP Pipeline</span>
            <span class="hero-badge">TF-IDF</span>
            <span class="hero-badge">Scikit-Learn</span>
            <span class="hero-badge">Probability Scoring</span>
            <span class="hero-badge">Binary Classification</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# =====================================================
# MAIN LAYOUT
# =====================================================
left_col, right_col = st.columns([2.1, 1], gap="large")

# ─────────────────────────────────────────────────────
# LEFT COLUMN — INPUT
# ─────────────────────────────────────────────────────
with left_col:

    st.markdown(
        '<div class="glass-card">'
        '<div class="glass-card-title">📰 News Content Analyzer</div>',
        unsafe_allow_html=True
    )

    news_text = st.text_area(
        "Paste a news article, headline, or social media post",
        placeholder="Paste a news article, headline, or social media post… "
                    "e.g. Scientists discover hidden government signals inside satellite transmissions.",
        label_visibility="collapsed"
    )

    c1, c2 = st.columns(2, gap="small")

    with c1:
        analyze_button = st.button("🔍  Run Analysis")

    with c2:
        clear_button = st.button("🧹  Clear Session")

    st.markdown('</div>', unsafe_allow_html=True)

    if clear_button:
        st.session_state.history = []
        st.rerun()

# ─────────────────────────────────────────────────────
# RIGHT COLUMN — QUICK STATS
# ─────────────────────────────────────────────────────
with right_col:

    # Quick Statistics card
    st.markdown(
        '<div class="glass-card">'
        '<div class="glass-card-title">📈 Quick Statistics</div>',
        unsafe_allow_html=True
    )

    word_count  = len(news_text.split()) if news_text else 0
    char_count  = len(news_text)          if news_text else 0
    reading_time = max(1, word_count // 200)

    st.metric("Word Count",        word_count)
    st.metric("Character Count",   char_count)
    st.metric("Est. Read Time",    f"{reading_time} min")

    st.markdown('</div>', unsafe_allow_html=True)

    # Sample inputs card
    st.markdown(
        '<div class="glass-card">'
        '<div class="glass-card-title">⚡ Sample Inputs</div>',
        unsafe_allow_html=True
    )

    samples = [
        "Breaking: Scientists confirm Earth is flat after secret NASA investigation.",
        "Government announces new renewable energy investment program.",
        "Celebrity cloned by hidden underground research facility."
    ]

    st.code(random.choice(samples), language="text")

    st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# ANALYSIS EXECUTION
# =====================================================
if analyze_button:

    if not news_text.strip():
        st.warning("⚠️ Please enter text for analysis.")

    elif len(news_text.split()) < 4:
        st.warning("⚠️ Please provide slightly more content for reliable analysis.")

    else:

        with st.spinner("Analysing linguistic patterns and semantic signals…"):
            time.sleep(1)
            result = predict_news(news_text)

        st.session_state.history.append({
            "timestamp":  datetime.now().strftime("%H:%M:%S"),
            "prediction": result["prediction"],
            "confidence": round(result["confidence"] * 100, 2)
        })

        # ── PREDICTION RESULT CARD ─────────────────
        if result["prediction"] == "FAKE":
            st.markdown(
                f"""
                <div class="prediction-card fake-card">
                    <div class="pred-icon">🚨</div>
                    <div>
                        <div class="pred-label">Fake News Detected</div>
                        <div class="pred-sub">Risk Level: {get_risk_level(result['fake_prob'])}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div class="prediction-card real-card">
                    <div class="pred-icon">✅</div>
                    <div>
                        <div class="pred-label">Content Appears Authentic</div>
                        <div class="pred-sub">Credibility Level: High</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # ── CONFIDENCE BAR ─────────────────────────
        st.progress(float(result["confidence"]))

        # ── METRICS ROW ────────────────────────────
        m1, m2, m3, m4 = st.columns(4, gap="small")

        m1.metric("Confidence",   f"{result['confidence'] * 100:.1f}%")
        m2.metric("Authenticity", f"{result['real_prob']  * 100:.1f}%")
        m3.metric("Deception",    f"{result['fake_prob']  * 100:.1f}%")
        m4.metric("Risk Level",   get_risk_level(result['fake_prob']).split(" ", 1)[1])

        # ── CHARTS ─────────────────────────────────
        st.markdown('<div class="section-label">Probability Distribution</div>', unsafe_allow_html=True)

        chart_col1, chart_col2 = st.columns([1.4, 1], gap="large")

        with chart_col1:

            fig = go.Figure(
                data=[
                    go.Bar(
                        x=["FAKE", "REAL"],
                        y=[result["fake_prob"], result["real_prob"]],
                        text=[
                            f"{result['fake_prob'] * 100:.1f}%",
                            f"{result['real_prob'] * 100:.1f}%"
                        ],
                        textposition="outside",
                        textfont=dict(family="DM Sans", size=13, color="#f0f4ff"),
                        marker=dict(
                            color=["#ef4444", "#10b981"],
                            opacity=0.90,
                            line=dict(width=0)
                        ),
                        width=[0.42, 0.42]
                    )
                ]
            )

            fig.update_layout(
                title=dict(
                    text="Prediction Probabilities",
                    font=dict(family="DM Sans", size=13, color="#8b98b8"),
                    x=0, xanchor="left"
                ),
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(
                    range=[0, 1.18],
                    tickformat=".0%",
                    tickfont=dict(family="DM Sans", size=11, color="#505a72"),
                    gridcolor="rgba(255,255,255,0.04)",
                    zeroline=False
                ),
                xaxis=dict(
                    tickfont=dict(family="DM Sans", size=12, color="#8b98b8")
                ),
                height=360,
                margin=dict(t=50, l=0, r=0, b=0),
                showlegend=False
            )

            st.plotly_chart(fig, use_container_width=True)

        with chart_col2:

            gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=result["fake_prob"] * 100,
                    number=dict(
                        suffix="%",
                        font=dict(family="DM Sans", size=32, color="#f0f4ff")
                    ),
                    title=dict(
                        text="Fake News Probability",
                        font=dict(family="DM Sans", size=13, color="#8b98b8")
                    ),
                    gauge={
                        "axis": {
                            "range": [0, 100],
                            "tickwidth": 1,
                            "tickcolor": "rgba(255,255,255,0.12)",
                            "tickfont": dict(family="DM Sans", size=10, color="#505a72")
                        },
                        "bar": {"color": "#ef4444", "thickness": 0.22},
                        "bgcolor": "rgba(0,0,0,0)",
                        "borderwidth": 0,
                        "steps": [
                            {"range": [0,  40],  "color": "rgba(16,185,129,0.15)"},
                            {"range": [40, 70],  "color": "rgba(245,158,11,0.15)"},
                            {"range": [70, 100], "color": "rgba(239,68,68,0.18)"}
                        ],
                        "threshold": {
                            "line": {"color": "#f87171", "width": 2},
                            "thickness": 0.75,
                            "value": result["fake_prob"] * 100
                        }
                    }
                )
            )

            gauge.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="DM Sans"),
                height=360,
                margin=dict(t=50, l=10, r=10, b=0)
            )

            st.plotly_chart(gauge, use_container_width=True)

        # ── DETAILED ANALYSIS EXPANDER ──────────────
        with st.expander("🧪 Detailed Analysis"):

            st.markdown("**Cleaned Text**")
            st.code(result["cleaned_text"][:1000], language="text")

            st.markdown("**Prediction JSON**")
            st.json(result)

        # ── SESSION HISTORY ─────────────────────────
        if st.session_state.history:

            st.markdown('<div class="history-header">🕓 Session Analysis History</div>', unsafe_allow_html=True)

            st.dataframe(
                st.session_state.history,
                use_container_width=True,
                hide_index=True
            )

# =====================================================
# ABOUT SECTION
# =====================================================
with st.expander("ℹ️ About The System"):

    st.markdown(
        """
        ### B.U.S.T. — Bogus Text Uncovering System

        B.U.S.T. is an NLP-powered misinformation detection system built using:

        - TF-IDF vectorisation
        - Classical machine learning classification
        - Text preprocessing pipelines
        - Probability-based inference
        - Streamlit visualisation interface

        ### Pipeline

        ```
        Raw Text
            ↓
        NLP Cleaning
            ↓
        TF-IDF Vectorisation
            ↓
        Trained ML Model
            ↓
        Fake / Real Prediction
        ```

        ### Important Note

        Predictions are probabilistic estimates and should not replace professional fact-checking.
        """
    )

# =====================================================
# FOOTER
# =====================================================
st.markdown(
    f"""
    <div class="footer">
        <strong>🛡️ B.U.S.T.</strong> — AI-Powered Bogus Text Uncovering System<br>
        Built with Streamlit · Plotly · Scikit-Learn · TF-IDF NLP Pipelines<br>
        Session Timestamp: {datetime.now().strftime('%d %b %Y · %I:%M %p')}
    </div>
    """,
    unsafe_allow_html=True
)
