"""
CarPrice AI — Professional Streamlit App
No sidebar. Full custom layout with permanent left panel.
Run: streamlit run car_prediction_app.py
"""

import os
import base64
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from PIL import Image

# ══════════════════════════════════════════════════════════════════
#  PAGE CONFIG  — must be first Streamlit call
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="CarPrice AI",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed",  # hide sidebar completely
)

# ══════════════════════════════════════════════════════════════════
#  GLOBAL CSS
# ══════════════════════════════════════════════════════════════════
st.markdown(
    """
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@400;600;700;800&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Variables ── */
:root{
  --bg:       #08090D;
  --panel:    #0E1018;
  --surface:  #141720;
  --border:   #1F2232;
  --accent:   #C8102E;
  --accent2:  #FF6B35;
  --gold:     #F5C518;
  --teal:     #00C9A7;
  --txt:      #E4E6F0;
  --muted:    #5A6080;
  --font-hd:  'Barlow Condensed', sans-serif;
  --font-bd:  'DM Sans', sans-serif;
}

/* ── Reset / global ── */
*,*::before,*::after{box-sizing:border-box;}
html,body,[class*="css"]{
  font-family:var(--font-bd);
  background:var(--bg) !important;
  color:var(--txt);
}
.stApp{background:var(--bg) !important;}

.hero-wrap{
  position:relative;
  border-radius:24px;
  overflow:hidden;
  min-height:360px;
  margin-bottom:1.75rem;
  background-size:cover;
  background-position:center;
  display:grid;
  place-items:center;
  box-shadow:0 32px 80px rgba(0,0,0,.42);
}
.hero-wrap::before{
  content:"";
  position:absolute;
  inset:0;
  background:linear-gradient(180deg,rgba(8,9,13,.28),rgba(8,9,13,.92));
  backdrop-filter:blur(4px);
}
.hero-content{
  position:relative;
  z-index:1;
  width:100%;
  max-width:980px;
  padding:2.5rem 2.25rem;
}
.hero-title{
  font-family:var(--font-hd);
  font-size:3.4rem;
  font-weight:800;
  line-height:1.02;
  letter-spacing:-0.03em;
  color:#fff;
  margin:0;
}
.hero-copy{
  margin:1rem 0 0;
  font-size:1.05rem;
  line-height:1.75;
  color:rgba(228,230,240,.88);
  max-width:760px;
}
.hero-chip{
  display:inline-flex;
  gap:0.55rem;
  align-items:center;
  padding:0.8rem 1.05rem;
  border-radius:999px;
  background:rgba(255,255,255,.08);
  border:1px solid rgba(255,255,255,.14);
  color:#fff;
  font-size:0.78rem;
  font-weight:700;
  text-transform:uppercase;
  letter-spacing:0.12em;
  margin-top:1.35rem;
}
.hero-actions{
  display:flex;
  flex-wrap:wrap;
  gap:0.75rem;
  margin-top:1.75rem;
}
.hero-btn{
  display:inline-flex;
  align-items:center;
  justify-content:center;
  min-width:180px;
  padding:0.95rem 1.45rem;
  border-radius:999px;
  font-family:var(--font-hd);
  font-weight:700;
  letter-spacing:0.1em;
  text-transform:uppercase;
  text-decoration:none;
}
.hero-btn.primary{
  background:linear-gradient(135deg,var(--accent),var(--accent2));
  color:#fff;
}
.hero-btn.secondary{
  background:rgba(255,255,255,.08);
  color:#fff;
  border:1px solid rgba(255,255,255,.14);
}
.hero-cards{
  display:grid;
  grid-template-columns:repeat(3,minmax(0,1fr));
  gap:1rem;
  margin-top:2rem;
}
.hero-card{
  background:rgba(255,255,255,.08);
  border:1px solid rgba(255,255,255,.1);
  border-radius:16px;
  padding:1.1rem 1rem;
}
.hero-card .label{
  display:block;
  font-size:0.7rem;
  letter-spacing:0.12em;
  text-transform:uppercase;
  color:rgba(228,230,240,.7);
}
.hero-card .value{
  margin-top:0.55rem;
  font-family:var(--font-hd);
  font-size:1.7rem;
  color:#fff;
}

/* ── Kill default padding ── */
.block-container{
  padding:0 !important;
  max-width:100% !important;
}

/* ── Hide Streamlit chrome & sidebar arrow ── */
#MainMenu,footer,header,
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapseButton"],
button[kind="header"]{display:none !important;}
[data-testid="stSidebar"]{display:none !important;}

/* ══════════  TOP NAV BAR  ══════════ */
.topbar{
  display:flex;
  align-items:center;
  justify-content:space-between;
  background:var(--panel);
  border-bottom:1px solid var(--border);
  padding:0 2rem;
  height:58px;
}
.topbar-logo{
  font-family:var(--font-hd);
  font-size:1.55rem;
  font-weight:800;
  letter-spacing:0.12em;
  text-transform:uppercase;
  color:#fff;
}
.topbar-logo span{color:var(--accent);}
.topbar-nav{display:flex;gap:0.25rem;}
.topbar-nav a{
  font-size:0.72rem;
  font-weight:600;
  letter-spacing:0.12em;
  text-transform:uppercase;
  color:var(--muted);
  text-decoration:none;
  padding:0.35rem 0.85rem;
  border-radius:6px;
}
.topbar-badge{
  font-size:0.65rem;
  font-weight:600;
  letter-spacing:0.1em;
  text-transform:uppercase;
  background:rgba(200,16,46,.12);
  border:1px solid rgba(200,16,46,.35);
  color:var(--accent);
  padding:0.28rem 0.75rem;
  border-radius:20px;
}

/* ══════════  KPI GRID  ══════════ */
.kpi-grid{
  display:grid;
  grid-template-columns:repeat(4,1fr);
  gap:0.85rem;
  margin-bottom:1.75rem;
}
.kpi-card{
  background:var(--surface);
  border:1px solid var(--border);
  border-radius:10px;
  padding:1.1rem 1.2rem;
  position:relative;
  overflow:hidden;
}
.kpi-card::after{
  content:"";
  position:absolute;
  top:0;left:0;right:0;
  height:2px;
  background:linear-gradient(90deg,var(--accent),transparent);
}
.kpi-label{
  font-size:0.62rem;
  font-weight:600;
  letter-spacing:0.14em;
  text-transform:uppercase;
  color:var(--muted);
  margin-bottom:0.35rem;
}
.kpi-value{
  font-family:var(--font-hd);
  font-size:1.75rem;
  font-weight:700;
  color:#fff;
  line-height:1;
}
.kpi-sub{font-size:0.64rem;color:var(--teal);margin-top:0.3rem;}

/* ══════════  PANEL SECTION LABEL  ══════════ */
.panel-section-label{
  font-family:var(--font-hd);
  font-size:0.72rem;
  font-weight:700;
  letter-spacing:0.18em;
  text-transform:uppercase;
  color:var(--muted);
  padding-bottom:0.6rem;
  border-bottom:1px solid var(--border);
  margin-bottom:0.75rem;
}

/* ══════════  SECTION HEADER  ══════════ */
.sec-hd{
  display:flex;
  align-items:center;
  gap:0.6rem;
  font-family:var(--font-hd);
  font-size:1rem;
  font-weight:700;
  letter-spacing:0.1em;
  text-transform:uppercase;
  color:var(--txt);
  margin:1.5rem 0 1rem;
}
.sec-hd::before{
  content:"";
  display:block;
  width:3px;
  height:18px;
  background:var(--accent);
  border-radius:2px;
  flex-shrink:0;
}

/* ══════════  PREDICT BOX  ══════════ */
.pred-wrap{
  background:linear-gradient(135deg,#0C0A10 0%,#10080C 60%,#080C14 100%);
  border:1px solid rgba(200,16,46,.4);
  border-radius:14px;
  padding:2.25rem 2rem 1.75rem;
  text-align:center;
  box-shadow:0 0 60px rgba(200,16,46,.07);
  margin-bottom:1.75rem;
}
.pred-eyebrow{
  font-size:0.65rem;
  font-weight:700;
  letter-spacing:0.22em;
  text-transform:uppercase;
  color:var(--muted);
  margin-bottom:0.6rem;
}
.pred-price{
  font-family:var(--font-hd);
  font-size:4.5rem;
  font-weight:800;
  color:var(--accent);
  line-height:1;
  letter-spacing:0.02em;
}
.pred-note{font-size:0.68rem;color:var(--muted);margin-top:0.5rem;}
.pred-sub-grid{
  display:flex;
  justify-content:center;
  gap:2.5rem;
  margin-top:1.4rem;
  padding-top:1.4rem;
  border-top:1px solid var(--border);
}
.pred-sub-item{text-align:center;}
.pred-sub-lbl{
  font-size:0.62rem;
  font-weight:600;
  letter-spacing:0.12em;
  text-transform:uppercase;
  color:var(--muted);
}
.pred-sub-val{
  font-family:var(--font-hd);
  font-size:1.6rem;
  font-weight:700;
  color:var(--gold);
}
.pred-placeholder{
  text-align:center;
  padding:3.5rem 1rem;
  color:var(--muted);
}
.pred-placeholder .icon{font-size:2.8rem;margin-bottom:0.75rem;}
.pred-placeholder p{
  font-family:var(--font-hd);
  font-size:1rem;
  font-weight:600;
  letter-spacing:0.1em;
  text-transform:uppercase;
  line-height:1.6;
}

/* ══════════  SPEC TABLE  ══════════ */
.spec-table{
  width:100%;
  border-collapse:collapse;
  font-size:0.82rem;
}
.spec-table td{
  padding:0.5rem 0.6rem;
  border-bottom:1px solid var(--border);
  color:var(--txt);
}
.spec-table td:first-child{
  color:var(--muted);
  font-size:0.72rem;
  font-weight:600;
  letter-spacing:0.1em;
  text-transform:uppercase;
  width:45%;
}
.spec-table td:last-child{font-weight:600;}

/* ══════════  SCORE BARS  ══════════ */
.score-bar-wrap{margin-bottom:0.9rem;}
.score-bar-label{
  display:flex;
  justify-content:space-between;
  font-size:0.7rem;
  color:var(--muted);
  margin-bottom:0.3rem;
  letter-spacing:0.06em;
  text-transform:uppercase;
}
.score-bar-track{
  background:var(--border);
  border-radius:4px;
  height:6px;
  overflow:hidden;
}
.score-bar-fill{
  height:100%;
  border-radius:4px;
  background:linear-gradient(90deg,var(--accent),var(--accent2));
}

/* ══════════  STREAMLIT WIDGET OVERRIDES  ══════════ */
[data-testid="stSlider"] label{
  font-size:0.72rem !important;
  font-weight:600 !important;
  letter-spacing:0.1em !important;
  text-transform:uppercase !important;
  color:var(--muted) !important;
}
[data-testid="stSlider"] > div > div{background:var(--border) !important;}
[data-testid="stSlider"] > div > div > div > div{background:var(--accent) !important;}

.stRadio label{
  font-size:0.72rem !important;
  color:var(--muted) !important;
  letter-spacing:0.08em !important;
}
.stRadio > div{gap:0.4rem !important;}

[data-testid="stFileUploader"]{
  border:1px dashed var(--border) !important;
  border-radius:10px !important;
  padding:0.5rem !important;
  background:var(--surface) !important;
}

.stButton > button{
  width:100% !important;
  background:linear-gradient(135deg,var(--accent),#9B0B22) !important;
  color:#fff !important;
  border:none !important;
  border-radius:9px !important;
  font-family:var(--font-hd) !important;
  font-size:1.05rem !important;
  font-weight:800 !important;
  letter-spacing:0.15em !important;
  text-transform:uppercase !important;
  padding:0.8rem !important;
  box-shadow:0 4px 22px rgba(200,16,46,.3) !important;
  transition:all .2s !important;
}
.stButton > button:hover{
  box-shadow:0 6px 30px rgba(200,16,46,.55) !important;
  transform:translateY(-1px) !important;
}

/* Tabs */
[data-baseweb="tab-list"]{
  background:var(--surface) !important;
  border-radius:10px 10px 0 0 !important;
  border:1px solid var(--border) !important;
  border-bottom:none !important;
  gap:0 !important;
  padding:0 !important;
}
[data-baseweb="tab"]{
  font-family:var(--font-hd) !important;
  font-size:0.82rem !important;
  font-weight:700 !important;
  letter-spacing:0.1em !important;
  text-transform:uppercase !important;
  color:var(--muted) !important;
  padding:0.75rem 1.4rem !important;
  border:none !important;
}
[aria-selected="true"]{
  color:var(--accent) !important;
  border-bottom:2px solid var(--accent) !important;
}
[data-baseweb="tab-panel"]{
  background:var(--surface) !important;
  border:1px solid var(--border) !important;
  border-top:none !important;
  border-radius:0 0 10px 10px !important;
  padding:1.5rem !important;
}

hr{border-color:var(--border) !important;}
</style>
""",
    unsafe_allow_html=True,
)


def get_image_base64(path: str):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def render_hero_background(image_path: str):
    encoded = get_image_base64(image_path)
    if not encoded:
        return
    st.markdown(
        f"""
        <style>
        .hero-wrap{{
          background-image:
            linear-gradient(180deg, rgba(8,9,13,.28), rgba(8,9,13,.92)),
            url('data:image/jpeg;base64,{encoded}');
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════
#  MODEL TRAINING  (cached)
# ══════════════════════════════════════════════════════════════════
@st.cache_resource
def train_model(csv_path: str):
    df = pd.read_csv(csv_path)
    drop_cols = [c for c in df.columns if c.startswith("Unnamed")]
    df = df.drop(columns=drop_cols, errors="ignore").dropna().copy()
    df["MSRP"] = (
        df["MSRP"]
        .astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .astype(float)
    )
    keep = ["MSRP", "Type", "Origin", "EngineSize", "Horsepower", "Weight"]
    df_m = pd.get_dummies(df[keep], columns=["Type", "Origin"], drop_first=True)
    for col in ["Type_SUV", "Type_Truck"]:
        if col in df_m.columns:
            df_m.drop(columns=[col], inplace=True)

    feats = [
        f
        for f in [
            "EngineSize",
            "Horsepower",
            "Weight",
            "Type_Sedan",
            "Type_Sports",
            "Type_Wagon",
            "Origin_Europe",
            "Origin_USA",
        ]
        if f in df_m.columns
    ]

    tr, te = train_test_split(df_m, train_size=0.80, random_state=0)
    Xtr, ytr = tr[feats], tr["MSRP"].astype(int)
    Xte, yte = te[feats], te["MSRP"].astype(int)

    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)

    lr = LinearRegression().fit(Xtr_s, ytr)

    poly = PolynomialFeatures(degree=2)
    pr = LinearRegression().fit(poly.fit_transform(Xtr_s), ytr)

    scores = {
        "lr_train": lr.score(Xtr_s, ytr) * 100,
        "lr_test": lr.score(Xte_s, yte) * 100,
        "pr_train": pr.score(poly.transform(Xtr_s), ytr) * 100,
        "pr_test": pr.score(poly.transform(Xte_s), yte) * 100,
    }
    return lr, pr, scaler, poly, df_m, df, feats, scores


# ══════════════════════════════════════════════════════════════════
#  TOP NAV
# ══════════════════════════════════════════════════════════════════
st.markdown(
    """
<div class="topbar">
  <div class="topbar-logo">Car<span>Price</span>&nbsp;AI</div>
  <div class="topbar-nav">
    <a href="#">Dashboard</a>
    <a href="#">Models</a>
    <a href="#">Analytics</a>
    <a href="#">Docs</a>
  </div>
  <div class="topbar-badge">⚡ ML Engine Active</div>
</div>
""",
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════
#  TWO-COLUMN LAYOUT  (no sidebar — permanent left panel)
# ══════════════════════════════════════════════════════════════════
left_col, main_col = st.columns([1, 3.2], gap="small")

# ─────────────────────────────────────────────
#  LEFT CONTROL PANEL
# ─────────────────────────────────────────────
with left_col:
    with st.container():
        # spacer to match panel feel
        st.markdown('<div style="padding:1.25rem 0.25rem;">', unsafe_allow_html=True)

        st.markdown(
            '<div class="panel-section-label">📂 Dataset</div>', unsafe_allow_html=True
        )
        uploaded = st.file_uploader(
            "Upload CSV", type=["csv"], label_visibility="collapsed"
        )
        csv_path = None
        if uploaded:
            tmp = "/tmp/cars_up.csv"
            with open(tmp, "wb") as f:
                f.write(uploaded.getbuffer())
            csv_path = tmp
        elif os.path.exists("Cars Data.csv"):
            csv_path = "Cars Data.csv"

        if not csv_path:
            st.warning("⚠️ Upload **Cars Data.csv** to begin.")
            st.stop()

        with st.spinner("Training models…"):
            lr, pr, scaler, poly, df_m, df_raw, feats, scores = train_model(csv_path)

        st.markdown(
            f"""
        <div style="background:rgba(0,201,167,.06);border:1px solid rgba(0,201,167,.2);
                    border-radius:8px;padding:0.6rem 0.85rem;font-size:0.72rem;
                    color:#00C9A7;letter-spacing:0.07em;margin:0.25rem 0 1rem;">
          ✔ {df_raw.shape[0]:,} cars loaded &nbsp;·&nbsp; {len(feats)} features
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="panel-section-label">🔧 Engine & Performance</div>',
            unsafe_allow_html=True,
        )
        engine_size = st.slider("Engine Size (L)", 1.0, 8.0, 3.0, 0.1)
        horsepower = st.slider("Horsepower (HP)", 50, 600, 200, 5)
        weight = st.slider("Weight (lbs)", 1500, 6500, 3200, 50)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            '<div class="panel-section-label">🏷️ Body Type</div>',
            unsafe_allow_html=True,
        )
        car_type = st.radio(
            "Type", ["Sedan", "Sports", "Wagon"], label_visibility="collapsed"
        )
        type_sedan = int(car_type == "Sedan")
        type_sports = int(car_type == "Sports")
        type_wagon = int(car_type == "Wagon")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            '<div class="panel-section-label">🌍 Market Origin</div>',
            unsafe_allow_html=True,
        )
        origin = st.radio(
            "Origin",
            ["Asia", "Europe", "USA"],
            label_visibility="collapsed",
            horizontal=True,
        )
        origin_europe = int(origin == "Europe")
        origin_usa = int(origin == "USA")

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("⚡  Predict Price")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            '<div class="panel-section-label">📈 Model Accuracy</div>',
            unsafe_allow_html=True,
        )
        for label, val in [
            ("LR Train", scores["lr_train"]),
            ("LR Test", scores["lr_test"]),
            ("Poly Train", scores["pr_train"]),
            ("Poly Test", scores["pr_test"]),
        ]:
            st.markdown(
                f"""
            <div class="score-bar-wrap">
              <div class="score-bar-label">
                <span>{label}</span><span>{val:.1f}%</span>
              </div>
              <div class="score-bar-track">
                <div class="score-bar-fill" style="width:{min(val,100):.1f}%"></div>
              </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  MAIN CONTENT AREA
# ─────────────────────────────────────────────
with main_col:
    with st.container():
        hero_image_path = os.path.join("image", "Sports Car.jpg")
        render_hero_background(hero_image_path)
        st.markdown(
            f"""
            <div class="hero-wrap">
              <div class="hero-content">
                <div class="hero-chip">Car Price Prediction · AI-powered insights</div>
                <h1 class="hero-title">Smart price estimates for every car in your dataset.</h1>
                <p class="hero-copy">Tune engine, horsepower, weight, body type and origin to forecast MSRP with a polished regression engine. See reliability metrics, price ranges, and market intelligence instantly.</p>
                <div class="hero-actions">
                  <a class="hero-btn primary" href="#prediction">Start predicting</a>
                  <a class="hero-btn secondary" href="#analytics">Explore analytics</a>
                </div>
                <div class="hero-cards">
                  <div class="hero-card"><span class="label">Dataset size</span><span class="value">{df_raw.shape[0]:,} cars</span></div>
                  <div class="hero-card"><span class="label">Model strength</span><span class="value">{scores['lr_test']:.1f}% test</span></div>
                  <div class="hero-card"><span class="label">Supported origins</span><span class="value">USA · Europe · Asia</span></div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<div style="padding:1.25rem 1rem 0;">', unsafe_allow_html=True)

        # ── KPI Row ──────────────────────────
        avg_msrp = df_raw["MSRP"].astype(float).mean()
        max_msrp = df_raw["MSRP"].astype(float).max()
        n_types = df_raw["Type"].nunique() if "Type" in df_raw.columns else "—"
        n_origins = df_raw["Origin"].nunique() if "Origin" in df_raw.columns else "—"

        st.markdown(
            f"""
        <div class="kpi-grid">
          <div class="kpi-card">
            <div class="kpi-label">Avg Market Price</div>
            <div class="kpi-value">${avg_msrp/1000:.1f}K</div>
            <div class="kpi-sub">across {df_raw.shape[0]:,} vehicles</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-label">Highest MSRP</div>
            <div class="kpi-value">${max_msrp/1000:.0f}K</div>
            <div class="kpi-sub">dataset maximum</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-label">Body Types</div>
            <div class="kpi-value">{n_types}</div>
            <div class="kpi-sub">distinct categories</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-label">Market Origins</div>
            <div class="kpi-value">{n_origins}</div>
            <div class="kpi-sub">USA · Europe · Asia</div>
          </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # ── TABS ─────────────────────────────

        car_images = {
            "Sedan": "image/Sedan Car.jpg",
            "Sports": "image/Sports Car.jpg",
            "Wagon": "image/Wagon Car.jpg",
        }
        tab1, tab2, tab3 = st.tabs(
            ["🎯  Prediction", "📊  Analytics", "🔥  Correlation"]
        )
        DARK = "#0E1018"
        BORDER = "#1F2232"

        # ── TAB 1 PREDICTION ─────────────────
        # ── TAB 1 PREDICTION ─────────────────
        # ── TAB 1 PREDICTION ─────────────────
        with tab1:
            if predict_btn:
                # 1. Feature Preparation & Extraction
                row = {k: [0] for k in feats}
                for k, v in [
                    ("EngineSize", engine_size),
                    ("Horsepower", horsepower),
                    ("Weight", weight),
                    ("Type_Sedan", type_sedan),
                    ("Type_Sports", type_sports),
                    ("Type_Wagon", type_wagon),
                    ("Origin_Europe", origin_europe),
                    ("Origin_USA", origin_usa),
                ]:
                    if k in row:
                        row[k] = [v]

                df_in = pd.DataFrame(row)[feats]
                scaled = scaler.transform(df_in)

                # 2. ML Model Execution
                lr_pred = lr.predict(scaled)[0]
                pr_pred = pr.predict(poly.transform(scaled))[0]
                avg_pred = (lr_pred + pr_pred) / 2

                # 3. Main Split Layout: Left side Preview / Right side Results
                prev_col, res_col = st.columns([1.3, 2], gap="medium")

                with prev_col:
                    st.markdown(
                        '<div class="sec-hd">Vehicle Preview</div>',
                        unsafe_allow_html=True,
                    )
                    image_path = car_images.get(car_type)

                    if image_path and os.path.exists(image_path):
                        image = Image.open(image_path)
                        st.markdown(
                            """
                            <div style="background:var(--surface); border:1px solid var(--border); 
                                        border-radius:14px; padding:0.75rem; margin-bottom:1.5rem; 
                                        box-shadow:0 10px 30px rgba(0,0,0,0.35);">
                            """,
                            unsafe_allow_html=True,
                        )
                        st.image(image, use_container_width=True)
                        st.markdown(
                            f"""
                            <div style="display:flex; justify-content:space-between; align-items:center; 
                                          padding:0.5rem 0.6rem 0.2rem; font-family:var(--font-hd); font-size:0.85rem; 
                                          letter-spacing:0.08em; text-transform:uppercase; color:var(--muted);">
                              <span>Class: <strong style="color:#fff;">{car_type}</strong></span>
                              <span style="color:var(--accent);">● <span style="color:var(--txt); margin-left:4px;">{origin}</span></span>
                            </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            """
                            <div style="background:var(--surface); border:1px dashed var(--accent); 
                                        border-radius:14px; padding:3rem 1rem; text-align:center; color:var(--muted); margin-bottom:1.5rem;">
                              <span style="font-size:1.5rem;">⚠️</span><br>
                              <p style="font-size:0.8rem; margin-top:0.5rem; text-transform:uppercase; letter-spacing:0.05em;">
                                Preview Image Missing<br><span style="font-size:0.7rem; color:#5A6080;">Verify 'image/' directory</span>
                              </p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                with res_col:
                    st.markdown(
                        '<div class="sec-hd">Estimation Results</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"""
                        <div class="pred-wrap" style="padding: 1.85rem 2rem 1.4rem;">
                          <div class="pred-eyebrow">Estimated Market Value</div>
                          <div class="pred-price">${avg_pred:,.0f}</div>
                          <div class="pred-note">Ensemble average · Linear + Polynomial Regression</div>
                          <div class="pred-sub-grid">
                            <div class="pred-sub-item">
                              <div class="pred-sub-lbl">Linear Regression</div>
                              <div class="pred-sub-val" style="font-size:1.3rem;">${lr_pred:,.0f}</div>
                            </div>
                            <div class="pred-sub-item">
                              <div class="pred-sub-lbl">Polynomial (deg 2)</div>
                              <div class="pred-sub-val" style="font-size:1.3rem;">${pr_pred:,.0f}</div>
                            </div>
                          </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # 4. Secondary Split Layout: Details & Model Discrepancy Breakdown
                c1, c2 = st.columns(2, gap="medium")
                with c1:
                    st.markdown(
                        '<div class="sec-hd">Input Specifications</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"""
                        <table class="spec-table">
                          <tr><td>Engine Size</td><td>{engine_size:.1f} L</td></tr>
                          <tr><td>Horsepower</td><td>{horsepower} HP</td></tr>
                          <tr><td>Weight</td><td>{weight:,} lbs</td></tr>
                          <tr><td>Body Type</td><td>{car_type}</td></tr>
                          <tr><td>Origin</td><td>{origin}</td></tr>
                        </table>
                        """,
                        unsafe_allow_html=True,
                    )

                with c2:
                    st.markdown(
                        '<div class="sec-hd">Model Agreement</div>',
                        unsafe_allow_html=True,
                    )
                    diff_pct = abs(lr_pred - pr_pred) / max(avg_pred, 1) * 100
                    agreement = max(0, 100 - diff_pct * 2)
                    st.markdown(
                        f"""
                        <div style="margin-top:0.5rem;">
                          <div style="background:var(--border);border-radius:6px;height:10px;
                                      margin-bottom:0.5rem;overflow:hidden;">
                            <div style="height:100%;width:{agreement:.0f}%;
                                        background:linear-gradient(90deg,#00C9A7,#6DECB9);
                                        border-radius:6px;"></div>
                          </div>
                          <div style="font-size:0.7rem;color:var(--muted);letter-spacing:0.08em;
                                      text-transform:uppercase;">
                            Agreement: <strong style="color:#E4E6F0;">{agreement:.0f}%</strong>
                            &nbsp;·&nbsp; Spread: <strong style="color:#F5C518;">${abs(lr_pred-pr_pred):,.0f}</strong>
                          </div>
                        </div>
                        <div style="margin-top:1.15rem;padding:0.85rem 1rem;background:var(--panel);
                                    border:1px solid var(--border);border-radius:10px;">
                          <div style="font-size:0.65rem;color:var(--muted);letter-spacing:0.12em;
                                      text-transform:uppercase;margin-bottom:0.3rem;">Price Range Estimate</div>
                          <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.1rem;
                                      font-weight:700;color:#E4E6F0;">
                            ${min(lr_pred,pr_pred):,.0f}
                            <span style="color:var(--muted);font-weight:400;font-size:0.85rem;"> → </span>
                            ${max(lr_pred,pr_pred):,.0f}
                          </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                # Welcome Placeholder when app loads fresh
                st.markdown(
                    """
                    <div class="pred-wrap pred-placeholder">
                      <div class="icon">🚘</div>
                      <p>Configure specs in the left panel<br>
                         and click <span style="color:#C8102E;">⚡ Predict Price</span></p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # ── TAB 2 ANALYTICS ──────────────────
        with tab2:
            st.markdown(
                '<div class="sec-hd">Price Distribution</div>', unsafe_allow_html=True
            )
            fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), facecolor=DARK)
            for ax in axes:
                ax.set_facecolor(DARK)
                for sp in ax.spines.values():
                    sp.set_edgecolor(BORDER)
                ax.tick_params(colors="#5A6080", labelsize=8)

            msrp_vals = df_raw["MSRP"].astype(float)
            axes[0].hist(
                msrp_vals,
                bins=45,
                color="#C8102E",
                alpha=0.85,
                edgecolor=DARK,
                linewidth=0.4,
            )
            axes[0].set_title("MSRP Distribution", color="#E4E6F0", fontsize=10, pad=10)
            axes[0].set_xlabel("Price ($)", color="#5A6080", fontsize=8)
            axes[0].set_ylabel("Vehicles", color="#5A6080", fontsize=8)
            axes[0].xaxis.set_major_formatter(
                mtick.FuncFormatter(lambda x, _: f"${x/1000:.0f}K")
            )
            axes[0].axvline(
                msrp_vals.mean(),
                color="#F5C518",
                linewidth=1.2,
                linestyle="--",
                alpha=0.8,
            )

            if "Type" in df_raw.columns:
                tdata = df_raw.groupby("Type")["MSRP"].mean().sort_values()
                palette = ["#C8102E", "#FF6B35", "#F5C518", "#00C9A7", "#6CA0DC"][
                    : len(tdata)
                ]
                bars = axes[1].barh(tdata.index, tdata.values, color=palette, alpha=0.9)
                axes[1].set_title(
                    "Avg Price by Body Type", color="#E4E6F0", fontsize=10, pad=10
                )
                axes[1].set_xlabel("Avg MSRP ($)", color="#5A6080", fontsize=8)
                axes[1].xaxis.set_major_formatter(
                    mtick.FuncFormatter(lambda x, _: f"${x/1000:.0f}K")
                )
                for bar, val in zip(bars, tdata.values):
                    axes[1].text(
                        val + 200,
                        bar.get_y() + bar.get_height() / 2,
                        f"${val/1000:.1f}K",
                        va="center",
                        color="#E4E6F0",
                        fontsize=7.5,
                    )

            plt.tight_layout(pad=1.5)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            st.markdown(
                '<div class="sec-hd">Feature Relationships</div>',
                unsafe_allow_html=True,
            )
            num_cols = [
                c
                for c in ["MSRP", "EngineSize", "Horsepower", "Weight"]
                if c in df_m.columns
            ]
            pp = sns.pairplot(
                df_m[num_cols].sample(min(300, len(df_m)), random_state=1),
                height=1.9,
                aspect=1,
                plot_kws={
                    "alpha": 0.45,
                    "color": "#C8102E",
                    "s": 9,
                    "edgecolors": "none",
                },
                diag_kws={"color": "#F5C518", "alpha": 0.7, "bins": 20},
            )
            pp.figure.patch.set_facecolor(DARK)
            for ax in pp.axes.flat:
                ax.set_facecolor(DARK)
                ax.tick_params(colors="#5A6080", labelsize=7)
                ax.xaxis.label.set_color("#5A6080")
                ax.yaxis.label.set_color("#5A6080")
                for sp in ax.spines.values():
                    sp.set_edgecolor(BORDER)
            st.pyplot(pp.figure, use_container_width=True)
            plt.close()

        # ── TAB 3 CORRELATION ────────────────
        with tab3:
            st.markdown(
                '<div class="sec-hd">Feature Correlation Matrix</div>',
                unsafe_allow_html=True,
            )
            corr = df_m.corr()
            fig2, ax2 = plt.subplots(figsize=(10, 6.5), facecolor=DARK)
            ax2.set_facecolor(DARK)
            mask = np.triu(np.ones_like(corr, dtype=bool))
            cmap = sns.diverging_palette(5, 210, s=90, l=40, as_cmap=True)
            sns.heatmap(
                corr,
                mask=mask,
                annot=True,
                fmt=".2f",
                cmap=cmap,
                linewidths=0.6,
                linecolor="#08090D",
                annot_kws={"size": 8, "color": "#E4E6F0", "weight": "600"},
                ax=ax2,
                cbar_kws={"shrink": 0.75, "pad": 0.02},
                vmin=-1,
                vmax=1,
            )
            ax2.tick_params(colors="#9098B0", labelsize=8)
            ax2.set_xticklabels(ax2.get_xticklabels(), rotation=35, ha="right")
            ax2.set_yticklabels(ax2.get_yticklabels(), fontsize=8)
            ax2.figure.axes[-1].tick_params(colors="#5A6080", labelsize=7)
            plt.title(
                "Pearson Correlation — All Features",
                color="#E4E6F0",
                fontsize=11,
                pad=14,
                loc="left",
            )
            plt.tight_layout()
            st.pyplot(fig2, use_container_width=True)
            plt.close(fig2)

            st.markdown(
                '<div class="sec-hd">MSRP Feature Impact Ranking</div>',
                unsafe_allow_html=True,
            )
            if "MSRP" in corr.columns:
                msrp_c = corr["MSRP"].drop("MSRP").sort_values()
                colors = ["#6CA0DC" if v < 0 else "#C8102E" for v in msrp_c.values]
                fig3, ax3 = plt.subplots(figsize=(9, 3.5), facecolor=DARK)
                ax3.set_facecolor(DARK)
                for sp in ax3.spines.values():
                    sp.set_edgecolor(BORDER)
                ax3.barh(
                    msrp_c.index, msrp_c.values, color=colors, alpha=0.88, height=0.55
                )
                ax3.axvline(0, color="#3A3E54", linewidth=1)
                ax3.tick_params(colors="#9098B0", labelsize=8)
                ax3.set_xlabel("Correlation Coefficient", color="#5A6080", fontsize=8)
                ax3.set_title(
                    "Correlation with MSRP (Price)",
                    color="#E4E6F0",
                    fontsize=10,
                    pad=10,
                    loc="left",
                )
                for i, (val, name) in enumerate(zip(msrp_c.values, msrp_c.index)):
                    ax3.text(
                        val + (0.01 if val >= 0 else -0.01),
                        i,
                        f"{val:+.2f}",
                        va="center",
                        ha="left" if val >= 0 else "right",
                        color="#E4E6F0",
                        fontsize=7.5,
                        fontweight="600",
                    )
                plt.tight_layout()
                st.pyplot(fig3, use_container_width=True)
                plt.close(fig3)

        st.markdown("</div>", unsafe_allow_html=True)

# ── Footer ──────────────────────────────────
st.markdown(
    """
<div style="text-align:center;padding:1.5rem 0 0.5rem;font-size:0.65rem;
            color:#2A2E42;letter-spacing:0.14em;text-transform:uppercase;
            border-top:1px solid #1F2232;margin-top:1rem;">
  CarPrice AI &nbsp;·&nbsp; Linear &amp; Polynomial Regression &nbsp;·&nbsp;
  Built with Streamlit &amp; scikit-learn
</div>
""",
    unsafe_allow_html=True,
)
