"""
CarPrice AI — Landing Page + Dashboard
Modern, professional slate/blue redesign.
Run: streamlit run car_prediction_app.py
"""

import os
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


# ── SCROLL FUNCTION ─────────────────────────────

def scroll_top():
    st.components.v1.html(
        """
        <script>
        window.parent.scrollTo(0,0);
        </script>
        """,
        height=0,
    )

# ── PAGE CONFIG ──────────────────────────────────────────────────
st.set_page_config(
    page_title="CarPrice AI",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

if "page" not in st.session_state:
    st.session_state.page = "landing"
if "csv_path" not in st.session_state:
    st.session_state.csv_path = None
if "uploaded_df" not in st.session_state:
    st.session_state.uploaded_df = None

# ── GLOBAL CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --bg:      #F7F8FB;
  --panel:   #FFFFFF;
  --surf:    #FFFFFF;
  --surf2:   #F2F4F8;
  --border:  #E5E8F0;
  --border2: #D6DBE6;
  --accent:  #3D5AFE;
  --accent2: #2A3FCC;
  --gold:    #B8842E;
  --teal:    #16916F;
  --slate:   #64708A;
  --txt:     #1A2233;
  --muted:   #8A93A8;
  --muted2:  #565F76;
  --fhd:     'Manrope', -apple-system, 'Segoe UI', system-ui, sans-serif;
  --fbd:     'Inter', -apple-system, 'Segoe UI', system-ui, sans-serif;
  --fmono:   'JetBrains Mono', 'Courier New', monospace;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body { color-scheme: light !important; }
html, body, [class*="css"] {
  font-family: var(--fbd);
  background: var(--bg) !important;
  color: var(--txt);
  -webkit-font-smoothing: antialiased;
}
.stApp { background: var(--bg) !important; color: var(--txt) !important; }
h1, h2, h3, h4, h5, h6, p, span, label, div { color: inherit; }
.block-container { padding: 0 !important; max-width: 100% !important; }
#MainMenu, footer, header,
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapseButton"] { display: none !important; }
[data-testid="stSidebar"] { display: none !important; }
::-webkit-scrollbar { width: 3px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 4px; }

/* ════════════════════════════════════
   LANDING
════════════════════════════════════ */
.lp-nav {
  display: flex; align-items: center; justify-content: space-between;
  padding: 0 3rem; height: 64px;
  border-bottom: 1px solid var(--border);
  background: rgba(255,255,255,0.9);
  backdrop-filter: blur(12px);
  position: sticky; top: 0; z-index: 99;
}
.lp-logo { font-family: var(--fhd); font-size: 1.15rem; font-weight: 800;
  letter-spacing: -0.01em; color: var(--txt); }
.lp-logo em { color: var(--accent); font-style: normal; }
.lp-pill {
  font-size: 0.62rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase;
  background: rgba(61,90,254,0.07); border: 1px solid rgba(61,90,254,0.25);
  color: var(--accent2); padding: 0.32rem 0.85rem; border-radius: 6px;
  font-family: var(--fmono);
}
.hero-wrap {
  padding: 6rem 3rem 4rem;
  background:
    radial-gradient(circle at 80% 20%, rgba(61,90,254,.08), transparent 35%),
    radial-gradient(circle at 20% 0%, rgba(22,145,111,.06), transparent 30%),
    var(--bg);
}
.hero-tag {
  display: inline-flex; align-items: center; gap: 0.55rem;
  font-size: 0.68rem; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase;
  color: var(--accent2); margin-bottom: 1.5rem; font-family: var(--fmono);
}
.hero-tag::before { content:''; width:18px; height:1px; background:var(--accent); display:block; }
.hero-h1, .hero-h1 * {
  font-family: var(--fhd) !important;
  font-size: clamp(2.6rem,4.6vw,4.2rem); font-weight: 800 !important;
  line-height: 1.05; letter-spacing: -0.03em; color: var(--txt) !important; margin-bottom: 1.5rem;
  max-width: 760px;
}
.hero-h1 { margin-bottom: 1.5rem; }
.hero-h1 .gr { color: var(--accent2) !important; }
.hero-sub {
  font-size: 1rem; line-height: 1.85; color: var(--muted2);
  max-width: 540px; font-weight: 400; margin-bottom: 2.25rem;
}
.hero-chips { display: flex; flex-wrap: wrap; gap: 0.6rem; margin-bottom: 3rem; }
.hero-chip {
  display: inline-flex; align-items: center; gap: 0.45rem;
  font-size: 0.7rem; font-weight: 500; letter-spacing: 0.02em;
  color: var(--muted2); background: var(--surf2); border: 1px solid var(--border);
  border-radius: 7px; padding: 0.4rem 0.85rem;
}
.hero-chip em { width:5px; height:5px; border-radius:50%; display:block; font-style:normal; background: var(--accent); }

.stat-row {
  display: grid; grid-template-columns: repeat(4,1fr);
  gap: 1px; background: var(--border);
  border: 1px solid var(--border); border-radius: 10px; overflow: hidden;
  margin-bottom: 3.5rem;
  box-shadow: 0 1px 3px rgba(26,34,51,0.04);
}
.stat-cell { background: var(--panel); padding: 1.5rem 1.4rem; }
.stat-num { font-family:var(--fhd); font-size:2.1rem; font-weight:800; color:var(--txt); line-height:1; margin-bottom:0.35rem; }
.stat-num em { color:var(--accent2); font-style:normal; }
.stat-lbl { font-size:0.65rem; font-weight:500; letter-spacing:0.06em; text-transform:uppercase; color:var(--muted); }

.how-band {
  background: var(--panel);
  padding: 4rem 3rem;
  border-bottom: 1px solid var(--border);
}
.band-tag { font-size:0.64rem; font-weight:600; letter-spacing:0.1em; text-transform:uppercase; color:var(--accent2); margin-bottom:0.5rem; font-family: var(--fmono); }
.band-h   { font-family:var(--fhd); font-size:1.7rem; font-weight:800; color:var(--txt); letter-spacing:-0.02em; margin-bottom:2.5rem; }
.steps { display:grid; grid-template-columns:repeat(4,1fr); gap:1.5rem; }
.step { border-left: 2px solid var(--border); padding-left: 1.1rem; }
.step-n {
  font-family:var(--fmono); font-size:0.72rem; font-weight:500; color:var(--accent2);
  margin-bottom:0.6rem; letter-spacing: 0.05em;
}
.step-h { font-family:var(--fhd); font-size:0.88rem; font-weight:700;
  color:var(--txt); margin-bottom:0.45rem; }
.step-p { font-size:0.78rem; line-height:1.7; color:var(--muted2); }

.feat-band { padding: 4rem 3rem 3rem; }
.feat-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:1.1rem; }
.feat-card {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 2rem;
  box-shadow: 0 1px 2px rgba(26,34,51,0.03);
  transition: all .25s ease;
}
.feat-card:hover {
  transform: translateY(-4px);
  border-color: rgba(61,90,254,0.35);
  box-shadow: 0 12px 28px rgba(26,34,51,0.08);
}
.feat-icon { width:38px; height:38px; border-radius:8px; display:flex;
  align-items:center; justify-content:center; font-size:1.05rem; margin-bottom:1.1rem;
  background: var(--surf2); border: 1px solid var(--border); }
.feat-h { font-family:var(--fhd); font-size:0.95rem; font-weight:700; color:var(--txt); margin-bottom:0.5rem; }
.feat-p { font-size:0.8rem; line-height:1.8; color:var(--muted2); font-weight:400; }

.lp-footer {
  text-align:center; padding:1.75rem 0 1.5rem;
  font-size:0.65rem; color:var(--muted); letter-spacing:0.06em;
  border-top:1px solid var(--border); margin-top:1rem;
  font-family: var(--fmono);
}

/* ════════════════════════════════════
   DASHBOARD TOPBAR
════════════════════════════════════ */
.dash-nav {
  display: flex; align-items: center; justify-content: space-between;
  padding: 0 2rem; height: 56px;
  background: var(--panel);
  border-bottom: 1px solid var(--border);
}
.dash-logo { font-family:var(--fhd); font-size:1.05rem; font-weight:800;
  letter-spacing:-0.01em; color:var(--txt); }
.dash-logo em { color:var(--accent2); font-style:normal; }
html { scroll-behavior: smooth; }
.dash-nav-links { display:flex; align-items:center; gap:0.25rem; }
.dash-nav-link {
  font-size:0.74rem; font-weight:600; letter-spacing:0.02em;
  color:var(--muted2); text-decoration:none;
  padding:0.42rem 0.9rem; border-radius:7px;
  cursor:pointer; transition: all .18s ease;
  border: 1px solid transparent;
}
.dash-nav-link:hover {
  color: var(--accent2); background: rgba(61,90,254,0.06);
}
.dash-nav-link.active {
  color: var(--accent2); background: rgba(61,90,254,0.08);
  border-color: rgba(61,90,254,0.18);
}
.dash-badge {
  font-size:0.62rem; font-weight:600; letter-spacing:0.04em;
  background:rgba(22,145,111,0.08); border:1px solid rgba(22,145,111,0.25);
  color:var(--teal); padding:0.28rem 0.75rem; border-radius:6px;
  font-family: var(--fmono);
}

/* ── BACK BUTTON (uses st.button type="secondary") ── */

div[data-testid="stButton"] button[kind="secondary"],
button[kind="secondary"],
div[data-testid="stButton"].back-home-btn > button,
.back-home-btn .stButton > button {
  background: var(--surf2) !important;
  border: 1px solid var(--border2) !important;
  color: var(--muted2) !important;
  font-family: var(--fbd) !important;
  font-size: 0.8rem !important;
  font-weight: 600 !important;
  padding: 0.65rem 1.4rem !important;
  height: auto !important;
  min-height: 42px !important;
  border-radius: 10px !important;
  width: auto !important;
  box-shadow: none !important;
  transition: all .2s ease !important;
}

div[data-testid="stButton"] button[kind="secondary"]:hover,
button[kind="secondary"]:hover,
.back-home-btn .stButton > button:hover {
  border-color: var(--accent) !important;
  color: var(--accent2) !important;
  background: rgba(61,90,254,.07) !important;
  transform: translateX(-2px);
  box-shadow: none !important;
}

/* ── REAL CONTAINER-BASED SPACING (st.container(key=...)) ── */
.st-key-left_panel {
  padding: 1.5rem 1.5rem 2rem 3rem !important;
}
.st-key-home_btn, .st-key-home_btn_nd {
  padding: 1.1rem 0 0.4rem 3rem !important;
}

/* ── DASHBOARD HEADER BANNER ── */
.dash-banner {
  padding: 1.4rem 2rem 1.2rem;
  background: var(--bg);
  border-bottom: 1px solid var(--border);
  display: flex; align-items: flex-end; justify-content: space-between;
}
.dash-banner-title {
  font-family:var(--fhd); font-size:1.35rem; font-weight:800;
  color:var(--txt); letter-spacing:-0.01em; margin-bottom:0.2rem;
}
.dash-banner-sub { font-size:0.76rem; color:var(--muted2); font-weight:400; }
.dash-banner-right { display:flex; gap:0.5rem; }
.dash-tag {
  font-size:0.6rem; font-weight:600; letter-spacing:0.03em;
  padding:0.28rem 0.65rem; border-radius:6px; border:1px solid;
  font-family: var(--fmono);
}
.dash-tag.r { color:var(--accent2); background:rgba(61,90,254,0.06); border-color:rgba(61,90,254,0.2); }
.dash-tag.t { color:var(--teal);    background:rgba(22,145,111,0.06); border-color:rgba(22,145,111,0.2); }
.dash-tag.g { color:var(--gold);    background:rgba(184,132,46,0.07); border-color:rgba(184,132,46,0.22); }

/* ── MODERN KPI CARDS ── */

.kpi-row {
  display:grid;
  grid-template-columns:repeat(4,1fr);
  gap:1rem;
  margin-bottom:1.5rem;
}

.kpi {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius:16px;
  padding:1.4rem 1.5rem;
  position:relative;
  overflow:hidden;
  box-shadow: 0 1px 2px rgba(26,34,51,0.03);
  transition: transform .25s ease, border-color .25s ease, box-shadow .25s ease;
}

/* subtle blue accent glow */
.kpi::before {
  content:"";
  position:absolute;
  top:-40px; right:-40px;
  width:90px; height:90px;
  background: radial-gradient(circle, rgba(61,90,254,.10), transparent 70%);
}

.kpi:hover {
  transform: translateY(-4px);
  border-color: rgba(61,90,254,.3);
  box-shadow: 0 12px 28px rgba(26,34,51,0.08);
}

.kpi-lbl {
  font-size:0.65rem;
  font-weight:600;
  letter-spacing:0.08em;
  text-transform:uppercase;
  color: var(--muted);
  margin-bottom:.7rem;
}

.kpi-val {
  font-family:var(--fhd);
  font-size:2rem;
  font-weight:800;
  color: var(--txt);
  line-height:1;
  letter-spacing:-0.03em;
}

.kpi-sub {
  font-size:.68rem;
  color: var(--accent2);
  margin-top:.5rem;
  font-weight:500;
}

/* ── LEFT PANEL ── */
.plbl {
  font-size:0.78rem; font-weight:700; letter-spacing:0.04em; text-transform:uppercase;
  color:var(--muted2); padding-bottom:0.5rem; border-bottom:1px solid var(--border);
  margin-bottom:0.85rem;
}

/* ── SECTION HEADER ── */
.sec {
  display:flex; align-items:center; gap:0.5rem;
  font-family:var(--fhd); font-size:0.8rem; font-weight:700;
  letter-spacing:0.01em; color:var(--txt);
  margin:1.1rem 0 0.75rem;
}
.sec::before { content:''; width:3px; height:12px; background:var(--accent);
  border-radius:2px; flex-shrink:0; }

/* ── SCORE BARS ── */
.sbar { margin-bottom:0.65rem; }
.sbar-top { display:flex; justify-content:space-between; font-size:0.6rem;
  color:var(--muted); margin-bottom:0.2rem; letter-spacing:0.02em; font-family: var(--fmono); }
.sbar-track { background:var(--border); border-radius:3px; height:4px; overflow:hidden; }
.sbar-fill { height:100%; border-radius:3px; }

/* ── PREDICTION BOX ── */
.pred-box {
  background: var(--panel);
  border: 1px solid var(--border); border-radius:12px;
  padding: 1.75rem 1.5rem 1.4rem; text-align:center;
  margin-bottom: 1.25rem;
  box-shadow: 0 1px 3px rgba(26,34,51,0.04);
}
.pred-eye  { font-size:0.62rem; font-weight:600; letter-spacing:0.08em; text-transform:uppercase; color:var(--muted); margin-bottom:0.5rem; }
.pred-amt  { font-family:var(--fhd); font-size:3rem; font-weight:800; color:var(--txt); line-height:1; }
.pred-note { font-size:0.66rem; color:var(--muted); margin-top:0.4rem; font-weight:400; }
.pred-row  { display:flex; justify-content:center; gap:2.5rem;
  margin-top:1.1rem; padding-top:1.1rem; border-top:1px solid var(--border); }
.pred-sub-lbl { font-size:0.58rem; font-weight:500; letter-spacing:0.04em; text-transform:uppercase; color:var(--muted); }
.pred-sub-val { font-family:var(--fhd); font-size:1.3rem; font-weight:700; color:var(--accent2); }

.pred-empty {
  text-align:center; padding:3rem 1rem; color:var(--muted);
  background: var(--surf2); border:1px dashed var(--border2); border-radius:12px;
  margin-bottom:1.25rem;
}
.pred-empty .eicon { font-size:2rem; margin-bottom:0.6rem; opacity: 0.6; }
.pred-empty p { font-family:var(--fhd); font-size:0.82rem; font-weight:600;
  letter-spacing:0.01em; line-height:1.7; color: var(--muted2); }

/* ── SPEC TABLE ── */
.spec { width:100%; border-collapse:collapse; font-size:0.78rem; }
.spec td { padding:0.5rem 0.5rem; border-bottom:1px solid var(--border); color:var(--txt); }
.spec td:first-child { color:var(--muted); font-size:0.64rem; font-weight:500;
  letter-spacing:0.04em; text-transform:uppercase; width:46%; }
.spec td:last-child { font-weight:500; }

/* ── IMAGE WRAPPER ── */
.img-wrap {
  background:var(--surf2); border:1px solid var(--border); border-radius:10px;
  padding:0.65rem; margin-bottom:0.85rem;
}
.img-foot {
  display:flex; justify-content:space-between; padding:0.4rem 0.4rem 0.1rem;
  font-size:0.7rem; letter-spacing:0.01em; color:var(--muted2);
}

/* ── STREAMLIT OVERRIDES ── */
[data-testid="stSlider"] label {
  font-size:0.66rem !important; font-weight:500 !important;
  letter-spacing:0.04em !important; text-transform:uppercase !important; color:var(--muted) !important;
}
[data-testid="stSlider"] > div > div { background:var(--border) !important; }
[data-testid="stSlider"] > div > div > div > div { background:var(--accent) !important; }

.stRadio label,
.stRadio label p,
.stRadio label span,
[data-testid="stRadio"] label,
[data-testid="stRadio"] label p,
[data-testid="stRadio"] label span,
[data-testid="stRadio"] div[role="radiogroup"] label {
  color: var(--muted2) !important;
  font-size: 0.85rem !important;
  letter-spacing: 0.01em !important;
  font-weight: 500 !important;
  opacity: 1 !important;
}
.stRadio > div,
[data-testid="stRadio"] > div { gap: 0.3rem !important; }

[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] label {
  color: var(--muted) !important;
}

[data-testid="stFileUploader"] {
  border:1px dashed var(--border2) !important;
  border-radius:8px !important; background:var(--surf2) !important;
}
[data-testid="stFileUploader"] label {
  font-size:0.66rem !important; font-weight:500 !important;
  letter-spacing:0.04em !important; text-transform:uppercase !important; color:var(--muted) !important;
}

/* Tabs */
[data-baseweb="tab-list"] {
  background:var(--surf2) !important; border-radius:8px 8px 0 0 !important;
  border:1px solid var(--border) !important; border-bottom:none !important; gap:0 !important; padding:0 !important;
}
[data-baseweb="tab"] {
  font-family:var(--fbd) !important; font-size:0.74rem !important; font-weight:500 !important;
  letter-spacing:0.01em !important;
  color:var(--muted) !important; padding:0.65rem 1.2rem !important; border:none !important;
}
[aria-selected="true"] { color:var(--accent2) !important; border-bottom:2px solid var(--accent) !important; }
[data-baseweb="tab-panel"] {
  background:var(--panel) !important; border:1px solid var(--border) !important;
  border-top:none !important; border-radius:0 0 8px 8px !important; padding:1.4rem !important;
}

hr { border-color:var(--border) !important; }

/* Upload success notice */
.upload-success {
  display:flex; align-items:center; gap:0.5rem;
  background:rgba(22,145,111,0.06); border:1px solid rgba(22,145,111,0.2);
  border-radius:7px; padding:0.5rem 0.75rem; margin-top:0.5rem;
  font-size:0.66rem; color:var(--teal); font-weight:500; letter-spacing:0.01em;
}

/* Generic button styling — clean, professional, no glow */
div[data-testid="stButton"] > button {
    width: 100% !important;
    height: 52px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    background: linear-gradient(135deg, #3D5AFE, #5B76FF) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 12px !important;
    font-size: 0.9rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.02em !important;
    box-shadow: 0 6px 18px rgba(61,90,254,0.22);
    transition: all .25s ease !important;
}

div[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 26px rgba(61,90,254,0.32);
}

.predict-btn-wrap .stButton > button,
div[data-testid="stButton"] .predict-btn-wrap button {
  width:100% !important;
  padding:0.75rem 1.25rem !important;
  font-size:0.9rem !important;
  height:52px !important;
  letter-spacing: 0.02em !important;
}

.launch-btn-wrap .stButton > button {
  width:100% !important;
  padding:0.9rem 1.5rem !important;
  font-size:0.9rem !important;
  height:56px !important;
}

</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  MODEL TRAINING
# ══════════════════════════════════════════════════════════════════
@st.cache_resource
def train_model(csv_path: str):
    df = pd.read_csv(csv_path)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore").dropna().copy()
    df["MSRP"] = (df["MSRP"].astype(str)
                  .str.replace("$", "", regex=False)
                  .str.replace(",", "", regex=False)
                  .astype(float))
    keep = ["MSRP", "Type", "Origin", "EngineSize", "Horsepower", "Weight"]
    df_m = pd.get_dummies(df[keep], columns=["Type", "Origin"], drop_first=True)
    for col in ["Type_SUV", "Type_Truck"]:
        if col in df_m.columns:
            df_m.drop(columns=[col], inplace=True)
    feats = [f for f in [
        "EngineSize", "Horsepower", "Weight",
        "Type_Sedan", "Type_Sports", "Type_Wagon",
        "Origin_Europe", "Origin_USA",
    ] if f in df_m.columns]
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
        "lr_test":  lr.score(Xte_s, yte) * 100,
        "pr_train": pr.score(poly.transform(Xtr_s), ytr) * 100,
        "pr_test":  pr.score(poly.transform(Xte_s), yte) * 100,
    }
    return lr, pr, scaler, poly, df_m, df, feats, scores


@st.cache_resource
def train_model_from_df(df_key: str, _df: pd.DataFrame):
    """Train from an in-memory dataframe (uploaded file)."""
    df = _df.copy()
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore").dropna()
    df["MSRP"] = (df["MSRP"].astype(str)
                  .str.replace("$", "", regex=False)
                  .str.replace(",", "", regex=False)
                  .astype(float))
    keep = ["MSRP", "Type", "Origin", "EngineSize", "Horsepower", "Weight"]
    df_m = pd.get_dummies(df[keep], columns=["Type", "Origin"], drop_first=True)
    for col in ["Type_SUV", "Type_Truck"]:
        if col in df_m.columns:
            df_m.drop(columns=[col], inplace=True)
    feats = [f for f in [
        "EngineSize", "Horsepower", "Weight",
        "Type_Sedan", "Type_Sports", "Type_Wagon",
        "Origin_Europe", "Origin_USA",
    ] if f in df_m.columns]
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
        "lr_test":  lr.score(Xte_s, yte) * 100,
        "pr_train": pr.score(poly.transform(Xtr_s), ytr) * 100,
        "pr_test":  pr.score(poly.transform(Xte_s), yte) * 100,
    }
    return lr, pr, scaler, poly, df_m, df, feats, scores


# ══════════════════════════════════════════════════════════════════
#  LANDING PAGE
# ══════════════════════════════════════════════════════════════════
def render_landing():

    st.markdown("""
    <div class="lp-nav">
      <div class="lp-logo">Car<em>Price</em>&nbsp;AI</div>
      <div style="display:flex;align-items:center;gap:2.2rem;">
        <a href="#" style="font-size:0.72rem;font-weight:500;color:var(--muted2);text-decoration:none;">Features</a>
        <a href="#" style="font-size:0.72rem;font-weight:500;color:var(--muted2);text-decoration:none;">How it works</a>
        <a href="#" style="font-size:0.72rem;font-weight:500;color:var(--muted2);text-decoration:none;">Docs</a>
      </div>
      <div class="lp-pill">● Model Ready</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="hero-wrap">
      <div class="hero-tag">Automotive Intelligence Platform</div>
      <h1 class="hero-h1">
        Predict any car's <span class="gr">market value instantly.</span>
      </h1>
      <p class="hero-sub">
        Harness dual-model machine learning — Linear Regression paired with Polynomial
        Regression — to forecast MSRP with precision. Configure specs and get
        instant AI-powered price estimates in seconds.
      </p>
      <div class="hero-chips">
        <span class="hero-chip"><em></em> Dual ML Models</span>
        <span class="hero-chip"><em></em> Real-Time Predictions</span>
        <span class="hero-chip"><em></em> Deep Analytics</span>
        <span class="hero-chip"><em></em> Correlation Heatmaps</span>
        <span class="hero-chip"><em></em> 8 Predictive Features</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="padding:0 3rem;">
    <div class="stat-row">
      <div class="stat-cell">
        <div class="stat-num">2<em>×</em></div>
        <div class="stat-lbl">ML Models — Linear &amp; Polynomial</div>
      </div>
      <div class="stat-cell">
        <div class="stat-num">8</div>
        <div class="stat-lbl">Predictive Features Used</div>
      </div>
      <div class="stat-cell">
        <div class="stat-num">3</div>
        <div class="stat-lbl">Market Origins Supported</div>
      </div>
      <div class="stat-cell">
        <div class="stat-num">80<em>%</em></div>
        <div class="stat-lbl">Training Split Ratio</div>
      </div>
    </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="how-band">
      <div class="band-tag">Process</div>
      <div class="band-h">How it works</div>
      <div class="steps">
        <div class="step">
          <div class="step-n">01 — Upload</div>
          <div class="step-h">Upload Dataset</div>
          <p class="step-p">Upload a CSV with MSRP, engine specs, weight, body type and origin directly in the app.</p>
        </div>
        <div class="step">
          <div class="step-n">02 — Train</div>
          <div class="step-h">Train Models</div>
          <p class="step-p">Both Linear and Polynomial Regression models train instantly on your data with an 80/20 split.</p>
        </div>
        <div class="step">
          <div class="step-n">03 — Configure</div>
          <div class="step-h">Configure Specs</div>
          <p class="step-p">Tune engine size, horsepower, weight, body type and market origin from the control panel.</p>
        </div>
        <div class="step">
          <div class="step-n">04 — Predict</div>
          <div class="step-h">Get Prediction</div>
          <p class="step-p">Receive an ensemble price estimate with agreement score, price range and full analytics.</p>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="feat-band">
      <div class="band-tag">Capabilities</div>
      <div class="band-h">Everything you need</div>
      <div class="feat-grid">
        <div class="feat-card">
          <div class="feat-icon">🤖</div>
          <div class="feat-h">Dual-Model Ensemble</div>
          <p class="feat-p">Linear Regression for baseline accuracy. Polynomial Regression captures non-linear price relationships. Both averaged for optimal estimates.</p>
        </div>
        <div class="feat-card">
          <div class="feat-icon">📊</div>
          <div class="feat-h">Deep Analytics</div>
          <p class="feat-p">Explore MSRP distributions, price by body type, feature pair plots and full Pearson correlation heatmaps for any uploaded dataset.</p>
        </div>
        <div class="feat-card">
          <div class="feat-icon">⚡</div>
          <div class="feat-h">Instant Predictions</div>
          <p class="feat-p">Sub-second price estimates with model agreement scores, confidence ranges and visual vehicle previews by body type.</p>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    csv_path = None
    if os.path.exists("Cars Data.csv"):
        csv_path = "Cars Data.csv"

    st.markdown("<br>", unsafe_allow_html=True)
    _, c, _ = st.columns([2.8, 2, 2])
    with c:
        st.markdown('<div class="launch-btn-wrap">', unsafe_allow_html=True)
        if st.button("Launch Dashboard →", key="launch_btn"):
            if csv_path:
                st.session_state.csv_path = csv_path
                st.session_state.uploaded_df = None
                st.session_state.page = "dashboard"
                scroll_top()
                st.rerun()
            elif st.session_state.uploaded_df is not None:
                st.session_state.csv_path = None
                st.session_state.page = "dashboard"
                scroll_top()
                st.rerun()
            else:
                st.session_state.page = "dashboard"
                scroll_top()
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("""
      <div style="
      text-align:center;
      margin-top:20px;
      font-size:0.75rem;
      color:var(--muted);
      ">
      Powered by Machine Learning • Linear Regression • Polynomial Regression
      </div>
      """,
    unsafe_allow_html=True)

    st.markdown("""
    <div class="lp-footer">
      CarPrice AI · Linear &amp; Polynomial Regression · Built with Streamlit &amp; scikit-learn
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  DASHBOARD
# ══════════════════════════════════════════════════════════════════
def render_dashboard():
    scroll_top()
    csv_path = st.session_state.get("csv_path")
    uploaded_df = st.session_state.get("uploaded_df")

    has_data = csv_path or (uploaded_df is not None)

    if not has_data:
        st.markdown("""
        <div class="dash-nav">
          <div class="dash-logo">Car<em>Price</em>&nbsp;AI</div>
          <div class="dash-badge">No Dataset</div>
        </div>
        """, unsafe_allow_html=True)
        with st.container(key="home_btn_nd"):
            if st.button("← Home", key="back_home_nd", type="secondary"):
                st.session_state.page = "landing"
                st.rerun()
        st.markdown("""
        <div style="max-width:480px;margin:3rem auto;text-align:center;padding:2.5rem;
                    background:var(--panel);border:1px dashed var(--border2);border-radius:12px;
                    box-shadow:0 1px 3px rgba(26,34,51,0.04);">
          <div style="font-size:1.8rem;margin-bottom:0.75rem;opacity:0.6;">📂</div>
          <div style="font-family:'Manrope',sans-serif;font-size:1rem;font-weight:700;color:var(--txt);margin-bottom:0.4rem;">
            Upload a dataset to continue
          </div>
          <div style="font-size:0.74rem;color:var(--muted2);line-height:1.7;">
            Drop a CSV with MSRP, Type, Origin,<br>EngineSize, Horsepower, Weight columns.
          </div>
        </div>
        """, unsafe_allow_html=True)
        up = st.file_uploader("Upload CSV", type=None, key="no_data_uploader")
        if up is not None and up.name.endswith(".csv"):
            try:
                df_up = pd.read_csv(up)
                required = {"MSRP","Type","Origin","EngineSize","Horsepower","Weight"}
                missing = required - set(df_up.columns)
                if missing:
                    st.error(f"Missing columns: {', '.join(missing)}")
                else:
                    st.session_state.uploaded_df = df_up
                    st.rerun()
            except Exception as e:
                st.error(str(e))
        return

    with st.spinner("Training models…"):
        if uploaded_df is not None:
            lr, pr, scaler, poly, df_m, df_raw, feats, scores = train_model_from_df(
                str(id(uploaded_df)), uploaded_df
            )
        else:
            lr, pr, scaler, poly, df_m, df_raw, feats, scores = train_model(csv_path)

    avg_msrp  = df_raw["MSRP"].astype(float).mean()
    max_msrp  = df_raw["MSRP"].astype(float).max()
    n_types   = df_raw["Type"].nunique()   if "Type"   in df_raw.columns else "—"
    n_origins = df_raw["Origin"].nunique() if "Origin" in df_raw.columns else "—"

    st.markdown(f"""
    <div class="dash-nav">
      <div class="dash-logo">Car<em>Price</em>&nbsp;AI</div>
      <div class="dash-nav-links">
        <a class="dash-nav-link active" href="#dash-top">Dashboard</a>
        <a class="dash-nav-link" href="#model-accuracy">Models</a>
        <a class="dash-nav-link" href="#analytics-tabs">Analytics</a>
      </div>
      <div style="display:flex;align-items:center;gap:0.85rem;">
        <span style="font-size:0.64rem;color:var(--muted);font-family:var(--fmono);">{df_raw.shape[0]:,} records</span>
        <div class="dash-badge">● Models Active</div>
      </div>
    </div>
    <div id="dash-top"></div>
    """, unsafe_allow_html=True)
    with st.container(key="home_btn"):
        if st.button("← Home", key="back_home", type="secondary"):
            st.session_state.page = "landing"
            st.rerun()

    st.markdown(f"""
    <div class="dash-banner">
      <div class="dash-banner-left">
        <div class="dash-banner-title">Vehicle Price Intelligence</div>
        <div class="dash-banner-sub">
          {df_raw.shape[0]:,} vehicles · {len(feats)} features · Poly R² {scores['pr_test']:.1f}%
        </div>
      </div>
      <div class="dash-banner-right">
        <span class="dash-tag r">Linear R² {scores['lr_test']:.1f}%</span>
        <span class="dash-tag t">Poly R² {scores['pr_test']:.1f}%</span>
        <span class="dash-tag g">{df_raw.shape[0]:,} Records</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    left_col, main_col = st.columns([1, 3.2], gap="large")

    with left_col:
      with st.container(key="left_panel"):

        st.markdown('<div class="plbl">Engine & Performance</div>', unsafe_allow_html=True)
        engine_size = st.slider("Engine Size (L)", 1.0, 8.0, 3.0, 0.1)
        horsepower  = st.slider("Horsepower (HP)", 50, 600, 200, 5)
        weight      = st.slider("Weight (lbs)", 1500, 6500, 3200, 50)

        st.markdown('<br><div class="plbl">Body Type</div>', unsafe_allow_html=True)
        car_type    = st.radio("Type", ["Sedan", "Sports", "Wagon"], label_visibility="collapsed")
        type_sedan  = int(car_type == "Sedan")
        type_sports = int(car_type == "Sports")
        type_wagon  = int(car_type == "Wagon")

        st.markdown('<br><div class="plbl">Market Origin</div>', unsafe_allow_html=True)
        origin        = st.radio("Origin", ["Asia", "Europe", "USA"],
                                 label_visibility="collapsed", horizontal=True)
        origin_europe = int(origin == "Europe")
        origin_usa    = int(origin == "USA")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="predict-btn-wrap">', unsafe_allow_html=True)
        predict_btn = st.button("Predict Price")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<br><div class="plbl" id="model-accuracy">Model Accuracy</div>', unsafe_allow_html=True)
        for lbl, val, clr in [
            ("LR Train",   scores["lr_train"],  "#3D5AFE"),
            ("LR Test",    scores["lr_test"],   "#5B76FF"),
            ("Poly Train", scores["pr_train"],  "#16916F"),
            ("Poly Test",  scores["pr_test"],   "#16916F"),
        ]:
            st.markdown(f"""
            <div class="sbar">
              <div class="sbar-top"><span>{lbl}</span><span>{val:.1f}%</span></div>
              <div class="sbar-track">
                <div class="sbar-fill" style="width:{min(val,100):.1f}%;
                  background:{clr};"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<br><div class="plbl">Dataset</div>', unsafe_allow_html=True)

        with st.expander("Upload / Replace Dataset", expanded=(uploaded_df is None and not csv_path)):
            new_file = st.file_uploader(
                "Drop CSV here",
                type=None,
                key="file_uploader_dash",
                label_visibility="collapsed",
            )
            if new_file is not None:
                ext = new_file.name.split(".")[-1].lower()
                if ext == "csv":
                    try:
                        df_new = pd.read_csv(new_file)
                        required = {"MSRP", "Type", "Origin", "EngineSize", "Horsepower", "Weight"}
                        missing = required - set(df_new.columns)
                        if missing:
                            st.error(f"Missing: {', '.join(missing)}")
                        else:
                            st.session_state.uploaded_df = df_new
                            st.session_state.csv_path = None
                            st.markdown(f"""
                            <div class="upload-success">
                              ✓ {new_file.name}<br>
                              <span style="opacity:0.7;">{len(df_new):,} records</span>
                            </div>""", unsafe_allow_html=True)
                            st.rerun()
                    except Exception as e:
                        st.error(str(e))
                else:
                    st.warning("CSV files only")

        if uploaded_df is not None:
            st.markdown(f"""
            <div class="upload-success" style="margin-top:0.4rem;">
              ✓ &nbsp;{df_raw.shape[0]:,} records loaded
            </div>""", unsafe_allow_html=True)
        elif csv_path:
            st.markdown(f"""
            <div style="font-size:0.62rem;color:var(--muted);margin-top:0.3rem;padding:0.4rem 0.5rem;
                        background:var(--surf2);border:1px solid var(--border);border-radius:6px;
                        font-family:var(--fmono);">
              Cars Data.csv
            </div>""", unsafe_allow_html=True)



    LIGHT_BG = "#FFFFFF"
    LIGHT_BORDER = "#E5E8F0"
    TXT_DARK = "#1A2233"
    TXT_MUTED = "#64708A"
    car_images = {
        "Sedan":  "image/Sedan Car.jpg",
        "Sports": "image/Sports Car.jpg",
        "Wagon":  "image/Wagon Car.jpg",
    }

    with main_col:
        st.markdown('<div style="padding:1rem 1rem 0;">', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="kpi-row">
          <div class="kpi">
            <div class="kpi-lbl">Avg Market Price</div>
            <div class="kpi-val">${avg_msrp/1000:.1f}K</div>
            <div class="kpi-sub">across {df_raw.shape[0]:,} vehicles</div>
          </div>
          <div class="kpi">
            <div class="kpi-lbl">Highest MSRP</div>
            <div class="kpi-val">${max_msrp/1000:.0f}K</div>
            <div class="kpi-sub">dataset peak</div>
          </div>
          <div class="kpi">
            <div class="kpi-lbl">Body Types</div>
            <div class="kpi-val">{n_types}</div>
            <div class="kpi-sub">distinct categories</div>
          </div>
          <div class="kpi">
            <div class="kpi-lbl">Origins</div>
            <div class="kpi-val">{n_origins}</div>
            <div class="kpi-sub">USA · Europe · Asia</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div id="analytics-tabs"></div>', unsafe_allow_html=True)
        tab1, tab2, tab3 = st.tabs(["Prediction", "Analytics", "Correlation"])

        with tab1:
            if predict_btn:
                row = {k: [0] for k in feats}
                for k, v in [
                    ("EngineSize", engine_size), ("Horsepower", horsepower), ("Weight", weight),
                    ("Type_Sedan", type_sedan),  ("Type_Sports", type_sports), ("Type_Wagon", type_wagon),
                    ("Origin_Europe", origin_europe), ("Origin_USA", origin_usa),
                ]:
                    if k in row:
                        row[k] = [v]

                df_in    = pd.DataFrame(row)[feats]
                scaled   = scaler.transform(df_in)
                lr_pred  = lr.predict(scaled)[0]
                pr_pred  = pr.predict(poly.transform(scaled))[0]
                avg_pred = (lr_pred + pr_pred) / 2

                prev_col, res_col = st.columns([1.2, 2], gap="medium")

                with prev_col:
                    st.markdown('<div class="sec">Vehicle Preview</div>', unsafe_allow_html=True)
                    img_path = car_images.get(car_type)
                    if img_path and os.path.exists(img_path):
                        st.markdown('<div class="img-wrap">', unsafe_allow_html=True)
                        st.image(Image.open(img_path), use_container_width=True)
                        st.markdown(f"""
                        <div class="img-foot">
                          <span>Class: <strong style="color:var(--txt);">{car_type}</strong></span>
                          <span style="color:var(--accent2);">●
                            <span style="color:var(--txt);margin-left:3px;">{origin}</span>
                          </span>
                        </div>
                        </div>""", unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style="background:var(--surf2);border:1px dashed var(--border2);
                                    border-radius:10px;padding:2.5rem 1rem;text-align:center;
                                    color:var(--muted);margin-bottom:0.85rem;">
                          <div style="font-size:1.3rem;opacity:0.5;">🚗</div>
                          <p style="font-size:0.74rem;margin-top:0.5rem;color:var(--muted2);font-weight:500;">
                            {car_type} · {origin}
                          </p>
                          <p style="font-size:0.64rem;margin-top:0.25rem;color:var(--muted);">
                            Add image/ folder for preview
                          </p>
                        </div>""", unsafe_allow_html=True)

                    st.markdown('<div class="sec" style="margin-top:0.5rem;">Specifications</div>', unsafe_allow_html=True)
                    st.markdown(f"""
                    <table class="spec">
                      <tr><td>Engine</td><td>{engine_size:.1f} L</td></tr>
                      <tr><td>Horsepower</td><td>{horsepower} HP</td></tr>
                      <tr><td>Weight</td><td>{weight:,} lbs</td></tr>
                      <tr><td>Type</td><td>{car_type}</td></tr>
                      <tr><td>Origin</td><td>{origin}</td></tr>
                    </table>
                    """, unsafe_allow_html=True)

                with res_col:
                    st.markdown('<div class="sec">Estimation Results</div>', unsafe_allow_html=True)

                    st.markdown(f"""
                    <div class="pred-box">
                      <div class="pred-eye">Estimated Market Value</div>
                      <div class="pred-amt">${avg_pred:,.0f}</div>
                      <div class="pred-note">Ensemble average · Linear + Polynomial Regression</div>
                      <div class="pred-row">
                        <div>
                          <div class="pred-sub-lbl">Linear Regression</div>
                          <div class="pred-sub-val">${lr_pred:,.0f}</div>
                        </div>
                        <div>
                          <div class="pred-sub-lbl">Polynomial (deg 2)</div>
                          <div class="pred-sub-val">${pr_pred:,.0f}</div>
                        </div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

                    diff_pct  = abs(lr_pred - pr_pred) / max(avg_pred, 1) * 100
                    agreement = max(0, 100 - diff_pct * 2)
                    agree_clr = ("#16916F" if agreement >= 70
                                 else "#B8842E" if agreement >= 40
                                 else "#C7466B")

                    ag_col, rng_col = st.columns(2, gap="medium")
                    with ag_col:
                        st.markdown('<div class="sec" style="margin-top:0.3rem;">Model Agreement</div>', unsafe_allow_html=True)
                        st.markdown(f"""
                        <div style="margin-bottom:0.4rem;">
                          <div style="background:var(--border);border-radius:4px;height:8px;overflow:hidden;margin-bottom:0.4rem;">
                            <div style="height:100%;width:{agreement:.0f}%;border-radius:4px;
                                        background:{agree_clr};"></div>
                          </div>
                          <div style="font-size:0.64rem;color:var(--muted);">
                            Agreement: <strong style="color:var(--txt);">{agreement:.0f}%</strong>
                            &nbsp;·&nbsp; Spread: <strong style="color:var(--gold);">${abs(lr_pred-pr_pred):,.0f}</strong>
                          </div>
                        </div>
                        """, unsafe_allow_html=True)

                    with rng_col:
                        st.markdown('<div class="sec" style="margin-top:0.3rem;">Price Range</div>', unsafe_allow_html=True)
                        lo = min(lr_pred, pr_pred)
                        hi = max(lr_pred, pr_pred)
                        st.markdown(f"""
                        <div style="background:var(--surf2);border:1px solid var(--border);
                                    border-radius:8px;padding:0.75rem 0.9rem;">
                          <div style="font-size:0.6rem;color:var(--muted);text-transform:uppercase;margin-bottom:0.3rem;">Low → High Estimate</div>
                          <div style="font-family:'Manrope',sans-serif;font-size:1.05rem;font-weight:700;color:var(--txt);">
                            ${lo:,.0f}
                            <span style="color:var(--muted);font-weight:300;font-size:0.8rem;"> → </span>
                            ${hi:,.0f}
                          </div>
                          <div style="font-size:0.6rem;color:var(--muted2);margin-top:0.3rem;">
                            Spread: ${hi-lo:,.0f}
                          </div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="pred-empty">
                  <div class="eicon">🚘</div>
                  <p>Configure specs in the left panel<br>
                     and click <span style="color:var(--accent2);">Predict Price</span></p>
                </div>
                """, unsafe_allow_html=True)

        with tab2:
            st.markdown('<div class="sec">Price Distribution</div>', unsafe_allow_html=True)
            fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), facecolor=LIGHT_BG)
            for ax in axes:
                ax.set_facecolor(LIGHT_BG)
                for sp in ax.spines.values(): sp.set_edgecolor(LIGHT_BORDER)
                ax.tick_params(colors=TXT_MUTED, labelsize=8)

            msrp_vals = df_raw["MSRP"].astype(float)
            axes[0].hist(msrp_vals, bins=45, color="#3D5AFE", alpha=0.85, edgecolor=LIGHT_BG, linewidth=0.3)
            axes[0].set_title("MSRP Distribution", color=TXT_DARK, fontsize=10, pad=10)
            axes[0].set_xlabel("Price ($)", color=TXT_MUTED, fontsize=8)
            axes[0].set_ylabel("Vehicles",  color=TXT_MUTED, fontsize=8)
            axes[0].xaxis.set_major_formatter(mtick.FuncFormatter(lambda x,_: f"${x/1000:.0f}K"))
            axes[0].axvline(msrp_vals.mean(), color="#B8842E", linewidth=1.2, linestyle="--", alpha=0.85)

            if "Type" in df_raw.columns:
                tdata   = df_raw.groupby("Type")["MSRP"].mean().sort_values()
                palette = ["#3D5AFE","#5B76FF","#16916F","#B8842E","#8A93A8"][:len(tdata)]
                bars    = axes[1].barh(tdata.index, tdata.values, color=palette, alpha=0.9)
                axes[1].set_title("Avg Price by Body Type", color=TXT_DARK, fontsize=10, pad=10)
                axes[1].set_xlabel("Avg MSRP ($)", color=TXT_MUTED, fontsize=8)
                axes[1].xaxis.set_major_formatter(mtick.FuncFormatter(lambda x,_: f"${x/1000:.0f}K"))
                for bar, val in zip(bars, tdata.values):
                    axes[1].text(val+200, bar.get_y()+bar.get_height()/2,
                                 f"${val/1000:.1f}K", va="center", color=TXT_DARK, fontsize=7.5)

            plt.tight_layout(pad=1.5)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            st.markdown('<div class="sec">Feature Relationships</div>', unsafe_allow_html=True)
            num_cols = [c for c in ["MSRP","EngineSize","Horsepower","Weight"] if c in df_m.columns]
            pp = sns.pairplot(
                df_m[num_cols].sample(min(300,len(df_m)), random_state=1),
                height=1.9, aspect=1,
                plot_kws={"alpha":0.45,"color":"#3D5AFE","s":8,"edgecolors":"none"},
                diag_kws={"color":"#B8842E","alpha":0.7,"bins":20},
            )
            pp.figure.patch.set_facecolor(LIGHT_BG)
            for ax in pp.axes.flat:
                ax.set_facecolor(LIGHT_BG)
                ax.tick_params(colors=TXT_MUTED, labelsize=7)
                ax.xaxis.label.set_color(TXT_MUTED)
                ax.yaxis.label.set_color(TXT_MUTED)
                for sp in ax.spines.values(): sp.set_edgecolor(LIGHT_BORDER)
            st.pyplot(pp.figure, use_container_width=True)
            plt.close()

        with tab3:
            st.markdown('<div class="sec">Feature Correlation Matrix</div>', unsafe_allow_html=True)
            corr = df_m.corr()
            fig2, ax2 = plt.subplots(figsize=(10, 6.5), facecolor=LIGHT_BG)
            ax2.set_facecolor(LIGHT_BG)
            mask = np.triu(np.ones_like(corr, dtype=bool))
            cmap = sns.diverging_palette(220, 20, s=75, l=50, as_cmap=True)
            sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap=cmap,
                        linewidths=0.5, linecolor=LIGHT_BG,
                        annot_kws={"size":8,"color":TXT_DARK,"weight":"600"},
                        ax=ax2, cbar_kws={"shrink":0.75,"pad":0.02}, vmin=-1, vmax=1)
            ax2.tick_params(colors=TXT_MUTED, labelsize=8)
            ax2.set_xticklabels(ax2.get_xticklabels(), rotation=35, ha="right")
            ax2.set_yticklabels(ax2.get_yticklabels(), fontsize=8)
            ax2.figure.axes[-1].tick_params(colors=TXT_MUTED, labelsize=7)
            plt.title("Pearson Correlation — All Features", color=TXT_DARK, fontsize=11, pad=14, loc="left")
            plt.tight_layout()
            st.pyplot(fig2, use_container_width=True)
            plt.close(fig2)

            st.markdown('<div class="sec">MSRP Feature Impact</div>', unsafe_allow_html=True)
            if "MSRP" in corr.columns:
                msrp_c = corr["MSRP"].drop("MSRP").sort_values()
                colors  = ["#8A93A8" if v < 0 else "#3D5AFE" for v in msrp_c.values]
                fig3, ax3 = plt.subplots(figsize=(9, 3.5), facecolor=LIGHT_BG)
                ax3.set_facecolor(LIGHT_BG)
                for sp in ax3.spines.values(): sp.set_edgecolor(LIGHT_BORDER)
                ax3.barh(msrp_c.index, msrp_c.values, color=colors, alpha=0.9, height=0.52)
                ax3.axvline(0, color=LIGHT_BORDER, linewidth=1)
                ax3.tick_params(colors=TXT_MUTED, labelsize=8)
                ax3.set_xlabel("Correlation Coefficient", color=TXT_MUTED, fontsize=8)
                ax3.set_title("Correlation with MSRP", color=TXT_DARK, fontsize=10, pad=10, loc="left")
                for i, (val, _) in enumerate(zip(msrp_c.values, msrp_c.index)):
                    ax3.text(val+(0.01 if val>=0 else -0.01), i, f"{val:+.2f}",
                             va="center", ha="left" if val>=0 else "right",
                             color=TXT_DARK, fontsize=7.5, fontweight="600")
                plt.tight_layout()
                st.pyplot(fig3, use_container_width=True)
                plt.close(fig3)

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center;padding:1.5rem 0 0.75rem;font-size:0.62rem;
                color:var(--muted);
                border-top:1px solid var(--border);margin-top:1rem;font-family:var(--fmono);">
      CarPrice AI · Linear &amp; Polynomial Regression · Streamlit &amp; scikit-learn
    </div>
    """, unsafe_allow_html=True)


# ── ROUTER ──────────────────────────────────────────────────────
if st.session_state.page == "landing":
    render_landing()
else:
    render_dashboard()