"""
CarPrice AI — Landing Page + Dashboard
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
import io
# import base64

# def get_base64(file_path):
#     with open(file_path, "rb") as f:
#         return base64.b64encode(f.read()).decode()
      
# img_base64 = get_base64("image/Sports Car.jpg")
      
      
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
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Inter:wght@300;400;500;600&display=swap');

:root {
  --bg:      #03040A;
  --panel:   #070A14;
  --surf:    #0A0E1A;
  --surf2:   #0F1422;
  --border:  #161E36;
  --border2: #1F2D50;
  --accent:  #E63946;
  --accent2: #FF6B6B;
  --gold:    #FFD166;
  --teal:    #06D6A0;
  --blue:    #118AB2;
  --purple:  #7B61FF;
  --txt:     #E8EAF2;
  --muted:   #3D4668;
  --muted2:  #5A6690;
  --fhd:     'Syne', sans-serif;
  --fbd:     'Inter', sans-serif;
}
# /* ── BACKGROUND CAR IMAGE ── */
# .stApp {
#   background: 
#     linear-gradient(rgba(3,4,10,0.92), rgba(3,4,10,0.97)),
#     url("data:image/jpg;base64,{img_base64}");
#   background-size: cover;
#   background-position: center;
#   background-attachment: fixed;
# }
# .stApp::before {
#   content: "";
#   position: fixed;
#   top: 0; left: 0;
#   width: 100%; height: 100%;
#   background: radial-gradient(circle at 20% 30%, rgba(230,57,70,0.12), transparent 40%),
#               radial-gradient(circle at 80% 70%, rgba(6,214,160,0.08), transparent 45%);
#   pointer-events: none;
# }
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body, [class*="css"] {
  font-family: var(--fbd);
  background: var(--bg) !important;
  color: var(--txt);
  -webkit-font-smoothing: antialiased;
}
.stApp { background: var(--bg) !important; }
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
  border-bottom: 1px solid rgba(255,255,255,0.04);
  background: rgba(3,4,10,0.9);
  backdrop-filter: blur(12px);
  position: sticky; top: 0; z-index: 99;
}
.lp-logo { font-family: var(--fhd); font-size: 1.3rem; font-weight: 800;
  letter-spacing: 0.08em; text-transform: uppercase; color: #fff; }
.lp-logo em { color: var(--accent); font-style: normal; }
.lp-pill {
  font-size: 0.6rem; font-weight: 700; letter-spacing: 0.14em; text-transform: uppercase;
  background: rgba(230,57,70,0.1); border: 1px solid rgba(230,57,70,0.28);
  color: var(--accent2); padding: 0.3rem 0.9rem; border-radius: 999px;
}

.hero-wrap {
  padding: 5rem 3rem 3rem;
  background:
    radial-gradient(ellipse 70% 55% at 75% -5%, rgba(230,57,70,0.13) 0%, transparent 55%),
    radial-gradient(ellipse 55% 45% at 5% 95%, rgba(6,214,160,0.07) 0%, transparent 55%);
}
.hero-tag {
  display: inline-flex; align-items: center; gap: 0.55rem;
  font-size: 0.65rem; font-weight: 600; letter-spacing: 0.2em; text-transform: uppercase;
  color: var(--teal); margin-bottom: 1.5rem;
}
.hero-tag::before { content:''; width:18px; height:1px; background:var(--teal); display:block; }
.hero-h1 {
  font-family: var(--fhd); font-size: clamp(3rem,5.5vw,5.2rem); font-weight: 800;
  line-height: 0.94; letter-spacing: -0.03em; color: #fff; margin-bottom: 1.5rem;
}
.hero-h1 .gr {
  display: block;
  background: linear-gradient(125deg, var(--accent) 0%, var(--gold) 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.hero-sub {
  font-size: 1rem; line-height: 1.85; color: rgba(232,234,242,0.55);
  max-width: 520px; font-weight: 300; margin-bottom: 2.5rem;
}
.hero-chips { display: flex; flex-wrap: wrap; gap: 0.6rem; margin-bottom: 3rem; }
.hero-chip {
  display: inline-flex; align-items: center; gap: 0.4rem;
  font-size: 0.7rem; font-weight: 500; letter-spacing: 0.06em;
  color: var(--muted2); background: var(--surf); border: 1px solid var(--border);
  border-radius: 999px; padding: 0.38rem 0.85rem;
}
.hero-chip em { width:6px; height:6px; border-radius:50%; display:block; font-style:normal; }
.hero-chip em.red  { background: var(--accent); }
.hero-chip em.teal { background: var(--teal); }
.hero-chip em.gold { background: var(--gold); }

.stat-row {
  display: grid; grid-template-columns: repeat(4,1fr);
  gap: 1px; background: var(--border);
  border: 1px solid var(--border); border-radius: 16px; overflow: hidden;
  margin-bottom: 5rem;
}
.stat-cell { background: var(--surf); padding: 1.6rem 1.4rem; }
.stat-num { font-family:var(--fhd); font-size:2.4rem; font-weight:800; color:#fff; line-height:1; margin-bottom:0.3rem; }
.stat-num em { color:var(--accent); font-style:normal; }
.stat-lbl { font-size:0.65rem; font-weight:500; letter-spacing:0.1em; text-transform:uppercase; color:var(--muted); }

.how-band {
  background: var(--surf); border-top:1px solid var(--border); border-bottom:1px solid var(--border);
  padding: 4rem 3rem;
}
.band-tag { font-size:0.62rem; font-weight:700; letter-spacing:0.2em; text-transform:uppercase; color:var(--muted); margin-bottom:0.4rem; }
.band-h   { font-family:var(--fhd); font-size:1.9rem; font-weight:800; color:#fff; letter-spacing:-0.02em; margin-bottom:2.5rem; }
.steps { display:grid; grid-template-columns:repeat(4,1fr); gap:1.5rem; position:relative; }
.steps::before {
  content:''; position:absolute; top:26px; left:80px; right:80px; height:1px;
  background: linear-gradient(90deg,transparent,var(--border2),var(--accent),var(--border2),transparent);
}
.step { text-align:center; position:relative; z-index:1; }
.step-n {
  width:52px; height:52px; border-radius:50%;
  background:var(--panel); border:2px solid var(--accent);
  display:flex; align-items:center; justify-content:center; margin:0 auto 1rem;
  font-family:var(--fhd); font-size:1.1rem; font-weight:800; color:var(--accent);
}
.step-h { font-family:var(--fhd); font-size:0.82rem; font-weight:700; letter-spacing:0.06em;
  text-transform:uppercase; color:#fff; margin-bottom:0.4rem; }
.step-p { font-size:0.73rem; line-height:1.7; color:var(--muted2); }

.feat-band { padding: 4rem 3rem 3rem; }
.feat-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:1.25rem; }
.feat-card {
  background:var(--surf); border:1px solid var(--border); border-radius:18px; padding:1.75rem;
  transition: border-color .25s, transform .25s;
}
.feat-card:hover { border-color:var(--border2); transform:translateY(-3px); }
.feat-icon { width:44px; height:44px; border-radius:10px; display:flex;
  align-items:center; justify-content:center; font-size:1.25rem; margin-bottom:1.1rem; }
.feat-icon.r { background:rgba(230,57,70,0.1); }
.feat-icon.t { background:rgba(6,214,160,0.08); }
.feat-icon.g { background:rgba(255,209,102,0.08); }
.feat-h { font-family:var(--fhd); font-size:1rem; font-weight:700; color:#fff; margin-bottom:0.5rem; }
.feat-p { font-size:0.78rem; line-height:1.8; color:var(--muted2); font-weight:300; }

.lp-footer {
  text-align:center; padding:1.75rem 0 1.25rem;
  font-size:0.6rem; color:var(--border2); letter-spacing:0.14em; text-transform:uppercase;
  border-top:1px solid var(--border); margin-top:1rem;
}

/* ════════════════════════════════════
   DASHBOARD TOPBAR — single bar only
════════════════════════════════════ */
.dash-nav {
  display: flex; align-items: center; justify-content: space-between;
  padding: 0 2rem; height: 58px;
  background: rgba(7,10,20,0.98);
  border-bottom: 1px solid var(--border);
  backdrop-filter: blur(16px);
}
.dash-logo { font-family:var(--fhd); font-size:1.2rem; font-weight:800;
  letter-spacing:0.08em; text-transform:uppercase; color:#fff; }
.dash-logo em { color:var(--accent); font-style:normal; }
.dash-nav-links { display:flex; align-items:center; gap:0.1rem; }
.dash-nav-link {
  font-size:0.68rem; font-weight:500; letter-spacing:0.06em; text-transform:uppercase;
  color:var(--muted); text-decoration:none;
  padding:0.3rem 0.75rem; border-radius:6px;
}
.dash-badge {
  font-size:0.6rem; font-weight:600; letter-spacing:0.08em; text-transform:uppercase;
  background:rgba(6,214,160,0.07); border:1px solid rgba(6,214,160,0.2);
  color:var(--teal); padding:0.25rem 0.75rem; border-radius:999px;
}

/* ── BACK BUTTON — small, inline, subtle ── */
.back-btn-wrap {
  display: inline-flex; align-items: center; gap: 0.35rem;
  font-size: 0.65rem; font-weight: 500; letter-spacing: 0.06em;
  color: var(--muted2); text-decoration: none; cursor: pointer;
  border: 1px solid var(--border2); border-radius: 6px;
  padding: 0.25rem 0.65rem; background: transparent;
  transition: border-color .2s, color .2s;
}
.back-btn-wrap:hover { color: var(--accent); border-color: rgba(230,57,70,0.4); }

/* Override Streamlit button specifically for back button */
div[data-testid="stButton"].back-home-btn > button {
  background: transparent !important;
  border: 1px solid var(--border2) !important;
  color: var(--muted2) !important;
  font-family: var(--fbd) !important;
  font-size: 0.65rem !important;
  font-weight: 500 !important;
  letter-spacing: 0.06em !important;
  text-transform: none !important;
  padding: 0.22rem 0.7rem !important;
  border-radius: 6px !important;
  width: auto !important;
  min-width: unset !important;
  box-shadow: none !important;
  line-height: 1.4 !important;
}
div[data-testid="stButton"].back-home-btn > button:hover {
  border-color: rgba(230,57,70,0.4) !important;
  color: var(--accent) !important;
  transform: none !important;
  box-shadow: none !important;
}

/* ── DASHBOARD HEADER BANNER ── */
.dash-banner {
  padding: 1.4rem 2rem 1.2rem;
  background: linear-gradient(180deg, rgba(230,57,70,0.04) 0%, transparent 100%);
  border-bottom: 1px solid var(--border);
  display: flex; align-items: flex-end; justify-content: space-between;
}
.dash-banner-title {
  font-family:var(--fhd); font-size:1.5rem; font-weight:800;
  color:#fff; letter-spacing:-0.01em; margin-bottom:0.15rem;
}
.dash-banner-sub { font-size:0.73rem; color:var(--muted2); font-weight:400; }
.dash-banner-right { display:flex; gap:0.5rem; }
.dash-tag {
  font-size:0.58rem; font-weight:600; letter-spacing:0.08em; text-transform:uppercase;
  padding:0.25rem 0.65rem; border-radius:6px; border:1px solid;
}
.dash-tag.r { color:var(--accent2); background:rgba(230,57,70,0.08); border-color:rgba(230,57,70,0.2); }
.dash-tag.t { color:var(--teal);    background:rgba(6,214,160,0.06); border-color:rgba(6,214,160,0.18); }
.dash-tag.g { color:var(--gold);    background:rgba(255,209,102,0.06); border-color:rgba(255,209,102,0.18); }

/* ── KPI CARDS ── */
.kpi-row { display:grid; grid-template-columns:repeat(4,1fr); gap:0.85rem; margin-bottom:1.25rem; }
.kpi {
  background:var(--surf2); border:1px solid var(--border); border-radius:12px;
  padding:1.1rem 1.2rem; position:relative; overflow:hidden;
  transition: border-color .2s;
}
.kpi:hover { border-color:var(--border2); }
.kpi::after { content:''; position:absolute; top:0;left:0;right:0;height:2px; }
.kpi.red::after    { background:linear-gradient(90deg,var(--accent),transparent); }
.kpi.gold::after   { background:linear-gradient(90deg,var(--gold),transparent); }
.kpi.teal::after   { background:linear-gradient(90deg,var(--teal),transparent); }
.kpi.purple::after { background:linear-gradient(90deg,var(--purple),transparent); }
.kpi-lbl { font-size:0.6rem; font-weight:500; letter-spacing:0.08em; text-transform:uppercase;
  color:var(--muted); margin-bottom:0.35rem; }
.kpi-val { font-family:var(--fbd); font-size:1.75rem; font-weight:600; color:#fff; line-height:1; letter-spacing:-0.01em; }
.kpi-sub { font-size:0.6rem; color:var(--teal); margin-top:0.25rem; font-weight:400; }

/* ── LEFT PANEL ── */
.plbl {
  font-size:0.6rem; font-weight:600; letter-spacing:0.12em; text-transform:uppercase;
  color:var(--muted); padding-bottom:0.4rem; border-bottom:1px solid var(--border);
  margin-bottom:0.7rem;
}

/* ── SECTION HEADER ── */
.sec {
  display:flex; align-items:center; gap:0.5rem;
  font-family:var(--fhd); font-size:0.78rem; font-weight:700;
  letter-spacing:0.08em; text-transform:uppercase; color:var(--txt);
  margin:1.1rem 0 0.75rem;
}
.sec::before { content:''; width:3px; height:12px; background:var(--accent);
  border-radius:2px; flex-shrink:0; }

/* ── SCORE BARS ── */
.sbar { margin-bottom:0.65rem; }
.sbar-top { display:flex; justify-content:space-between; font-size:0.58rem;
  color:var(--muted); margin-bottom:0.2rem; letter-spacing:0.04em; text-transform:uppercase; }
.sbar-track { background:var(--border); border-radius:3px; height:4px; overflow:hidden; }
.sbar-fill { height:100%; border-radius:3px; }

/* ── PREDICTION BOX ── */
.pred-box {
  background: linear-gradient(135deg, #080614, #0A0810 55%, #07090F);
  border: 1px solid rgba(230,57,70,0.28); border-radius:16px;
  padding: 1.75rem 1.5rem 1.4rem; text-align:center;
  box-shadow: 0 0 80px rgba(230,57,70,0.04);
  margin-bottom: 1.25rem;
}
.pred-eye  { font-size:0.58rem; font-weight:600; letter-spacing:0.16em; text-transform:uppercase; color:var(--muted); margin-bottom:0.5rem; }
.pred-amt  { font-family:var(--fhd); font-size:3.8rem; font-weight:800; color:var(--accent); line-height:1;
  text-shadow: 0 0 50px rgba(230,57,70,0.38); }
.pred-note { font-size:0.6rem; color:var(--muted); margin-top:0.35rem; font-weight:400; }
.pred-row  { display:flex; justify-content:center; gap:2.5rem;
  margin-top:1.1rem; padding-top:1.1rem; border-top:1px solid var(--border); }
.pred-sub-lbl { font-size:0.55rem; font-weight:500; letter-spacing:0.1em; text-transform:uppercase; color:var(--muted); }
.pred-sub-val { font-family:var(--fhd); font-size:1.4rem; font-weight:700; color:var(--gold); }

.pred-empty {
  text-align:center; padding:3rem 1rem; color:var(--muted);
  background: var(--surf2); border:1px dashed var(--border2); border-radius:16px;
  margin-bottom:1.25rem;
}
.pred-empty .eicon { font-size:2.5rem; margin-bottom:0.6rem; }
.pred-empty p { font-family:var(--fhd); font-size:0.8rem; font-weight:600;
  letter-spacing:0.06em; text-transform:uppercase; line-height:1.65; }

/* ── SPEC TABLE ── */
.spec { width:100%; border-collapse:collapse; font-size:0.77rem; }
.spec td { padding:0.48rem 0.5rem; border-bottom:1px solid var(--border); color:var(--txt); }
.spec td:first-child { color:var(--muted); font-size:0.62rem; font-weight:500;
  letter-spacing:0.08em; text-transform:uppercase; width:46%; }
.spec td:last-child { font-weight:500; }

/* ── IMAGE WRAPPER ── */
.img-wrap {
  background:var(--surf2); border:1px solid var(--border); border-radius:14px;
  padding:0.65rem; margin-bottom:0.85rem; box-shadow:0 12px 40px rgba(0,0,0,0.45);
}
.img-foot {
  display:flex; justify-content:space-between; padding:0.35rem 0.4rem 0.1rem;
  font-size:0.7rem; letter-spacing:0.04em; text-transform:uppercase; color:var(--muted);
}

/* ── STREAMLIT OVERRIDES ── */
[data-testid="stSlider"] label {
  font-size:0.64rem !important; font-weight:500 !important;
  letter-spacing:0.08em !important; text-transform:uppercase !important; color:var(--muted) !important;
}
[data-testid="stSlider"] > div > div { background:var(--border) !important; }
[data-testid="stSlider"] > div > div > div > div { background:var(--accent) !important; }

.stRadio label { font-size:0.7rem !important; color:var(--muted2) !important; letter-spacing:0.03em !important; font-weight:400 !important; }
.stRadio > div { gap:0.3rem !important; }

[data-testid="stFileUploader"] {
  border:1px dashed var(--border2) !important;
  border-radius:10px !important; background:var(--surf2) !important;
}
[data-testid="stFileUploader"] label {
  font-size:0.64rem !important; font-weight:500 !important;
  letter-spacing:0.08em !important; text-transform:uppercase !important; color:var(--muted) !important;
}

/* Primary action button (Predict) - scoped to predict container */
.predict-btn-wrap .stButton > button {
  width:100% !important;
  background:linear-gradient(135deg, var(--accent), #9E1520) !important;
  color:#fff !important; border:none !important; border-radius:10px !important;
  font-family:var(--fbd) !important; font-size:0.85rem !important; font-weight:600 !important;
  letter-spacing:0.06em !important; text-transform:none !important;
  padding:0.72rem !important; box-shadow:0 4px 18px rgba(230,57,70,0.3) !important;
  transition:all .2s !important;
}
.predict-btn-wrap .stButton > button:hover {
  box-shadow:0 6px 28px rgba(230,57,70,0.52) !important; transform:translateY(-1px) !important;
}
/* Launch button on landing */
.launch-btn-wrap .stButton > button {
  width:100% !important;
  background:linear-gradient(135deg, var(--accent), #9E1520) !important;
  color:#fff !important; border:none !important; border-radius:10px !important;
  font-family:var(--fbd) !important; font-size:0.85rem !important; font-weight:600 !important;
  letter-spacing:0.06em !important; text-transform:none !important;
  padding:0.72rem !important; box-shadow:0 4px 18px rgba(230,57,70,0.3) !important;
  transition:all .2s !important;
}
.launch-btn-wrap .stButton > button:hover {
  box-shadow:0 6px 28px rgba(230,57,70,0.52) !important; transform:translateY(-1px) !important;
}
/* Back home — tiny ghost button, embedded in navbar */
.back-home-btn .stButton > button {
  background: transparent !important;
  border: 1px solid var(--border2) !important;
  color: var(--muted2) !important;
  font-family: var(--fbd) !important;
  font-size: 0.62rem !important;
  font-weight: 500 !important;
  letter-spacing: 0.04em !important;
  text-transform: none !important;
  padding: 0.18rem 0.55rem !important;
  border-radius: 5px !important;
  width: auto !important;
  min-width: unset !important;
  height: auto !important;
  line-height: 1.5 !important;
  box-shadow: none !important;
}
.back-home-btn .stButton > button:hover {
  border-color: rgba(230,57,70,0.35) !important;
  color: var(--accent2) !important;
  transform: none !important;
  box-shadow: none !important;
}

/* Tabs */
[data-baseweb="tab-list"] {
  background:var(--surf2) !important; border-radius:10px 10px 0 0 !important;
  border:1px solid var(--border) !important; border-bottom:none !important; gap:0 !important; padding:0 !important;
}
[data-baseweb="tab"] {
  font-family:var(--fbd) !important; font-size:0.72rem !important; font-weight:500 !important;
  letter-spacing:0.06em !important; text-transform:uppercase !important;
  color:var(--muted) !important; padding:0.65rem 1.2rem !important; border:none !important;
}
[aria-selected="true"] { color:var(--accent) !important; border-bottom:2px solid var(--accent) !important; }
[data-baseweb="tab-panel"] {
  background:var(--surf2) !important; border:1px solid var(--border) !important;
  border-top:none !important; border-radius:0 0 10px 10px !important; padding:1.4rem !important;
}

hr { border-color:var(--border) !important; }

/* Upload success notice */
.upload-success {
  display:flex; align-items:center; gap:0.5rem;
  background:rgba(6,214,160,0.06); border:1px solid rgba(6,214,160,0.2);
  border-radius:8px; padding:0.5rem 0.75rem; margin-top:0.5rem;
  font-size:0.65rem; color:var(--teal); font-weight:500; letter-spacing:0.04em;
}

/* GLOBAL GLOW BUTTON FIX */
div[data-testid="stButton"] > button {
  position: relative !important;
  border-radius: 12px !important;
  background: linear-gradient(135deg, #E63946, #9E1520) !important;
  color: white !important;
  font-weight: 600 !important;

  box-shadow: 0 0 12px rgba(230,57,70,0.4),
              0 0 30px rgba(230,57,70,0.2) !important;

  transition: 0.25s ease-in-out !important;
}

/* hover glow */
div[data-testid="stButton"] > button:hover {
  transform: translateY(-2px) scale(1.02) !important;
  box-shadow: 0 0 18px rgba(230,57,70,0.7),
              0 0 60px rgba(230,57,70,0.35) !important;
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

    # NAV
    st.markdown("""
    <div class="lp-nav">
      <div class="lp-logo">Car<em>Price</em>&nbsp;AI</div>
      <div style="display:flex;align-items:center;gap:2.2rem;">
        <a href="#" style="font-size:0.7rem;font-weight:500;letter-spacing:0.08em;text-transform:uppercase;color:var(--muted2);text-decoration:none;">Features</a>
        <a href="#" style="font-size:0.7rem;font-weight:500;letter-spacing:0.08em;text-transform:uppercase;color:var(--muted2);text-decoration:none;">How it works</a>
        <a href="#" style="font-size:0.7rem;font-weight:500;letter-spacing:0.08em;text-transform:uppercase;color:var(--muted2);text-decoration:none;">Docs</a>
      </div>
      <div class="lp-pill">⚡ ML Engine Active</div>
    </div>
    """, unsafe_allow_html=True)

    # HERO
    st.markdown("""
    <div class="hero-wrap">
      <div class="hero-tag">Automotive Intelligence Platform</div>
      <h1 class="hero-h1">
        Predict any car's
        <span class="gr">market value instantly.</span>
      </h1>
      <p class="hero-sub">
        Harness dual-model machine learning — Linear Regression paired with Polynomial
        Regression — to forecast MSRP with precision. Configure specs and get
        instant AI-powered price estimates in seconds.
      </p>
      <div class="hero-chips">
        <span class="hero-chip"><em class="red"></em> Dual ML Models</span>
        <span class="hero-chip"><em class="teal"></em> Real-Time Predictions</span>
        <span class="hero-chip"><em class="gold"></em> Deep Analytics</span>
        <span class="hero-chip"><em class="teal"></em> Correlation Heatmaps</span>
        <span class="hero-chip"><em class="red"></em> 8 Predictive Features</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # STATS
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

    # HOW IT WORKS
    st.markdown("""
    <div class="how-band">
      <div class="band-tag">Process</div>
      <div class="band-h">How it works</div>
      <div class="steps">
        <div class="step">
          <div class="step-n">01</div>
          <div class="step-h">Upload Dataset</div>
          <p class="step-p">Upload your CSV file with MSRP, engine specs, weight, body type and origin directly in the app.</p>
        </div>
        <div class="step">
          <div class="step-n">02</div>
          <div class="step-h">Train Models</div>
          <p class="step-p">Both Linear and Polynomial Regression models train instantly on your data with an 80/20 split.</p>
        </div>
        <div class="step">
          <div class="step-n">03</div>
          <div class="step-h">Configure Specs</div>
          <p class="step-p">Tune engine size, horsepower, weight, body type and market origin from the control panel.</p>
        </div>
        <div class="step">
          <div class="step-n">04</div>
          <div class="step-h">Get Prediction</div>
          <p class="step-p">Receive an ensemble price estimate with agreement score, price range and full analytics.</p>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # FEATURES
    st.markdown("""
    <div class="feat-band">
      <div class="band-tag">Capabilities</div>
      <div class="band-h">Everything you need</div>
      <div class="feat-grid">
        <div class="feat-card">
          <div class="feat-icon r">🤖</div>
          <div class="feat-h">Dual-Model Ensemble</div>
          <p class="feat-p">Linear Regression for baseline accuracy. Polynomial Regression captures non-linear price relationships. Both averaged for optimal estimates.</p>
        </div>
        <div class="feat-card">
          <div class="feat-icon t">📊</div>
          <div class="feat-h">Deep Analytics</div>
          <p class="feat-p">Explore MSRP distributions, price by body type, feature pair plots and full Pearson correlation heatmaps for any uploaded dataset.</p>
        </div>
        <div class="feat-card">
          <div class="feat-icon g">⚡</div>
          <div class="feat-h">Instant Predictions</div>
          <p class="feat-p">Sub-second price estimates with model agreement scores, confidence ranges and visual vehicle previews by body type.</p>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── LAUNCH SECTION (no upload — upload lives in dashboard) ───
    csv_path = None
    if os.path.exists("Cars Data.csv"):
        csv_path = "Cars Data.csv"

    st.markdown("<br>", unsafe_allow_html=True)
    _, c, _ = st.columns([2.5, 2, 2.5])
    with c:
        st.markdown('<div class="glow-btn">', unsafe_allow_html=True)
        if st.button("🚀  Launch Dashboard", key="launch_btn"):
            if csv_path:
                st.session_state.csv_path = csv_path
                st.session_state.uploaded_df = None
                st.session_state.page = "dashboard"
                st.rerun()
            elif st.session_state.uploaded_df is not None:
                st.session_state.csv_path = None
                st.session_state.page = "dashboard"
                st.rerun()
            else:
                st.session_state.page = "dashboard"
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center;margin-top:0.6rem;font-size:0.68rem;color:var(--muted);letter-spacing:0.04em;">
      You can upload your dataset inside the dashboard
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="lp-footer">
      CarPrice AI &nbsp;·&nbsp; Linear &amp; Polynomial Regression &nbsp;·&nbsp; Built with Streamlit &amp; scikit-learn
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  DASHBOARD
# ══════════════════════════════════════════════════════════════════
def render_dashboard():
    csv_path = st.session_state.get("csv_path")
    uploaded_df = st.session_state.get("uploaded_df")

    has_data = csv_path or (uploaded_df is not None)

    if not has_data:
        # Show navbar + upload-only state
        st.markdown("""
        <div class="dash-nav">
          <div class="dash-logo">Car<em>Price</em>&nbsp;AI</div>
          <div class="dash-badge">⚠ No Dataset</div>
        </div>
        """, unsafe_allow_html=True)
        back_c, _ = st.columns([0.08, 0.92])
        with back_c:
            st.markdown('<div class="glow-btn" style="padding:0.35rem 0 0 0.75rem;">', unsafe_allow_html=True)
            if st.button("← Home", key="back_home_nd"):
                st.session_state.page = "landing"
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="max-width:480px;margin:3rem auto;text-align:center;padding:2.5rem;
                    background:var(--surf2);border:1px dashed var(--border2);border-radius:16px;">
          <div style="font-size:2rem;margin-bottom:0.75rem;">📂</div>
          <div style="font-family:'Syne',sans-serif;font-size:1rem;font-weight:700;color:#fff;margin-bottom:0.4rem;">
            Upload a dataset to continue
          </div>
          <div style="font-size:0.72rem;color:var(--muted2);line-height:1.7;">
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

    # ── SINGLE TOPBAR with inline back button ──────────────────────
    # Render navbar HTML (back btn sits in right side via Streamlit column trick)
    st.markdown(f"""
    <div class="dash-nav">
      <div class="dash-logo">Car<em>Price</em>&nbsp;AI</div>
      <div class="dash-nav-links">
        <a class="dash-nav-link" href="#">Dashboard</a>
        <a class="dash-nav-link" href="#">Models</a>
        <a class="dash-nav-link" href="#">Analytics</a>
      </div>
      <div style="display:flex;align-items:center;gap:0.85rem;">
        <span style="font-size:0.62rem;color:var(--muted);font-weight:400;">{df_raw.shape[0]:,} records</span>
        <div class="dash-badge">● Models Active</div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    # Back button row — rendered below navbar, left-aligned, tiny
    back_col, _ = st.columns([0.08, 0.92])
    with back_col:
        st.markdown('<div class="glow-btn" style="padding:0.35rem 0 0 0.75rem;">', unsafe_allow_html=True)
        if st.button("← Home", key="back_home"):
            st.session_state.page = "landing"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # ── DASHBOARD BANNER ─────────────────────────────────────────
    st.markdown(f"""
    <div class="dash-banner">
      <div class="dash-banner-left">
        <div class="dash-banner-title">Vehicle Price Intelligence</div>
        <div class="dash-banner-sub">
          {df_raw.shape[0]:,} vehicles &nbsp;·&nbsp;
          {len(feats)} features &nbsp;·&nbsp;
          Poly R² {scores['pr_test']:.1f}%
        </div>
      </div>
      <div class="dash-banner-right">
        <span class="dash-tag r">Linear R² {scores['lr_test']:.1f}%</span>
        <span class="dash-tag t">Poly R² {scores['pr_test']:.1f}%</span>
        <span class="dash-tag g">{df_raw.shape[0]:,} Records</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── TWO-COLUMN LAYOUT ─────────────────────────────────────────
    left_col, main_col = st.columns([1, 3.2], gap="small")

    # LEFT PANEL
    with left_col:
        st.markdown('<div style="padding:1rem 0.6rem;">', unsafe_allow_html=True)

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
        st.markdown('<div class="glow-btn">', unsafe_allow_html=True)
        predict_btn = st.button("⚡  Predict Price")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<br><div class="plbl">Model Accuracy</div>', unsafe_allow_html=True)
        for lbl, val, clr in [
            ("LR Train",   scores["lr_train"],  "#E63946"),
            ("LR Test",    scores["lr_test"],   "#FF6B6B"),
            ("Poly Train", scores["pr_train"],  "#06D6A0"),
            ("Poly Test",  scores["pr_test"],   "#06D6A0"),
        ]:
            st.markdown(f"""
            <div class="sbar">
              <div class="sbar-top"><span>{lbl}</span><span>{val:.1f}%</span></div>
              <div class="sbar-track">
                <div class="sbar-fill" style="width:{min(val,100):.1f}%;
                  background:linear-gradient(90deg,{clr},transparent);"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<br><div class="plbl">Dataset</div>', unsafe_allow_html=True)

        with st.expander("📂  Upload / Replace Dataset", expanded=(uploaded_df is None and not csv_path)):
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
            <div style="font-size:0.6rem;color:var(--muted);margin-top:0.3rem;padding:0.35rem 0.5rem;
                        background:var(--surf2);border:1px solid var(--border);border-radius:6px;">
              📁 Cars Data.csv
            </div>""", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # MAIN AREA
    DARK   = "#070A14"
    BORDER = "#161E36"
    car_images = {
        "Sedan":  "image/Sedan Car.jpg",
        "Sports": "image/Sports Car.jpg",
        "Wagon":  "image/Wagon Car.jpg",
    }

    with main_col:
        st.markdown('<div style="padding:1rem 1rem 0;">', unsafe_allow_html=True)

        # KPI Row
        st.markdown(f"""
        <div class="kpi-row">
          <div class="kpi red">
            <div class="kpi-lbl">Avg Market Price</div>
            <div class="kpi-val">${avg_msrp/1000:.1f}K</div>
            <div class="kpi-sub">across {df_raw.shape[0]:,} vehicles</div>
          </div>
          <div class="kpi gold">
            <div class="kpi-lbl">Highest MSRP</div>
            <div class="kpi-val">${max_msrp/1000:.0f}K</div>
            <div class="kpi-sub">dataset peak</div>
          </div>
          <div class="kpi teal">
            <div class="kpi-lbl">Body Types</div>
            <div class="kpi-val">{n_types}</div>
            <div class="kpi-sub">distinct categories</div>
          </div>
          <div class="kpi purple">
            <div class="kpi-lbl">Origins</div>
            <div class="kpi-val">{n_origins}</div>
            <div class="kpi-sub">USA · Europe · Asia</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs(["🎯  Prediction", "📊  Analytics", "🔥  Correlation"])

        # ── TAB 1 ──────────────────────────────────────
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
                          <span>Class: <strong style="color:#fff;">{car_type}</strong></span>
                          <span style="color:var(--accent);">●
                            <span style="color:var(--txt);margin-left:3px;">{origin}</span>
                          </span>
                        </div>
                        </div>""", unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style="background:var(--surf2);border:1px dashed var(--border2);
                                    border-radius:14px;padding:2.5rem 1rem;text-align:center;
                                    color:var(--muted);margin-bottom:0.85rem;">
                          <div style="font-size:1.4rem;">🚗</div>
                          <p style="font-size:0.72rem;margin-top:0.5rem;color:var(--muted2);font-weight:500;">
                            {car_type} · {origin}
                          </p>
                          <p style="font-size:0.62rem;margin-top:0.25rem;color:var(--muted);">
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
                    agree_clr = ("#06D6A0" if agreement >= 70
                                 else "#FFD166" if agreement >= 40
                                 else "#E63946")

                    ag_col, rng_col = st.columns(2, gap="medium")
                    with ag_col:
                        st.markdown('<div class="sec" style="margin-top:0.3rem;">Model Agreement</div>', unsafe_allow_html=True)
                        st.markdown(f"""
                        <div style="margin-bottom:0.4rem;">
                          <div style="background:var(--border);border-radius:4px;height:8px;overflow:hidden;margin-bottom:0.4rem;">
                            <div style="height:100%;width:{agreement:.0f}%;border-radius:4px;
                                        background:linear-gradient(90deg,{agree_clr},transparent);"></div>
                          </div>
                          <div style="font-size:0.62rem;color:var(--muted);letter-spacing:0.06em;">
                            Agreement: <strong style="color:#E8EAF2;">{agreement:.0f}%</strong>
                            &nbsp;·&nbsp; Spread: <strong style="color:var(--gold);">${abs(lr_pred-pr_pred):,.0f}</strong>
                          </div>
                        </div>
                        """, unsafe_allow_html=True)

                    with rng_col:
                        st.markdown('<div class="sec" style="margin-top:0.3rem;">Price Range</div>', unsafe_allow_html=True)
                        lo = min(lr_pred, pr_pred)
                        hi = max(lr_pred, pr_pred)
                        st.markdown(f"""
                        <div style="background:var(--panel);border:1px solid var(--border);
                                    border-radius:10px;padding:0.75rem 0.9rem;">
                          <div style="font-size:0.58rem;color:var(--muted);letter-spacing:0.1em;
                                      text-transform:uppercase;margin-bottom:0.3rem;">Low → High Estimate</div>
                          <div style="font-family:'Syne',sans-serif;font-size:1.05rem;font-weight:700;color:#E8EAF2;">
                            ${lo:,.0f}
                            <span style="color:var(--muted);font-weight:300;font-size:0.8rem;"> → </span>
                            ${hi:,.0f}
                          </div>
                          <div style="font-size:0.58rem;color:var(--muted2);margin-top:0.3rem;">
                            Spread: ${hi-lo:,.0f}
                          </div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="pred-empty">
                  <div class="eicon">🚘</div>
                  <p>Configure specs in the left panel<br>
                     and click <span style="color:#E63946;">⚡ Predict Price</span></p>
                </div>
                """, unsafe_allow_html=True)

        # ── TAB 2 ANALYTICS ────────────────────────────
        with tab2:
            st.markdown('<div class="sec">Price Distribution</div>', unsafe_allow_html=True)
            fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), facecolor=DARK)
            for ax in axes:
                ax.set_facecolor(DARK)
                for sp in ax.spines.values(): sp.set_edgecolor(BORDER)
                ax.tick_params(colors="#3D4668", labelsize=8)

            msrp_vals = df_raw["MSRP"].astype(float)
            axes[0].hist(msrp_vals, bins=45, color="#E63946", alpha=0.82, edgecolor=DARK, linewidth=0.3)
            axes[0].set_title("MSRP Distribution", color="#E8EAF2", fontsize=10, pad=10)
            axes[0].set_xlabel("Price ($)", color="#3D4668", fontsize=8)
            axes[0].set_ylabel("Vehicles",  color="#3D4668", fontsize=8)
            axes[0].xaxis.set_major_formatter(mtick.FuncFormatter(lambda x,_: f"${x/1000:.0f}K"))
            axes[0].axvline(msrp_vals.mean(), color="#FFD166", linewidth=1.2, linestyle="--", alpha=0.75)

            if "Type" in df_raw.columns:
                tdata   = df_raw.groupby("Type")["MSRP"].mean().sort_values()
                palette = ["#E63946","#FF6B6B","#FFD166","#06D6A0","#118AB2"][:len(tdata)]
                bars    = axes[1].barh(tdata.index, tdata.values, color=palette, alpha=0.88)
                axes[1].set_title("Avg Price by Body Type", color="#E8EAF2", fontsize=10, pad=10)
                axes[1].set_xlabel("Avg MSRP ($)", color="#3D4668", fontsize=8)
                axes[1].xaxis.set_major_formatter(mtick.FuncFormatter(lambda x,_: f"${x/1000:.0f}K"))
                for bar, val in zip(bars, tdata.values):
                    axes[1].text(val+200, bar.get_y()+bar.get_height()/2,
                                 f"${val/1000:.1f}K", va="center", color="#E8EAF2", fontsize=7.5)

            plt.tight_layout(pad=1.5)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            st.markdown('<div class="sec">Feature Relationships</div>', unsafe_allow_html=True)
            num_cols = [c for c in ["MSRP","EngineSize","Horsepower","Weight"] if c in df_m.columns]
            pp = sns.pairplot(
                df_m[num_cols].sample(min(300,len(df_m)), random_state=1),
                height=1.9, aspect=1,
                plot_kws={"alpha":0.42,"color":"#E63946","s":8,"edgecolors":"none"},
                diag_kws={"color":"#FFD166","alpha":0.7,"bins":20},
            )
            pp.figure.patch.set_facecolor(DARK)
            for ax in pp.axes.flat:
                ax.set_facecolor(DARK)
                ax.tick_params(colors="#3D4668", labelsize=7)
                ax.xaxis.label.set_color("#3D4668")
                ax.yaxis.label.set_color("#3D4668")
                for sp in ax.spines.values(): sp.set_edgecolor(BORDER)
            st.pyplot(pp.figure, use_container_width=True)
            plt.close()

        # ── TAB 3 CORRELATION ──────────────────────────
        with tab3:
            st.markdown('<div class="sec">Feature Correlation Matrix</div>', unsafe_allow_html=True)
            corr = df_m.corr()
            fig2, ax2 = plt.subplots(figsize=(10, 6.5), facecolor=DARK)
            ax2.set_facecolor(DARK)
            mask = np.triu(np.ones_like(corr, dtype=bool))
            cmap = sns.diverging_palette(5, 210, s=90, l=40, as_cmap=True)
            sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap=cmap,
                        linewidths=0.5, linecolor="#03040A",
                        annot_kws={"size":8,"color":"#E8EAF2","weight":"600"},
                        ax=ax2, cbar_kws={"shrink":0.75,"pad":0.02}, vmin=-1, vmax=1)
            ax2.tick_params(colors="#5A6690", labelsize=8)
            ax2.set_xticklabels(ax2.get_xticklabels(), rotation=35, ha="right")
            ax2.set_yticklabels(ax2.get_yticklabels(), fontsize=8)
            ax2.figure.axes[-1].tick_params(colors="#3D4668", labelsize=7)
            plt.title("Pearson Correlation — All Features", color="#E8EAF2", fontsize=11, pad=14, loc="left")
            plt.tight_layout()
            st.pyplot(fig2, use_container_width=True)
            plt.close(fig2)

            st.markdown('<div class="sec">MSRP Feature Impact</div>', unsafe_allow_html=True)
            if "MSRP" in corr.columns:
                msrp_c = corr["MSRP"].drop("MSRP").sort_values()
                colors  = ["#118AB2" if v < 0 else "#E63946" for v in msrp_c.values]
                fig3, ax3 = plt.subplots(figsize=(9, 3.5), facecolor=DARK)
                ax3.set_facecolor(DARK)
                for sp in ax3.spines.values(): sp.set_edgecolor(BORDER)
                ax3.barh(msrp_c.index, msrp_c.values, color=colors, alpha=0.85, height=0.52)
                ax3.axvline(0, color="#161E36", linewidth=1)
                ax3.tick_params(colors="#5A6690", labelsize=8)
                ax3.set_xlabel("Correlation Coefficient", color="#3D4668", fontsize=8)
                ax3.set_title("Correlation with MSRP", color="#E8EAF2", fontsize=10, pad=10, loc="left")
                for i, (val, _) in enumerate(zip(msrp_c.values, msrp_c.index)):
                    ax3.text(val+(0.01 if val>=0 else -0.01), i, f"{val:+.2f}",
                             va="center", ha="left" if val>=0 else "right",
                             color="#E8EAF2", fontsize=7.5, fontweight="600")
                plt.tight_layout()
                st.pyplot(fig3, use_container_width=True)
                plt.close(fig3)

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center;padding:1.5rem 0 0.75rem;font-size:0.58rem;
                color:var(--border2);letter-spacing:0.1em;text-transform:uppercase;
                border-top:1px solid var(--border);margin-top:1rem;">
      CarPrice AI &nbsp;·&nbsp; Linear &amp; Polynomial Regression &nbsp;·&nbsp; Streamlit &amp; scikit-learn
    </div>
    """, unsafe_allow_html=True)


# ── ROUTER ──────────────────────────────────────────────────────
if st.session_state.page == "landing":
    render_landing()
else:
    render_dashboard()