# introquest_app.py
# Ultra-premium UI with TF-IDF backend (safe for Streamlit Cloud)
# Run: streamlit run introquest_app.py

import streamlit as st
st.set_page_config(page_title="IntroQuest AI — Premium", page_icon="✨", layout="wide")

import pandas as pd
import re
import numpy as np
import base64
from io import BytesIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go

# ---------------------------
# Configuration
# ---------------------------
# Use the uploaded rubric file path as default fallback (local path you provided)
DEFAULT_RUBRIC_PATH = "/mnt/data/Case study for interns.xlsx"
REQUIRED_RUBRIC_COLS = ["criterion", "description", "keywords", "weight"]

# ---------------------------
# Premium CSS (glassmorphism + minimal sidebar)
# ---------------------------
def load_premium_css():
    st.markdown(
        """
    <style>
    /* Page background */
    body { background: linear-gradient(135deg, #020617 0%, #08112b 45%, #071734 100%); color: #e8eef8; }

    /* Minimal sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.02));
        border-right: 1px solid rgba(255,255,255,0.03);
        padding-top: 24px;
    }

    /* Glass container */
    .glass {
        background: rgba(255,255,255,0.04);
        border-radius: 16px;
        padding: 20px;
        border: 1px solid rgba(255,255,255,0.06);
        backdrop-filter: blur(8px);
        box-shadow: 0 10px 30px rgba(2,6,23,0.6);
        color: #eaf2ff;
    }

    /* Header */
    .brand-title {
        font-size: 34px;
        font-weight: 700;
        background: linear-gradient(90deg,#42e695,#3bb2f8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Score circle */
    .score-circle {
        width: 160px;
        height: 160px;
        border-radius: 50%;
        background: conic-gradient(#7c4dff calc(var(--p) * 1%), rgba(255,255,255,0.04) 0%);
        display:flex;align-items:center;justify-content:center;
        font-weight:700;color:#fff;font-size:28px;margin:0 auto;
        box-shadow: 0 8px 30px rgba(124,77,255,0.18);
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg,#7c4dff,#3bb2f8);
        color: white;
        border-radius: 10px;
        padding: 8px 18px;
        font-weight: 700;
        border: none;
    }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 12px 36px rgba(59,178,248,0.12); }

    /* Compact cards for criteria */
    .criterion-card {
        background: rgba(255,255,255,0.02);
        padding: 12px;
        border-radius: 10px;
        border-left: 6px solid rgba(124,77,255,0.9);
        margin-bottom:10px;
    }

    /* Dataframe tweaks */
    .stDataFrame table { border-radius: 8px; overflow: hidden; }

    </style>
    """,
        unsafe_allow_html=True,
    )

load_premium_css()

# ---------------------------
# Rubric loader (structured + robust fallback)
# ---------------------------
def load_rubric_from_excel_like(obj):
    """
    Accepts an UploadedFile or a local path; returns rubric dict:
      { "Criterion Name": {"description":..., "keywords":[...], "weight":float, "min_words":int, "max_words":int}, ... }
    """
    import pandas as pd

    # read workbook (file-like or path)
    if hasattr(obj, "read"):
        content = obj.read()
        df_raw = pd.read_excel(BytesIO(content), header=None)
    else:
        df_raw = pd.read_excel(obj, header=None)

    # try to find a header row that contains 'criterion' and 'weight'
    header_row = None
    for i in range(min(10, df_raw.shape[0])):
        row = df_raw.iloc[i].astype(str).str.lower().tolist()
        if any("criterion" in c for c in row) and any("weight" in c or "score" in c for c in row):
            header_row = i
            break

    try:
        if header_row is not None:
            if hasattr(obj, "read"):
                df = pd.read_excel(BytesIO(content), header=header_row)
            else:
                df = pd.read_excel(obj, header=header_row)
        else:
            # try header=0 parse
            maybe = pd.read_excel(obj, header=0) if not hasattr(obj, "read") else pd.read_excel(BytesIO(content), header=0)
            lower_cols = [str(c).strip().lower() for c in maybe.columns]
            if any("criterion" in c for c in lower_cols) or any("weight" in c for c in lower_cols):
                df = maybe
            else:
                df = df_raw
    except Exception:
        df = df_raw

    # normalize column names
    df_cols = [str(c).strip().lower() for c in df.columns]
    df.columns = df_cols

    def find_col(names):
        for n in names:
            for c in df.columns:
                if n in str(c).lower():
                    return c
        return None

    col_crit = find_col(["criterion", "creteria", "criteria", "creterion"])
    col_desc = find_col(["description", "metric", "metric description", "details"])
    col_kw = find_col(["keyword", "key words", "keywords", "key_words"])
    col_weight = find_col(["weight", "score attributed", "score", "weightage", "total"])
    col_min = find_col(["min", "min_words", "min words"])
    col_max = find_col(["max", "max_words", "max words"])

    rubric = {}

    # structured parse if criterion column exists
    if col_crit:
        for _, row in df.iterrows():
            crit = str(row.get(col_crit, "")).strip()
            if crit and crit.lower() not in ["nan", "none"]:
                desc = str(row.get(col_desc, "")).strip() if col_desc else ""
                kwcell = row.get(col_kw, "") if col_kw else ""
                if pd.isna(kwcell) or str(kwcell).strip() == "":
                    keywords = []
                else:
                    keywords = [k.strip().lower() for k in re.split(r"[;,]", str(kwcell)) if k.strip()]
                try:
                    weight = float(row.get(col_weight, 0)) if col_weight else 10.0
                except Exception:
                    weight = 10.0
                try:
                    min_w = int(row.get(col_min, 0)) if col_min else 0
                except Exception:
                    min_w = 0
                try:
                    max_w = int(row.get(col_max, 200)) if col_max else 200
                except Exception:
                    max_w = 200

                rubric[crit] = {
                    "description": desc or f"Assess elements of {crit}",
                    "keywords": keywords,
                    "weight": weight,
                    "min_words": min_w,
                    "max_words": max_w,
                }

        if rubric:
            return rubric

    # fallback "smart" parse: find candidate headings and numeric weights
    df_text = df_raw.applymap(lambda x: "" if pd.isna(x) else str(x).strip())
    candidate = []
    pattern = re.compile(r"(content|structure|speech rate|language|grammar|clarity|engagement|presentation|delivery)", re.I)
    for i in range(df_text.shape[0]):
        joined = " ".join(df_text.iloc[i].tolist())
        if pattern.search(joined) and len(joined.split()) <= 6:
            candidate.append((joined, i))

    names = []
    for name, idx in candidate:
        cleaned = re.sub(r"[:\n\r]+", "", name).strip()
        if cleaned and cleaned.lower() not in [n.lower() for n in names]:
            names.append(cleaned)

    # collect numeric weights from a "Total score" block or bottom rows
    weight_values = []
    total_idx = None
    for i in range(df_text.shape[0]):
        if "total score" in " ".join(df_text.iloc[i].tolist()).lower():
            total_idx = i
            break
    if total_idx is not None:
        for i in range(total_idx, min(total_idx + 25, df_text.shape[0])):
            for cell in df_text.iloc[i].tolist():
                if re.fullmatch(r"\d{1,3}", cell):
                    weight_values.append(int(cell))
    if not weight_values:
        tail = df_text.tail(40).values.flatten().tolist()
        for cell in tail:
            if isinstance(cell, str) and re.fullmatch(r"\d{1,3}", cell.strip()):
                weight_values.append(int(cell.strip()))
        weight_values = list(dict.fromkeys(weight_values))

    if not names:
        defaults = {
            "Content & Structure": 20,
            "Speech Rate": 10,
            "Language & Grammar": 10,
            "Clarity": 10,
            "Engagement": 10,
        }
        return {k: {"description": f"{k}", "keywords": [], "weight": float(w), "min_words": 0, "max_words": 200} for k, w in defaults.items()}

    while len(weight_values) < len(names):
        weight_values.append(10)
    weight_values = weight_values[:len(names)]

    for i, name in enumerate(names):
        rubric[name] = {
            "description": f"Assess performance for {name}",
            "keywords": [],
            "weight": float(weight_values[i]),
            "min_words": 0,
            "max_words": 200,
        }

    return rubric

# ---------------------------
# Scoring building blocks (TF-IDF based)
# ---------------------------
def compute_rule_based(transcript, keywords, min_w, max_w):
    t_low = transcript.lower()
    token_matches = 0
    for kw in keywords:
        if not kw:
            continue
        kw = kw.lower().strip()
        if re.search(r"\b" + re.escape(kw) + r"\b", t_low):
            token_matches += 1
        elif kw in t_low:
            token_matches += 0.8
    kw_ratio = token_matches / max(1, len(keywords))

    wc = len(transcript.split())
    if min_w <= wc <= max_w:
        length_score = 1.0
    else:
        diff = min(abs(wc - min_w), abs(wc - max_w))
        length_score = max(0.0, 1.0 - (diff / max(200, min_w + 1)))

    rule_score = 0.7 * kw_ratio + 0.3 * length_score
    return {"rule_score": max(0.0, min(1.0, rule_score)), "keyword_matches": int(token_matches), "keyword_ratio": kw_ratio, "word_count": wc, "length_score": length_score}

def compute_nlp_similarity(transcript, description):
    """
    TF-IDF based semantic similarity between transcript and description.
    Returns similarity in 0..1.
    """
    if not transcript or not description:
        return 0.0
    docs = [transcript, description]
    tfidf = TfidfVectorizer(stop_words="english")
    try:
        mat = tfidf.fit_transform(docs)
        sim = cosine_similarity(mat[0:1], mat[1:2])[0][0]
    except ValueError:
        # happens if vectorizer fails (e.g., empty docs), fallback 0
        sim = 0.0
    return max(0.0, min(1.0, float(sim)))

def combine_scores(rule_score, nlp_sim, weight, weight_sum):
    combined_01 = 0.5 * rule_score + 0.5 * nlp_sim
    return combined_01 * (weight / max(1.0, weight_sum))

# ---------------------------
# UI helpers
# ---------------------------
def radar_chart(categories, values):
    fig = go.Figure(data=go.Scatterpolar(r=values, theta=categories, fill='toself', hoverinfo='all'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 5])), margin=dict(l=10, r=10, t=20, b=10), template="plotly_dark")
    return fig

def df_to_csv_download(df, filename="results.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a download="{filename}" href="data:file/csv;base64,{b64}">⬇️ Download results CSV</a>'
    return href

# ---------------------------
# Minimal sidebar (Option C) - logo + nav
# ---------------------------
with st.sidebar:
    st.markdown("<div style='padding:10px; text-align:center;'>", unsafe_allow_html=True)
    st.markdown("<img src='https://img.icons8.com/fluency/48/000000/rocket.png' alt='logo'/>", unsafe_allow_html=True)
    st.markdown("<h3 style='margin:6px 0 0 0; color:#dfe9ff;'>IntroQuest AI</h3>", unsafe_allow_html=True)
    st.markdown("<small style='color:#b8c6ff;'>Premium Evaluator</small>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    page = st.radio("", ["Dashboard", "About"], index=0, key="nav_radio")
    st.markdown("---")
    st.markdown("<small style='color:#aebef7;'>Need help? Contact your admin.</small>", unsafe_allow_html=True)

# ---------------------------
# Main layout (center content)
# ---------------------------
if page == "About":
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.markdown("<h2>About IntroQuest AI — Premium</h2>", unsafe_allow_html=True)
    st.markdown("<p>Ultra-premium evaluator combining rule-based checks and TF-IDF semantic similarity. Paste a transcript, upload a rubric Excel (optional), and press Score.</p>", unsafe_allow_html=True)
    st.markdown(f"<p>Fallback rubric path (local): <code>{DEFAULT_RUBRIC_PATH}</code></p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
else:
    st.markdown("<div class='glass' style='max-width:1100px; margin:auto;'>", unsafe_allow_html=True)
    st.markdown("<div style='display:flex; justify-content:space-between; align-items:center;'>", unsafe_allow_html=True)
    st.markdown("<div><h1 class='brand-title'>IntroQuest AI</h1><div style='color:#bcd7ff'>Premium Transcript Evaluator</div></div>", unsafe_allow_html=True)
    st.markdown("</div><hr>", unsafe_allow_html=True)

    # Upload rubric & transcript area in main center
    col_upload, col_input = st.columns([1, 2], gap="large")

    with col_upload:
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.subheader("📁 Rubric")
        uploaded_rubric = st.file_uploader("Upload rubric Excel (optional)", type=["xlsx"])
        if uploaded_rubric:
            try:
                rubric = load_rubric_from_excel_like(uploaded_rubric)
                st.success("Rubric loaded from uploaded file.")
            except Exception as e:
                st.error(f"Could not parse uploaded rubric: {e}")
                rubric = None
        else:
            # try fallback local path (use the uploaded file path you provided)
            try:
                rubric = load_rubric_from_excel_like(DEFAULT_RUBRIC_PATH)
                st.info("Using local fallback rubric.")
            except Exception:
                rubric = None
                st.warning("No rubric found; default will be used when scoring.")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_input:
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.subheader("🎤 Transcript")
        transcript_input = st.text_area("Paste or type your transcript here", height=260)
        uploaded_transcript_file = st.file_uploader("Or upload transcript (.txt)", type=["txt"])
        if uploaded_transcript_file:
            try:
                content = uploaded_transcript_file.read().decode(errors="ignore")
                if not transcript_input.strip():
                    transcript_input = content
                    st.success("Transcript loaded from file.")
            except Exception:
                st.error("Could not read transcript file.")
        st.markdown("</div>", unsafe_allow_html=True)

    # Score button centered
    st.markdown("<div style='text-align:center; margin-top:18px;'>", unsafe_allow_html=True)
    score_clicked = st.button("✨ Score Transcript", key="score_btn")
    st.markdown("</div>", unsafe_allow_html=True)

    # Run analysis when button clicked
    if score_clicked:
        if not transcript_input or not transcript_input.strip():
            st.error("Please paste or upload a transcript to analyze.")
        else:
            transcript = transcript_input.strip()
            if not rubric:
                # default rubric if nothing parsed
                rubric = {
                    "Content & Structure": {"description": "Content & Structure", "keywords": [], "weight": 20, "min_words": 0, "max_words": 200},
                    "Speech Rate": {"description": "Speech Rate", "keywords": [], "weight": 10, "min_words": 0, "max_words": 200},
                    "Language & Grammar": {"description": "Language & Grammar", "keywords": [], "weight": 10, "min_words": 0, "max_words": 200},
                    "Clarity": {"description": "Clarity", "keywords": [], "weight": 10, "min_words": 0, "max_words": 200},
                    "Engagement": {"description": "Engagement", "keywords": [], "weight": 10, "min_words": 0, "max_words": 200},
                }

            weight_sum = sum([float(v.get("weight", 10)) for v in rubric.values()]) or 1.0

            per_criterion_results = []
            overall_percent = 0.0
            categories = []
            values_for_radar = []

            for crit, details in rubric.items():
                desc = details.get("description", crit)
                kws = details.get("keywords", []) or []
                wmin = int(details.get("min_words", 0) or 0)
                wmax = int(details.get("max_words", 200) or 200)
                weight = float(details.get("weight", 10) or 10)

                rule = compute_rule_based(transcript, kws, wmin, wmax)
                nlp_sim = compute_nlp_similarity(transcript, desc)

                combined_01 = 0.5 * rule["rule_score"] + 0.5 * nlp_sim
                combined_weighted_contrib = combine_scores(rule["rule_score"], nlp_sim, weight, weight_sum)

                per_score_0_100 = combined_01 * 100.0

                per_criterion_results.append({
                    "criterion": crit,
                    "weight": weight,
                    "rule_score_01": round(rule["rule_score"], 3),
                    "nlp_sim_01": round(nlp_sim, 3),
                    "combined_01": round(combined_01, 3),
                    "final_score_pct": round(per_score_0_100, 2),
                    "keyword_matches": rule["keyword_matches"],
                    "keyword_ratio": round(rule["keyword_ratio"], 3),
                    "word_count": rule["word_count"],
                    "length_score": round(rule["length_score"], 3),
                    "description": desc,
                })

                overall_percent += combined_weighted_contrib * 100.0
                categories.append(crit)
                values_for_radar.append(round(combined_01 * 5, 3))

            overall_percent = round(max(0.0, min(100.0, overall_percent)), 2)

            if overall_percent >= 90:
                badge = "Quest Master"
            elif overall_percent >= 80:
                badge = "Apprentice Speaker"
            elif overall_percent >= 60:
                badge = "Rising Star"
            else:
                badge = "Novice Adventurer"

            # ---------- Display results ----------
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<div class='glass' style='max-width:1100px; margin:auto;'>", unsafe_allow_html=True)
            st.markdown("<div style='display:flex; gap:18px; justify-content:space-between; align-items:center;'>", unsafe_allow_html=True)

            # Score card, Badge, Word count
            c1, c2, c3 = st.columns([1,1,1])
            with c1:
                st.markdown(f"<div class='glass' style='text-align:center; padding:18px;'><div class='score-circle' style='--p:{overall_percent};'>{overall_percent}%</div><div style='margin-top:8px;color:#dbe9ff;'>Overall Score</div></div>", unsafe_allow_html=True)
            with c2:
                st.markdown(f"<div class='glass' style='text-align:center; padding:18px;'><h3 style='margin:4px 0;'>🏅 {badge}</h3><div style='color:#cfe6ff;'>Achievement</div></div>", unsafe_allow_html=True)
            with c3:
                st.markdown(f"<div class='glass' style='text-align:center; padding:18px;'><h3 style='margin:4px 0;'>{len(transcript.split())} words</h3><div style='color:#cfe6ff;'>Transcript Length</div></div>", unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

            # Radar chart
            st.markdown("<div style='margin-top:12px;'>", unsafe_allow_html=True)
            st.subheader("🌀 Performance Radar")
            fig = radar_chart(categories, values_for_radar)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # Per-criterion cards
            st.subheader("🧩 Criterion Breakdown")
            for r in per_criterion_results:
                st.markdown(f"<div class='criterion-card'><b>{r['criterion']}</b> — weight: {r['weight']}<br> Final: <b>{r['final_score_pct']}%</b> • Rule: {r['rule_score_01']} • NLP: {r['nlp_sim_01']} • Matches: {r['keyword_matches']} • Words: {r['word_count']}</div>", unsafe_allow_html=True)

            # Results table and download
            results_df = pd.DataFrame(per_criterion_results)
            results_df["overall_score"] = overall_percent
            st.markdown("<hr>", unsafe_allow_html=True)
            st.subheader("📥 Results & Export")
            st.write(results_df[["criterion", "weight", "final_score_pct", "rule_score_01", "nlp_sim_01", "keyword_matches", "word_count"]])
            st.markdown(df_to_csv_download(results_df, filename="introquest_results.csv"), unsafe_allow_html=True)

            # Quick textual suggestions
            st.subheader("🤖 Quick Suggestions")
            overall_msgs = []
            if overall_percent >= 85:
                overall_msgs.append("Strong introduction — structure and content align well with the rubric.")
            elif overall_percent >= 60:
                overall_msgs.append("Solid start; with targeted examples and slightly more alignment you can improve rapidly.")
            else:
                overall_msgs.append("Work on aligning content and including rubric keywords; add a clear hook and an example.")
            overall_msgs.append("Practice pausing for emphasis and vary sentence length for engagement.")
            st.success(" • ".join(overall_msgs))

            st.markdown("</div>", unsafe_allow_html=True)
