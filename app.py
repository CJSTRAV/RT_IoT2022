"""
RT-IoT2022 Cyberattack Classifier - Streamlit presentation app
Run:  streamlit run app.py

Visual theme follows the Ulticon "soft cards" design: a cool grey canvas, flat
white 28px cards, one ink hero card and one lime accent card per screen, violet
for active states, raspberry for danger, pill-shaped controls, Bricolage
Grotesque for display text and Figtree for body copy.
"""
import json
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import core

st.set_page_config(page_title="IoT Attack Classifier", layout="wide")

# ---------------------------------------------------------------- theme tokens
INK = "#1b1a2e"
CANVAS = "#eeeef4"
SURFACE = "#ffffff"
INSET = "#f4f4f9"
MUTED = "#5a5870"
BORDER = "#cfcee0"
VIOLET = "#4b45c6"
LIME = "#d4ee5e"
LIME_FG = "#2f3311"
INK_MUTED = "#b9b7d0"
TINT = "#d9d8ea"
WASH = "#e4e3ef"
DANGER = "#b3263f"
DANGER_SOFT = "#f8dde1"

ATTACK = DANGER  # attack traffic everywhere in the app
NORMAL = VIOLET  # normal traffic
NEUTRAL = INK_MUTED

FONT_BODY = "Figtree, system-ui, -apple-system, Segoe UI, sans-serif"
FONT_DISPLAY = "'Bricolage Grotesque', Figtree, system-ui, sans-serif"

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,700;12..96,800&family=Figtree:wght@400;500;600;700&display=swap');

/* ---- canvas & type ---- */
.stApp {background: #eeeef4; color: #1b1a2e;}
.stApp, .stApp p, .stApp li, .stApp label, .stApp input, .stApp textarea, .stApp button,
.stApp td, .stApp th, .stMarkdown {font-family: Figtree, system-ui, -apple-system, "Segoe UI", sans-serif;}
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5 {
  font-family: 'Bricolage Grotesque', Figtree, system-ui, sans-serif; color: #1b1a2e;}
.stApp h1 {font-weight: 800; letter-spacing: -0.03em;}
.stApp h2, .stApp h3 {font-weight: 700; letter-spacing: -0.01em;}
header[data-testid="stHeader"] {background: transparent;}
.block-container {max-width: 1280px; padding-top: 2rem; padding-bottom: 3rem;}
code {color: #4b45c6; background: #f4f4f9; border-radius: 6px;}

/* ---- cards (containers created with key="card_...") ---- */
[class*="st-key-card_"] {background: #ffffff; border-radius: 28px; padding: 26px 30px; gap: 14px;}
.card-title {font-family: 'Bricolage Grotesque', Figtree, sans-serif; font-weight: 700; font-size: 22px;
  letter-spacing: -0.01em; margin: 0 0 2px 0; color: #1b1a2e;}
.muted {color: #5a5870;}
.small {font-size: 14px;}

/* ---- page header ---- */
.page-head {display: flex; flex-wrap: wrap; align-items: baseline; gap: 4px 14px; margin: 0 0 18px 0;}
.page-head .t {font-family: 'Bricolage Grotesque', Figtree, sans-serif; font-weight: 800; font-size: 34px;
  letter-spacing: -0.03em; line-height: 1.05; color: #1b1a2e;}
.page-head .s {color: #5a5870; font-size: 16px;}

/* ---- grid rows of HTML cards ---- */
.grid {display: grid; gap: 20px; margin-bottom: 20px;}
.grid.hero-row {grid-template-columns: 2fr 1fr;}
.grid.kpis {grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));}
@media (max-width: 900px) {.grid.hero-row {grid-template-columns: 1fr;}}
.c {background: #ffffff; border-radius: 28px; padding: 24px 28px;}
.c.ink {background: #1b1a2e; color: #ffffff; padding: 34px 36px;}
.c.lime {background: #d4ee5e; color: #1b1a2e; padding: 34px 36px; display: flex; flex-direction: column;
  justify-content: space-between; gap: 22px;}
.c .lbl {font-size: 15px; font-weight: 600; color: #5a5870;}
.c.ink .lbl {color: #b9b7d0;}
.c.lime .lbl {color: #2f3311;}
.big {font-family: 'Bricolage Grotesque', Figtree, sans-serif; font-weight: 800; letter-spacing: -0.04em;
  line-height: 0.9;}
.kpi .v {font-family: 'Bricolage Grotesque', Figtree, sans-serif; font-weight: 800; font-size: 38px;
  letter-spacing: -0.03em; line-height: 1; margin-bottom: 8px;}
.kpi .l {color: #5a5870; font-size: 15px; line-height: 1.3;}
.hero-title {font-family: 'Bricolage Grotesque', Figtree, sans-serif; font-weight: 800; font-size: 40px;
  line-height: 1.02; letter-spacing: -0.03em; margin: 0 0 14px 0; color: #ffffff;}
.hero-sub {color: #b9b7d0; font-size: 17px; margin: 0;}

/* ---- wells, badges, notes ---- */
.well {background: #f4f4f9; border-radius: 16px; padding: 12px 16px;}
.tiles {display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px;}
.tiles .well .l {color: #5a5870; font-size: 13px; font-weight: 600;}
.tiles .well .v {font-family: 'Bricolage Grotesque', Figtree, sans-serif; font-weight: 700; font-size: 20px;}
.badge {display: inline-flex; align-items: center; border-radius: 999px; padding: 2px 12px; font-size: 13px;
  font-weight: 700; white-space: nowrap;}
.b-attack {background: #f8dde1; color: #b3263f;}
.b-normal {background: #4b45c6; color: #ffffff;}
.b-lime {background: #d4ee5e; color: #2f3311;}
.b-tint {background: #d9d8ea; color: #1b1a2e;}
.note {border-radius: 16px; padding: 14px 18px; font-size: 15.5px; margin: 6px 0 4px 0;}
.note.info {background: #e4e3ef; color: #1b1a2e;}
.note.ok {background: #d4ee5e; color: #2f3311;}
.note.bad {background: #f8dde1; color: #b3263f;}
.note.ink {background: #1b1a2e; color: #ffffff;}
.rowlist {display: flex; flex-direction: column; gap: 8px;}
.rowlist .well {display: flex; align-items: center; gap: 10px; flex-wrap: wrap;}

/* ---- verdict card ---- */
.verdict {background: #1b1a2e; color: #fff; border-radius: 28px; padding: 30px 32px;}
.verdict .name {font-family: 'Bricolage Grotesque', Figtree, sans-serif; font-weight: 800; font-size: 36px;
  letter-spacing: -0.03em; line-height: 1.02; margin: 14px 0 8px 0;}
.verdict .desc {color: #b9b7d0;}
.verdict .conf {display: flex; align-items: baseline; gap: 12px; margin-top: 22px;}
.verdict .conf .big {color: #d4ee5e; font-size: 64px;}

/* ---- controls: pills ---- */
.stButton > button, .stDownloadButton > button {border-radius: 999px; min-height: 44px; padding: 0 22px;
  font-weight: 600; border: 1px solid #cfcee0; background: #ffffff; color: #1b1a2e;}
.stButton > button:hover, .stDownloadButton > button:hover {border-color: #1b1a2e; color: #1b1a2e;}
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"],
[data-testid="stBaseButton-primary"] {background: #1b1a2e; border-color: #1b1a2e; color: #ffffff;}
.stButton > button[kind="primary"]:hover, .stDownloadButton > button[kind="primary"]:hover,
[data-testid="stBaseButton-primary"]:hover {background: #4b45c6; border-color: #4b45c6; color: #ffffff;}
[data-baseweb="select"] > div, [data-testid="stNumberInputContainer"], [data-baseweb="input"] {
  border-radius: 12px !important; background: #f4f4f9;}
[data-testid="stFileUploaderDropzone"] {border-radius: 20px; background: #f4f4f9;}

/* tabs drawn as chips */
.stTabs [data-baseweb="tab-list"] {gap: 8px; flex-wrap: wrap;}
.stTabs [data-baseweb="tab"] {border: 1px solid #cfcee0; background: #ffffff; border-radius: 999px;
  padding: 8px 20px; height: auto; font-weight: 600;}
.stTabs [data-baseweb="tab"] p {font-weight: 600; font-size: 15px;}
.stTabs [aria-selected="true"] {background: #1b1a2e; border-color: #1b1a2e;}
.stTabs [aria-selected="true"] p {color: #ffffff;}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {display: none;}
.stTabs [data-baseweb="tab-panel"] {padding-top: 18px;}

/* expanders as soft cards */
[data-testid="stExpander"] details {background: #ffffff; border: none; border-radius: 22px;}
[data-testid="stExpander"] summary p {font-weight: 600; font-size: 16px;}

/* sidebar: white panel, nav radio drawn as chips */
[data-testid="stSidebar"] {background: #ffffff;}
[data-testid="stSidebar"] [role="radiogroup"] {gap: 8px;}
[data-testid="stSidebar"] [role="radiogroup"] label {border: 1px solid #cfcee0; border-radius: 999px;
  padding: 9px 18px; width: 100%; margin: 0; background: #ffffff;}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {border-color: #1b1a2e;}
[data-testid="stSidebar"] [role="radiogroup"] label > div:first-child {display: none;}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {background: #1b1a2e; border-color: #1b1a2e;}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p {color: #ffffff;}
[data-testid="stSidebar"] [role="radiogroup"] label p {font-weight: 600; font-size: 15px;}
.wordmark {font-family: 'Bricolage Grotesque', Figtree, sans-serif; font-weight: 800; font-size: 26px;
  letter-spacing: -0.02em; line-height: 1.05; color: #1b1a2e;}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- UI helpers
def html(s):
    st.markdown(s, unsafe_allow_html=True)


def page_head(title, sub=""):
    html(f'<div class="page-head"><div class="t">{title}</div><div class="s">{sub}</div></div>')


def card(key):
    """White 28px card for widgets and charts."""
    return st.container(key=f"card_{key}")


def card_title(text, sub=None):
    html(f'<div class="card-title">{text}</div>' + (f'<div class="muted small">{sub}</div>' if sub else ""))


def kpi_row(items):
    """items: (value, label, color) -> one responsive row of white KPI cards."""
    cells = "".join(f'<div class="c kpi"><div class="v" style="color:{c}">{v}</div><div class="l">{l}</div></div>'
                    for v, l, c in items)
    html(f'<div class="grid kpis">{cells}</div>')


def note(text, kind="info"):
    html(f'<div class="note {kind}">{text}</div>')


def badge(kind):
    return f'<span class="badge {"b-normal" if kind == "Normal" else "b-attack"}">{kind}</span>'


def tiles(items):
    cells = "".join(f'<div class="well"><div class="l">{l}</div><div class="v">{v}</div></div>' for l, v in items)
    html(f'<div class="tiles">{cells}</div>')


def nice(name):
    return name.replace("_", " ")


def type_color(cls):
    return NORMAL if cls in core.NORMAL_CLASSES else ATTACK


def style_fig(fig, height=420):
    fig.update_layout(
        height=height, template="plotly_white", margin=dict(l=8, r=30, t=46, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_BODY, size=14, color=INK),
        title_font=dict(family=FONT_DISPLAY, size=18, color=INK),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title_text=""),
        hoverlabel=dict(font_family=FONT_BODY),
    )
    fig.update_xaxes(gridcolor=WASH, zerolinecolor=BORDER, linecolor=BORDER)
    fig.update_yaxes(gridcolor=WASH, zerolinecolor=BORDER, linecolor=BORDER)
    return fig


def chart(fig, height=420, key=None, target=st, legend=None):
    fig = style_fig(fig, height)
    if legend:
        fig.update_layout(legend=legend)
    target.plotly_chart(fig, width="stretch", theme=None, key=key,
                        config={"displayModeBar": False})


def type_style(t):
    return f"color:{NORMAL if t == 'Normal' else ATTACK}; font-weight:700"


# ---------------------------------------------------------------- data
@st.cache_resource(show_spinner="Loading Random Forest model...")
def get_bundle():
    return core.load_bundle()


@st.cache_data
def get_csv(name, **kw):
    return pd.read_csv(core.DATA / name, **kw)


@st.cache_data
def get_results():
    return json.loads((core.DATA / "results.json").read_text())


@st.cache_data
def predict_cached(df):
    return core.predict(df, get_bundle())


bundle = get_bundle()

# ---------------------------------------------------------------- sidebar
PAGES = ["Overview", "Dataset & Pipeline", "Model Results",
         "Classify a Flow", "Live Traffic Simulation", "Batch Prediction"]
with st.sidebar:
    html('<div class="wordmark">Attack Classifier</div>'
         '<div class="muted" style="margin:2px 0 18px 0">RT-IoT2022 &middot; Random Forest</div>')
    page = st.radio("Go to", PAGES, label_visibility="collapsed")
    html('<div style="height:18px"></div>'
         '<div class="well small"><b>Cyberattacks on Real-Time IoT Classification Using Random Forest</b><br>'
         '<span class="muted">Chris John Sam III V. Travilla<br>CSTL9 &middot; University of Mindanao<br>'
         'S.Y. 2026&ndash;2027</span></div>'
         '<div class="muted small" style="margin-top:12px">200 trees &middot; 62 features &middot; seed 42</div>')


# ================================================================ OVERVIEW
def page_overview():
    page_head("Overview", "Classifying IoT network flows as normal traffic or one of nine attacks")
    html(
        '<div class="grid hero-row">'
        '<div class="c ink">'
        '<div class="hero-title">Cyberattacks on Real-Time IoT Classification Using Random Forest</div>'
        '<p class="hero-sub">One network flow in, one of 12 traffic classes out &mdash; trained and tested on the '
        'RT-IoT2022 dataset.</p>'
        '<div style="display:flex;flex-wrap:wrap;align-items:flex-end;gap:18px;margin-top:30px">'
        f'<div class="big" style="font-size:110px;color:{LIME}">99.78%</div>'
        '<div style="font-size:19px;font-weight:600;padding-bottom:10px">accuracy on<br>23,582 unseen flows</div>'
        '</div></div>'
        '<div class="c lime">'
        '<div class="lbl" style="font-size:18px">As an attack detector</div>'
        '<div><div class="big" style="font-size:84px">99.96%</div>'
        '<div style="font-size:17px;margin-top:10px">of attack flows caught,<br>with a <b>0.62%</b> false-alarm rate '
        'on normal traffic</div></div>'
        '</div></div>'
    )
    kpi_row([
        ("0.9741", "Macro F1-score &mdash; the main measure, every class weighted equally", VIOLET),
        ("52", "Errors out of 23,582 test flows (0.22%)", INK),
        ("1st of 6", "Ranking among the classifiers compared", INK),
        ("62", "Model inputs: 53 selected + 9 engineered features", INK),
    ])

    left, right = st.columns([1, 1.25], gap="medium")
    with left:
        with card("problem"):
            card_title("The problem")
            st.markdown(
                "- IoT devices (sensors, bulbs, smart appliances) often ship with **weak security** and run unattended.\n"
                "- Signature-based intrusion detection **misses attacks** that don't match a stored rule.\n"
                "- A classifier can **learn traffic patterns** — packet counts, timing, header sizes, flags — "
                "and apply them to traffic it has never seen."
            )
            card_title("What we built")
            st.markdown(
                "- Cleaned **123,117 → 117,909** flows and split them **80/20** (stratified).\n"
                "- Selected **53** features and engineered **9** more → **62** inputs.\n"
                "- Trained a **Random Forest** (200 trees, balanced class weights) and compared it with five other classifiers."
            )
    with right:
        with card("classes"):
            card_title("The 12 traffic classes", "3 normal, 9 attack")
            rows = "".join(
                f'<div class="well">{badge(v[0])}<b>{nice(k)}</b><span class="muted small">{v[1]}</span></div>'
                for k, v in core.CLASS_INFO.items())
            html(f'<div class="rowlist">{rows}</div>')

    note("<b>Key takeaway:</b> Random Forest ranked first of six classifiers &mdash; only slightly ahead of a single "
         "decision tree. Most of its 52 errors come from two pairs of similar classes.", "ink")


# ================================================================ DATASET
def page_dataset():
    page_head("Dataset & Pipeline", "RT-IoT2022, UCI Machine Learning Repository &middot; real IoT network traffic")

    with card("framework"):
        card_title("Conceptual framework",
                   "Feature selection and training use the training set only &mdash; nothing is learned from the test set.")
        st.image(str(core.ROOT / "assets" / "conceptual_framework.png"), width="stretch")

    html('<div style="height:20px"></div>')
    kpi_row([
        ("123,117", "Raw flows, 83 features + label", INK),
        ("&minus;5,195", "Exact duplicates removed", MUTED),
        ("&minus;13", "Contradictory records removed", MUTED),
        ("117,909", "Clean flows", VIOLET),
        ("80 / 20", "Train / test split: 94,327 / 23,582 flows, stratified", INK),
    ])
    with st.expander("Cleaning steps in detail"):
        st.markdown(
            "1. No missing values. The placeholder `-` in **service** was renamed `none` (it is a real category).\n"
            "2. Dropped the constant column `bwd_URG_flag_count`.\n"
            "3. Removed 5,195 exact duplicates **before** dropping the source port — flood attacks create flows "
            "that differ only by source port.\n"
            "4. Removed 13 contradictory records (same features, different labels).\n"
            "5. Dropped the row number and **source port** (`id.orig_p`): it is random, so the model would "
            "memorise port numbers instead of learning behaviour."
        )
    html('<div style="height:20px"></div>')
    before_after_section()
    html('<div style="height:20px"></div>')

    with card("dist"):
        top = st.columns([3, 1])
        with top[0]:
            card_title("Class distribution &mdash; heavily imbalanced")
        with top[1]:
            log = st.toggle("Logarithmic scale", value=True)
        dist = get_csv("class_distribution.csv").sort_values("cleaned")
        dist["Type"] = dist["Attack_type"].map(lambda c: core.CLASS_INFO[c][0])
        dist["share"] = dist["cleaned"] / dist["cleaned"].sum() * 100
        fig = px.bar(dist, x="cleaned", y=dist["Attack_type"].map(nice), color="Type", orientation="h",
                     color_discrete_map={"Normal": NORMAL, "Attack": ATTACK}, log_x=log,
                     text=dist.apply(lambda r: f"{r.cleaned:,} ({r.share:.2f}%)", axis=1),
                     labels={"cleaned": "Flows after cleaning", "y": ""})
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_yaxes(categoryorder="total ascending")
        fig.update_xaxes(range=[np.log10(15), np.log10(600000)] if log else [0, 115000])
        chart(fig, 470)
        note("DOS_SYN_Hping is <b>76.41%</b> of all flows, NMAP_FIN_SCAN only <b>0.02%</b>. A model that always "
             "answered DOS_SYN_Hping would already score 76% accuracy &mdash; that is why the <b>macro F1-score</b> "
             "is the main measure.")

    html('<div style="height:20px"></div>')
    a, b = st.columns([1.25, 1], gap="medium")
    with a:
        with card("features"):
            card_title("From 92 columns to 62 model inputs")
            steps = pd.DataFrame({
                "Step": ["One-hot encoded", "Correlation filter", "Importance cut-off", "+ 9 engineered"],
                "Features": [92, 67, 53, 62],
            })
            fig = go.Figure(go.Bar(x=steps["Step"], y=steps["Features"], text=steps["Features"],
                                   marker_color=[NEUTRAL, NEUTRAL, INK, VIOLET], textposition="outside"))
            fig.update_yaxes(title="Number of features", range=[0, 105])
            chart(fig, 340)
            st.caption("Correlation filter drops one of each pair above 0.95; the importance cut-off keeps the "
                       "features carrying 99% of a 100-tree forest's importance.")
    with b:
        with card("engineered"):
            card_title("Engineered features", "Computed from each flow's own values")
            items = [
                ("total_pkts", "forward + backward packets"),
                ("fwd_bwd_pkt_ratio", "forward / backward packets"),
                ("fwd_bwd_payload_ratio", "forward / backward payload"),
                ("header_payload_ratio", "header bytes per payload byte"),
                ("data_pkt_fraction", "share of packets carrying data"),
                ("syn_per_pkt", "SYN flags per packet"),
                ("fin_per_pkt", "FIN flags per packet"),
                ("rst_per_pkt", "RST flags per packet"),
                ("no_response", "flow got no reply"),
            ]
            rows = "".join(f'<div class="well" style="padding:8px 14px"><code>{f}</code>'
                           f'<span class="muted small">{d}</span></div>' for f, d in items)
            html(f'<div class="rowlist">{rows}</div>')



# ================================================================ BEFORE & AFTER
def split_donut(normal, attack, title):
    fig = go.Figure(go.Pie(labels=["Attack", "Normal"], values=[attack, normal], hole=0.62, sort=False,
                           marker=dict(colors=[ATTACK, NORMAL], line=dict(color=SURFACE, width=3)),
                           texttemplate="%{label}<br>%{percent:.2%}", textposition="outside"))
    fig.update_layout(showlegend=False, margin=dict(l=30, r=30, t=30, b=20),
                      annotations=[dict(text=f"<b>{normal + attack:,}</b><br>flows", showarrow=False,
                                        font=dict(family=FONT_DISPLAY, size=18, color=INK))])
    card_title(title)
    chart(fig, 300, key=f"donut_{title}")


def before_after_section():
    summ = json.loads((core.DATA / "data_summary.json").read_text())
    dist = get_csv("class_distribution.csv")
    dist["Type"] = dist["Attack_type"].map(lambda c: core.CLASS_INFO[c][0])
    dist["removed"] = dist["raw"] - dist["cleaned"]

    with card("ba_table"):
        card_title("Before and after preprocessing", "What the raw file looked like, and what the model was trained on")
        table = pd.DataFrame([
            ("Rows (flows)", f"{summ['raw_rows']:,}", f"{summ['clean_rows']:,}", "5,208 rows removed (4.2%)"),
            ("Exact duplicate rows", f"{summ['duplicates_removed']:,}", "0", "Kept one copy of each"),
            ("Contradictory rows (same features, different label)", f"{summ['contradictions_removed']}", "0",
             "Removed entirely"),
            ("Constant columns", "1 (bwd_URG_flag_count)", "0", "Carries no information"),
            ("Source port (id.orig_p)", "Included", "Dropped", "Random per connection; would be memorised"),
            ("Value '-' in service", "'-'", "'none'", "Treated as a real category"),
            ("Feature columns", f"{summ['raw_cols']}", f"{summ['final_features']}",
             "81 cleaned → 92 one-hot → 53 selected + 9 engineered"),
            ("Categorical columns", "proto, service (text)", "13 binary columns", "One-hot encoded"),
            ("Missing values", "0", "0", "None to fix"),
        ], columns=["Item", "Before", "After", "Note"])
        st.dataframe(table, hide_index=True, width="stretch")

    html('<div style="height:20px"></div>')
    with card("ba_classes"):
        top = st.columns([3, 1])
        with top[0]:
            card_title("Flows per class, before and after cleaning",
                       "Most removed rows were repeated SYN-flood flows")
        with top[1]:
            log = st.toggle("Logarithmic scale", value=True, key="ba_log")
        d = dist.sort_values("raw", ascending=False)
        fig = go.Figure([
            go.Bar(name="Before (raw)", x=d["Attack_type"].map(nice), y=d["raw"], marker_color=NEUTRAL),
            go.Bar(name="After (cleaned)", x=d["Attack_type"].map(nice), y=d["cleaned"],
                   marker_color=[type_color(c) for c in d["Attack_type"]],
                   text=[f"−{r:,}" if r else "" for r in d["removed"]], textposition="outside",
                   textfont=dict(size=11, color=MUTED)),
        ])
        fig.update_layout(barmode="group")
        fig.update_xaxes(tickangle=-35)
        fig.update_yaxes(type="log" if log else "linear", title="Flows")
        chart(fig, 430)
        removed = dist[dist["removed"] > 0].sort_values("removed", ascending=False)
        st.dataframe(removed[["Attack_type", "Type", "raw", "cleaned", "removed"]]
                     .assign(pct=lambda x: x["removed"] / x["raw"])
                     .rename(columns={"Attack_type": "Class", "raw": "Before", "cleaned": "After",
                                      "removed": "Removed", "pct": "% removed"})
                     .style.format({"Before": "{:,}", "After": "{:,}", "Removed": "{:,}", "% removed": "{:.1%}"})
                     .map(type_style, subset=["Type"]),
                     hide_index=True, width="stretch")

    html('<div style="height:20px"></div>')
    a, b, c = st.columns(3, gap="medium")
    raw_n = int(dist.loc[dist["Type"] == "Normal", "raw"].sum()); raw_a = int(dist.loc[dist["Type"] == "Attack", "raw"].sum())
    cl_n = int(dist.loc[dist["Type"] == "Normal", "cleaned"].sum()); cl_a = int(dist.loc[dist["Type"] == "Attack", "cleaned"].sum())
    with a:
        with card("donut_before"):
            split_donut(raw_n, raw_a, "Before cleaning")
    with b:
        with card("donut_after"):
            split_donut(cl_n, cl_a, "After cleaning")
    with c:
        with card("donut_test"):
            split_donut(2403, 21179, "Test set")
    note("Normal vs attack stays at roughly <b>10% normal / 90% attack</b> through every step &mdash; cleaning and the "
         "stratified split did not change the balance.")

    html('<div style="height:20px"></div>')
    with card("ba_flow"):
        card_title("One flow, before and after transformation",
                   "The same record as the raw file stores it, and as the Random Forest receives it")
        demo = get_csv("demo_flows.csv")
        cls = st.selectbox("Example class", sorted(demo["Attack_type"].unique()), key="ba_cls",
                           index=sorted(demo["Attack_type"].unique()).index("MQTT_Publish"), format_func=nice)
        row = demo[demo["Attack_type"] == cls].head(1)
        flow = row.drop(columns=["flow_id", "Attack_type"])
        X = core.prepare(flow, bundle)
        l, r = st.columns(2, gap="medium")
        with l:
            html('<div class="card-title" style="font-size:17px">Before: raw record</div>')
            raw_cols = ["id.orig_p", "id.resp_p", "proto", "service", "flow_duration", "fwd_pkts_tot", "bwd_pkts_tot",
                        "fwd_pkts_payload.tot", "bwd_pkts_payload.tot", "flow_SYN_flag_count", "flow_FIN_flag_count",
                        "flow_RST_flag_count", "fwd_header_size_tot", "bwd_header_size_tot"]
            st.dataframe(flow[raw_cols].T.rename(columns={flow.index[0]: "value"}).astype(str),
                         width="stretch", height=420)
            st.caption(f"83 columns in total. Text values in proto and service; source port {int(flow['id.orig_p'].iloc[0])} "
                       "is still present.")
        with r:
            html('<div class="card-title" style="font-size:17px">After: model input</div>')
            onehot = [c for c in X.columns if c.startswith(("proto_", "service_"))]
            show = X.iloc[0][onehot + core.ENGINEERED + ["id.resp_p", "flow_duration", "fwd_pkts_tot"]]
            kind = ["One-hot" if c in onehot else "Engineered" if c in core.ENGINEERED else "Selected original"
                    for c in show.index]
            out = pd.DataFrame({"feature": show.index, "value": show.values.round(4), "kind": kind})
            st.dataframe(out.style.map(lambda k: f"color:{VIOLET if k == 'Engineered' else INK if k == 'One-hot' else MUTED};"
                                                 "font-weight:600", subset=["kind"]),
                         hide_index=True, width="stretch", height=420)
            st.caption(f"{X.shape[1]} numeric columns, no text, no source port. A sample of them is shown.")


# ================================================================ ROC
ROC_COLORS = [VIOLET, DANGER, INK, "#8a85e0", "#e07a8c", "#7d7a99", "#2f3311", "#b4c94a", "#5a5870",
              "#c7a6e8", "#e8a33d", "#3a8f7b"]


def roc_section():
    roc = get_csv("roc_curves.csv")
    auc = json.loads((core.DATA / "roc_auc.json").read_text())
    zoom = st.toggle("Zoom into the top-left corner (false-positive rate 0 to 5%)", value=False, key="roc_zoom")
    xr, yr = ([0, 0.05], [0.8, 1.005]) if zoom else ([0, 1], [0, 1.02])

    a, b = st.columns([1, 1], gap="medium")
    with a:
        with card("roc_bin"):
            card_title("Attack vs normal", "Score = share of trees voting for any attack class")
            d = roc[roc["curve"] == "ATTACK_VS_NORMAL"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random guess",
                                     line=dict(color=BORDER, dash="dash")))
            fig.add_trace(go.Scatter(x=d["fpr"], y=d["tpr"], mode="lines", name=f"Random Forest (AUC {auc['ATTACK_VS_NORMAL']:.5f})",
                                     line=dict(color=VIOLET, width=3, shape="hv"), fill="tozeroy",
                                     fillcolor="rgba(75,69,198,0.10)"))
            fig.add_trace(go.Scatter(x=[15 / 2403], y=[21171 / 21179], mode="markers", name="Model's decision point",
                                     marker=dict(color=LIME, size=13, line=dict(color=INK, width=2))))
            fig.update_xaxes(title="False-positive rate (normal flagged as attack)", range=xr)
            fig.update_yaxes(title="True-positive rate (attacks caught)", range=yr)
            chart(fig, 440, key="roc_bin_chart",
                  legend=dict(orientation="v", yanchor="bottom", y=0.04, xanchor="right", x=0.98,
                              bgcolor="rgba(255,255,255,0.9)"))
    with b:
        with card("roc_cls"):
            card_title("One class vs the rest", "Each curve treats one class as positive and all others as negative")
            classes = sorted([c for c in roc["curve"].unique() if c != "ATTACK_VS_NORMAL"])
            default = ["Metasploit_Brute_Force_SSH", "NMAP_FIN_SCAN", "DDOS_Slowloris", "NMAP_UDP_SCAN", "DOS_SYN_Hping"]
            pick = st.multiselect("Classes", classes, default=default, format_func=nice, key="roc_pick")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", showlegend=False,
                                     line=dict(color=BORDER, dash="dash")))
            for i, c in enumerate(pick):
                d = roc[roc["curve"] == c]
                fig.add_trace(go.Scatter(x=d["fpr"], y=d["tpr"], mode="lines", name=f"{nice(c)} ({auc[c]:.4f})",
                                         line=dict(color=ROC_COLORS[classes.index(c) % len(ROC_COLORS)], width=2.5,
                                                   shape="hv")))
            fig.update_xaxes(title="False-positive rate", range=xr)
            fig.update_yaxes(title="True-positive rate", range=yr)
            chart(fig, 440, key="roc_cls_chart",
                  legend=dict(orientation="v", yanchor="bottom", y=0.04, xanchor="right", x=0.98,
                              bgcolor="rgba(255,255,255,0.9)", font=dict(size=12)))

    html('<div style="height:20px"></div>')
    kpi_row([
        (f"{auc['ATTACK_VS_NORMAL']:.5f}", "AUC, attack vs normal", VIOLET),
        (f"{auc['weighted_ovr']:.4f}", "Weighted one-vs-rest AUC (by class size)", INK),
        (f"{auc['macro_ovr']:.4f}", "Macro one-vs-rest AUC (every class equal)", INK),
        ("0.917", "Lowest class AUC: NMAP FIN SCAN (6 test flows)", DANGER),
    ])
    with card("roc_table"):
        card_title("AUC per class")
        t = pd.DataFrame([(c, core.CLASS_INFO[c][0], auc[c]) for c in sorted(core.CLASS_INFO)],
                         columns=["Class", "Type", "AUC"]).sort_values("AUC")
        st.dataframe(t.style.format({"AUC": "{:.6f}"}).map(type_style, subset=["Type"]),
                     hide_index=True, width="stretch")
        note("An AUC of 1.0 means the model ranks every flow of that class above every other flow. Ten classes are at "
             "0.999 or higher. The two rare classes score lower (0.928 and 0.917) because each has one missed flow out "
             "of 6&ndash;7, and the forest gave that flow almost no votes for its true class &mdash; so no threshold "
             "can catch it without many false alarms. With so few test flows, these two values are uncertain.")

# ================================================================ RESULTS
def page_results():
    page_head("Model Results", "Final Random Forest on the held-out 20% test set")
    res = get_results()["final"]
    kpi_row([
        (f"{res['accuracy']:.4f}", "Accuracy", INK),
        (f"{res['macro_precision']:.4f}", "Macro precision", INK),
        (f"{res['macro_recall']:.4f}", "Macro recall", INK),
        (f"{res['macro_f1']:.4f}", "Macro F1", VIOLET),
        (f"{res['n_errors']}", f"Errors out of {res['n_test']:,} test flows", DANGER),
    ])

    tabs = st.tabs(["Model comparison", "Per-class results", "Confusion matrix", "ROC curves",
                    "Feature importance", "Experiments"])

    with tabs[0]:
        with card("compare"):
            card_title("Six classifiers, same 62 features")
            mc = get_csv("model_comparison.csv").sort_values("macro_f1", ascending=False)
            long = mc.melt(id_vars="model", value_vars=["accuracy", "macro_f1"], var_name="Metric", value_name="Score")
            long["Metric"] = long["Metric"].map({"accuracy": "Accuracy", "macro_f1": "Macro F1"})
            fig = px.bar(long, x="model", y="Score", color="Metric", barmode="group",
                         text=long["Score"].map("{:.4f}".format),
                         color_discrete_map={"Accuracy": NEUTRAL, "Macro F1": VIOLET}, labels={"model": ""})
            fig.update_traces(textposition="outside", textfont_size=11)
            fig.update_yaxes(range=[0.55, 1.03])
            chart(fig, 430)
            show = mc[["model", "accuracy", "macro_f1", "mcc", "cv_macro_f1_mean", "fit_seconds", "n_errors"]].rename(columns={
                "model": "Model", "accuracy": "Accuracy", "macro_f1": "Macro F1", "mcc": "MCC",
                "cv_macro_f1_mean": "CV macro F1", "fit_seconds": "Train time (s)", "n_errors": "Errors"})
            st.dataframe(show.style.format({"Accuracy": "{:.4f}", "Macro F1": "{:.4f}", "MCC": "{:.4f}",
                                            "CV macro F1": "{:.4f}", "Train time (s)": "{:.1f}"})
                         .highlight_max(subset=["Macro F1"], color="#eef7c8"),
                         hide_index=True, width="stretch")
            note("Random Forest is best, but a <b>single decision tree</b> is close behind (&minus;0.0012 macro F1) and "
                 "trains about 14&times; faster. Logistic Regression and Naive Bayes look fine on accuracy but collapse "
                 "on macro F1 &mdash; accuracy hides failures on small classes.")

    with tabs[1]:
        with card("perclass"):
            card_title("F1-score per class", "n = number of test flows in that class")
            pc = get_csv("per_class_metrics.csv", index_col=0).reset_index(names="Class")
            pc["Type"] = pc["Class"].map(lambda c: core.CLASS_INFO[c][0])
            pc = pc.sort_values("f1")
            fig = px.bar(pc, x="f1", y=pc["Class"].map(nice), orientation="h", color="Type",
                         color_discrete_map={"Normal": NORMAL, "Attack": ATTACK},
                         text=pc.apply(lambda r: f"{r.f1:.4f}  (n={int(r.support):,})", axis=1),
                         labels={"f1": "F1-score", "y": ""})
            fig.update_traces(textposition="outside", cliponaxis=False)
            fig.update_xaxes(range=[0.85, 1.04])
            fig.update_yaxes(categoryorder="total ascending")
            chart(fig, 500)
            st.dataframe(pc.sort_values("Class")[["Class", "Type", "support", "precision", "recall", "f1"]]
                         .rename(columns={"support": "Test flows", "precision": "Precision", "recall": "Recall", "f1": "F1"})
                         .style.format({"Precision": "{:.4f}", "Recall": "{:.4f}", "F1": "{:.4f}", "Test flows": "{:,}"})
                         .map(type_style, subset=["Type"]),
                         hide_index=True, width="stretch")
            note("Four classes are perfect. The weakest is DDOS_Slowloris (0.9032) &mdash; with 106 test flows that is "
                 "not just a small-sample effect. The two rarest classes have only 7 and 6 test flows.")

    with tabs[2]:
        with card("cm"):
            top = st.columns([2, 1.3])
            with top[0]:
                card_title("Confusion matrix", "Rows are the true class, columns the prediction")
            with top[1]:
                mode = st.radio("Colour by", ["Row % (recall)", "Raw counts"], horizontal=True)
            cm = get_csv("confusion_matrix_counts.csv", index_col=0)
            pct = cm.div(cm.sum(axis=1), axis=0) * 100
            z = pct.values if mode.startswith("Row") else np.log10(cm.values + 1)
            text = np.where(cm.values == 0, "",
                            np.vectorize(lambda p, n: f"{p:.1f}%<br>({n:,})")(pct.values, cm.values))
            labels = [nice(c) for c in cm.columns]
            fig = go.Figure(go.Heatmap(
                z=z, x=labels, y=labels, text=text, texttemplate="%{text}", textfont={"size": 10},
                colorscale=[[0, INSET], [0.25, TINT], [0.6, VIOLET], [1, INK]], showscale=False, xgap=2, ygap=2,
                hovertemplate="True: %{y}<br>Predicted: %{x}<extra></extra>"))
            fig.update_yaxes(autorange="reversed", title="True class", showgrid=False)
            fig.update_xaxes(title="Predicted class", tickangle=-40, showgrid=False)
            chart(fig, 680)
        html('<div style="height:20px"></div>')
        a, b = st.columns([1, 1.2], gap="medium")
        with a:
            with card("mixups"):
                card_title("Most frequent mix-ups")
                tc = get_csv("top_confusions.csv").head(6)
                rows = "".join(
                    f'<div class="well"><span class="badge b-tint">{int(r.flows)}</span>'
                    f'<b>{nice(r.true_class)}</b><span class="muted">predicted as</span><b>{nice(r.predicted_as)}</b></div>'
                    for r in tc.itertuples())
                html(f'<div class="rowlist">{rows}</div>')
        with b:
            html('<div class="c lime" style="height:100%">'
                 '<div class="lbl" style="font-size:18px">Reading the matrix</div>'
                 '<div><div class="big" style="font-size:76px">40 of 52</div>'
                 '<div style="font-size:16.5px;margin-top:10px">errors come from two pairs of similar classes: '
                 '<b>Thing Speak &harr; ARP poisioning</b> (13 + 7) and <b>NMAP UDP SCAN &harr; DDOS Slowloris</b> '
                 '(13 + 7).<br><br>As a normal-vs-attack detector: <b>21,171 / 21,179</b> attacks caught, '
                 '<b>15 / 2,403</b> normal flows falsely flagged.</div></div></div>')

    with tabs[3]:
        roc_section()

    with tabs[4]:
        with card("importance"):
            fi = get_csv("feature_importance.csv", index_col=0).reset_index(names="Feature")
            top = st.columns([2, 1.2])
            with top[0]:
                card_title("Feature importance", "Mean decrease in impurity across the 200 trees")
            with top[1]:
                n = st.slider("Features shown", 5, len(fi), 15)
            sel = fi.head(n).iloc[::-1].copy()
            sel["Kind"] = np.where(sel["Feature"].isin(core.ENGINEERED), "Engineered",
                                   np.where(sel["Feature"].str.startswith(("service_", "proto_")), "Encoded category", "Original"))
            fig = px.bar(sel, x="importance", y="Feature", orientation="h", color="Kind",
                         color_discrete_map={"Engineered": VIOLET, "Original": NEUTRAL, "Encoded category": INK},
                         text=sel["importance"].map("{:.4f}".format), labels={"importance": "Importance", "Feature": ""})
            fig.update_traces(textposition="outside", cliponaxis=False)
            fig.update_yaxes(categoryorder="total ascending")
            chart(fig, max(380, 26 * n + 80))
            note(f"The top 10 features carry <b>{fi.head(10)['importance'].sum() * 100:.1f}%</b> of the total &mdash; the "
                 "model combines many weak clues. Three of the top five are <b>engineered</b> (header_payload_ratio, "
                 "syn_per_pkt, fin_per_pkt), which fits SYN floods and FIN scans.")

    with tabs[5]:
        with card("configs"):
            card_title("Feature-set experiments (E1&ndash;E6)", "Same Random Forest, different inputs")
            cfg = get_csv("config_results.csv")
            fig = px.bar(cfg, x="config", y="macro_f1", text=cfg["macro_f1"].map("{:.4f}".format),
                         hover_data=["description", "n_features", "n_errors"],
                         color=np.where(cfg["config"] == "E5", "Final model", "Other"),
                         color_discrete_map={"Final model": VIOLET, "Other": NEUTRAL},
                         category_orders={"config": list(cfg["config"])},
                         labels={"config": "", "macro_f1": "Macro F1", "color": ""})
            fig.update_traces(textposition="outside")
            fig.update_yaxes(range=[0.94, 0.98])
            chart(fig, 350)
            st.dataframe(cfg[["config", "description", "n_features", "accuracy", "macro_f1", "n_errors"]]
                         .rename(columns={"config": "ID", "description": "Description", "n_features": "Features",
                                          "accuracy": "Accuracy", "macro_f1": "Macro F1", "n_errors": "Errors"})
                         .style.format({"Accuracy": "{:.4f}", "Macro F1": "{:.4f}"}),
                         hide_index=True, width="stretch")
            note("E6 removes destination port and service: macro F1 drops 0.9741 &rarr; 0.9609 and errors rise "
                 "52 &rarr; 96. The model leans on them but still works from flow behaviour alone.")
        html('<div style="height:20px"></div>')
        a, b = st.columns(2, gap="medium")
        with a:
            with card("splits"):
                card_title("Train/test split sensitivity")
                sp = get_csv("split_results.csv")
                st.dataframe(sp[["split", "train", "test", "rare_test_flows", "accuracy", "macro_f1"]]
                             .rename(columns={"split": "Split", "train": "Train", "test": "Test",
                                              "rare_test_flows": "Rare-class test flows", "accuracy": "Accuracy",
                                              "macro_f1": "Macro F1"})
                             .style.format({"Accuracy": "{:.4f}", "Macro F1": "{:.4f}", "Train": "{:,}", "Test": "{:,}"}),
                             hide_index=True, width="stretch")
        with b:
            with card("tuning"):
                card_title("Hyper-parameter search", "5-fold cross-validation")
                tu = get_csv("tuning_results.csv")
                st.dataframe(tu[["candidate", "n_estimators", "max_features", "min_samples_leaf", "cv_macro_f1_mean"]]
                             .rename(columns={"candidate": "Candidate", "n_estimators": "Trees",
                                              "max_features": "Max features", "min_samples_leaf": "Min leaf",
                                              "cv_macro_f1_mean": "CV macro F1"})
                             .style.format({"CV macro F1": "{:.4f}"}),
                             hide_index=True, width="stretch")
                st.caption("No candidate beat the default by a meaningful margin, so the default 200-tree forest was kept.")


# ================================================================ CLASSIFY ONE FLOW
KEY_FIELDS = [
    ("id.resp_p", "Destination port", 1),
    ("flow_duration", "Flow duration (s)", 0.001),
    ("fwd_pkts_tot", "Forward packets", 1),
    ("bwd_pkts_tot", "Backward packets", 1),
    ("fwd_pkts_payload.tot", "Forward payload bytes", 1.0),
    ("flow_SYN_flag_count", "SYN flags", 1),
    ("flow_FIN_flag_count", "FIN flags", 1),
    ("flow_RST_flag_count", "RST flags", 1),
    ("fwd_URG_flag_count", "Forward URG flags", 1),
    ("fwd_last_window_size", "Forward last window size", 1),
]


def show_prediction(row, true_label=None, key="main"):
    res, proba = core.predict(row, bundle)
    pred, conf, traffic = res.iloc[0]["predicted_class"], res.iloc[0]["confidence"], res.iloc[0]["traffic"]
    kind = "Normal" if traffic == "Normal" else "Attack"
    left, right = st.columns([1, 1.35], gap="medium")
    with left:
        html(
            f'<div class="verdict">{badge(kind)}'
            f'<div class="name">{nice(pred)}</div>'
            f'<div class="desc">{core.CLASS_INFO[pred][1]}</div>'
            f'<div class="conf"><div class="big">{conf * 100:.1f}%</div>'
            f'<div class="desc">of the 200 trees<br>voted for this class</div></div></div>'
        )
        if true_label is not None:
            if true_label == pred:
                note(f"<b>Correct.</b> The true label is {nice(true_label)}.", "ok")
            else:
                note(f"<b>Misclassified.</b> The true label is {nice(true_label)} &mdash; one of the model's 52 "
                     "test-set errors.", "bad")
    with right:
        with card(f"proba_{key}"):
            card_title("Class probabilities", "Classes with more than 0.1% of the vote")
            p = proba.iloc[0].sort_values(ascending=True)
            p = p[p > 0.001]
            fig = go.Figure(go.Bar(x=p.values, y=[nice(c) for c in p.index], orientation="h",
                                   marker_color=[type_color(c) for c in p.index],
                                   text=[f"{v * 100:.1f}%" for v in p.values], textposition="outside",
                                   cliponaxis=False, width=0.6))
            fig.update_xaxes(range=[0, 1.14], tickformat=".0%", title="Share of tree votes")
            fig.update_layout(margin=dict(t=10))
            chart(fig, max(200, 46 * len(p) + 90), key=f"proba_chart_{key}")


def page_classify():
    page_head("Classify a Flow", "Flows from the held-out test set &mdash; the model never saw them in training")
    demo = get_csv("demo_flows.csv")
    preds, _ = predict_cached(demo)
    demo = demo.assign(_pred=preds["predicted_class"].values)

    with card("picker"):
        c1, c2, c3, c4 = st.columns([1.3, 1, 1.3, 0.8], vertical_alignment="bottom")
        with c1:
            cls = st.selectbox("True class", ["Any class"] + sorted(demo["Attack_type"].unique()),
                               format_func=lambda c: c if c == "Any class" else f"{nice(c)} ({core.CLASS_INFO[c][0]})")
        with c2:
            which = st.selectbox("Show", ["All flows", "Only misclassified flows"])
        pool = demo if cls == "Any class" else demo[demo["Attack_type"] == cls]
        if which.startswith("Only"):
            pool = pool[pool["_pred"] != pool["Attack_type"]]
        if pool.empty:
            note("No flows match this filter &mdash; the model made no errors on this class in the test set.")
            return
        ids = pool["flow_id"].tolist()
        if st.session_state.get("flow_pick") not in ids:
            good = pool[pool["_pred"] == pool["Attack_type"]]["flow_id"].tolist()
            st.session_state["flow_pick"] = good[0] if good else ids[0]  # open on a correct example
        with c4:
            if st.button("Random flow", width="stretch"):
                st.session_state["flow_pick"] = str(np.random.choice(ids))
        with c3:
            fid = st.selectbox(f"Flow ({len(ids)} available)", ids, key="flow_pick")

        row = demo[demo["flow_id"] == fid].drop(columns="_pred")
        true_label = row["Attack_type"].iloc[0]
        flow = row.drop(columns=["Attack_type"])
        r = flow.iloc[0]
        tiles([
            ("Protocol", str(r["proto"]).upper()),
            ("Service", "none" if r["service"] == "-" else str(r["service"])),
            ("Destination port", f"{int(r['id.resp_p'])}"),
            ("Packets fwd / bwd", f"{int(r['fwd_pkts_tot'])} / {int(r['bwd_pkts_tot'])}"),
            ("Duration", f"{r['flow_duration']:.3f} s"),
            ("SYN / FIN / RST", f"{int(r['flow_SYN_flag_count'])} / {int(r['flow_FIN_flag_count'])} / "
                                f"{int(r['flow_RST_flag_count'])}"),
        ])

    html('<div style="height:20px"></div>')
    show_prediction(flow, true_label, key="main")
    html('<div style="height:20px"></div>')

    with st.expander("What-if: edit this flow and re-classify"):
        st.caption("Change a few key features. The other 70+ features keep their original values.")
        edited = flow.copy()
        cols = st.columns(4)
        protos = core.known_values(bundle, "proto_")
        services = ["-" if s == "none" else s for s in core.known_values(bundle, "service_")]
        edited["proto"] = cols[0].selectbox("Protocol", protos,
                                            index=protos.index(r["proto"]) if r["proto"] in protos else 0,
                                            key=f"wi_proto_{fid}")
        svc = r["service"] if r["service"] in services else "-"
        edited["service"] = cols[1].selectbox("Service", services, index=services.index(svc), key=f"wi_svc_{fid}")
        for i, (col, label, step) in enumerate(KEY_FIELDS):
            val = float(r[col])
            if isinstance(step, int) and float(val).is_integer():
                new = cols[(i + 2) % 4].number_input(label, min_value=0, value=int(val), step=1, key=f"wi_{col}_{fid}")
            else:
                new = cols[(i + 2) % 4].number_input(label, min_value=0.0, value=val, step=float(step),
                                                     format="%.4f", key=f"wi_{col}_{fid}")
            edited[col] = new
        changed = [label for col, label, _ in KEY_FIELDS if float(edited[col].iloc[0]) != float(r[col])]
        changed += [n for n, c in (("Protocol", "proto"), ("Service", "service")) if edited[c].iloc[0] != r[c]]
        html(f'<div class="well small"><b>Changed:</b> {", ".join(changed) if changed else "nothing yet"}</div>')
        html('<div style="height:10px"></div>')
        show_prediction(edited, key="whatif")
        st.caption("Demo tip: take a normal MQTT flow, raise the SYN flags or set backward packets to zero, "
                   "and watch the vote shift toward an attack class.")

    with st.expander("All raw features of this flow"):
        st.dataframe(flow.T.rename(columns={flow.index[0]: "value"}), width="stretch", height=360)


# ================================================================ LIVE SIMULATION
def page_live():
    page_head("Live Traffic Simulation", "Held-out test flows replayed as if they were arriving on an IoT network")
    stream = get_csv("stream_flows.csv")

    with card("controls"):
        c = st.columns([1.2, 1.2, 1.2, 1], vertical_alignment="bottom")
        n = c[0].slider("Flows to replay", 50, len(stream), 200, step=50)
        speed = c[1].select_slider("Speed (flows per second)", [2, 5, 10, 20, 50, 100], value=10)
        thin = c[2].toggle("Thin out SYN-flood flows", value=True,
                           help="Keeps 1 in 8 DOS_SYN_Hping flows so the other classes appear more often.")
        start = c[3].button("Start simulation", type="primary", width="stretch")
        st.caption("The traffic mix follows the real test set, which is mostly SYN-flood flows. "
                   "Every flow is classified by the Random Forest the moment it arrives.")

    if thin:
        dos = stream["Attack_type"] == "DOS_SYN_Hping"
        stream = pd.concat([stream[~dos], stream[dos].iloc[::8]]).sort_values("flow_id")
    flows = stream.sample(min(n, len(stream)), random_state=int(time.time()) if start else 1).reset_index(drop=True)

    html('<div style="height:20px"></div>')
    kpi_box = st.empty()
    a, b = st.columns([1.5, 1], gap="medium")
    with a:
        chart_card = card("live_chart")
        feed_card = card("live_feed")
    with b:
        alert_card = card("live_alerts")
    with chart_card:
        card_title("Predicted classes so far")
        chart_box = st.empty()
    with feed_card:
        card_title("Latest flows")
        feed_box = st.empty()
    with alert_card:
        card_title("Alert log", "Latest first")
        alert_box = st.empty()

    if not start:
        kpi_row([("0", "Flows processed", INK), ("0", "Attacks detected", DANGER),
                 ("0", "Normal flows", VIOLET), ("&ndash;", "Correct so far", INK)])
        chart_box.markdown('<div class="muted">Press <b>Start simulation</b> to begin.</div>', unsafe_allow_html=True)
        feed_box.markdown('<div class="muted">No flows yet.</div>', unsafe_allow_html=True)
        alert_box.markdown('<div class="muted">No alerts yet.</div>', unsafe_allow_html=True)
        return

    res, _ = core.predict(flows, bundle)
    res = res.assign(flow_id=flows["flow_id"], true=flows["Attack_type"], proto=flows["proto"].str.upper(),
                     port=flows["id.resp_p"].astype(int))
    alerts = []
    batch = max(1, speed // 10)  # redraw at most ~10 times per second
    for i in range(len(res)):
        r = res.iloc[i]
        if r["traffic"] == "ATTACK":
            alerts.append(f'<div class="well"><span class="badge b-attack">{r.confidence * 100:.0f}%</span>'
                          f'<b>{nice(r.predicted_class)}</b><span class="muted small">{r.flow_id} &middot; '
                          f'{r.proto} port {r.port}</span></div>')
        if (i + 1) % batch and i + 1 != len(res):
            continue
        seen = res.iloc[: i + 1]
        attacks = int((seen["traffic"] == "ATTACK").sum())
        correct = int((seen["predicted_class"] == seen["true"]).sum())
        with kpi_box.container():
            kpi_row([
                (f"{i + 1:,}", "Flows processed", INK),
                (f"{attacks:,}", "Attacks detected", DANGER),
                (f"{i + 1 - attacks:,}", "Normal flows", VIOLET),
                (f"{correct / (i + 1) * 100:.1f}%", "Correct so far, against the true label", INK),
            ])
        counts = seen["predicted_class"].value_counts()
        fig = go.Figure(go.Bar(x=counts.values, y=[nice(x) for x in counts.index], orientation="h",
                               marker_color=[type_color(x) for x in counts.index],
                               text=counts.values, textposition="outside", cliponaxis=False))
        fig.update_layout(yaxis=dict(autorange="reversed"), margin=dict(t=10))
        chart(fig, 320, key=f"live_chart_{i}", target=chart_box)
        feed = seen.tail(8).iloc[::-1][["flow_id", "proto", "port", "predicted_class", "confidence", "traffic", "true"]]
        feed = feed.rename(columns={"flow_id": "Flow", "proto": "Proto", "port": "Port", "predicted_class": "Prediction",
                                    "confidence": "Confidence", "traffic": "Verdict", "true": "True label"})
        feed_box.dataframe(
            feed.style.format({"Confidence": "{:.0%}"})
            .map(lambda v: f"color:{ATTACK if v == 'ATTACK' else NORMAL}; font-weight:700", subset=["Verdict"]),
            hide_index=True, width="stretch",
        )
        alert_box.markdown('<div class="rowlist">' + ("".join(alerts[::-1][:9]) or
                           '<div class="muted">No attacks yet.</div>') + "</div>", unsafe_allow_html=True)
        time.sleep(batch / speed)
    note(f"<b>Simulation finished.</b> {len(res)} flows classified, {int((res['traffic'] == 'ATTACK').sum())} "
         "flagged as attacks.", "ok")


# ================================================================ BATCH
def page_batch():
    page_head("Batch Prediction", "Classify a whole CSV of flows at once")
    with card("upload"):
        c1, c2 = st.columns([1.5, 1], gap="medium", vertical_alignment="center")
        with c1:
            up = st.file_uploader("CSV file with the RT-IoT2022 flow columns", type=["csv"])
        with c2:
            st.download_button("Download a sample CSV", (core.DATA / "sample_upload_unlabeled.csv").read_bytes(),
                               "sample_flows.csv", "text/csv", width="stretch")
            st.caption("Extra columns (row number, id.orig_p, Attack_type) are ignored. Without an upload the page "
                       "uses the bundled sample of 1,123 labelled test flows. If Attack_type is present, the "
                       "predictions are scored against it.")

    if up is not None:
        df = pd.read_csv(up)
        name = up.name
    else:
        df = get_csv("demo_flows.csv")
        name = "the bundled sample"

    if "service" in df.columns:
        df["service"] = df["service"].fillna("-")
    missing = core.missing_columns(df, bundle)
    if missing:
        note(f"<b>The file is missing {len(missing)} required column(s):</b> "
             f"{', '.join(missing[:10])}{' ...' if len(missing) > 10 else ''}", "bad")
        with st.expander("All required columns"):
            st.code(", ".join(core.required_columns(bundle)))
        return

    with st.spinner(f"Classifying {len(df):,} flows..."):
        res, proba = core.predict(df, bundle)

    attacks = int((res["traffic"] == "ATTACK").sum())
    html('<div style="height:20px"></div>')
    last = ((f"{(res['predicted_class'].values == df['Attack_type'].values).mean() * 100:.2f}%",
             "Accuracy against the Attack_type column", VIOLET) if "Attack_type" in df.columns
            else (f"{res['confidence'].mean() * 100:.1f}%", "Average confidence", VIOLET))
    kpi_row([
        (f"{len(df):,}", f"Flows in {name}", INK),
        (f"{attacks:,}", "Predicted attacks", DANGER),
        (f"{len(df) - attacks:,}", "Predicted normal", VIOLET),
        last,
    ])

    a, b = st.columns([1, 1.2], gap="medium")
    with a:
        with card("counts"):
            card_title("Predicted class counts")
            counts = res["predicted_class"].value_counts().iloc[::-1]
            fig = go.Figure(go.Bar(x=counts.values, y=[nice(x) for x in counts.index], orientation="h",
                                   marker_color=[type_color(x) for x in counts.index],
                                   text=counts.values, textposition="outside", cliponaxis=False))
            fig.update_layout(margin=dict(t=10))
            chart(fig, 400)
    with b:
        with card("conf"):
            card_title("Confidence distribution", "Log scale &mdash; most flows get a unanimous vote")
            fig = px.histogram(res, x="confidence", nbins=20, color="traffic",
                               color_discrete_map={"Normal": NORMAL, "ATTACK": ATTACK},
                               labels={"confidence": "Confidence", "traffic": ""}, log_y=True)
            fig.update_xaxes(tickformat=".0%")
            chart(fig, 400)

    html('<div style="height:20px"></div>')
    with card("table"):
        out = pd.concat([df[[c for c in ("flow_id", "proto", "service", "id.resp_p") if c in df.columns]], res], axis=1)
        if "Attack_type" in df.columns:
            out["true_label"] = df["Attack_type"]
            out["correct"] = out["predicted_class"] == out["true_label"]
        top = st.columns([2, 1, 1], vertical_alignment="center")
        with top[0]:
            card_title("Predictions")
        with top[1]:
            only_attacks = st.toggle("Predicted attacks only")
        with top[2]:
            st.download_button("Download predictions", out.to_csv(index=False).encode(),
                               "predictions.csv", "text/csv", type="primary", width="stretch")
        view = out[out["traffic"] == "ATTACK"] if only_attacks else out
        st.dataframe(view.style.format({"confidence": "{:.1%}"})
                     .map(lambda v: f"color:{ATTACK if v == 'ATTACK' else NORMAL}; font-weight:700",
                          subset=["traffic"]),
                     width="stretch", height=380)


# ---------------------------------------------------------------- router
{
    PAGES[0]: page_overview,
    PAGES[1]: page_dataset,
    PAGES[2]: page_results,
    PAGES[3]: page_classify,
    PAGES[4]: page_live,
    PAGES[5]: page_batch,
}[page]()
