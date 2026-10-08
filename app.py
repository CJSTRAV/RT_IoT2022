"""
RT-IoT2022 Cyberattack Classifier - Streamlit presentation app
Run:  streamlit run app.py
"""
import json
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import core

st.set_page_config(page_title="IoT Attack Classifier", page_icon="🛡️", layout="wide")

# ---------------------------------------------------------------- style
ATTACK = "#D64545"
NORMAL = "#2E9E6A"
ACCENT = "#1F5FAD"
MUTED = "#8A94A6"

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1300px;}
    h1, h2, h3 {letter-spacing: -0.01em;}
    .hero {padding: 1.6rem 1.8rem; border-radius: 14px;
           background: linear-gradient(120deg, #0F2A4A 0%, #1F5FAD 100%); color: #fff; margin-bottom: 1.2rem;}
    .hero h1 {color: #fff; margin: 0 0 .35rem 0; font-size: 2.1rem;}
    .hero p {color: #DCE7F5; margin: 0; font-size: 1.05rem;}
    .kpi {border: 1px solid rgba(128,128,128,.25); border-radius: 12px; padding: 1rem 1.1rem; height: 100%;}
    .kpi .v {font-size: 2rem; font-weight: 700; line-height: 1.1;}
    .kpi .l {font-size: .9rem; color: #6B7686; margin-top: .25rem;}
    .verdict {border-radius: 14px; padding: 1.2rem 1.4rem; color: #fff;}
    .verdict .big {font-size: 1.9rem; font-weight: 700;}
    .verdict .small {font-size: 1rem; opacity: .92;}
    .pill {display: inline-block; padding: .1rem .55rem; border-radius: 999px; font-size: .8rem;
           font-weight: 600; color: #fff;}
    .note {color: #6B7686; font-size: .9rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


def kpi(col, value, label, color=ACCENT):
    col.markdown(f'<div class="kpi"><div class="v" style="color:{color}">{value}</div>'
                 f'<div class="l">{label}</div></div>', unsafe_allow_html=True)


def pill(kind):
    color = NORMAL if kind == "Normal" else ATTACK
    return f'<span class="pill" style="background:{color}">{kind}</span>'


def nice(name):
    return name.replace("_", " ")


def style_fig(fig, height=420):
    fig.update_layout(height=height, margin=dict(l=10, r=30, t=50, b=10), template="plotly_white",
                      font=dict(size=14), legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                                      xanchor="right", x=1, title_text=""))
    return fig


# ---------------------------------------------------------------- data
@st.cache_resource(show_spinner="Loading Random Forest model…")
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
CLASSES = list(bundle["model"].classes_)

# ---------------------------------------------------------------- sidebar
PAGES = ["🏠 Overview", "🗂️ Dataset & Pipeline", "📈 Model Results",
         "🔍 Classify a Flow", "📡 Live Traffic Simulation", "📂 Batch Prediction"]
with st.sidebar:
    st.markdown("### 🛡️ IoT Attack Classifier")
    page = st.radio("Go to", PAGES, label_visibility="collapsed")
    st.divider()
    st.markdown(
        "**Cyberattacks on Real-Time IoT Classification Using Random Forest**\n\n"
        "Chris John Sam III V. Travilla  \nCSTL9 · University of Mindanao  \nS.Y. 2026–2027"
    )
    st.caption("Model: Random Forest · 200 trees · 62 features · seed 42")


# ================================================================ OVERVIEW
def page_overview():
    st.markdown(
        '<div class="hero"><h1>Cyberattacks on Real-Time IoT Classification Using Random Forest</h1>'
        "<p>A machine-learning model that reads one network flow and decides whether it is normal "
        "IoT traffic or one of nine attack types — trained and tested on the RT-IoT2022 dataset.</p></div>",
        unsafe_allow_html=True,
    )
    c = st.columns(4)
    kpi(c[0], "99.78%", "Accuracy on 23,582 unseen flows")
    kpi(c[1], "0.9741", "Macro F1-score (main measure)")
    kpi(c[2], "99.96%", "Attack flows detected", ATTACK)
    kpi(c[3], "0.62%", "False-alarm rate on normal traffic", NORMAL)

    st.write("")
    left, right = st.columns([1.15, 1])
    with left:
        st.subheader("The problem")
        st.markdown(
            "- IoT devices (sensors, bulbs, smart appliances) often ship with **weak security** and run unattended.\n"
            "- Signature-based intrusion detection **misses attacks** that don't match a stored rule.\n"
            "- A classifier can **learn traffic patterns** — packet counts, timing, header sizes, flags — "
            "and apply them to traffic it has never seen."
        )
        st.subheader("What we built")
        st.markdown(
            "- Cleaned **123,117 → 117,909** flows and split them **80/20** (stratified).\n"
            "- Selected **53** features and engineered **9** more → **62** inputs.\n"
            "- Trained a **Random Forest** (200 trees, balanced class weights) and compared it with five other classifiers."
        )
    with right:
        st.subheader("The 12 traffic classes")
        rows = [{"Class": nice(k), "Type": v[0], "What it is": v[1]} for k, v in core.CLASS_INFO.items()]
        info = pd.DataFrame(rows)
        st.dataframe(
            info.style.map(lambda t: f"color:{NORMAL if t == 'Normal' else ATTACK}; font-weight:600", subset=["Type"]),
            hide_index=True, width="stretch", height=458,
        )

    st.info("**Key takeaway:** 52 errors out of 23,582 test flows. Random Forest ranked first of six "
            "classifiers — only slightly ahead of a single decision tree. Most errors come from two pairs "
            "of similar classes.", icon="💡")


# ================================================================ DATASET
def page_dataset():
    st.title("Dataset & Pipeline")
    st.markdown("**RT-IoT2022** (UCI Machine Learning Repository) — real IoT network traffic. "
                "Each row is one network flow described by 83 features and labelled with its traffic class.")

    st.subheader("Conceptual framework")
    st.image(str(core.ROOT / "assets" / "conceptual_framework.png"), width="stretch")
    st.caption("Feature selection and training use the training set only — nothing is learned from the test set.")

    st.subheader("Data cleaning")
    c = st.columns(5)
    kpi(c[0], "123,117", "Raw flows")
    kpi(c[1], "−5,195", "Exact duplicates removed", MUTED)
    kpi(c[2], "−13", "Contradictory records removed", MUTED)
    kpi(c[3], "117,909", "Clean flows")
    kpi(c[4], "94,327 / 23,582", "Train / test (80/20)")
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

    st.subheader("Class distribution — heavily imbalanced")
    dist = get_csv("class_distribution.csv").sort_values("cleaned")
    dist["Type"] = dist["Attack_type"].map(lambda c: core.CLASS_INFO[c][0])
    dist["share"] = dist["cleaned"] / dist["cleaned"].sum() * 100
    log = st.toggle("Logarithmic scale", value=True)
    fig = px.bar(dist, x="cleaned", y=dist["Attack_type"].map(nice), color="Type", orientation="h",
                 color_discrete_map={"Normal": NORMAL, "Attack": ATTACK}, log_x=log,
                 text=dist.apply(lambda r: f"{r.cleaned:,} ({r.share:.2f}%)", axis=1),
                 labels={"cleaned": "Flows after cleaning", "y": ""})
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_yaxes(categoryorder="total ascending")
    fig.update_xaxes(range=[np.log10(15), np.log10(600000)] if log else [0, 115000])
    st.plotly_chart(style_fig(fig, 480), width="stretch")
    st.caption("DOS_SYN_Hping is 76.41% of all flows, NMAP_FIN_SCAN only 0.02%. That is why accuracy alone "
               "is misleading and the **macro F1-score** (every class weighted equally) is the main measure.")

    st.subheader("From 92 columns to 62 model inputs")
    steps = pd.DataFrame({
        "Step": ["One-hot encoded", "Correlation filter (> 0.95)", "Importance cut-off (99%)", "+ 9 engineered"],
        "Features": [92, 67, 53, 62],
    })
    a, b = st.columns([1.2, 1])
    with a:
        fig = go.Figure(go.Bar(x=steps["Step"], y=steps["Features"], text=steps["Features"],
                               marker_color=[MUTED, MUTED, ACCENT, ATTACK], textposition="outside"))
        fig.update_yaxes(title="Number of features", range=[0, 105])
        st.plotly_chart(style_fig(fig, 360), width="stretch")
    with b:
        st.markdown("**Engineered features** (computed from each flow's own values):")
        st.markdown(
            "- `total_pkts` — forward + backward packets\n"
            "- `fwd_bwd_pkt_ratio`, `fwd_bwd_payload_ratio`\n"
            "- `header_payload_ratio` — header bytes per payload byte\n"
            "- `data_pkt_fraction` — share of packets carrying data\n"
            "- `syn_per_pkt`, `fin_per_pkt`, `rst_per_pkt` — flag rates\n"
            "- `no_response` — flow got no reply"
        )


# ================================================================ RESULTS
def page_results():
    st.title("Model Results")
    res = get_results()["final"]
    c = st.columns(5)
    kpi(c[0], f"{res['accuracy']:.4f}", "Accuracy")
    kpi(c[1], f"{res['macro_precision']:.4f}", "Macro precision")
    kpi(c[2], f"{res['macro_recall']:.4f}", "Macro recall")
    kpi(c[3], f"{res['macro_f1']:.4f}", "Macro F1", ATTACK)
    kpi(c[4], f"{res['n_errors']} / {res['n_test']:,}", "Errors on the test set", MUTED)
    st.write("")

    tabs = st.tabs(["Model comparison", "Per-class results", "Confusion matrix",
                    "Feature importance", "Experiments"])

    with tabs[0]:
        mc = get_csv("model_comparison.csv").sort_values("macro_f1", ascending=False)
        long = mc.melt(id_vars="model", value_vars=["accuracy", "macro_f1"], var_name="Metric", value_name="Score")
        long["Metric"] = long["Metric"].map({"accuracy": "Accuracy", "macro_f1": "Macro F1"})
        fig = px.bar(long, x="model", y="Score", color="Metric", barmode="group", text=long["Score"].map("{:.4f}".format),
                     color_discrete_map={"Accuracy": "#A9BCD6", "Macro F1": ACCENT}, labels={"model": ""})
        fig.update_traces(textposition="outside", textfont_size=11)
        fig.update_yaxes(range=[0.55, 1.03])
        st.plotly_chart(style_fig(fig, 440), width="stretch")
        show = mc[["model", "accuracy", "macro_f1", "mcc", "cv_macro_f1_mean", "fit_seconds", "n_errors"]].rename(columns={
            "model": "Model", "accuracy": "Accuracy", "macro_f1": "Macro F1", "mcc": "MCC",
            "cv_macro_f1_mean": "CV macro F1", "fit_seconds": "Train time (s)", "n_errors": "Errors"})
        st.dataframe(show.style.format({"Accuracy": "{:.4f}", "Macro F1": "{:.4f}", "MCC": "{:.4f}",
                                        "CV macro F1": "{:.4f}", "Train time (s)": "{:.1f}"})
                     .highlight_max(subset=["Macro F1"], color="#D9E7F7"),
                     hide_index=True, width="stretch")
        st.markdown("Random Forest is best, but a **single decision tree** is close behind (−0.0012 macro F1) and "
                    "trains ~14× faster. Logistic Regression and Naive Bayes look fine on accuracy but collapse on "
                    "macro F1 — accuracy hides failures on small classes.")

    with tabs[1]:
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
        st.plotly_chart(style_fig(fig, 500), width="stretch")
        st.dataframe(pc.sort_values("Class")[["Class", "Type", "support", "precision", "recall", "f1"]]
                     .rename(columns={"support": "Test flows", "precision": "Precision", "recall": "Recall", "f1": "F1"})
                     .style.format({"Precision": "{:.4f}", "Recall": "{:.4f}", "F1": "{:.4f}", "Test flows": "{:,}"}),
                     hide_index=True, width="stretch")
        st.caption("Four classes are perfect. The weakest is DDOS_Slowloris (0.9032) — with 106 test flows, "
                   "that is not just a small-sample effect. The two rarest classes have only 7 and 6 test flows.")

    with tabs[2]:
        cm = get_csv("confusion_matrix_counts.csv", index_col=0)
        pct = cm.div(cm.sum(axis=1), axis=0) * 100
        mode = st.radio("Show", ["Row % (recall per class)", "Raw counts"], horizontal=True)
        z = pct.values if mode.startswith("Row") else np.log10(cm.values + 1)
        text = np.where(cm.values == 0, "",
                        np.vectorize(lambda p, n: f"{p:.1f}%<br>({n:,})")(pct.values, cm.values))
        labels = [nice(c) for c in cm.columns]
        fig = go.Figure(go.Heatmap(z=z, x=labels, y=labels, text=text, texttemplate="%{text}",
                                   textfont={"size": 10}, colorscale="Blues", showscale=False,
                                   hovertemplate="True: %{y}<br>Predicted: %{x}<extra></extra>"))
        fig.update_yaxes(autorange="reversed", title="True class")
        fig.update_xaxes(title="Predicted class", tickangle=-40)
        st.plotly_chart(style_fig(fig, 680), width="stretch")
        a, b = st.columns([1, 1.3])
        with a:
            st.markdown("**Most frequent mix-ups**")
            tc = get_csv("top_confusions.csv").head(6)
            st.dataframe(tc.rename(columns={"true_class": "True", "predicted_as": "Predicted as", "flows": "Flows"}),
                         hide_index=True, width="stretch")
        with b:
            st.markdown("**Reading the matrix**")
            st.markdown("Two pairs of similar classes cause **40 of the 52 errors**:\n"
                        "- Thing Speak ↔ ARP poisioning (13 + 7)\n"
                        "- NMAP UDP SCAN ↔ DDOS Slowloris (13 + 7)\n\n"
                        "As a normal-vs-attack detector: **21,171 / 21,179** attacks caught, "
                        "**15 / 2,403** normal flows falsely flagged.")

    with tabs[3]:
        fi = get_csv("feature_importance.csv", index_col=0).reset_index(names="Feature")
        n = st.slider("Number of features to show", 5, len(fi), 15)
        top = fi.head(n).iloc[::-1].copy()
        top["Kind"] = np.where(top["Feature"].isin(core.ENGINEERED), "Engineered",
                               np.where(top["Feature"].str.startswith(("service_", "proto_")), "Encoded category", "Original"))
        fig = px.bar(top, x="importance", y="Feature", orientation="h", color="Kind",
                     color_discrete_map={"Engineered": ATTACK, "Original": ACCENT, "Encoded category": "#E0A526"},
                     text=top["importance"].map("{:.4f}".format), labels={"importance": "Importance (mean decrease in impurity)"})
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_yaxes(categoryorder="total ascending")
        st.plotly_chart(style_fig(fig, max(380, 26 * n + 80)), width="stretch")
        st.markdown(f"Top 10 features carry **{fi.head(10)['importance'].sum() * 100:.1f}%** of the total — the model "
                    "combines many weak clues. Three of the top five are **engineered** (`header_payload_ratio`, "
                    "`syn_per_pkt`, `fin_per_pkt`), which fits SYN floods and FIN scans.")

    with tabs[4]:
        st.markdown("**Feature-set experiments (E1–E6)** — same Random Forest, different inputs")
        cfg = get_csv("config_results.csv")
        fig = px.bar(cfg, x="config", y="macro_f1", text=cfg["macro_f1"].map("{:.4f}".format),
                     hover_data=["description", "n_features", "n_errors"],
                     color=np.where(cfg["config"] == "E5", "Final model", "Other"),
                     color_discrete_map={"Final model": ACCENT, "Other": "#A9BCD6"},
                     category_orders={"config": list(cfg["config"])},
                     labels={"config": "", "macro_f1": "Macro F1", "color": ""})
        fig.update_traces(textposition="outside")
        fig.update_yaxes(range=[0.94, 0.98])
        st.plotly_chart(style_fig(fig, 360), width="stretch")
        st.dataframe(cfg[["config", "description", "n_features", "accuracy", "macro_f1", "n_errors"]]
                     .rename(columns={"config": "ID", "description": "Description", "n_features": "Features",
                                      "accuracy": "Accuracy", "macro_f1": "Macro F1", "n_errors": "Errors"})
                     .style.format({"Accuracy": "{:.4f}", "Macro F1": "{:.4f}"}),
                     hide_index=True, width="stretch")
        st.caption("E6 removes destination port and service: macro F1 drops 0.9741 → 0.9609 and errors rise "
                   "52 → 96 — the model leans on them but still works from flow behaviour alone.")
        a, b = st.columns(2)
        with a:
            st.markdown("**Train/test split sensitivity**")
            sp = get_csv("split_results.csv")
            st.dataframe(sp[["split", "train", "test", "rare_test_flows", "accuracy", "macro_f1"]]
                         .rename(columns={"split": "Split", "train": "Train", "test": "Test",
                                          "rare_test_flows": "Rare-class test flows", "accuracy": "Accuracy", "macro_f1": "Macro F1"})
                         .style.format({"Accuracy": "{:.4f}", "Macro F1": "{:.4f}", "Train": "{:,}", "Test": "{:,}"}),
                         hide_index=True, width="stretch")
        with b:
            st.markdown("**Hyper-parameter search (5-fold CV)**")
            tu = get_csv("tuning_results.csv")
            st.dataframe(tu[["candidate", "n_estimators", "max_features", "min_samples_leaf", "cv_macro_f1_mean"]]
                         .rename(columns={"candidate": "Candidate", "n_estimators": "Trees", "max_features": "Max features",
                                          "min_samples_leaf": "Min leaf", "cv_macro_f1_mean": "CV macro F1"})
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


def show_prediction(row, true_label=None):
    res, proba = core.predict(row, bundle)
    pred, conf, traffic = res.iloc[0]["predicted_class"], res.iloc[0]["confidence"], res.iloc[0]["traffic"]
    color = NORMAL if traffic == "Normal" else ATTACK
    icon = "✅" if traffic == "Normal" else "🚨"
    left, right = st.columns([1, 1.4])
    with left:
        st.markdown(
            f'<div class="verdict" style="background:{color}"><div class="small">{icon} {traffic} traffic</div>'
            f'<div class="big">{nice(pred)}</div>'
            f'<div class="small">{core.CLASS_INFO[pred][1]}</div>'
            f'<div class="small" style="margin-top:.6rem">Confidence: <b>{conf * 100:.1f}%</b> of 200 trees\' vote</div></div>',
            unsafe_allow_html=True,
        )
        if true_label is not None:
            st.write("")
            if true_label == pred:
                st.success(f"Correct — the true label is **{nice(true_label)}**.", icon="✔️")
            else:
                st.error(f"Misclassified — the true label is **{nice(true_label)}**. "
                         "This is one of the model's 52 test-set errors.", icon="✖️")
    with right:
        p = proba.iloc[0].sort_values(ascending=True)
        p = p[p > 0.001] if (p > 0.001).sum() >= 1 else p
        fig = go.Figure(go.Bar(x=p.values, y=[nice(c) for c in p.index], orientation="h",
                               marker_color=[NORMAL if c in core.NORMAL_CLASSES else ATTACK for c in p.index],
                               text=[f"{v * 100:.1f}%" for v in p.values], textposition="outside", cliponaxis=False))
        fig.update_xaxes(range=[0, 1.12], tickformat=".0%", title="Share of tree votes")
        fig.update_layout(title="Class probabilities (classes with > 0.1%)")
        st.plotly_chart(style_fig(fig, max(240, 48 * len(p) + 110)), width="stretch")


def page_classify():
    st.title("Classify a Flow")
    st.markdown("Pick a flow from the **held-out test set** (the model never saw these during training), "
                "see how the Random Forest labels it, then tweak its features to see the decision change.")
    demo = get_csv("demo_flows.csv")
    preds, _ = predict_cached(demo)
    demo = demo.assign(_pred=preds["predicted_class"].values)

    c1, c2, c3 = st.columns([1.3, 1, 1])
    with c1:
        cls = st.selectbox("True class", ["Any class"] + sorted(demo["Attack_type"].unique()),
                           format_func=lambda c: c if c == "Any class" else f"{nice(c)} ({core.CLASS_INFO[c][0]})")
    with c2:
        which = st.selectbox("Show", ["All flows", "Only misclassified flows"])
    pool = demo if cls == "Any class" else demo[demo["Attack_type"] == cls]
    if which.startswith("Only"):
        pool = pool[pool["_pred"] != pool["Attack_type"]]
    if pool.empty:
        st.warning("No flows match this filter — the model made no errors on this class in the test set.")
        return
    ids = pool["flow_id"].tolist()
    if st.session_state.get("flow_pick") not in ids:
        good = pool[pool["_pred"] == pool["Attack_type"]]["flow_id"].tolist()
        st.session_state["flow_pick"] = good[0] if good else ids[0]  # open on a correct example
    with c3:
        st.write("")
        st.write("")
        if st.button("🎲 Random flow", width="stretch"):
            st.session_state["flow_pick"] = str(np.random.choice(ids))
    fid = st.selectbox(f"Flow ({len(ids)} available)", ids, key="flow_pick")

    row = demo[demo["flow_id"] == fid].drop(columns="_pred")
    true_label = row["Attack_type"].iloc[0]
    flow = row.drop(columns=["Attack_type"])

    st.markdown("##### Flow summary")
    r = flow.iloc[0]
    m = st.columns(6)
    m[0].metric("Protocol", str(r["proto"]).upper())
    m[1].metric("Service", "none" if r["service"] == "-" else str(r["service"]))
    m[2].metric("Dest. port", f"{int(r['id.resp_p'])}")
    m[3].metric("Packets fwd / bwd", f"{int(r['fwd_pkts_tot'])} / {int(r['bwd_pkts_tot'])}")
    m[4].metric("Duration", f"{r['flow_duration']:.3f} s")
    m[5].metric("SYN / FIN / RST", f"{int(r['flow_SYN_flag_count'])} / {int(r['flow_FIN_flag_count'])} / {int(r['flow_RST_flag_count'])}")

    st.markdown("##### Model decision")
    show_prediction(flow, true_label)

    with st.expander("🧪 What-if: edit this flow and re-classify", expanded=False):
        st.caption("Change a few key features. The other 70+ features keep their original values.")
        edited = flow.copy()
        cols = st.columns(4)
        protos = core.known_values(bundle, "proto_")
        services = ["-" if s == "none" else s for s in core.known_values(bundle, "service_")]
        edited["proto"] = cols[0].selectbox("Protocol", protos, index=protos.index(r["proto"]) if r["proto"] in protos else 0,
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
        st.markdown(f"**Changed:** {', '.join(changed) if changed else 'nothing yet'}")
        show_prediction(edited)
        st.caption("Tip for the demo: take a normal MQTT flow and raise SYN flags / drop backward packets to zero "
                   "and watch the vote shift toward an attack class.")

    with st.expander("All raw features of this flow"):
        st.dataframe(flow.T.rename(columns={flow.index[0]: "value"}), width="stretch", height=360)


# ================================================================ LIVE SIMULATION
def page_live():
    st.title("Live Traffic Simulation")
    st.markdown("Replays held-out test flows one by one, as if they were arriving on an IoT network. "
                "The traffic mix follows the real test set (mostly SYN-flood flows). "
                "Every flow is classified by the Random Forest the moment it arrives.")
    stream = get_csv("stream_flows.csv")

    c = st.columns([1, 1, 1, 1])
    n = c[0].slider("Flows to replay", 50, len(stream), 200, step=50)
    speed = c[1].select_slider("Speed (flows / second)", [2, 5, 10, 20, 50, 100], value=10)
    hide_dos = c[2].toggle("Thin out SYN-flood flows", value=True,
                           help="Keeps only 1 in 8 DOS_SYN_Hping flows so the other classes appear more often.")
    c[3].write("")
    start = c[3].button("▶ Start simulation", type="primary", width="stretch")

    if hide_dos:
        dos = stream["Attack_type"] == "DOS_SYN_Hping"
        stream = pd.concat([stream[~dos], stream[dos].iloc[::8]]).sort_values("flow_id")
    flows = stream.sample(min(n, len(stream)), random_state=int(time.time()) if start else 1).reset_index(drop=True)

    kpi_box = st.empty()
    chart_box = st.empty()
    a, b = st.columns([1.6, 1])
    feed_box = a.empty()
    alert_box = b.empty()

    if not start:
        kpi_box.info("Press **Start simulation** to begin.", icon="📡")
        return

    res, _ = core.predict(flows, bundle)
    res = res.assign(flow_id=flows["flow_id"], true=flows["Attack_type"], proto=flows["proto"].str.upper(),
                     port=flows["id.resp_p"].astype(int))
    timeline = []
    alerts = []
    batch = max(1, speed // 10)  # redraw at most ~10 times per second
    for i in range(len(res)):
        r = res.iloc[i]
        if r["traffic"] == "ATTACK":
            alerts.append(f"🚨 `{r.flow_id}` **{nice(r.predicted_class)}** — {r.proto} port {r.port} "
                          f"({r.confidence * 100:.0f}%)")
        timeline.append(r["traffic"])
        if (i + 1) % batch and i + 1 != len(res):
            continue
        seen = res.iloc[: i + 1]
        attacks = int((seen["traffic"] == "ATTACK").sum())
        correct = int((seen["predicted_class"] == seen["true"]).sum())
        with kpi_box.container():
            k = st.columns(4)
            kpi(k[0], f"{i + 1:,}", "Flows processed")
            kpi(k[1], f"{attacks:,}", "Attacks detected", ATTACK)
            kpi(k[2], f"{i + 1 - attacks:,}", "Normal flows", NORMAL)
            kpi(k[3], f"{correct / (i + 1) * 100:.1f}%", "Correct so far (vs. true label)")
        counts = seen["predicted_class"].value_counts()
        fig = go.Figure(go.Bar(x=counts.values, y=[nice(x) for x in counts.index], orientation="h",
                               marker_color=[NORMAL if x in core.NORMAL_CLASSES else ATTACK for x in counts.index],
                               text=counts.values, textposition="outside", cliponaxis=False))
        fig.update_layout(title="Predicted classes so far", yaxis=dict(autorange="reversed"))
        chart_box.plotly_chart(style_fig(fig, 330), width="stretch", key=f"live_chart_{i}")
        feed = seen.tail(10).iloc[::-1][["flow_id", "proto", "port", "predicted_class", "confidence", "traffic", "true"]]
        feed = feed.rename(columns={"flow_id": "Flow", "proto": "Proto", "port": "Port", "predicted_class": "Prediction",
                                    "confidence": "Confidence", "traffic": "Verdict", "true": "True label"})
        feed_box.dataframe(
            feed.style.format({"Confidence": "{:.0%}"})
            .map(lambda v: f"color:{ATTACK if v == 'ATTACK' else NORMAL}; font-weight:700", subset=["Verdict"]),
            hide_index=True, width="stretch",
        )
        alert_box.markdown("**Alert log (latest first)**\n\n" + ("\n\n".join(alerts[::-1][:8]) or "_No attacks yet_"))
        time.sleep(batch / speed)
    st.success(f"Simulation finished — {len(res)} flows classified.", icon="✅")


# ================================================================ BATCH
def page_batch():
    st.title("Batch Prediction")
    st.markdown("Upload a CSV of network flows with the same columns as RT-IoT2022. Extra columns "
                "(row number, `id.orig_p`, `Attack_type`) are ignored. If `Attack_type` is present, the app "
                "also scores the predictions.")
    c1, c2 = st.columns([1.4, 1])
    with c1:
        up = st.file_uploader("CSV file", type=["csv"])
    with c2:
        st.write("")
        st.download_button("⬇ Download an unlabelled sample CSV", (core.DATA / "sample_upload_unlabeled.csv").read_bytes(),
                           "sample_flows.csv", "text/csv", width="stretch")
        st.caption("No file uploaded? The page uses the bundled sample of 1,123 labelled test flows.")

    if up is not None:
        df = pd.read_csv(up)
        name = up.name
    else:
        df = get_csv("demo_flows.csv")
        name = "bundled sample"

    if "service" in df.columns:
        df["service"] = df["service"].fillna("-")
    missing = core.missing_columns(df, bundle)
    if missing:
        st.error(f"The file is missing {len(missing)} required column(s): "
                 f"{', '.join(missing[:10])}{' …' if len(missing) > 10 else ''}")
        with st.expander("All required columns"):
            st.code(", ".join(core.required_columns(bundle)))
        return

    with st.spinner(f"Classifying {len(df):,} flows…"):
        res, proba = core.predict(df, bundle)

    attacks = int((res["traffic"] == "ATTACK").sum())
    k = st.columns(4)
    kpi(k[0], f"{len(df):,}", f"Flows in {name}")
    kpi(k[1], f"{attacks:,}", "Predicted attacks", ATTACK)
    kpi(k[2], f"{len(df) - attacks:,}", "Predicted normal", NORMAL)
    if "Attack_type" in df.columns:
        acc = (res["predicted_class"].values == df["Attack_type"].values).mean()
        kpi(k[3], f"{acc * 100:.2f}%", "Accuracy vs. Attack_type column")
    else:
        kpi(k[3], f"{res['confidence'].mean() * 100:.1f}%", "Average confidence")

    st.write("")
    a, b = st.columns([1, 1.2])
    with a:
        counts = res["predicted_class"].value_counts().iloc[::-1]
        fig = go.Figure(go.Bar(x=counts.values, y=[nice(x) for x in counts.index], orientation="h",
                               marker_color=[NORMAL if x in core.NORMAL_CLASSES else ATTACK for x in counts.index],
                               text=counts.values, textposition="outside", cliponaxis=False))
        fig.update_layout(title="Predicted class counts")
        st.plotly_chart(style_fig(fig, 420), width="stretch")
    with b:
        fig = px.histogram(res, x="confidence", nbins=20, color="traffic",
                           color_discrete_map={"Normal": NORMAL, "ATTACK": ATTACK},
                           labels={"confidence": "Confidence", "traffic": ""}, log_y=True,
                           title="Confidence distribution (log scale)")
        fig.update_xaxes(tickformat=".0%")
        st.plotly_chart(style_fig(fig, 420), width="stretch")

    out = pd.concat([df[[c for c in ("flow_id", "proto", "service", "id.resp_p") if c in df.columns]], res], axis=1)
    if "Attack_type" in df.columns:
        out["true_label"] = df["Attack_type"]
        out["correct"] = out["predicted_class"] == out["true_label"]
    only_attacks = st.toggle("Show predicted attacks only")
    view = out[out["traffic"] == "ATTACK"] if only_attacks else out
    st.dataframe(view.style.format({"confidence": "{:.1%}"}), width="stretch", height=380)
    st.download_button("⬇ Download predictions (CSV)", out.to_csv(index=False).encode(),
                       "predictions.csv", "text/csv", type="primary")


# ---------------------------------------------------------------- router
{
    PAGES[0]: page_overview,
    PAGES[1]: page_dataset,
    PAGES[2]: page_results,
    PAGES[3]: page_classify,
    PAGES[4]: page_live,
    PAGES[5]: page_batch,
}[page]()
