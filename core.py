"""
Model logic for the RT-IoT2022 Streamlit app.

The preprocessing here is identical to rt_iot2022_model.py (encode + engineer),
so a flow is classified exactly the way it was during evaluation in the paper.
"""
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
MODEL_FILE = ROOT / "model" / "rt_iot2022_rf.joblib"
DATA = ROOT / "data"

NORMAL_CLASSES = {"Thing_Speak", "MQTT_Publish", "Wipro_bulb"}

# Short, presenter-friendly descriptions of the 12 classes.
CLASS_INFO = {
    "Thing_Speak": ("Normal", "IoT cloud analytics traffic (ThingSpeak sensor uploads)"),
    "MQTT_Publish": ("Normal", "IoT sensor publishing messages over MQTT"),
    "Wipro_bulb": ("Normal", "Smart light bulb traffic"),
    "DOS_SYN_Hping": ("Attack", "SYN flood denial of service made with hping3"),
    "DDOS_Slowloris": ("Attack", "Slow HTTP DoS that keeps many connections half open"),
    "ARP_poisioning": ("Attack", "ARP spoofing / man-in-the-middle on the local network"),
    "Metasploit_Brute_Force_SSH": ("Attack", "SSH password brute force using Metasploit"),
    "NMAP_UDP_SCAN": ("Attack", "Reconnaissance: Nmap UDP port scan"),
    "NMAP_XMAS_TREE_SCAN": ("Attack", "Reconnaissance: Nmap XMAS scan (FIN, PSH, URG flags set)"),
    "NMAP_OS_DETECTION": ("Attack", "Reconnaissance: Nmap operating-system fingerprinting"),
    "NMAP_TCP_scan": ("Attack", "Reconnaissance: Nmap TCP connect scan"),
    "NMAP_FIN_SCAN": ("Attack", "Reconnaissance: Nmap FIN scan"),
}

ENGINEERED = [
    "total_pkts", "fwd_bwd_pkt_ratio", "fwd_bwd_payload_ratio", "header_payload_ratio",
    "data_pkt_fraction", "syn_per_pkt", "fin_per_pkt", "rst_per_pkt", "no_response",
]

# Columns that are never model inputs and are ignored if present.
IGNORED = ("Unnamed: 0", "", "flow_id", "id.orig_p", "Attack_type")


def load_bundle(path=MODEL_FILE):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # sklearn version-mismatch notice on unpickle
        return joblib.load(path)


def _safe_div(a, b):
    return pd.Series(np.where(b == 0, 0.0, a / b.where(b != 0, 1)), index=a.index)


def engineer(d):
    """The nine engineered features of the paper."""
    pkts = d["fwd_pkts_tot"] + d["bwd_pkts_tot"]
    new = pd.DataFrame(index=d.index)
    new["total_pkts"] = pkts
    new["fwd_bwd_pkt_ratio"] = d["fwd_pkts_tot"] / (d["bwd_pkts_tot"] + 1)
    new["fwd_bwd_payload_ratio"] = d["fwd_pkts_payload.tot"] / (d["bwd_pkts_payload.tot"] + 1)
    new["header_payload_ratio"] = (d["fwd_header_size_tot"] + d["bwd_header_size_tot"]) / (d["flow_pkts_payload.tot"] + 1)
    new["data_pkt_fraction"] = _safe_div(d["fwd_data_pkts_tot"] + d["bwd_data_pkts_tot"], pkts)
    new["syn_per_pkt"] = _safe_div(d["flow_SYN_flag_count"], pkts)
    new["fin_per_pkt"] = _safe_div(d["flow_FIN_flag_count"], pkts)
    new["rst_per_pkt"] = _safe_div(d["flow_RST_flag_count"], pkts)
    new["no_response"] = (d["bwd_pkts_tot"] == 0).astype(int)
    return new


def required_columns(bundle):
    """Raw flow columns a CSV must contain."""
    base = {c for c in bundle["encoded_columns"] if not c.startswith(("proto_", "service_"))}
    return sorted(base | {"proto", "service", "bwd_pkts_tot"})


def missing_columns(df, bundle):
    return [c for c in required_columns(bundle) if c not in df.columns]


def known_values(bundle, prefix):
    return [c[len(prefix):] for c in bundle["encoded_columns"] if c.startswith(prefix)]


def prepare(df, bundle):
    """Raw flows -> the 62-column matrix the forest was trained on."""
    d = df.drop(columns=[c for c in IGNORED if c in df.columns]).copy()
    d["service"] = d["service"].fillna("-").astype(str).replace("-", "none")
    d["proto"] = d["proto"].astype(str)
    d = pd.get_dummies(d, columns=["proto", "service"], dtype=int)
    d = d.reindex(columns=bundle["encoded_columns"], fill_value=0)  # unseen service -> all zeros
    d = d.apply(pd.to_numeric, errors="coerce").fillna(0)
    return pd.concat([d[bundle["selected"]], engineer(d)], axis=1)[bundle["final_columns"]]


def predict(df, bundle):
    """Returns (results DataFrame, probability DataFrame)."""
    model = bundle["model"]
    X = prepare(df, bundle)
    proba = model.predict_proba(X)
    classes = model.classes_
    out = pd.DataFrame({
        "predicted_class": classes[proba.argmax(axis=1)],
        "confidence": proba.max(axis=1),
    }, index=df.index)
    out["traffic"] = np.where(out["predicted_class"].isin(NORMAL_CLASSES), "Normal", "ATTACK")
    return out, pd.DataFrame(proba, columns=classes, index=df.index)


def load_csv(name):
    return pd.read_csv(DATA / name)
