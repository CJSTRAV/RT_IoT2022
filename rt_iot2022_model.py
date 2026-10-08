"""
RT-IoT2022 Random Forest - train once, then classify new flows.

    python rt_iot2022_model.py train   RT_IOT2022.csv
    python rt_iot2022_model.py predict new_flows.csv [predictions.csv]

'train' repeats the cleaning, split, feature selection and feature engineering of the
paper (seed 42, 80/20 stratified, 200 trees, balanced_subsample), prints the held-out
score, and saves everything the model needs to rt_iot2022_rf.joblib.

'predict' loads that file and labels every row of a CSV of flows. The CSV needs the same
flow columns as RT_IOT2022 (extra columns such as the row number, id.orig_p or
Attack_type are ignored).

Needs: pandas, numpy, scikit-learn, joblib
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

SEED = 42
CORR_LIMIT = 0.95
IMPORTANCE_KEEP = 0.99
MODEL_FILE = Path("rt_iot2022_rf.joblib")
NORMAL_CLASSES = {"Thing_Speak", "MQTT_Publish", "Wipro_bulb"}


def make_rf(n_trees):
    return RandomForestClassifier(n_estimators=n_trees, class_weight="balanced_subsample",
                                  n_jobs=-1, random_state=SEED)


def safe_div(a, b):
    return pd.Series(np.where(b == 0, 0.0, a / b.where(b != 0, 1)), index=a.index)


def engineer(d):
    """The nine engineered features of Table 3."""
    pkts = d["fwd_pkts_tot"] + d["bwd_pkts_tot"]
    new = pd.DataFrame(index=d.index)
    new["total_pkts"] = pkts
    new["fwd_bwd_pkt_ratio"] = d["fwd_pkts_tot"] / (d["bwd_pkts_tot"] + 1)
    new["fwd_bwd_payload_ratio"] = d["fwd_pkts_payload.tot"] / (d["bwd_pkts_payload.tot"] + 1)
    new["header_payload_ratio"] = (d["fwd_header_size_tot"] + d["bwd_header_size_tot"]) / (d["flow_pkts_payload.tot"] + 1)
    new["data_pkt_fraction"] = safe_div(d["fwd_data_pkts_tot"] + d["bwd_data_pkts_tot"], pkts)
    new["syn_per_pkt"] = safe_div(d["flow_SYN_flag_count"], pkts)
    new["fin_per_pkt"] = safe_div(d["flow_FIN_flag_count"], pkts)
    new["rst_per_pkt"] = safe_div(d["flow_RST_flag_count"], pkts)
    new["no_response"] = (d["bwd_pkts_tot"] == 0).astype(int)
    return new


def encode(d, encoded_columns):
    """service '-' -> 'none', one-hot proto/service, same columns as training."""
    d = d.copy()
    d["service"] = d["service"].replace("-", "none")
    d = pd.get_dummies(d, columns=["proto", "service"], dtype=int)
    # a service never seen in training just becomes all zeros
    return d.reindex(columns=encoded_columns, fill_value=0)


def train(csv_path):
    raw = pd.read_csv(csv_path, index_col=0)

    # cleaning (same order as the paper: duplicates before dropping the source port)
    df = raw.copy()
    df["service"] = df["service"].replace("-", "none")
    df = df.drop(columns=[c for c in df.columns if df[c].nunique() <= 1])
    df = df.drop_duplicates()
    feats = [c for c in df.columns if c != "Attack_type"]
    df = df[df.groupby(feats)["Attack_type"].transform("nunique") == 1].reset_index(drop=True)
    df = df.drop(columns=["id.orig_p"])
    print(f"cleaned: {len(df):,} flows")

    y = df["Attack_type"]
    X = pd.get_dummies(df.drop(columns="Attack_type"), columns=["proto", "service"], dtype=int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=SEED)

    # feature selection on the training set only
    numeric = [c for c in X_train.columns if not c.startswith(("proto_", "service_"))]
    corr = X_train[numeric].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    too_correlated = [c for c in upper.columns if (upper[c] > CORR_LIMIT).any()]
    step1 = [c for c in X_train.columns if c not in too_correlated]
    ranker = make_rf(100).fit(X_train[step1], y_train)
    importance = pd.Series(ranker.feature_importances_, index=step1).sort_values(ascending=False)
    selected = list(importance.index[:int((importance.cumsum() < IMPORTANCE_KEEP).sum()) + 1])

    def finish(d):
        return pd.concat([d[selected], engineer(d)], axis=1)

    X_train_final, X_test_final = finish(X_train), finish(X_test)
    print(f"features: {X_train_final.shape[1]} ({len(selected)} selected + 9 engineered)")

    model = make_rf(200).fit(X_train_final, y_train)
    pred = model.predict(X_test_final)
    print(f"held-out test: accuracy {accuracy_score(y_test, pred):.4f}, "
          f"macro F1 {f1_score(y_test, pred, average='macro'):.4f}")

    joblib.dump(dict(model=model, encoded_columns=list(X.columns), selected=selected,
                     final_columns=list(X_train_final.columns)), MODEL_FILE, compress=3)
    print(f"saved {MODEL_FILE} ({MODEL_FILE.stat().st_size / 1e6:.1f} MB)")


def predict(csv_path, out_path=None):
    bundle = joblib.load(MODEL_FILE)
    new = pd.read_csv(csv_path)
    new = new.drop(columns=[c for c in ("Unnamed: 0", "", "id.orig_p", "Attack_type") if c in new.columns])

    needed = set(bundle["encoded_columns"]) | {"proto", "service", "bwd_pkts_tot"}
    base = {c for c in needed if not c.startswith(("proto_", "service_"))}
    missing = sorted(base - set(new.columns))
    if missing:
        sys.exit(f"missing columns in {csv_path}: {missing[:8]}{' ...' if len(missing) > 8 else ''}")

    X = encode(new, bundle["encoded_columns"])
    X = pd.concat([X[bundle["selected"]], engineer(X)], axis=1)[bundle["final_columns"]]

    model = bundle["model"]
    proba = model.predict_proba(X)
    out = pd.DataFrame({
        "predicted_class": model.classes_[proba.argmax(axis=1)],
        "confidence": proba.max(axis=1).round(4),
    })
    out["traffic"] = np.where(out["predicted_class"].isin(NORMAL_CLASSES), "normal", "ATTACK")

    print(out["predicted_class"].value_counts().to_string())
    if out_path:
        out.to_csv(out_path, index=False)
        print(f"wrote {out_path}")
    return out


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] not in ("train", "predict"):
        sys.exit(__doc__)
    if sys.argv[1] == "train":
        train(sys.argv[2])
    else:
        predict(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
