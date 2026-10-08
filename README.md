# IoT Attack Classifier — Streamlit app

Presentation app (styled with the Ulticon "soft cards" theme) for **"Cyberattacks on Real-Time IoT Classification Using Random Forest"**
(Chris John Sam III V. Travilla · CSTL9 · University of Mindanao).

## Run it

Needs Python 3.11 or newer.

- **Windows:** double-click `run_app.bat`
- **Mac / Linux:** `./run_app.sh`
- **Manually:**
  ```
  pip install -r requirements.txt
  streamlit run app.py
  ```

The app opens at http://localhost:8501. Everything it needs is in this folder. The theme fonts (Bricolage Grotesque,
Figtree) load from Google Fonts; offline, the app falls back to your system font and still works.

`scikit-learn` is pinned to 1.8.0 because that is the version the model was saved with.

## Pages

| Page | What it shows |
|---|---|
| Overview | Problem, approach, headline results, the 12 traffic classes |
| Background of the Study | Background, general and specific objectives, scope and limitations, dataset description, why Random Forest |
| Dataset & Pipeline | Conceptual framework, before-and-after preprocessing, normal vs attack split, class imbalance, 92 → 62 features |
| Model Results | Model comparison, per-class F1, confusion matrix, ROC curves and AUC, feature importance, experiments E1–E6 |
| Classify a Flow | Pick a held-out test flow, see the prediction and tree votes, edit features (what-if) |
| Live Traffic Simulation | Replays test flows as live traffic with counters, feed and alert log |
| Batch Prediction | Upload a CSV of flows, get predictions, download results |

## Suggested demo script (≈ 5 minutes)

1. **Overview** — read the four headline numbers; point out macro F1 is the main measure.
1. **Background of the Study** — background, objectives, scope and limitations.
2. **Dataset & Pipeline** — show the class-imbalance chart (76% is one class) to explain why accuracy alone misleads.
3. **Model Results → Confusion matrix** — two pairs of similar classes cause 40 of the 52 errors.
4. **Classify a Flow** — click *Random flow* a few times; then set *Show → Only misclassified flows* to be honest about the errors.
5. **What-if** — open the expander, change SYN flags / backward packets and watch the vote move.
6. **Live Traffic Simulation** — 200 flows at 10 flows/s; let it run while you talk about deployment.
7. **Batch Prediction** — download the sample CSV, upload it, download predictions.

## Files

```
app.py                     Streamlit app
core.py                    Preprocessing + prediction (same steps as rt_iot2022_model.py)
rt_iot2022_model.py        Original training / CLI script
model/rt_iot2022_rf.joblib Trained Random Forest (200 trees, 62 features)
data/demo_flows.csv        1,123 held-out test flows (≤120 per class, includes all 52 misclassified)
data/stream_flows.csv      1,500 random test flows in the real class mix (live simulation)
data/sample_upload_unlabeled.csv  300 unlabelled flows for the batch-upload demo
data/*.csv, results.json   Experiment results from the paper
assets/*.png               Figures from the paper
```

All demo flows come from the paper's own 20% test split (seed 42), so the model has never seen them.
On the full test set the model reproduces the paper exactly: 52 errors out of 23,582 flows.
