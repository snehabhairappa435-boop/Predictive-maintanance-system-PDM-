import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(page_title="Predictive Maintenance System", layout="wide")

# ---------------------------------------------------
# LOAD MODELS
# ---------------------------------------------------
scaler = joblib.load("models/scaler.pkl")
iso_model = joblib.load("models/isolation_forest.pkl")
autoencoder = load_model("models/autoencoder_model.keras")

# ---------------------------------------------------
# CONSTANTS
# ---------------------------------------------------
CRITICAL_LIMITS = {
    "temperature": 350,
    "vibration": 7,
    "pressure": 10,
    "humidity": 80
}

WEIGHTS = {
    "temperature": 0.30,
    "vibration": 0.40,
    "pressure": 0.20,
    "humidity": 0.10
}

# ---------------------------------------------------
# HEALTH FUNCTIONS
# ---------------------------------------------------
def compute_health(row):
    scores = {}
    for p in CRITICAL_LIMITS:
        limit = CRITICAL_LIMITS[p]
        current = row[p]
        scores[p] = max(0, (1 - current / limit) * 100)
    return scores

def compute_weighted_health(scores):
    return sum(scores[p] * WEIGHTS[p] for p in scores)

# ---------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------
pages = [
    "Upload CSV & Predict",
    "Machine Failure Probability",
    "Parameter Health Analysis",
    "Alerts & Notifications",
    "Cost & Savings Analysis",
    "Monitoring Dashboard",
    "Model Comparison"
]

choice = st.sidebar.selectbox("Navigate", pages)
# ---------------------------------------------------
# PAGE 1 — UPLOAD CSV & PREDICT (runs both models)
# ---------------------------------------------------
if choice == "Upload CSV & Predict":
    st.title(" Upload CSV File for Prediction")

    uploaded_file = st.file_uploader("Upload Machine Sensor CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        st.write("### Uploaded Data")
        st.dataframe(df.head())

        # -----------------------
        # FEATURE EXTRACTION
        # -----------------------
        X = df[["temperature", "vibration", "pressure", "humidity"]]
        X_scaled = scaler.transform(X)

        st.subheader("Running Predictions with Both Models...")

        # ---------------------------------------------------
        # 1. Isolation Forest
        # ---------------------------------------------------
        iso_scores = iso_model.decision_function(X_scaled)
        iso_prob = (1 - (iso_scores - iso_scores.min()) /
                    (iso_scores.max() - iso_scores.min())) * 100
        df["iso_failure_probability"] = iso_prob.clip(0, 100)

        # ---------------------------------------------------
        # 2. Autoencoder
        # ---------------------------------------------------
        reconstructed = autoencoder.predict(X_scaled)
        mse = np.mean((X_scaled - reconstructed) ** 2, axis=1)
        ae_prob = (mse - mse.min()) / (mse.max() - mse.min()) * 100
        df["ae_failure_probability"] = ae_prob.clip(0, 100)

        # ---------------------------------------------------
        # SAVE TO SESSION
        # ---------------------------------------------------
        st.session_state["data"] = df

        # ---------------------------------------------------
        # SHOW FULL TABLE (10,000 ROWS)
        # ---------------------------------------------------
        st.success("Prediction Completed with BOTH MODELS!")

        st.write("### Full Prediction Table")
        st.dataframe(df, use_container_width=True)   # FULL TABLE

        # ---------------------------------------------------
        # DOWNLOAD BUTTON
        # ---------------------------------------------------
        st.download_button(
            "Download Predictions CSV",
            df.to_csv(index=False).encode("utf-8"),
            file_name="predictions.csv"
        )

# ---------------------------------------------------
# PAGE 2 — FAILURE PROBABILITY (both models)
# ---------------------------------------------------
elif choice == "Machine Failure Probability":
    st.title(" Machine Failure Probability (Isolation Forest & Autoencoder)")

    if "data" not in st.session_state:
        st.warning("Upload CSV first.")
        st.stop()

    df = st.session_state["data"].copy()

    # Convert timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Partition time of day
    def get_day_part(hour):
        if 0 <= hour < 6: return "00:00–06:00"
        if 6 <= hour < 12: return "06:00–12:00"
        if 12 <= hour < 18: return "12:00–18:00"
        return "18:00–24:00"

    df["day_part"] = df["timestamp"].dt.hour.apply(get_day_part)
    parts = ["00:00–06:00", "06:00–12:00", "12:00–18:00", "18:00–24:00"]

    # --------- Helper: Dynamic y-axis range ---------
    def auto_yaxis_range(series):
        minimum = series.min()
        maximum = series.max()
        if maximum == minimum:   # avoid flat line issue
            return [0, maximum + 10]
        padding = (maximum - minimum) * 0.25
        return [max(0, minimum - padding), maximum + padding]

    # --------- Machine selector ---------
    st.subheader(" Select Machine")
    selected_machine = st.selectbox("Machine ID", sorted(df["machine_id"].unique()))

    df_m = df[df["machine_id"] == selected_machine]

    # Compute mean per time segment
    def compute_parts(col):
        g = df_m.groupby("day_part")[col].mean().reset_index()
        return g.set_index("day_part").reindex(parts, fill_value=0).reset_index()

    m_iso = compute_parts("iso_failure_probability")
    m_ae = compute_parts("ae_failure_probability")

    # -----------------------
    # MACHINE-SPECIFIC GRAPHS
    # -----------------------
    st.subheader(f" Failure Probability — Machine {selected_machine}")

    # --- Isolation Forest ---
    fig_iso = px.bar(
        m_iso,
        x="day_part",
        y="iso_failure_probability",
        title="Isolation Forest Failure Probability",
        color="day_part"
    )
    fig_iso.update_yaxes(range=auto_yaxis_range(m_iso["iso_failure_probability"]))
    st.plotly_chart(fig_iso, use_container_width=True)

    # --- Autoencoder ---
    fig_ae = px.bar(
        m_ae,
        x="day_part",
        y="ae_failure_probability",
        title="Autoencoder Failure Probability",
        color="day_part"
    )
    fig_ae.update_yaxes(range=auto_yaxis_range(m_ae["ae_failure_probability"]))
    st.plotly_chart(fig_ae, use_container_width=True)

    # -----------------------
    # OVERALL GRAPHS
    # -----------------------
    st.subheader("Overall Failure Probability (All Machines)")

    overall_iso = (
        df.groupby("day_part")["iso_failure_probability"]
        .mean()
        .reset_index()
        .set_index("day_part")
        .reindex(parts, fill_value=0)
        .reset_index()
    )

    overall_ae = (
        df.groupby("day_part")["ae_failure_probability"]
        .mean()
        .reset_index()
        .set_index("day_part")
        .reindex(parts, fill_value=0)
        .reset_index()
    )

    # --- Overall Isolation Forest ---
    fig_iso_overall = px.bar(
        overall_iso,
        x="day_part",
        y="iso_failure_probability",
        title="Overall Isolation Forest Failure Probability",
        color="day_part"
    )
    fig_iso_overall.update_yaxes(range=auto_yaxis_range(overall_iso["iso_failure_probability"]))
    st.plotly_chart(fig_iso_overall, use_container_width=True)

    # --- Overall Autoencoder ---
    fig_ae_overall = px.bar(
        overall_ae,
        x="day_part",
        y="ae_failure_probability",
        title="Overall Autoencoder Failure Probability",
        color="day_part"
    )
    fig_ae_overall.update_yaxes(range=auto_yaxis_range(overall_ae["ae_failure_probability"]))
    st.plotly_chart(fig_ae_overall, use_container_width=True)

# ---------------------------------------------------
# PAGE 3 — PARAMETER HEALTH ANALYSIS
# ---------------------------------------------------
elif choice == "Parameter Health Analysis":
    st.title(" Parameter Health Analysis")

    if "data" not in st.session_state:
        st.warning("Upload CSV first.")
        st.stop()

    df = st.session_state["data"]

    machine = st.selectbox("Select Machine ID", df["machine_id"].unique())
    row = df[df["machine_id"] == machine].iloc[0]

    scores = compute_health(row)
    weighted = compute_weighted_health(scores)
    worst = min(scores, key=scores.get)

    st.subheader(f"Overall Health: {weighted:.2f}%")
    st.subheader(f"Most Critical Parameter: {worst.upper()}")

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=list(scores.values()),
        theta=list(scores.keys()),
        fill='toself'
    ))
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------
# PAGE 4 — ALERTS
# ---------------------------------------------------
elif choice == "Alerts & Notifications":
    st.title("Alerts & Notifications")

    if "data" not in st.session_state:
        st.warning("Upload CSV first.")
        st.stop()

    df = st.session_state["data"]
    alerts = []

    for _, r in df.iterrows():
        for p in CRITICAL_LIMITS:
            if r[p] >= 0.9 * CRITICAL_LIMITS[p]:
                alerts.append((r["machine_id"], p, r[p]))

    if alerts:
        st.error(" Critical Alerts!")
        for m, p, v in alerts:
            st.write(f"Machine {m}: {p.upper()} = {v}")
    else:
        st.success("All machines are in safe condition.")

# ---------------------------------------------------
# PAGE 5 — COST & SAVINGS ANALYSIS
# ---------------------------------------------------
elif choice == "Cost & Savings Analysis":
    st.title("Cost & Savings Analysis ")

    if "data" not in st.session_state:
        st.warning("Upload CSV first.")
        st.stop()

    df = st.session_state["data"].copy()

    df["iso_cost_saved"] = df["iso_failure_probability"] / 100 * 50
    df["ae_cost_saved"] = df["ae_failure_probability"] / 100 * 50

    ranges = [(i, i + 1000) for i in range(0, 10000, 1000)]

    for start, end in ranges:
        sub = df[(df.machine_id >= start) & (df.machine_id < end)]
        st.markdown(f"### Machines {start+1} – {end}")

        fig1 = px.line(sub, x="machine_id", y="iso_failure_probability",
                       title="Isolation Forest Failure Probability")
        st.plotly_chart(fig1)

        fig2 = px.line(sub, x="machine_id", y="ae_failure_probability",
                       title="Autoencoder Failure Probability")
        st.plotly_chart(fig2)

    iso_total = df["iso_cost_saved"].sum()
    ae_total = df["ae_cost_saved"].sum()

    best = "Isolation Forest" if iso_total > ae_total else "Autoencoder"



# ---------------------------------------------------
# PAGE 6 — MONITORING DASHBOARD (Using All 12 Machines)
# ---------------------------------------------------
elif choice == "Monitoring Dashboard":
    st.title("Machine Monitoring Dashboard")

    if "data" not in st.session_state:
        st.warning("Upload CSV first.")
        st.stop()

    df = st.session_state["data"]

    # Full machine group mapping
    MACHINE_GROUPS = {
        "temperature": [
            "Iron Machine",
            "Single Needle Machine",
            "Keyhole Machine"
        ],
        "vibration": [
            "Overlock Machine",
            "Double Needle Machine",
            "Bartack Machine"
        ],
        "pressure": [
            "Cutting Machine",
            "Edge Cutter Machine",
            "Feed of the Arm Machine"
        ],
        "humidity": [
            "Thread Sucking Machine",
            "Kansi Machine",
            "Shank and Rivet Machine"
        ]
    }

    for m in df["machine_id"].unique():

        row = df[df["machine_id"] == m].iloc[0]
        scores = compute_health(row)
        weighted = compute_weighted_health(scores)

        # Find most critical parameter
        critical_param = min(scores, key=scores.get)

        # Machine list for this parameter
        machine_list = MACHINE_GROUPS[critical_param]

        # Rotate machine selection based on machine_id
        machine_name = machine_list[m % len(machine_list)]

        # Health color coding
        color = (
            "green" if weighted > 80
            else "orange" if weighted > 50
            else "red"
        )

        # Display Dashboard Card
        st.markdown(f"""
        <div style="background:{color}; padding:15px; 
             border-radius:10px; color:white; margin:10px 0;">
            <h3>Machine ID: {m}</h3>
            <p><b>Machine Type:</b> {machine_name}</p>
            <p><b>Health Score:</b> {weighted:.2f}%</p>
            <p><b>Critical Parameter:</b> {critical_param.upper()}</p>
        </div>
        """, unsafe_allow_html=True)



elif choice == "Model Comparison":
    st.title("Model Comparison — Isolation Forest vs Autoencoder")

    if "data" not in st.session_state:
        st.warning("Upload CSV first.")
        st.stop()

    df_o = st.session_state["data"].copy()

    # ------------------------------
    # AVERAGE FAILURE & COST
    # ------------------------------
    iso_avg_fail = df_o["iso_failure_probability"].mean()
    ae_avg_fail = df_o["ae_failure_probability"].mean()

    iso_avg_cost = iso_avg_fail * 0.5
    ae_avg_cost = ae_avg_fail * 0.5

    # Final Score = (100 - failure%) + cost_saved
    iso_score = (100 - iso_avg_fail) + iso_avg_cost
    ae_score = (100 - ae_avg_fail) + ae_avg_cost

    # Dataframe for grouped graphs
    compare_df = pd.DataFrame({
        "Metric": ["Failure Probability (%)", "Cost Saved", "Overall Score"],
        "Isolation Forest": [iso_avg_fail, iso_avg_cost, iso_score],
        "Autoencoder": [ae_avg_fail, ae_avg_cost, ae_score]
    })

    # Melt for Plotly
    compare_melt = compare_df.melt(id_vars="Metric",
                                   var_name="Model",
                                   value_name="Value")

    # ------------------------------
    # GROUPED BAR GRAPH
    # ------------------------------
    st.subheader(" Model Comparison Chart (ISO vs AE)")

    fig = px.bar(
        compare_melt,
        x="Metric",
        y="Value",
        color="Model",
        barmode="group",
        text_auto='.2f',
        title=" Model Comparison"
    )

    fig.update_layout(
        xaxis_title="Metrics",
        yaxis_title="Score / Value",
        legend_title="Model",
        bargap=0.25
    )

    st.plotly_chart(fig, use_container_width=True)

    # ------------------------------
    # SUMMARY
    # ------------------------------
    st.subheader(" Numerical Summary")

    col1, col2 = st.columns(2)

    with col1:
        st.write("### Isolation Forest")
        st.write(f"- Avg Failure Probability: **{iso_avg_fail:.2f}%**")
        st.write(f"- Avg Cost Saved: **{iso_avg_cost:.2f}**")
        st.write(f"- Overall Score: **{iso_score:.2f}**")

    with col2:
        st.write("### Autoencoder")
        st.write(f"- Avg Failure Probability: **{ae_avg_fail:.2f}%**")
        st.write(f"- Avg Cost Saved: **{ae_avg_cost:.2f}**")
        st.write(f"- Overall Score: **{ae_score:.2f}**")

    # ------------------------------
    # BEST MODEL RECOMMENDATION
    # ------------------------------
    st.subheader(" Best Model Recommendation")

    if ae_score > iso_score:
        best = "Autoencoder"
        color = "green"
    else:
        best = "Isolation Forest"
        color = "blue"

    st.markdown(
        f"""
        <div style='padding:20px; background:{color}; color:white; border-radius:10px;'>
            <h2>Best Performing Model: {best} </h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Show full table
    st.write("### Full Comparison Table")
    st.dataframe(compare_df)
