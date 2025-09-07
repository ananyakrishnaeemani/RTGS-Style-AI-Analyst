# app.py - Streamlit dashboard for Telangana Literacy (full-featured)
from pathlib import Path
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from textwrap import shorten
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression

# ----------------------------
# Configuration
# ----------------------------
ARTIFACTS_DIR = Path("artifacts/full-run")
RAW_CSV = ARTIFACTS_DIR / "literacy.csv"
TRANSFORMED_CSV = ARTIFACTS_DIR / "transformed_literacy.csv"

st.set_page_config(page_title="Telangana Literacy Dashboard", layout="wide")
sns.set_theme(style="whitegrid")

# ----------------------------
# Utility functions
# ----------------------------
def standardize_df(df):
    # Normalize column names to lowercase snake_case
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df

def ensure_transformed():
    """
    Return dataframe that has:
    - districts
    - literacy_rate_males
    - literacy_rate_females
    - literacy_rate_avg
    - gender_gap
    If TRANSFORMED_CSV exists, load that, otherwise try to build from RAW_CSV.
    """
    if TRANSFORMED_CSV.exists():
        df = pd.read_csv(TRANSFORMED_CSV)
        df = standardize_df(df)
    elif RAW_CSV.exists():
        df = pd.read_csv(RAW_CSV)
        df = standardize_df(df)
        # attempt to detect literacy columns
        required = ['literacy_rate_males', 'literacy_rate_females']
        if not all(c in df.columns for c in required):
            st.warning("Input CSV does not contain expected literacy columns. Please check column names.")
        # derive columns if missing
        if 'literacy_rate_avg' not in df.columns:
            if all(c in df.columns for c in required):
                df['literacy_rate_avg'] = df[['literacy_rate_males', 'literacy_rate_females']].mean(axis=1)
        if 'gender_gap' not in df.columns:
            if all(c in df.columns for c in required):
                df['gender_gap'] = df['literacy_rate_males'] - df['literacy_rate_females']
        # save transformed for reproducibility
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(TRANSFORMED_CSV, index=False)
    else:
        st.error(f"No dataset found. Place your `literacy.csv` at {RAW_CSV} or produce {TRANSFORMED_CSV}")
        st.stop()
    # final standardize column names one more time
    df = standardize_df(df)
    # ensure essential columns exist
    for col in ['districts', 'literacy_rate_males', 'literacy_rate_females', 'literacy_rate_avg', 'gender_gap']:
        if col not in df.columns:
            df[col] = np.nan
    return df

def df_to_csv_bytes(df):
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return buf

def fig_to_png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return buf

# ----------------------------
# Load / prepare data
# ----------------------------
st.title("📊 Telangana Literacy Dashboard")
st.markdown("Interactive dashboard for district-level literacy. Data is loaded from `artifacts/full-run/transformed_literacy.csv` (or `literacy.csv` to be processed).")

df = ensure_transformed()

# basic sanity: drop rows without district
df = df.dropna(subset=['districts']).reset_index(drop=True)
# shorten district names for plots but keep original
df['district_short'] = df['districts'].astype(str).apply(lambda x: shorten(x, width=30, placeholder="..."))

# ----------------------------
# Sidebar controls
# ----------------------------
st.sidebar.header("Controls")
show_raw = st.sidebar.checkbox("Show raw table (first 10 rows)", value=False)
min_lit = st.sidebar.number_input("Min literacy % (filter)", value=float(df['literacy_rate_avg'].min()), step=0.1)
max_lit = st.sidebar.number_input("Max literacy % (filter)", value=float(df['literacy_rate_avg'].max()), step=0.1)
name_filter = st.sidebar.text_input("District name contains (filter)", value="")

# Top/Bottom N
top_n = st.sidebar.slider("Top/Bottom N", min_value=1, max_value=15, value=5)

# Clustering config
st.sidebar.markdown("---")
st.sidebar.subheader("AI / ML Options")
k_clusters = st.sidebar.slider("K for KMeans", 2, 6, 3)
anomaly_contamination = st.sidebar.slider("Anomaly contamination", 0.01, 0.3, 0.10, step=0.01)

# Exports
st.sidebar.markdown("---")
if st.sidebar.button("Save current transformed CSV"):
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TRANSFORMED_CSV, index=False)
    st.sidebar.success(f"Saved {TRANSFORMED_CSV}")

# ----------------------------
# Apply filters
# ----------------------------
filtered = df[(df['literacy_rate_avg'] >= min_lit) & (df['literacy_rate_avg'] <= max_lit)]
if name_filter.strip():
    filtered = filtered[filtered['districts'].str.contains(name_filter.strip(), case=False, na=False)]

# ----------------------------
# Show raw / filtered tables
# ----------------------------
if show_raw:
    st.subheader("Raw / Filtered Data (first 50 rows)")
    st.dataframe(filtered.head(50))
else:
    st.subheader("Dataset Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Districts", int(df['districts'].nunique()))
    col2.metric("Mean Literacy (%)", f"{df['literacy_rate_avg'].mean():.2f}")
    col3.metric("Mean Gender Gap (%)", f"{df['gender_gap'].mean():.2f}")

# ----------------------------
# Top / Bottom panels
# ----------------------------
st.markdown("### Top & Bottom Districts")
c1, c2 = st.columns(2)

with c1:
    st.markdown(f"**Top {top_n} by average literacy**")
    top_df = filtered.nlargest(top_n, 'literacy_rate_avg')[['districts', 'literacy_rate_avg', 'gender_gap']]
    st.table(top_df.reset_index(drop=True))

with c2:
    st.markdown(f"**Bottom {top_n} by average literacy**")
    bottom_df = filtered.nsmallest(top_n, 'literacy_rate_avg')[['districts', 'literacy_rate_avg', 'gender_gap']]
    st.table(bottom_df.reset_index(drop=True))

# download top/bottom CSVs
col_dl_top, col_dl_bot = st.columns(2)
with col_dl_top:
    st.download_button("Download Top CSV", data=df_to_csv_bytes(top_df), file_name="top_districts.csv", mime="text/csv")
with col_dl_bot:
    st.download_button("Download Bottom CSV", data=df_to_csv_bytes(bottom_df), file_name="bottom_districts.csv", mime="text/csv")

# ----------------------------
# Enhanced histogram + gap bars
# ----------------------------
st.markdown("### Enhanced Literacy Histogram (bars show literacy, overlay shows gender gap)")
fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(filtered))
width = 0.6
ax.bar(x, filtered['literacy_rate_avg'], width=width, label='Avg Literacy', color='skyblue')
# overlay gender gap as thin bar (scaled)
gap_scale = filtered['gender_gap'].abs().max() if not filtered['gender_gap'].empty else 1
ax.bar(x, filtered['gender_gap'], width=width*0.4, label='Gender Gap (M-F)', color='tomato', alpha=0.7)
ax.set_xticks(x)
ax.set_xticklabels(filtered['district_short'], rotation=90)
ax.set_ylabel("Percentage")
ax.set_title("Districts: Average Literacy and Gender Gap")
ax.legend()
plt.tight_layout()
st.pyplot(fig)

# export histogram
png_bytes = fig_to_png_bytes(fig)
st.download_button("Download Histogram PNG", data=png_bytes, file_name="literacy_histogram.png", mime="image/png")

# ----------------------------
# Top N bar chart
# ----------------------------
st.markdown("### Plot Top N Districts")
try:
    N_plot = st.number_input("Number of districts to plot (Top N)", min_value=1, max_value=31, value=top_n)
    plot_df = df.sort_values('literacy_rate_avg', ascending=False).head(int(N_plot))
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.bar(plot_df['district_short'], plot_df['literacy_rate_avg'], color='skyblue')
    ax2.set_xticklabels(plot_df['district_short'], rotation=45, ha='right')
    ax2.set_ylabel("Avg Literacy (%)")
    ax2.set_title(f"Top {int(N_plot)} Districts by Literacy Rate")
    plt.tight_layout()
    st.pyplot(fig2)
    st.download_button("Download Top-N PNG", data=fig_to_png_bytes(fig2), file_name="top_n_literacy.png", mime="image/png")
except Exception as e:
    st.error(f"Plot error: {e}")

# ----------------------------
# Clustering section
# ----------------------------
st.markdown("### Clustering (KMeans)")
if st.button("Run KMeans clustering"):
    kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init="auto")
    # avoid NaNs
    km_df = df[['literacy_rate_avg']].fillna(df['literacy_rate_avg'].mean())
    df['cluster'] = kmeans.fit_predict(km_df)
    st.success("KMeans clustering complete.")
    st.dataframe(df[['districts', 'literacy_rate_avg', 'cluster']].sort_values('cluster').reset_index(drop=True))
    st.download_button("Download Clusters CSV", data=df_to_csv_bytes(df[['districts','literacy_rate_avg','cluster']]), file_name="clustering_results.csv", mime="text/csv")

# ----------------------------
# Anomaly detection
# ----------------------------
st.markdown("### Anomaly Detection (IsolationForest)")
if st.button("Run Anomaly Detection"):
    iso = IsolationForest(contamination=anomaly_contamination, random_state=42)
    iso_df = df[['literacy_rate_avg']].fillna(df['literacy_rate_avg'].mean())
    df['anomaly'] = iso.fit_predict(iso_df)
    anomalies = df[df['anomaly'] == -1][['districts', 'literacy_rate_avg']]
    if anomalies.empty:
        st.info("No anomalies detected at this contamination level.")
    else:
        st.warning("Anomalies detected:")
        st.dataframe(anomalies.reset_index(drop=True))
        st.download_button("Download Anomalies CSV", data=df_to_csv_bytes(anomalies), file_name="anomalies.csv", mime="text/csv")

# ----------------------------
# Forecasting
# ----------------------------
st.markdown("### Forecast literacy trend (simple linear regression)")
if st.button("Run Forecast"):
    X = np.arange(len(df)).reshape(-1, 1)
    y = df['literacy_rate_avg'].fillna(df['literacy_rate_avg'].mean()).values
    model = LinearRegression().fit(X, y)
    future_x = np.arange(len(df) + 5).reshape(-1, 1)
    preds = model.predict(future_x)
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    ax3.plot(np.arange(len(df)), y, marker='o', label='Actual')
    ax3.plot(future_x.flatten(), preds, linestyle='--', color='orange', label='Forecast')
    ax3.set_xlabel("District index (ordered)")
    ax3.set_ylabel("Avg Literacy (%)")
    ax3.set_title("Literacy Forecast (linear regression over district index)")
    ax3.legend()
    st.pyplot(fig3)
    st.download_button("Download Forecast PNG", data=fig_to_png_bytes(fig3), file_name="forecast.png", mime="image/png")

# ----------------------------
# Policy recommendations
# ----------------------------
st.markdown("### Policy Recommendations (simple rules)")
if st.button("Show Policy Recommendations"):
    high_gap = df[df['gender_gap'] > 10]
    low_lit = df[df['literacy_rate_avg'] < df['literacy_rate_avg'].mean()]
    st.write("Policy suggestions based on simple heuristics:")
    if not high_gap.empty:
        st.warning(f"- {len(high_gap)} districts have gender gap > 10%. Recommend female literacy & empowerment campaigns.")
        st.dataframe(high_gap[['districts','literacy_rate_avg','gender_gap']].reset_index(drop=True))
    if not low_lit.empty:
        st.warning(f"- {len(low_lit)} districts are below state average literacy. Recommend targeted education programs.")
        st.dataframe(low_lit[['districts','literacy_rate_avg','gender_gap']].reset_index(drop=True))
    if high_gap.empty and low_lit.empty:
        st.success("No specific policy actions suggested based on current heuristics.")

# ----------------------------
# Natural language (rule-based) queries
# ----------------------------
st.markdown("### Ask a question (natural-language) — rule-based")
q = st.text_input("Ask (examples: 'largest gap', 'top literacy', 'bottom literacy'):")
if q:
    ql = q.lower()
    if "gap" in ql and ("largest" in ql or "highest" in ql):
        df['gap'] = (df['literacy_rate_males'] - df['literacy_rate_females']).abs()
        st.write("Districts with largest gender gaps:")
        st.dataframe(df.nlargest(5, 'gap')[['districts','gap']].reset_index(drop=True))
    elif "top" in ql and "literacy" in ql:
        st.write("Top 5 districts by average literacy:")
        st.dataframe(df.nlargest(5, 'literacy_rate_avg')[['districts','literacy_rate_avg']].reset_index(drop=True))
    elif "bottom" in ql and "literacy" in ql:
        st.write("Bottom 5 districts by average literacy:")
        st.dataframe(df.nsmallest(5, 'literacy_rate_avg')[['districts','literacy_rate_avg']].reset_index(drop=True))
    else:
        st.info("Try queries such as: 'largest gap', 'top literacy', 'bottom literacy'.")

# ----------------------------
# Footer / provenance
# ----------------------------
st.markdown("---")
st.markdown("**Provenance:** Data loaded from `artifacts/full-run/transformed_literacy.csv` if present or `artifacts/full-run/literacy.csv` otherwise. All outputs and saved artifacts are stored in `artifacts/full-run/`.")
