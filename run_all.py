from pathlib import Path
from rtgs.utils.logging_utils import RunLogger
from rtgs.agents.standardize import StandardizationAgent
from rtgs.agents.transform import TransformAgent
from rtgs.agents.insights import InsightsAgent
import pandas as pd
import shutil
from texttable import Texttable
from termcolor import colored

# 🔹 New imports for AI features
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
import numpy as np
from transformers import pipeline

# -----------------------------
# Directories & dataset
# -----------------------------
ARTIFACTS_DIR = Path("artifacts/full-run")
LOCAL_CSV = ARTIFACTS_DIR / "literacy.csv"  # local CSV path
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
logger = RunLogger(ARTIFACTS_DIR)

# -----------------------------
# 1️⃣ Ingest (local CSV)
# -----------------------------
logger.info("Starting ingestion from local CSV...")
raw_file_path = ARTIFACTS_DIR / "raw.csv"
if LOCAL_CSV.exists():
    shutil.copy(LOCAL_CSV, raw_file_path)
    logger.info(f"Local CSV copied to {raw_file_path}")
else:
    raise FileNotFoundError(f"Local CSV not found at {LOCAL_CSV}")

# -----------------------------
# 2️⃣ Standardize
# -----------------------------
logger.info("Starting standardization...")
std_agent = StandardizationAgent(logger)
standardized_file = std_agent.run(infile=raw_file_path, outdir=ARTIFACTS_DIR)

# -----------------------------
# 3️⃣ Transform
# -----------------------------
logger.info("Starting transformation...")
df = pd.read_csv(standardized_file)
df['literacy_rate_avg'] = df[['literacy_rate_males', 'literacy_rate_females']].mean(axis=1)
df['gender_gap'] = df['literacy_rate_males'] - df['literacy_rate_females']
transformed_file = ARTIFACTS_DIR / "transformed.csv"
df.to_csv(transformed_file, index=False)
logger.info(f"Transformed data saved to {transformed_file}")

# -----------------------------
# 4️⃣ Insights
# -----------------------------
logger.info("Generating insights...")
insights_agent = InsightsAgent(logger)
insights_agent.run(transformed_file=transformed_file)

# -----------------------------
# 5️⃣ Helper: display table
# -----------------------------
def display_table(df_subset, title, value_col, scale=1, color_threshold=None):
    print(f"\n{title}")
    table = Texttable()
    table.add_row(["District", value_col, "Bar"])
    for _, row in df_subset.iterrows():
        value = row[value_col]
        bar = "█" * int(value * scale / 5)
        if color_threshold:
            bar_color = "red" if value >= color_threshold else "yellow"
            bar = colored(bar, bar_color)
        table.add_row([row['districts'], round(value, 2), bar])
    print(table.draw())

# -----------------------------
# 6️⃣ AI Analysis Functions
# -----------------------------
def run_clustering(df):
    kmeans = KMeans(n_clusters=3, random_state=42)
    df["cluster"] = kmeans.fit_predict(df[["literacy_rate_avg"]])
    print("\nClustering Results (3 groups of districts by literacy rate):")
    print(df[["districts", "literacy_rate_avg", "cluster"]].head(10))
    return df

def run_anomaly_detection(df):
    iso = IsolationForest(random_state=42)
    df["outlier"] = iso.fit_predict(df[["literacy_rate_avg"]])
    anomalies = df[df["outlier"] == -1]
    print("\n🚨 Districts flagged as anomalies (unusually low literacy):")
    print(anomalies[["districts", "literacy_rate_avg"]])
    return anomalies

def run_regression_forecast(df):
    X = np.arange(len(df)).reshape(-1, 1)
    y = df["literacy_rate_avg"].values
    model = LinearRegression().fit(X, y)
    next_idx = np.array([[len(df)]])
    forecast = model.predict(next_idx)[0]
    print(f"\n📈 Forecasted average literacy rate for next district index: {forecast:.2f}%")

def run_nl_query(df, query):
    query = query.lower()

    # normalize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Now your columns are:
    # 'districts', 'males', 'females', 'literacy_rate_males', 'literacy_rate_females'

    if "gap" in query and ("largest" in query or "highest" in query):
        df["gap"] = (df["literacy_rate_males"] - df["literacy_rate_females"]).abs()
        top_gap = df.nlargest(5, "gap")[["districts", "gap"]]
        print("\n🤖 Answer: Districts with the largest male-female literacy gap:")
        print(top_gap.to_string(index=False))
        return

    elif "top" in query and "literacy" in query:
        df["literacy_rate_avg"] = df[["literacy_rate_males", "literacy_rate_females"]].mean(axis=1)
        top = df.nlargest(5, "literacy_rate_avg")[["districts", "literacy_rate_avg"]]
        print("\n🤖 Answer: Top 5 districts by average literacy rate:")
        print(top.to_string(index=False))
        return

    elif "bottom" in query and "literacy" in query:
        df["literacy_rate_avg"] = df[["literacy_rate_males", "literacy_rate_females"]].mean(axis=1)
        bottom = df.nsmallest(5, "literacy_rate_avg")[["districts", "literacy_rate_avg"]]
        print("\n🤖 Answer: Bottom 5 districts by average literacy rate:")
        print(bottom.to_string(index=False))
        return

    else:
        print("\n🤖 Sorry, I couldn't understand your question. Try asking about:")
        print("- Largest male-female gap")
        print("- Top/Bottom districts by literacy rate")
        print("- Average literacy")


# -----------------------------
# 7️⃣ Terminal Histogram Summary
# -----------------------------
def display_summary_histogram(df):
    print("\n--- Summary Histogram: All Districts ---")
    
    print("\nAverage Literacy Rates:")
    for _, row in df.iterrows():
        bar = "█" * int(row['literacy_rate_avg'] / 2)
        print(f"{row['districts'][:15]:15} | {bar} {row['literacy_rate_avg']:.2f}%")
    
    print("\nGender Gap (male-female):")
    for _, row in df.iterrows():
        bar = "█" * int(abs(row['gender_gap']) / 2)
        bar_color = "red" if row['gender_gap'] > 10 else "yellow"
        print(f"{row['districts'][:15]:15} | {colored(bar, bar_color)} {row['gender_gap']:.2f}%")

# -----------------------------
# 8️⃣ Interactive CLI Loop
# -----------------------------
while True:
    print("\n--- Telangana Literacy Dashboard ---")
    print("1: Top N districts by average literacy")
    print("2: Bottom N districts by average literacy")
    print("3: Top N districts by gender gap")
    print("4: Filter by district name")
    print("5: Show summary histogram for all districts")
    print("6: Run AI clustering")
    print("7: Run anomaly detection")
    print("8: Forecast literacy trend")
    print("9: Ask a natural-language question")
    print("10: Exit")
    choice = input("Enter option: ").strip()

    if choice == "1":
        N = int(input("Enter N: "))
        display_table(df.sort_values(by='literacy_rate_avg', ascending=False).head(N),
                      f"Top {N} Districts by Average Literacy", 'literacy_rate_avg')
    elif choice == "2":
        N = int(input("Enter N: "))
        display_table(df.sort_values(by='literacy_rate_avg', ascending=True).head(N),
                      f"Bottom {N} Districts by Average Literacy", 'literacy_rate_avg')
    elif choice == "3":
        N = int(input("Enter N: "))
        display_table(df.sort_values(by='gender_gap', ascending=False).head(N),
                      f"Top {N} Districts by Gender Gap (male-female)", 'gender_gap', color_threshold=10)
    elif choice == "4":
        name = input("Enter district name keyword: ").strip()
        filtered = df[df['districts'].str.contains(name, case=False)]
        if filtered.empty:
            print(f"No districts match '{name}'")
        else:
            display_table(filtered, f"Districts matching '{name}'", 'literacy_rate_avg')
    elif choice == "5":
        display_summary_histogram(df)
    elif choice == "6":
        df = run_clustering(df)
    elif choice == "7":
        run_anomaly_detection(df)
    elif choice == "8":
        run_regression_forecast(df)
    elif choice == "9":
        query = input("Enter your question: ")
        run_nl_query(df, query)
    elif choice == "10":
        print("Exiting...")
        break
    else:
        print("Invalid choice. Try again.")
