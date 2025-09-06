from pathlib import Path
from rtgs.utils.logging_utils import RunLogger
from rtgs.agents.standardize import StandardizationAgent
from rtgs.agents.insights import InsightsAgent
import pandas as pd
import shutil
from texttable import Texttable
from termcolor import colored
import matplotlib.pyplot as plt

# 🔹 AI & analysis imports
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
import numpy as np

# -----------------------------
# Directories & dataset
# -----------------------------
ARTIFACTS_DIR = Path("artifacts/full-run")
LOCAL_CSV = ARTIFACTS_DIR / "literacy.csv"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
logger = RunLogger(ARTIFACTS_DIR)

# -----------------------------
# 1️⃣ Ingest dataset
# -----------------------------
logger.info("Starting ingestion from literacy CSV...")

def copy_file(src_path, dest_path):
    if src_path.exists():
        shutil.copy(src_path, dest_path)
        logger.info(f"Copied {src_path.name} to {dest_path}")
    else:
        raise FileNotFoundError(f"{src_path} not found.")

copy_file(LOCAL_CSV, ARTIFACTS_DIR / "raw_literacy.csv")

# -----------------------------
# 2️⃣ Standardize
# -----------------------------
logger.info("Starting standardization...")
std_agent = StandardizationAgent(logger)
standardized_lit = std_agent.run(infile=ARTIFACTS_DIR / "raw_literacy.csv", outdir=ARTIFACTS_DIR)

# -----------------------------
# 3️⃣ Transform
# -----------------------------
logger.info("Starting transformation...")
df = pd.read_csv(standardized_lit)
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Compute additional columns
df['literacy_rate_avg'] = df[['literacy_rate_males','literacy_rate_females']].mean(axis=1)
df['gender_gap'] = df['literacy_rate_males'] - df['literacy_rate_females']

# Save transformed CSV
transformed_file = ARTIFACTS_DIR / "transformed_literacy.csv"
df.to_csv(transformed_file, index=False)
logger.info(f"Transformed dataset saved to {transformed_file}")

# -----------------------------
# 4️⃣ Insights
# -----------------------------
insights_agent = InsightsAgent(logger)
insights_agent.run(transformed_file=transformed_file)

# -----------------------------
# 5️⃣ Helper functions
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

def display_summary_histogram(df, save_png=False):
    print("\n--- Summary Histogram: All Districts ---")
    districts = df['districts']
    literacy = df['literacy_rate_avg']
    gap = df['gender_gap']

    # ASCII histogram
    for i, d in enumerate(districts):
        bar = "█" * int(literacy.iloc[i] / 2)
        print(f"{d[:15]:15} | {bar} {literacy.iloc[i]:.2f}%")

    # Plot and save PNG
    if save_png:
        fig, ax = plt.subplots(figsize=(10,6))
        ax.bar(districts, literacy, color='skyblue', label='Avg Literacy')
        ax.bar(districts, gap, color='salmon', alpha=0.5, label='Gender Gap')
        plt.xticks(rotation=90)
        plt.ylabel("Percentage")
        plt.title("Telangana District Literacy & Gender Gap")
        plt.legend()
        png_file = ARTIFACTS_DIR / "literacy_summary.png"
        plt.tight_layout()
        plt.savefig(png_file)
        plt.close()
        print(f"\n📊 Histogram saved as {png_file}")

# -----------------------------
# 6️⃣ AI & Prescriptive Functions
# -----------------------------
def run_clustering(df, save_csv=False):
    kmeans = KMeans(n_clusters=3, random_state=42)
    df["cluster"] = kmeans.fit_predict(df[["literacy_rate_avg"]])
    print("\nClustering Results (3 groups by literacy rate):")
    print(df[["districts", "literacy_rate_avg", "cluster"]].head(10))
    if save_csv:
        cluster_file = ARTIFACTS_DIR / "clustering_results.csv"
        df.to_csv(cluster_file, index=False)
        print(f"✅ Clustering results saved as {cluster_file}")
    return df

def run_anomaly_detection(df, save_csv=False):
    iso = IsolationForest(random_state=42)
    df["outlier"] = iso.fit_predict(df[["literacy_rate_avg"]])
    anomalies = df[df["outlier"] == -1]
    print("\n🚨 Districts flagged as anomalies (unusually low literacy):")
    print(anomalies[["districts", "literacy_rate_avg"]])
    if save_csv:
        anomaly_file = ARTIFACTS_DIR / "anomalies.csv"
        anomalies.to_csv(anomaly_file, index=False)
        print(f"✅ Anomaly results saved as {anomaly_file}")
    return anomalies

def run_regression_forecast(df):
    X = np.arange(len(df)).reshape(-1, 1)
    y = df["literacy_rate_avg"].values
    model = LinearRegression().fit(X, y)
    next_idx = np.array([[len(df)]])
    forecast = model.predict(next_idx)[0]
    print(f"\n📈 Forecasted average literacy rate for next district index: {forecast:.2f}%")
    if forecast < df['literacy_rate_avg'].mean():
        print("⚠️ Forecast below average. Consider targeted literacy programs.")

def run_policy_recommendations(df):
    high_gap = df[df['gender_gap'] > 10]
    low_lit = df[df['literacy_rate_avg'] < df['literacy_rate_avg'].mean()]
    print("\n📝 Policy Recommendations:")
    if not high_gap.empty:
        print(f"- {len(high_gap)} districts have gender gap > 10%. Recommend female literacy campaigns.")
    if not low_lit.empty:
        print(f"- {len(low_lit)} districts have below-average literacy. Recommend focused educational programs.")

def run_nl_query(df, query):
    query = query.lower()
    if "gap" in query and ("largest" in query or "highest" in query):
        df["gap"] = (df["literacy_rate_males"] - df["literacy_rate_females"]).abs()
        top_gap = df.nlargest(5, "gap")[["districts", "gap"]]
        print("\n🤖 Answer: Districts with the largest male-female literacy gap:")
        print(top_gap.to_string(index=False))
    elif "top" in query and "literacy" in query:
        top = df.nlargest(5, "literacy_rate_avg")[["districts", "literacy_rate_avg"]]
        print("\n🤖 Answer: Top 5 districts by average literacy rate:")
        print(top.to_string(index=False))
    elif "bottom" in query and "literacy" in query:
        bottom = df.nsmallest(5, "literacy_rate_avg")[["districts", "literacy_rate_avg"]]
        print("\n🤖 Answer: Bottom 5 districts by average literacy rate:")
        print(bottom.to_string(index=False))
    else:
        print("\n🤖 Sorry, I couldn't understand your question. Try asking about:")
        print("- Largest male-female gap")
        print("- Top/Bottom districts by literacy rate")

def filter_by_literacy(df):
    try:
        min_val = float(input("Enter minimum literacy rate (%): "))
        max_val = float(input("Enter maximum literacy rate (%): "))
        filtered = df[(df['literacy_rate_avg'] >= min_val) & (df['literacy_rate_avg'] <= max_val)]
        if filtered.empty:
            print(f"No districts found between {min_val}% and {max_val}% literacy.")
        else:
            display_table(filtered, f"Districts with literacy between {min_val}% and {max_val}%", 'literacy_rate_avg')
    except ValueError:
        print("Invalid input. Enter numeric values.")

def plot_top_districts(df):
    try:
        N = int(input("Enter number of top districts to plot: "))
        top_df = df.sort_values(by='literacy_rate_avg', ascending=False).head(N)
        plt.figure(figsize=(10,6))
        plt.bar(top_df['districts'], top_df['literacy_rate_avg'], color='skyblue')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel("Average Literacy Rate (%)")
        plt.title(f"Top {N} Districts by Literacy Rate")
        plt.tight_layout()
        plt.show()
    except ValueError:
        print("Invalid number.")

def display_enhanced_histogram(df):
    print("\n--- Enhanced ASCII Histogram ---")
    for i, row in df.iterrows():
        literacy_bar = "█" * int(row['literacy_rate_avg'] / 2)
        gap_bar = "▓" * int(abs(row['gender_gap']) / 2)
        gap_color = 'red' if abs(row['gender_gap']) > 10 else 'yellow'
        gap_bar = colored(gap_bar, gap_color)
        print(f"{row['districts'][:15]:15} | {literacy_bar} {row['literacy_rate_avg']:.2f}% | {gap_bar} Gap {row['gender_gap']:.2f}")


# -----------------------------
# 7️⃣ Interactive CLI Loop
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
    print("10: Show policy recommendations")
    print("11: Export histogram as PNG")
    print("12: Filter districts by literacy range")
    print("13: Plot top N districts by literacy")
    print("14: Show enhanced ASCII histogram")
    print("15: Exit")
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
        save_png = input("Save histogram as PNG? (y/n): ").strip().lower() == 'y'
        display_summary_histogram(df, save_png)
    elif choice == "6":
        save_csv = input("Save clustering results? (y/n): ").strip().lower() == 'y'
        df = run_clustering(df, save_csv)
    elif choice == "7":
        save_csv = input("Save anomaly detection results? (y/n): ").strip().lower() == 'y'
        run_anomaly_detection(df, save_csv)
    elif choice == "8":
        run_regression_forecast(df)
    elif choice == "9":
        query = input("Enter your question: ")
        run_nl_query(df, query)
    elif choice == "10":
        run_policy_recommendations(df)
    elif choice == "11":
        display_summary_histogram(df, save_png=True)
    elif choice == "12":
        filter_by_literacy(df)
    elif choice == "13":
        plot_top_districts(df)
    elif choice == "14":
        display_enhanced_histogram(df)
    elif choice == "15":
        print("Exiting...")
        break
    else:
        print("Invalid choice. Try again.")
