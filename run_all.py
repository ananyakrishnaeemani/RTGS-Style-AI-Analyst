from pathlib import Path
from rtgs.utils.logging_utils import RunLogger
from rtgs.agents.standardize import StandardizationAgent
from rtgs.agents.transform import TransformAgent
from rtgs.agents.insights import InsightsAgent
import pandas as pd
import shutil
from texttable import Texttable
from termcolor import colored

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
# 7️⃣ Terminal Histogram Summary
# -----------------------------
def display_summary_histogram(df):
    print("\n--- Summary Histogram: All Districts ---")
    
    # Literacy rates
    print("\nAverage Literacy Rates:")
    for _, row in df.iterrows():
        bar = "█" * int(row['literacy_rate_avg'] / 2)  # scale factor for visibility
        print(f"{row['districts'][:15]:15} | {bar} {row['literacy_rate_avg']:.2f}%")
    
    # Gender gap
    print("\nGender Gap (male-female):")
    for _, row in df.iterrows():
        bar = "█" * int(abs(row['gender_gap']) / 2)
        bar_color = "red" if row['gender_gap'] > 10 else "yellow"
        print(f"{row['districts'][:15]:15} | {colored(bar, bar_color)} {row['gender_gap']:.2f}%")


# -----------------------------
# 6️⃣ Interactive CLI Loop
# -----------------------------
while True:
    print("\n--- Telangana Literacy Dashboard ---")
    print("1: Top N districts by average literacy")
    print("2: Bottom N districts by average literacy")
    print("3: Top N districts by gender gap")
    print("4: Filter by district name")
    print("5: Show summary histogram for all districts")
    print("6: Exit")
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
        break
    elif choice == "6":
        print("Exiting...")
        break
    else:
        print("Invalid choice. Try again.")
