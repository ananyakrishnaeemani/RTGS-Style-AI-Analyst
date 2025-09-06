Project Overview

This project prototypes a Real-Time Governance System (RTGS) for policymakers using Telangana open government data. It is a terminal-first agentic system that ingests raw CSV datasets, cleans and standardizes the data, performs transformations, and surfaces actionable insights via the CLI.

Dataset Used:

Literacy and Literacy Rate (Telangana Open Data Portal
)

Time Range: 2011–2022

Scope: All districts (state-level aggregation optional for future extensions)

Governance Relevance:

Understand literacy trends by district.

Identify gender disparities in literacy.

Support evidence-based policy interventions.
Agentic Architecture

The pipeline is modular with four main agents:

Stage	Agent	Responsibility
1️⃣ Ingest	IngestAgent	Load raw CSV (from URL or local) and save as raw.csv.
2️⃣ Standardize	StandardizationAgent	Standardize column names, types, and units.
3️⃣ Transform	TransformAgent	Compute derived fields: average literacy, gender gap.
4️⃣ Insights	InsightsAgent	Generate top/bottom districts, gender gap tables, and ASCII charts in CLI.

📁 Repo Structure
Telangana-Literacy-Dashboard/
│
├─ artifacts/
│   └─ full-run/
│       ├─ raw_literacy.csv
│       ├─ standardized.csv
│       ├─ transformed_literacy.csv
│       ├─ literacy_summary.png
│       ├─ clustering_results.csv
│       └─ anomalies.csv
│
├─ rtgs/
│   ├─ agents/
│   │   ├─ standardize.py
│   │   └─ insights.py
│   ├─ utils/
│   │   └─ logging_utils.py
│   └─ __init__.py
│
├─ datasets.md
├─ run_all.py
├─ README.md
└─ requirements.txt

README.md (Draft)
Telangana Literacy Dashboard

📺 Demo Video

1️⃣ Overview

The Telangana Literacy Dashboard is an agentic AI-driven pipeline for analyzing district-level literacy data. It performs:

Data ingestion from local CSVs

Standardization and cleaning

Transformations (computing average literacy and gender gaps)

Insights generation

Prescriptive analytics with AI agents (clustering, anomaly detection, forecasting)

Interactive CLI for exploration

2️⃣ Architecture

Agents:

Agent	Purpose
StandardizationAgent	Cleans and standardizes raw CSVs
InsightsAgent	Generates insights, top/bottom districts, histograms, and policy recommendations

Models & Libraries:

scikit-learn – KMeans, IsolationForest, LinearRegression

pandas, numpy – data manipulation

matplotlib – visualizations

texttable, termcolor – CLI tables and colored output

3️⃣ Installation
# Clone repo
git clone https://github.com/YOUR_USERNAME/Telangana-Literacy-Dashboard.git
cd Telangana-Literacy-Dashboard

# Create virtual environment
python -m venv venv
source venv/bin/activate     # Linux/Mac
venv\Scripts\activate        # Windows

# Install dependencies
pip install -r requirements.txt

4️⃣ Running the Project
python run_all.py


Follows a step-by-step CLI for:

Viewing top/bottom districts

Viewing gender gaps

Summary histograms

Clustering & anomaly detection

Forecasting literacy

Policy recommendations

All outputs (plots, tables, CSVs) are saved in artifacts/full-run/.

5️⃣ Expected Outputs

artifacts/full-run/transformed_literacy.csv – final cleaned dataset

artifacts/full-run/literacy_summary.png – literacy & gender gap histogram

artifacts/full-run/clustering_results.csv – cluster assignments

artifacts/full-run/anomalies.csv – low literacy anomalies

logs/ – detailed run logs

6️⃣ Dataset Manifest

See datasets.md
 for details about CSV sources, columns, and notes.

7️⃣ Config Samples
{
  "input_file": "artifacts/full-run/literacy.csv",
  "output_dir": "artifacts/full-run",
  "agents": ["StandardizationAgent", "InsightsAgent"]
}


Note: No secrets are required; this is a fully local pipeline.
