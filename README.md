# RTGS-Style-AI-Analyst

## Project Overview
This project prototypes a Real-Time Governance System (RTGS) for policymakers using Telangana open government data. It is a terminal-first agentic system that ingests raw CSV datasets, cleans and standardizes the data, performs transformations, and surfaces actionable insights via the CLI.

## Dataset Used:

- Literacy and Literacy Rate (Telangana Open Data Portal)
- Time Range: 2011–2022
- Scope: All districts (state-level aggregation optional for future extensions)
Governance Relevance:
- Understand literacy trends by district.
- Identify gender disparities in literacy.
- Support evidence-based policy interventions.
  
## AI Features & Models

We integrated AI/ML models (scikit-learn) to extract deeper insights from literacy data:
Clustering – KMeans 
- Groups districts into literacy clusters (e.g., high, medium, low).
- Helps identify regions with similar literacy patterns.
Anomaly Detection – IsolationForest
- Flags unusual districts with very low or unexpected literacy rates.
- Useful for detecting outliers that require special attention.
Forecasting – Linear Regression
- Predicts literacy trends for upcoming districts (index-based).
- Demonstrates time-series–like forecasting on static data.
Predictive Modeling – Linear Regression
- Learns from population, male literacy, and female literacy.
- Predicts the overall average literacy rate.

## Architecture
```
Ingestion Agent → Loads dataset(s)
Cleaning Agent → Handles missing values, type conversions
Transformation Agent → Normalization, derived columns
Insights Agent → Generates descriptive & AI-powered insights
Visualization Agent → Saves charts and plots
Export Agent → Produces documentation + logs
```
📁 Repo Structure
```
RTGS-Style-AI-Analyst/
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
```

## Models & Libraries:
- scikit-learn – KMeans, IsolationForest, LinearRegression
- pandas, numpy – data manipulation
- matplotlib – visualizations
- texttable, termcolor – CLI tables and colored output

## Installation
### Clone repo
```
git clone https://github.com/ananyakrishnaeemani/RTGS-Style-AI-Analyst.git
cd RTGS-Style-AI-Analyst
```
### Create virtual environment
```
python -m venv venv
source venv/bin/activate     # Linux/Mac
venv\Scripts\activate        # Windows
```
### Install dependencies
```
pip install -r requirements.txt
```
### Running the Project
```
python run_all.py
```

Follows a step-by-step CLI for:
- Viewing top/bottom districts
- Viewing gender gaps
- Summary histograms
- Clustering & anomaly detection
- Forecasting literacy
- Policy recommendations
- All outputs (plots, tables, CSVs) are saved in artifacts/full-run/.

## Expected Outputs

-artifacts/full-run/transformed_literacy.csv – final cleaned dataset
- artifacts/full-run/literacy_summary.png – literacy & gender gap histogram
- artifacts/full-run/clustering_results.csv – cluster assignments
- artifacts/full-run/anomalies.csv – low literacy anomalies
- logs/ – detailed run logs

## Dataset Manifest
See datasets.md for details about CSV sources, columns, and notes.

## Config Samples
```
{
  "input_file": "artifacts/full-run/literacy.csv",
  "output_dir": "artifacts/full-run",
  "agents": ["StandardizationAgent", "InsightsAgent"]
}
```

Note: No secrets are required; this is a fully local pipeline.
