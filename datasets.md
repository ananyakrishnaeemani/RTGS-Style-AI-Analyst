# Dataset Manifest

This project uses literacy data for Telangana districts to generate insights, visualizations, and policy recommendations.
```
This project uses Telangana literacy data to compute insights and visualizations.

| Dataset Name     | Source / Link             | Time Range / Year | Key Fields                                 | Notes                       |
|------------------|---------------------------|-------------------|--------------------------------------------|-----------------------------|
| literacy.csv     | [Local CSV or Link]       | Latest available  | districts, males, females,                 | Base dataset for dashboard; |
|                  |                           |                   | literacy_rate_males, literacy_rate_females | used for all analysis steps |
```


## Telangana Literacy Dataset

**Source:** Local CSV file: `artifacts/full-run/literacy.csv`  
**Coverage:** 31 districts of Telangana  
**Key columns:**
- `districts` – Name of the district
- `males` – Number of literate males
- `females` – Number of literate females
- `literacy_rate_males` – Literacy rate (%) of males
- `literacy_rate_females` – Literacy rate (%) of females

**Derived columns (from pipeline):**
- `literacy_rate_avg` – Average literacy rate across genders
- `gender_gap` – Difference between male and female literacy rates

**Notes:**
- No missing values in this dataset
- Preprocessed for consistency: column names standardized, spaces removed, lowercased
