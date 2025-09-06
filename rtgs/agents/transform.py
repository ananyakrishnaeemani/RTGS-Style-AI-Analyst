import pandas as pd
from pathlib import Path

class TransformAgent:
    def __init__(self, logger):
        self.logger = logger

    def run(self, infile: Path, outdir: Path):
        self.logger.info(f"Loading standardized data from {infile}")
        df = pd.read_csv(infile)

        # Convert literacy rates to numeric
        df["literacy_rate_males"] = pd.to_numeric(df["literacy_rate_males"], errors="coerce")
        df["literacy_rate_females"] = pd.to_numeric(df["literacy_rate_females"], errors="coerce")

        # Derived columns
        df["literacy_rate_avg"] = (df["literacy_rate_males"] + df["literacy_rate_females"]) / 2
        df["gender_gap"] = df["literacy_rate_males"] - df["literacy_rate_females"]

        # Save transformed data
        outdir.mkdir(parents=True, exist_ok=True)
        df_out = outdir / "transformed.csv"
        df.to_csv(df_out, index=False)
        self.logger.info(f"Transformed dataset saved: {df_out}")

        return df_out
