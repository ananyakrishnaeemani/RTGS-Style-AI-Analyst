import pandas as pd
from pathlib import Path

class StandardizationAgent:
    def __init__(self, logger):
        self.logger = logger

    def run(self, infile: Path, outdir: Path):
        self.logger.info(f"Loading raw data from {infile}")
        df = pd.read_csv(infile)

        # --- Standardize column names ---
        df.columns = (
            df.columns.str.strip()      # remove extra spaces
                      .str.lower()      # lowercase
                      .str.replace(" ", "_")  # replace spaces with _
        )
        self.logger.info(f"Standardized columns: {list(df.columns)}")

        # --- Handle missing values ---
        missing_before = df.isna().sum().sum()
        df = df.dropna(how="all")  # drop empty rows
        df = df.fillna("NA")       # simple fill
        missing_after = df.isna().sum().sum()
        self.logger.info(f"Missing values before: {missing_before}, after: {missing_after}")

        # --- Remove duplicates ---
        dupes = df.duplicated().sum()
        if dupes > 0:
            self.logger.info(f"Found {dupes} duplicate rows → dropping")
            df = df.drop_duplicates()

        # Save standardized file
        outdir.mkdir(parents=True, exist_ok=True)
        outfile = outdir / "standardized.csv"
        df.to_csv(outfile, index=False)
        self.logger.info(f"Standardized dataset saved: {outfile}")
        self.logger.info(f"Rows: {len(df)}, Columns: {len(df.columns)}")

        return outfile
