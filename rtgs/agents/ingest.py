from pathlib import Path
import requests
import shutil
import pandas as pd

class IngestionAgent:
    def __init__(self, logger):
        self.logger = logger

    def run(self, outdir: Path, dataset_url=None, local_csv=None):
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        raw_path = outdir / "raw.csv"

        if local_csv:
            self.logger.info(f"Copying local file {local_csv}")
            shutil.copy(local_csv, raw_path)
        elif dataset_url:
            self.logger.info(f"Downloading from {dataset_url}")
            r = requests.get(dataset_url, timeout=60)
            r.raise_for_status()
            with open(raw_path, "wb") as f:
                f.write(r.content)
        else:
            raise ValueError("Must provide either dataset_url or local_csv")

        # Quick check with pandas
        df = pd.read_csv(raw_path)
        self.logger.info(f"Ingested {len(df)} rows, {len(df.columns)} columns")
        return raw_path
