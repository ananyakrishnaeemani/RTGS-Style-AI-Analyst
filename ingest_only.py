from pathlib import Path
from rtgs.utils.logging_utils import RunLogger
from rtgs.agents.ingest import IngestionAgent

outdir = Path("artifacts/ingest-only")
logger = RunLogger(outdir)
agent = IngestionAgent(logger)

raw_file = agent.run(outdir=outdir, local_csv="literacy.csv")

print(f"\n✅ Saved raw dataset to: {raw_file}")
