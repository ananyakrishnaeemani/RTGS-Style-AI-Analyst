from pathlib import Path
from rtgs.utils.logging_utils import RunLogger
from rtgs.agents.standardize import StandardizationAgent

outdir = Path("artifacts/standardize-only")
logger = RunLogger(outdir)
agent = StandardizationAgent(logger)

# Take the output from ingestion stage
infile = Path("artifacts/ingest-only/raw.csv")

std_file = agent.run(infile=infile, outdir=outdir)

print(f"\n✅ Standardization done. File: {std_file}")
