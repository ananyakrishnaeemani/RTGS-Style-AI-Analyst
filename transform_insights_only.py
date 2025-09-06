from pathlib import Path
from rtgs.utils.logging_utils import RunLogger
from rtgs.agents.transform import TransformAgent
from rtgs.agents.insights import InsightsAgent

outdir = Path("artifacts/transform-insights")
logger = RunLogger(outdir)

transform_agent = TransformAgent(logger)
infile = Path("artifacts/standardize-only/standardized.csv")
transformed_file = transform_agent.run(infile=infile, outdir=outdir)

insights_agent = InsightsAgent(logger)
insights_agent.run(transformed_file=transformed_file)
