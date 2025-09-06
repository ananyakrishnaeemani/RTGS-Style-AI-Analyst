from pathlib import Path
import datetime

class RunLogger:
    def __init__(self, outdir: Path):
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.outdir / "run.log"
        self._log("Logger initialized")

    def _log(self, message: str):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {message}"
        print(line)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def info(self, msg: str):
        self._log(msg)
