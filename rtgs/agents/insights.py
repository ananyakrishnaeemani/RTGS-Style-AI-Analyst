import pandas as pd
from pathlib import Path
from rich.console import Console
from rich.table import Table

class InsightsAgent:
    def __init__(self, logger):
        self.logger = logger
        self.console = Console()

    def run(self, transformed_file: Path):
        df = pd.read_csv(transformed_file)
        self.logger.info("=== Literacy Insights ===")
        self.logger.info(f"Total districts: {len(df)}")

        # Top & bottom 5 by avg literacy
        top5_lit = df.sort_values("literacy_rate_avg", ascending=False).head(5)
        bottom5_lit = df.sort_values("literacy_rate_avg").head(5)

        # Top & bottom 5 by gender gap
        top5_gap = df.sort_values("gender_gap", ascending=False).head(5)
        bottom5_gap = df.sort_values("gender_gap").head(5)

        # --- Print summaries ---
        self.console.print("\n[bold underline]Top 5 Districts: Average Literacy[/]\n")
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("District")
        table.add_column("Avg Literacy")
        table.add_column("Gender Gap")
        for _, row in top5_lit.iterrows():
            table.add_row(row["districts"], f"{row['literacy_rate_avg']:.2f}%", f"{row['gender_gap']:.2f}%")
        self.console.print(table)

        self.console.print("\n[bold underline]Bottom 5 Districts: Average Literacy[/]\n")
        table = Table(show_header=True, header_style="bold red")
        table.add_column("District")
        table.add_column("Avg Literacy")
        table.add_column("Gender Gap")
        for _, row in bottom5_lit.iterrows():
            table.add_row(row["districts"], f"{row['literacy_rate_avg']:.2f}%", f"{row['gender_gap']:.2f}%")
        self.console.print(table)

        self.console.print("\n[bold underline]Top 5 Districts: Gender Gap[/]\n")
        table = Table(show_header=True, header_style="bold yellow")
        table.add_column("District")
        table.add_column("Avg Literacy")
        table.add_column("Gender Gap")
        for _, row in top5_gap.iterrows():
            table.add_row(row["districts"], f"{row['literacy_rate_avg']:.2f}%", f"{row['gender_gap']:.2f}%")
        self.console.print(table)

        self.console.print("\n[bold underline]Bottom 5 Districts: Gender Gap[/]\n")
        table = Table(show_header=True, header_style="bold green")
        table.add_column("District")
        table.add_column("Avg Literacy")
        table.add_column("Gender Gap")
        for _, row in bottom5_gap.iterrows():
            table.add_row(row["districts"], f"{row['literacy_rate_avg']:.2f}%", f"{row['gender_gap']:.2f}%")
        self.console.print(table)

        # --- Optional: full ASCII bar charts (as before) ---
        max_bar_len = 40
        self.console.print("\n[bold underline]District Literacy Rates:[/]\n")
        for _, row in df.iterrows():
            bar_len = int((row['literacy_rate_avg']/100)*max_bar_len)
            bar = "█" * bar_len
            self.console.print(f"{row['districts'][:15]:15} | {bar} {row['literacy_rate_avg']:.2f}%")

        self.console.print("\n[bold underline]District Gender Gaps:[/]\n")
        max_gap = df['gender_gap'].max()
        for _, row in df.iterrows():
            bar_len = int((row['gender_gap']/max_gap)*max_bar_len)
            bar = "█" * bar_len
            self.console.print(f"{row['districts'][:15]:15} | {bar} {row['gender_gap']:.2f}%")
