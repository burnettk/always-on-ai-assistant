import typer
import os
import json
import glob
from modules.benchmarks import run_benchmarks

app = typer.Typer()

@app.command()
def run(
    llm: str = typer.Option(..., "--llm", "-l", help="LLM to use ('deepseek' or 'gemini')"),
):
    """
    Run benchmarks for the specified LLM.
    """
    run_benchmarks(llm)

