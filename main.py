"""MOXY – single‑command LLM evaluation app.

> `python main.py` runs everything:
>  • loads / autogenerates `config.yaml`
>  • pulls `OPENAI_API_KEY` & optional `OXEN_API_KEY` from env (Replit secrets)
>  • executes the first eval block
>  • logs all results to console while still maintaining file logs for Oxen
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import tiktoken
import yaml
from mirascope.core import openai
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table

console = Console()
enc = tiktoken.get_encoding("cl100k_base")

ROOT = Path.cwd()
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
CONFIG_PATH = ROOT / "config.yaml"
OXEN_DIR = ROOT / ".oxen"

# ───────────────────────── Secrets / API keys ────────────────────
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
OXEN_TOKEN = os.getenv("OXEN_API_KEY")  # optional, used by oxen CLI if configured
OXEN_AVAILABLE = os.getenv("OXEN_AVAILABLE", "").lower() == "true"

if not OPENAI_KEY:
    console.print("[red]OPENAI_API_KEY missing – set it in Replit secrets.[/]")
    sys.exit(1)

os.environ["OPENAI_API_KEY"] = OPENAI_KEY
if OXEN_TOKEN:
    os.environ["OXEN_API_KEY"] = OXEN_TOKEN

# ───────────────────── default sample config ----------------------
DEFAULT_CFG = {
    "capital_test": {
        "prompt": "What is the capital of {country}?",
        "inputs": [
            {"country": "France"}, 
            {"country": "Japan"}, 
            {"country": "Brazil"}
        ],
    }
}

if not CONFIG_PATH.exists():
    CONFIG_PATH.write_text(yaml.safe_dump(DEFAULT_CFG))
    console.print("[cyan]Generated starter config.yaml[/]")

CONFIG: Dict[str, Any] = yaml.safe_load(CONFIG_PATH.read_text())

@openai.call("gpt-4o-mini")
def _ask(prompt: str) -> str:
    """LLM call using Mirascope."""
    return prompt

# ─────────────────────────── utilities ───────────────────────────

def _tok(text: str) -> int:
    return len(enc.encode(text))

def _oxen_add_commit(file_path: Path, msg: str, branch: str) -> None:
    """Add and commit a file to Oxen."""
    subprocess.run(["oxen", "add", str(file_path)], check=True)
    subprocess.run(["oxen", "commit", "-m", msg, "-b", branch], check=True)

def _write_markdown(payload: Dict[str, Any], json_path: Path) -> Path:
    """Write evaluation results as a markdown file."""
    md_path = json_path.with_suffix('.md')
    
    lines = [
        f"# Evaluation: {payload['eval_id']}",
        f"\nTimestamp: {payload['ts']}",
        f"\nModel: {payload['model']}",
        f"\nTotal tokens: {payload['token_usage']}",
        f"\nElapsed time: {payload['elapsed_s']} seconds",
        "\n## Results\n"
    ]
    
    for i, r in enumerate(payload['results'], 1):
        lines.extend([
            f"\n### Input {i}",
            "```json",
            json.dumps(r['input'], indent=2),
            "```",
            "\n**Response:**",
            f"\n{r['output']}\n",
            f"\nTokens: {r['total_tokens']} ({r['input_tokens']} input + {r['output_tokens']} output)"
        ])
    
    md_path.write_text('\n'.join(lines))
    return md_path

def _display_results(payload: Dict[str, Any]) -> None:
    """Display evaluation results in the console using rich formatting."""
    console.print(Panel(f"[bold blue]Eval `{payload['eval_id']}` — {payload['ts']}[/bold blue]"))

    # Create a summary table
    summary = Table(show_header=True, header_style="bold magenta")
    summary.add_column("Model")
    summary.add_column("Tokens")
    summary.add_column("Time (s)")
    summary.add_row(
        payload["model"],
        str(payload["token_usage"]),
        str(payload["elapsed_s"])
    )
    console.print(summary)

    console.print("\n[bold green]Results:[/bold green]")

    # Display each result
    for i, r in enumerate(payload["results"], 1):
        console.print(f"\n[bold cyan]Result #{i}[/bold cyan]")

        # Input display
        console.print("[yellow]Input:[/yellow]")
        console.print(json.dumps(r["input"], indent=2))

        # Output display
        console.print("[yellow]Output:[/yellow]")
        console.print(Panel(r["output"], border_style="green"))

        # Token usage for this input-output pair
        prompt = payload["prompt_template"].format(**r["input"])
        input_tokens = _tok(prompt)
        output_tokens = _tok(r["output"])
        console.print(f"[dim]Tokens: {input_tokens} (input) + {output_tokens} (output) = {input_tokens + output_tokens}[/dim]")

# ─────────────────────────── main run ────────────────────────────

def run_first_eval():
    console.print("[bold]Starting MOXY evaluation...[/bold]")

    eval_id = next(iter(CONFIG))
    spec = CONFIG[eval_id]
    tmpl: str = spec["prompt"]
    inputs = spec.get("inputs", [])

    console.print(f"[cyan]Running eval: [bold]{eval_id}[/bold][/cyan]")
    console.print(f"[cyan]Prompt template: [italic]{tmpl}[/italic][/cyan]")
    console.print(f"[cyan]Number of inputs: {len(inputs)}[/cyan]")

    results = []
    tokens = 0
    t0 = time.time()

    for i, inp in enumerate(inputs, 1):
        prompt = tmpl.format(**inp)
        console.print(f"\n[bold yellow]Processing input #{i}:[/bold yellow] {json.dumps(inp)}")
        console.print(f"[dim]Formatted prompt: {prompt}[/dim]")

        # Call the LLM
        console.print("[dim]Calling LLM...[/dim]")
        try:
            out = _ask(prompt).content.strip()
            console.print(f"[green]LLM response received ✓[/green]")
        except Exception as e:
            console.print(f"[red]Error calling LLM: {e}[/red]")
            out = f"ERROR: {str(e)}"

        # Calculate tokens
        input_tokens = _tok(prompt)
        output_tokens = _tok(out)
        pair_tokens = input_tokens + output_tokens
        tokens += pair_tokens

        console.print(f"[dim]Tokens for this pair: {input_tokens} (input) + {output_tokens} (output) = {pair_tokens}[/dim]")

        # Store result
        results.append({
            "input": inp,
            "output": out,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": pair_tokens
        })

    elapsed = round(time.time() - t0, 3)

    payload = {
        "eval_id": eval_id,
        "model": "gpt-4o-mini",
        "ts": datetime.utcnow().isoformat(timespec="seconds"),
        "elapsed_s": elapsed,
        "token_usage": tokens,
        "prompt_template": tmpl,
        "results": results,
    }

    # Console output
    console.print("\n[bold green]Evaluation complete![/bold green]")
    _display_results(payload)

    console.print(f"\n[bold]Summary:[/bold]")
    console.print(f"Total time: {elapsed} seconds")
    console.print(f"Total tokens: {tokens}")
    console.print(f"Average tokens per query: {tokens / len(inputs) if inputs else 0:.2f}")

    # File output for Oxen version control
    out_dir = RESULTS_DIR / eval_id
    out_dir.mkdir(exist_ok=True)
    fname = f"{datetime.utcnow():%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex}.json"
    json_fp = out_dir / fname

    # Write JSON results file
    json_fp.write_text(json.dumps(payload, indent=2))
    console.print(f"[dim]JSON log written to: {json_fp}[/dim]")

    # Write Markdown results file
    md_fp = _write_markdown(payload, json_fp)
    console.print(f"[dim]Markdown log written to: {md_fp}[/dim]")

    # Commit to Oxen if available
    if OXEN_AVAILABLE:
        console.print("[dim]Committing to Oxen...[/dim]")
        try:
            _oxen_add_commit(json_fp, f"run {eval_id}", eval_id)
            _oxen_add_commit(md_fp, f"log {eval_id}", eval_id)
            console.print("[green]✓ Committed to Oxen successfully[/green]")
        except Exception as e:
            console.print(f"[yellow]Warning: Failed to commit to Oxen: {e}[/yellow]")

# ───────────────────────── entrypoint ────────────────────────────
if __name__ == "__main__":
    run_first_eval()