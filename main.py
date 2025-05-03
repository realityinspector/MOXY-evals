"""MOXY Server – microscope × oxen × yaml
Feature‑max edition. YAML‑defined LLM evals, Oxen DVCS (optional), async, stats, logs.

Run   : uvicorn moxy_server:app --host 0.0.0.0 --port 8000
Deps  : pip install fastapi uvicorn pyyaml "mirascope[openai]" rich tiktoken
ENV   : OPENAI_API_KEY · MOXY_HOME · OXEN_REMOTE (optional)
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import tiktoken
import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from mirascope.core import openai
from pydantic import BaseModel
from rich.console import Console
from rich.table import Table

console = Console()
enc = tiktoken.get_encoding("cl100k_base")

# ───────────────────────────── Paths ──────────────────────────────
ROOT = Path(os.getenv("MOXY_HOME", Path.cwd()))
OXEN_DIR = ROOT / ".oxen"
CONFIG_PATH = ROOT / "config.yaml"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ────────────────────────── Oxen helpers ──────────────────────────

OXEN_AVAILABLE = shutil.which("oxen") is not None

def _run(cmd: List[str]):
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _ensure_repo():
    if not OXEN_AVAILABLE:
        console.log("[yellow]oxen CLI not found – version control disabled[/]")
        return
    if not OXEN_DIR.exists():
        _run(["oxen", "init", "--directory", str(ROOT)])
        remote = os.getenv("OXEN_REMOTE")
        if remote:
            _run(["oxen", "config", "--set-remote", remote])
    _run(["oxen", "config", "user", "--email", "moxy@local", "--name", "MOXY"])
    if CONFIG_PATH.exists():
        _oxen_commit(CONFIG_PATH, "sync config")


def _oxen_checkout(branch: str):
    if not OXEN_AVAILABLE:
        return
    try:
        _run(["oxen", "checkout", branch])
    except subprocess.CalledProcessError:
        _run(["oxen", "checkout", "-b", branch])


def _oxen_commit(path: Path, msg: str, branch: str | None = None):
    if not OXEN_AVAILABLE:
        return
    if branch:
        _oxen_checkout(branch)
    try:
        _run(["oxen", "add", str(path)])
        _run(["oxen", "commit", "-m", msg])
        _run(["oxen", "push"])
    except Exception:
        pass


def _oxen_log(branch: str, n: int = 20) -> str:
    if not OXEN_AVAILABLE:
        return "[oxen not installed]"
    try:
        return subprocess.check_output(["oxen", "log", branch, f"-n{n}"], text=True)
    except Exception:
        return "[log unavailable]"


_ensure_repo()

# ────────────────────────── LLM wrappers ─────────────────────────

@openai.call("gpt-4o-mini")
def _llm_sync(prompt: str) -> str:
    return prompt


@openai.call("gpt-4o-mini", stream=True)
def _llm_stream(prompt: str) -> str:
    return prompt


async def _llm_async(prompt: str, model: str = "gpt-4o-mini") -> str:
    call = openai.chat(model=model)
    resp = await call(prompt)
    return resp.content.strip()


# ───────────────────────────── Models ─────────────────────────────

class EvalRequest(BaseModel):
    eval_id: str
    dry_run: bool = False
    model: str | None = None
    inputs: List[Dict[str, Any]] | None = None


class BatchRequest(BaseModel):
    eval_ids: List[str]


class CompareRequest(BaseModel):
    eval_id: str
    base_file: str
    head_file: str


CONFIG: Dict[str, Any] = yaml.safe_load(CONFIG_PATH.read_text()) if CONFIG_PATH.exists() else {}

# ───────────────────────────── Utils ──────────────────────────────

def _count_tokens(text: str) -> int:
    return len(enc.encode(text))


# ───────────────────────────── FastAPI ────────────────────────────

app = FastAPI(title="MOXY Server", version="0.3.2")

# ───────────────────────────── Common helpers ─────────────────────

def _prepare_inputs(req: EvalRequest):
    if req.eval_id not in CONFIG:
        raise HTTPException(404, "eval_id not found")
    spec = CONFIG[req.eval_id]
    tmpl: str = spec["prompt"]
    inputs = req.inputs if req.inputs is not None else spec.get("inputs", [])
    return tmpl, inputs


def _save_result(eval_id: str, payload: Dict[str, Any]) -> Path:
    branch = eval_id
    save_dir = RESULTS_DIR / branch
    save_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{datetime.utcnow():%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex}.json"
    fpath = save_dir / fname
    fpath.write_text(json.dumps(payload, indent=2))
    _oxen_commit(fpath, f"{eval_id} results", branch)
    return fpath


# ───────────────────────────── Endpoints ─────────────────────────

@app.post("/run")
def run(req: EvalRequest):
    tmpl, inputs = _prepare_inputs(req)
    model = req.model or "gpt-4o-mini"
    start = time.time()
    results = []
    token_usage = 0

    for inp in inputs:
        prompt = tmpl.format(**inp)
        output = "[dry-run]" if req.dry_run else _llm_sync(prompt).content.strip()
        results.append({"input": inp, "output": output})
        token_usage += _count_tokens(prompt) + _count_tokens(output)

    payload = {
        "eval_id": req.eval_id,
        "model": model,
        "ts": datetime.utcnow().isoformat(timespec="seconds"),
        "elapsed_s": round(time.time() - start, 3),
        "token_usage": token_usage,
        "results": results,
    }
    _save_result(req.eval_id, payload)
    return payload


@app.post("/run_async")
async def run_async(req: EvalRequest):
    tmpl, inputs = _prepare_inputs(req)
    model = req.model or "gpt-4o-mini"

    async def worker(inp):
        prompt = tmpl.format(**inp)
        if req.dry_run:
            return inp, "[dry-run]"
        out = await _llm_async(prompt, model)
        return inp, out

    tasks = [worker(i) for i in inputs]
    start = time.time()
    results = []
    token_usage = 0
    for inp, out in await asyncio.gather(*tasks):
        results.append({"input": inp, "output": out})
        token_usage += _count_tokens(tmpl.format(**inp)) + _count_tokens(out)

    payload = {
        "eval_id": req.eval_id,
        "model": model,
        "ts": datetime.utcnow().isoformat(timespec="seconds"),
        "elapsed_s": round(time.time() - start, 3),
        "token_usage": token_usage,
        "results": results,
    }
    _save_result(req.eval_id, payload)
    return payload


@app.post("/stream")
async def stream(req: EvalRequest):
    tmpl, inputs = _prepare_inputs(req)

    async def gen():
        for inp in inputs:
            prompt = tmpl.format(**inp)
            if req.dry_run:
                yield json.dumps({"input": inp, "chunk": "[dry-run]"}) + "\n"
            else:
                for chunk, _ in _llm_stream(prompt):
                    yield json.dumps({"input": inp, "chunk": chunk.content}) + "\n"
                yield "\n"  # delimiter
    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/batch")
async def batch(req: BatchRequest):
    return {eid: run(EvalRequest(eval_id=eid)) for eid in req.eval_ids}  # type: ignore


@app.get("/metrics/{eval_id}")
def metrics(eval_id: str):
    run_dir = RESULTS_DIR / eval_id
    if not run_dir.exists():
        raise HTTPException(404, "no runs yet")
    tokens = elapsed = runs = 0
    for fp in run_dir.glob("*.json"):
        data = json.loads(fp.read_text())
        tokens += data.get("token_usage", 0)
        elapsed += data.get("elapsed_s", 0)
        runs += 1
    return {
        "runs": runs,
        "tokens": tokens,
        "avg_tokens": tokens // max(runs, 1),
        "total_seconds": elapsed,
    }


@app.get("/log/{eval_id}")
def log(eval_id: str, n: int = 20):
    return {"log": _oxen_log(eval_id, n)}


@app.get("/runs/{eval_id}")
def list_runs(eval_id: str):
    run_dir = RESULTS_DIR / eval_id
    if not run_dir.exists():
        raise HTTPException(404, "no runs yet")
    files = sorted(p.name for p in run_dir.glob("*.json"))
    return {"files": files}


@app.get("/runs/{eval_id}/{fname}")
def get_run(eval_id: str, fname: str):
    fp = RESULTS_DIR / eval_id / fname
    if not fp.exists():
        raise HTTPException(404, "file not found")
    return json.loads(fp.read_text())


@app.post("/compare")
def compare(req: CompareRequest):
    base = RESULTS_DIR / req.eval_id / req.base_file
    head = RESULTS_DIR / req.eval_id / req.head_file
    if not base.exists() or not head.exists():
        raise HTTPException(404, "file(s) missing")
    if not OXEN_AVAILABLE:
        return {"diff": "[oxen not installed]"}
    try:
        diff = subprocess.check_output(["oxen", "diff", str(base), str(head)], text=True)
    except Exception:
        diff = "[diff unavailable]"
    return {"diff": diff}


@app.get("/config")
def get_config():
    return CONFIG


@app.put("/config")
async def update_config(request: Request):
    yaml_text = await request.body()
    CONFIG_PATH.write_bytes(yaml_text)
    global CONFIG
    CONFIG = yaml.safe_load(yaml_text)
    _oxen_commit(CONFIG_PATH, "update config")
    return {"updated": True}


@app.get("/health")
def health():
    return {"ok": True}


# ───────────────────────────── Banner ─────────────────────────────

table = Table(show_header=False, box=None)
table.add_row("MOXY", "ready @ http://localhost:8000")
console.print(table)
