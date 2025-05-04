#main.py 
"""
MOXY – single‑command LLM evaluation app
run:  python main.py
"""

from __future__ import annotations

import json, os, shutil, subprocess, sys, time, uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import tiktoken, yaml
from mirascope.core import openai
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import importlib, types, shutil

# ───────────────────────── basic paths / setup ──────────────────
ROOT         = Path.cwd()
RESULTS_DIR  = ROOT / "results";  RESULTS_DIR.mkdir(exist_ok=True)
CONFIG_PATH  = ROOT / "config.yaml"
console      = Console()
enc          = tiktoken.get_encoding("cl100k_base")

# ─────────────────────────── API KEYS ────────────────────────────
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
OXEN_TOKEN = os.getenv("OXEN_API_KEY")          # auth for hub / server
OXEN_REMOTE = os.getenv("OXEN_REMOTE")          # e.g. https://hub.oxen.ai/ns/repo
DEFAULT_NAME  = "MOXY Bot"
DEFAULT_EMAIL = "moxy@example.com"

if not OPENAI_KEY:
    console.print("[red]Missing OPENAI_API_KEY[/red]"); sys.exit(1)
os.environ["OPENAI_API_KEY"] = OPENAI_KEY

# ──────────────────────── Oxen handling  ───────────────────────
def load_oxen() -> types.ModuleType | None:
    """Return oxenai (preferred) or full oxen client, else None."""
    for name in ("oxenai", "oxen"):            # order matters
        try:
            mod = importlib.import_module(name)
            # the real client has both .init and .Repo
            if hasattr(mod, "init") and hasattr(mod, "Repo"):
                return mod
        except ModuleNotFoundError:
            pass
    return None

_ox = load_oxen()                    # None => no python client
_have_ox_py = _ox is not None
_ox_cli = shutil.which("oxen")           # compiled CLI if on PATH
_have_ox_cli = bool(_ox_cli)
OXEN_ENABLED = bool(OXEN_TOKEN) and (_have_ox_py or _have_ox_cli)

# Safe wrapper to check if a remote repository exists
def _remote_exists(repo_obj, remote_name: str = "origin") -> bool:
    """Safely check if a remote exists in the repository."""
    try:
        # Try different methods to get remotes depending on API version
        if hasattr(repo_obj, 'remote_list'):
            remotes = repo_obj.remote_list()
            return any(r.name == remote_name for r in remotes)
        elif hasattr(repo_obj, 'remotes'):
            remotes = repo_obj.remotes()
            return any(r.name == remote_name for r in remotes)
        elif hasattr(repo_obj, 'get_remote'):
            try:
                repo_obj.get_remote(remote_name)
                return True
            except:
                return False
        return False
    except Exception:
        return False

# Create remote repository on Oxen.ai
def _create_remote_repo(repo_name: str) -> bool:
    """Create a repository on Oxen.ai with proper namespace."""
    try:
        # Get username for namespace
        username = "realityinspector"  # Use your actual Oxen.ai username here

        # Full repo name with namespace
        full_repo_name = f"{username}/{repo_name}"

        if _have_ox_py:
            try:
                # Try to import the remote_repo module
                try:
                    from oxen.remote_repo import create_repo
                    remote_repo = create_repo(full_repo_name)
                    console.print(f"[green]Created remote repository: {full_repo_name}[/green]")
                    return True
                except ImportError:
                    # Try alternative API if available
                    if hasattr(_ox, 'create_remote_repo'):
                        _ox.create_remote_repo(full_repo_name)
                        console.print(f"[green]Created remote repository: {full_repo_name}[/green]")
                        return True
                    # Try RemoteRepo class if available
                    elif hasattr(_ox, 'RemoteRepo'):
                        try:
                            remote_repo = _ox.RemoteRepo(full_repo_name)
                            remote_repo.create(empty=False, is_public=True)
                            console.print(f"[green]Created remote repository using RemoteRepo: {full_repo_name}[/green]")
                            return True
                        except Exception as re:
                            console.print(f"[yellow]RemoteRepo creation failed: {re}[/yellow]")
            except Exception as e:
                console.print(f"[yellow]Failed to create remote repository: {e}[/yellow]")

        # Fall back to CLI if Python API fails
        if _have_ox_cli:
            try:
                # First try to create the repo
                result = subprocess.run(
                    [_ox_cli, "remote", "create", full_repo_name],
                    check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                )

                # Check if creation succeeded or already exists
                if result.returncode == 0 or "already exists" in result.stderr:
                    console.print(f"[green]Repository {full_repo_name} ready on Oxen.ai[/green]")
                    return True
                else:
                    console.print(f"[yellow]CLI remote creation failed: {result.stderr}[/yellow]")
            except Exception as e:
                console.print(f"[yellow]CLI remote creation error: {e}[/yellow]")

        return False
    except Exception as e:
        console.print(f"[yellow]Remote repository creation error: {e}[/yellow]")
        return False

# ─────────────────────── default config  ────────────────────────
DEFAULT_CFG = {
    "capital_test": {
        "prompt": "What is the capital of {country}?",
        "inputs": [{"country": "France"}, {"country": "Japan"}, {"country": "Brazil"}],
    }
}
if not CONFIG_PATH.exists():
    CONFIG_PATH.write_text(yaml.safe_dump(DEFAULT_CFG))
CONFIG: Dict[str, Any] = yaml.safe_load(CONFIG_PATH.read_text())

# ───────────────────────── Mirascope call ───────────────────────
@openai.call("gpt-4o-mini")
def _ask(prompt: str) -> str: return prompt

_tok = lambda txt: len(enc.encode(txt))

# ────────────────────── Oxen repository functions ───────────────────────────
def _ensure_repo(branch: str) -> None:
    """Init repo, set user/auth/remote, create branch, ready for commit/push."""
    try:
        if _have_ox_py:                             # python client
            new = not (ROOT / ".oxen").exists()
            repo = _ox.init(str(ROOT)) if new else _ox.Repo(str(ROOT))

            if new:
                if hasattr(_ox, 'user') and hasattr(_ox.user, 'config_user'):
                    _ox.user.config_user(DEFAULT_NAME, DEFAULT_EMAIL)
                else:
                    # Fallback method for older oxen versions
                    try:
                        from oxen.user import config_user
                        config_user(DEFAULT_NAME, DEFAULT_EMAIL)
                    except ImportError:
                        console.print("[yellow]Unable to configure user, skipping[/yellow]")

                # Create an initial dummy file if none exists
                readme_file = ROOT / "README.md"
                if not readme_file.exists():
                    readme_file.write_text("# MOXY Evaluation\n\nAutomatic LLM evaluation tool.")

                try:
                    repo.add(str(readme_file))
                    repo.commit("initial commit")
                except Exception as e:
                    console.print(f"[yellow]Initial commit error: {e}[/yellow]")

            if OXEN_TOKEN:
                try:
                    if hasattr(_ox, 'auth') and hasattr(_ox.auth, 'config_auth'):
                        _ox.auth.config_auth(OXEN_TOKEN)
                    else:
                        # Fallback method
                        try:
                            from oxen.auth import config_auth
                            config_auth(OXEN_TOKEN)
                        except ImportError:
                            console.print("[yellow]Unable to configure auth, skipping[/yellow]")
                except Exception as e:
                    console.print(f"[yellow]Auth configuration error: {e}[/yellow]")

                # Create or configure remote connection
                if OXEN_REMOTE:
                    # First, check if a remote with the name "origin" exists
                    try:
                        remotes = []

                        # Try different methods to list remotes
                        if hasattr(repo, 'remote_list'):
                            remotes = repo.remote_list()
                        elif hasattr(repo, 'remotes'):
                            remotes = repo.remotes()

                        # Check if origin exists in the remotes
                        origin_exists = False
                        for remote in remotes:
                            if hasattr(remote, 'name') and remote.name == "origin":
                                origin_exists = True
                                break

                        # If origin doesn't exist, add it
                        if not origin_exists:
                            if hasattr(repo, 'remote_add'):
                                repo.remote_add("origin", OXEN_REMOTE)
                                console.print("[green]Added remote 'origin'[/green]")
                            elif hasattr(repo, 'set_remote'):
                                repo.set_remote("origin", OXEN_REMOTE)
                                console.print("[green]Set remote 'origin'[/green]")
                            else:
                                # Fallback to CLI if Python API methods aren't available
                                if _have_ox_cli:
                                    try:
                                        subprocess.run([_ox_cli, "remote", "add", "origin", OXEN_REMOTE],
                                                     check=True, stderr=subprocess.PIPE)
                                        console.print("[green]Added remote 'origin' using CLI[/green]")
                                    except Exception as cli_err:
                                        console.print(f"[yellow]CLI remote add failed: {str(cli_err)}[/yellow]")
                        else:
                            # Update the existing remote
                            if hasattr(repo, 'remote_set_url'):
                                repo.remote_set_url("origin", OXEN_REMOTE)
                            elif hasattr(repo, 'set_remote'):
                                repo.set_remote("origin", OXEN_REMOTE)
                            console.print("[green]Updated remote 'origin'[/green]")

                    except Exception as e:
                        console.print(f"[yellow]Remote configuration error: {str(e)}[/yellow]")

                        # Last resort: try command-line interface if available
                        if _have_ox_cli:
                            try:
                                # Remove any existing origin
                                subprocess.run([_ox_cli, "remote", "remove", "origin"], 
                                              check=False, stderr=subprocess.PIPE)
                                # Add origin with correct URL
                                subprocess.run([_ox_cli, "remote", "add", "origin", OXEN_REMOTE], 
                                              check=True)
                                console.print("[green]Added remote 'origin' using CLI[/green]")
                            except Exception as cli_err:
                                console.print(f"[red]Failed to configure remote with CLI: {str(cli_err)}[/red]")

            # Make sure we have at least one commit before branch operations
            try:
                try:
                    # Try to get HEAD commit
                    repo.rev_parse("HEAD")
                except:
                    # No commits yet, create a dummy file and commit
                    readme_file = ROOT / "README.md"
                    if not readme_file.exists():
                        readme_file.write_text("# MOXY Evaluation\n\nAutomatic LLM evaluation tool.")
                    repo.add(str(readme_file))
                    repo.commit("initial commit")
            except Exception as e:
                console.print(f"[yellow]Initial commit check error: {e}[/yellow]")

            # Branch handling
            try:
                branch_exists = False

                # Get current branch
                current_branch = None
                try:
                    current = repo.current_branch()
                    # Handle different return types
                    if hasattr(current, 'name'):
                        current_branch = current.name
                    else:
                        current_branch = str(current)
                except Exception:
                    current_branch = "main"  # Default assumption

                # Check if the branch exists
                try:
                    branches = repo.branches()

                    # Handle different branch return types
                    branch_names = []
                    for b in branches:
                        if hasattr(b, 'name'):
                            branch_names.append(b.name)
                        elif isinstance(b, str):
                            branch_names.append(b)
                        else:
                            branch_names.append(str(b))

                    branch_exists = branch in branch_names
                except Exception as e:
                    console.print(f"[yellow]Branch listing error: {e}[/yellow]")

                # Switch to branch or create it
                if branch_exists:
                    try:
                        repo.checkout(branch)
                        console.print(f"[dim]Switched to branch: {branch}[/dim]")
                    except Exception as e:
                        console.print(f"[yellow]Branch checkout error: {e}[/yellow]")
                else:
                    try:
                        # Try to create branch
                        if hasattr(repo, 'branch_create'):
                            repo.branch_create(branch)
                            repo.checkout(branch)
                        elif hasattr(repo, 'branch') and callable(repo.branch):
                            repo.branch(branch)
                            repo.checkout(branch)
                        else:
                            # Try checkout with create flag
                            repo.checkout(branch, create=True)
                        console.print(f"[dim]Created and switched to branch: {branch}[/dim]")
                    except Exception as e:
                        console.print(f"[yellow]Branch creation error: {e}[/yellow]")
                        # Last resort - use current branch
                        console.print(f"[dim]Using current branch: {current_branch}[/dim]")
            except Exception as e:
                console.print(f"[yellow]Branch handling error: {e}[/yellow]")

        elif _have_ox_cli:                          # CLI fallback
            new = not (ROOT / ".oxen").exists()
            if new:
                subprocess.run([_ox_cli, "init"], check=True)
                subprocess.run([_ox_cli, "config", "--name", DEFAULT_NAME], check=True)
                subprocess.run([_ox_cli, "config", "--email", DEFAULT_EMAIL], check=True)

                # Create a dummy file if none exists
                readme_file = ROOT / "README.md"
                if not readme_file.exists():
                    readme_file.write_text("# MOXY Evaluation\n\nAutomatic LLM evaluation tool.")

                subprocess.run([_ox_cli, "add", str(readme_file)], check=True)
                subprocess.run([_ox_cli, "commit", "-m", "initial commit"], check=True)
            else:
                # Check if we have at least one commit
                try:
                    subprocess.run([_ox_cli, "rev-parse", "HEAD"], check=True, 
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                except subprocess.CalledProcessError:
                    # No commits yet, make an initial commit
                    readme_file = ROOT / "README.md"
                    if not readme_file.exists():
                        readme_file.write_text("# MOXY Evaluation\n\nAutomatic LLM evaluation tool.")
                    subprocess.run([_ox_cli, "add", str(readme_file)], check=True)
                    subprocess.run([_ox_cli, "commit", "-m", "initial commit"], check=True)

            if OXEN_TOKEN:
                try:
                    subprocess.run([_ox_cli, "config", "--auth", "hub.oxen.ai", OXEN_TOKEN],
                                check=True)
                except Exception as e:
                    console.print(f"[yellow]CLI auth error: {e}[/yellow]")

            if OXEN_REMOTE:
                try:
                    # Check if remote exists
                    result = subprocess.run([_ox_cli, "remote", "list"], check=False,
                                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    if "origin" not in result.stdout:
                        # Add remote
                        subprocess.run([_ox_cli, "remote", "add", "origin", OXEN_REMOTE],
                                    check=True)
                    else:
                        # Update remote
                        subprocess.run([_ox_cli, "remote", "set-url", "origin", OXEN_REMOTE],
                                    check=True)
                except Exception as e:
                    console.print(f"[yellow]CLI remote config error: {e}[/yellow]")

            # Branch handling
            try:
                # Check if branch exists
                result = subprocess.run([_ox_cli, "branch", "--list"], check=False, 
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                branch_exists = branch in result.stdout

                # Current branch
                curr_branch_result = subprocess.run([_ox_cli, "branch", "--show-current"], check=False,
                                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                current_branch = curr_branch_result.stdout.strip()

                if branch_exists:
                    # Just checkout the branch
                    subprocess.run([_ox_cli, "checkout", branch], check=False)
                    console.print(f"[dim]Switched to branch: {branch}[/dim]")
                else:
                    # Create and checkout the branch
                    subprocess.run([_ox_cli, "checkout", "-b", branch], check=False)
                    console.print(f"[dim]Created and switched to branch: {branch}[/dim]")
            except Exception as e:
                console.print(f"[yellow]CLI branch operation error: {e}[/yellow]")
    except Exception as e:
        console.print(f"[yellow]Error setting up repository: {e}[/yellow]")
        # Continue execution even if repository setup fails

def _commit_paths(paths: list[Path], msg: str, branch: str) -> None:
    """Commit and push changes to the oxen repository."""
    try:
        if _have_ox_py:
            repo = _ox.Repo(str(ROOT))

            # Add all files first
            for p in paths: 
                if p.exists():  # Make sure the file exists
                    try:
                        repo.add(str(p))
                    except Exception as e:
                        console.print(f"[yellow]Error adding file {p}: {e}[/yellow]")

            # Check if we have any changes to commit
            try:
                status = repo.status()
                has_changes = False

                # Different ways to detect changes based on API version
                if hasattr(status, 'staged') and status.staged:
                    has_changes = True
                elif hasattr(status, 'get'):
                    status_data = status.get()
                    if isinstance(status_data, dict):
                        has_changes = bool(status_data.get('staged', []))
                    else:
                        # Just try to commit anyway
                        has_changes = True
                else:
                    # Try to commit anyway
                    has_changes = True

                if has_changes:
                    try:
                        repo.commit(msg)
                        console.print(f"[dim]Committed changes: {msg}[/dim]")
                    except Exception as e:
                        console.print(f"[yellow]Commit error: {e}[/yellow]")
                else:
                    console.print("[dim]No changes to commit[/dim]")
            except Exception as e:
                console.print(f"[yellow]Status check error: {e}[/yellow]")
                # Try to commit anyway
                try:
                    repo.commit(msg)
                except Exception:
                    pass

            # Push changes if remote is configured
            if OXEN_REMOTE:
                # First ensure remote is properly configured
                try:
                    # Check if origin exists and reconfigure if needed
                    remotes = []

                    # Get list of remotes
                    if hasattr(repo, 'remote_list'):
                        remotes = repo.remote_list()
                    elif hasattr(repo, 'remotes'):
                        remotes = repo.remotes()

                    # Check if origin exists
                    origin_exists = False
                    for remote in remotes:
                        if hasattr(remote, 'name') and remote.name == "origin":
                            origin_exists = True
                            break

                    # Configure if needed
                    if not origin_exists:
                        if hasattr(repo, 'remote_add'):
                            repo.remote_add("origin", OXEN_REMOTE)
                            console.print("[green]Added missing remote 'origin'[/green]")
                        elif hasattr(repo, 'set_remote'):
                            repo.set_remote("origin", OXEN_REMOTE)
                            console.print("[green]Set missing remote 'origin'[/green]")

                    # Now try to push
                    try:
                        # Get current branch name safely
                        current_branch = None
                        try:
                            current = repo.current_branch()
                            # Handle different return types
                            if hasattr(current, 'name'):
                                current_branch = current.name
                            elif isinstance(current, str):
                                current_branch = current
                            else:
                                # Convert to string as last resort
                                current_branch = str(current)
                        except Exception:
                            # Default to the branch parameter if we can't get current branch
                            current_branch = branch

                        # Try to push using the branch name as a string
                        if current_branch:
                            try:
                                if hasattr(repo, 'push') and callable(repo.push):
                                    repo.push("origin", current_branch)
                                    console.print(f"[green]Pushed to origin/{current_branch}[/green]")
                                else:
                                    console.print("[yellow]Repository has no push method[/yellow]")
                            except Exception as push_error:
                                console.print(f"[yellow]Detailed push error: {push_error}[/yellow]")
                                # Try the simple push method
                                try:
                                    repo.push("origin")
                                    console.print("[green]Pushed to origin (default branch)[/green]")
                                except Exception:
                                    # If all else fails, try the CLI
                                    raise Exception("Python API push failed, trying CLI")
                        else:
                            # No branch name available, try simple push
                            repo.push("origin")
                            console.print("[green]Pushed to origin (default branch)[/green]")
                    except Exception as e:
                        console.print(f"[yellow]Push error: {e}[/yellow]")

                        # Try CLI fallback for pushing
                        if _have_ox_cli:
                            try:
                                # Try explicit branch push first
                                try:
                                    subprocess.run([_ox_cli, "push", "origin", branch], 
                                                check=False, stderr=subprocess.PIPE)
                                    console.print(f"[green]Successfully pushed to {branch} using CLI[/green]")
                                except Exception:
                                    # Try default push
                                    subprocess.run([_ox_cli, "push", "origin"], 
                                                check=False, stderr=subprocess.PIPE)
                                    console.print("[green]Successfully pushed using CLI[/green]")
                            except Exception as cli_err:
                                console.print(f"[yellow]CLI push failed: {str(cli_err)}[/yellow]")
                except Exception as e:
                    console.print(f"[yellow]Remote configuration error during push: {e}[/yellow]")

        elif _have_ox_cli:
            # Add all files
            for p in paths:
                if p.exists():  # Make sure the file exists
                    try:
                        subprocess.run([_ox_cli, "add", str(p)], check=True)
                    except Exception as e:
                        console.print(f"[yellow]Error adding file {p}: {e}[/yellow]")

            # Check if we have changes to commit
            status_result = subprocess.run([_ox_cli, "status"], check=False,
                                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            has_changes = "Changes to be committed" in status_result.stdout

            if has_changes:
                # Commit
                try:
                    subprocess.run([_ox_cli, "commit", "-m", msg], check=True)
                    console.print(f"[dim]Committed changes: {msg}[/dim]")
                except Exception as e:
                    console.print(f"[yellow]Commit error: {e}[/yellow]")
            else:
                console.print("[dim]No changes to commit[/dim]")

            # Push if remote is configured
            if OXEN_REMOTE:
                try:
                    # Get current branch
                    curr_branch_result = subprocess.run([_ox_cli, "branch", "--show-current"], check=False,
                                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    current_branch = curr_branch_result.stdout.strip()

                    # Push to current branch
                    subprocess.run([_ox_cli, "push", "origin", current_branch], check=False)
                except Exception as e:
                    console.print(f"[yellow]Push error: {e}[/yellow]")

    except Exception as e:
        console.print(f"[yellow]Error committing files: {e}[/yellow]")

# ───────────────────────── main eval  ───────────────────────────
def run_first_eval() -> None:
    global OXEN_REMOTE  # Global declaration

    eval_id, spec = next(iter(CONFIG.items()))
    tmpl, inputs = spec["prompt"], spec.get("inputs", [])
    console.print(f"[bold]Eval:[/bold] {eval_id}   [italic]{tmpl}[/italic]")

    results, total_tokens, t0 = [], 0, time.time()
    for inp in inputs:
        prompt = tmpl.format(**inp)
        out = _ask(prompt).content.strip()
        tk_in, tk_out = _tok(prompt), _tok(out)
        results.append({
            "input": inp, "output": out,
            "input_tokens": tk_in, "output_tokens": tk_out,
            "total_tokens": tk_in + tk_out
        })
        total_tokens += tk_in + tk_out

    payload = {
        "eval_id": eval_id, "model": "gpt-4o-mini",
        "ts": datetime.utcnow().isoformat(timespec="seconds"),
        "elapsed_s": round(time.time() - t0, 3),
        "token_usage": total_tokens,
        "prompt_template": tmpl, "results": results,
    }

    # pretty console print
    console.print(Panel(f"[bold blue]{eval_id}[/]  tokens={total_tokens}  "
                        f"time={payload['elapsed_s']}s"))

    # write logs
    out_dir = RESULTS_DIR / eval_id; out_dir.mkdir(exist_ok=True)
    stem = f"{datetime.utcnow():%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex}"
    json_fp = out_dir / f"{stem}.json"
    md_fp   = out_dir / f"{stem}.md"

    json_fp.write_text(json.dumps(payload, indent=2))
    md_fp.write_text(_mk_markdown(payload))

    console.print(f"[dim]logs → {json_fp.name} / {md_fp.name}[/dim]")

    # version‑control
    if OXEN_ENABLED:
        try:
            # If OXEN_REMOTE isn't set, create a default one using the eval_id
            if not OXEN_REMOTE and OXEN_TOKEN:
                # Update the global variable - fix URL format
                OXEN_REMOTE = f"https://hub.oxen.ai/realityinspector/{eval_id}"
                console.print(f"[dim]Using default remote: {OXEN_REMOTE}[/dim]")

                # Try to create the remote repository
                if _create_remote_repo(eval_id):
                    console.print(f"[green]Created remote repository for {eval_id}[/green]")

            # Fix common URL issues
            if OXEN_REMOTE and "www.hub.oxen.ai" in OXEN_REMOTE:
                # Remove 'www.' from URL as it causes DNS issues
                OXEN_REMOTE = OXEN_REMOTE.replace("www.hub.oxen.ai", "hub.oxen.ai")
                console.print(f"[dim]Fixed remote URL format: {OXEN_REMOTE}[/dim]")

            # Initialize the local repository
            _ensure_repo(eval_id)

            # Force remote configuration if using CLI
            if _have_ox_cli and OXEN_REMOTE:
                try:
                    # Remove any existing origin
                    subprocess.run(
                        [_ox_cli, "remote", "remove", "origin"],
                        check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                    )
                    # Make sure URL doesn't have www.
                    clean_url = OXEN_REMOTE.replace("www.hub.oxen.ai", "hub.oxen.ai")
                    # Add origin with correct URL
                    subprocess.run(
                        [_ox_cli, "remote", "add", "origin", clean_url],
                        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                    )
                    console.print(f"[green]Configured remote 'origin' with CLI: {clean_url}[/green]")
                except Exception:
                    pass

            # Commit and push the result files
            _commit_paths([json_fp, md_fp], f"eval {eval_id}", eval_id)

            # Final try - direct CLI push
            if _have_ox_cli and OXEN_REMOTE:
                try:
                    # Clean URL
                    clean_url = OXEN_REMOTE.replace("www.hub.oxen.ai", "hub.oxen.ai")
                    # Explicit push command with proper URL
                    result = subprocess.run(
                        [_ox_cli, "push", "origin", eval_id],
                        check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                    )
                    if "error" in result.stderr.lower() or "failed" in result.stderr.lower():
                        console.print(f"[yellow]CLI push to {eval_id} failed: {result.stderr}[/yellow]")
                        # Try main branch as fallback
                        subprocess.run(
                            [_ox_cli, "push", "origin", "main"],
                            check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                        )
                except Exception as e:
                    console.print(f"[yellow]Final CLI push attempt failed: {e}[/yellow]")

            console.print("[green]✓ Oxen commit/push ok[/green]")

            # Generate and print web URL for the repository
            if OXEN_REMOTE:
                # Extract namespace and repo name from remote URL
                namespace = "realityinspector"  # Default namespace
                repo_name = eval_id  # Default to eval_id

                # Try to parse from OXEN_REMOTE
                if "/" in OXEN_REMOTE:
                    parts = OXEN_REMOTE.replace("https://", "").replace("http://", "").split("/")
                    if len(parts) >= 2:
                        # Last part might be the repo name
                        potential_repo = parts[-1].replace(".git", "")
                        if potential_repo and potential_repo != "":
                            repo_name = potential_repo

                        # Check if we have namespace info
                        if len(parts) >= 3:
                            potential_namespace = parts[-2] 
                            if potential_namespace and potential_namespace != "hub.oxen.ai":
                                namespace = potential_namespace

                # Construct the web URL in the correct format
                results_url = f"https://www.oxen.ai/{namespace}/{repo_name}/dir/{eval_id}/results/{eval_id}"
                tree_url = f"https://www.oxen.ai/{namespace}/{repo_name}/tree/{eval_id}"

                # Print both URLs
                console.print(f"[bold blue]View results:[/bold blue] [link={results_url}]{results_url}[/link]")
                console.print(f"[bold blue]View repository tree:[/bold blue] [link={tree_url}]{tree_url}[/link]")

        except Exception as e:
            console.print(f"[yellow]Oxen error: {e}[/yellow]")

def _mk_markdown(p: Dict[str, Any]) -> str:
    out = [f"# {p['eval_id']}\n",
           f"*time*: {p['ts']}\n",
           f"*model*: {p['model']}\n",
           f"*tokens*: {p['token_usage']}\n",
           f"*elapsed*: {p['elapsed_s']} s\n\n## Results\n"]
    for i,r in enumerate(p["results"],1):
        out += [f"### {i}\n",
                f"`input`: `{json.dumps(r['input'])}`\n\n",
                f"> {r['output']}\n\n",
                (f"*tokens*: {r['input_tokens']}+{r['output_tokens']}"
                 f"={r['total_tokens']}\n")]
    return "\n".join(out)

# ───────────────────────── entrypoint ────────────────────────────
if __name__ == "__main__":
    run_first_eval()