"""
fetch_readme_signals.py
Run this LOCALLY on your machine.

Fetches README files for all repos in repos_to_predict.csv and extracts:
  1. readme_length       — character count (maturity signal)
  2. readme_lines        — line count
  3. ethereum_keywords   — count of ethereum-specific terms
  4. protocol_keywords   — count of protocol-level terms (beacon, validator, eip...)
  5. has_docs_link       — 1 if repo links to external documentation
  6. has_changelog       — 1 if repo mentions changelog/releases
  7. code_blocks         — number of code examples (developer tool signal)
  8. has_install         — 1 if repo has install instructions
  9. has_badge           — 1 if repo has CI/coverage badges

Token is loaded automatically from .env file in the same folder.

Usage:
    python fetch_readme_signals.py

    The script looks for .env in the same directory as this script.
    Your .env file should contain:
        GITHUB_TOKEN=ghp_yourtoken

Output:
    readme_signals.csv  (saved in same folder as this script)

Requirements:
    pip install requests pandas python-dotenv
"""

import os
import re
import time
import base64
import requests
import pandas as pd
from pathlib import Path

# ── Load .env token ───────────────────────────────────────────────────────────
# Looks for .env in the same directory as this script
SCRIPT_DIR = Path(__file__).parent.resolve()
ENV_FILE   = SCRIPT_DIR / ".env"

GITHUB_TOKEN = ""

if ENV_FILE.exists():
    with open(ENV_FILE) as f:
        for line in f:
            line = line.strip()
            if line.startswith("GITHUB_TOKEN"):
                # handles: GITHUB_TOKEN=value  or  GITHUB_TOKEN = value
                GITHUB_TOKEN = line.split("=", 1)[-1].strip().strip('"').strip("'")
                break
    if GITHUB_TOKEN:
        print(f"✅ Token loaded from {ENV_FILE}")
    else:
        print(f"⚠️  .env found at {ENV_FILE} but GITHUB_TOKEN not set inside it")
else:
    print(f"⚠️  No .env file found at {ENV_FILE}")
    print(f"   Create one with:  GITHUB_TOKEN=ghp_yourtoken")

# Also check environment variable as fallback
if not GITHUB_TOKEN:
    GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
    if GITHUB_TOKEN:
        print(f"✅ Token loaded from environment variable")

# ── Config ────────────────────────────────────────────────────────────────────
OUTPUT_FILE       = SCRIPT_DIR / "readme_signals.csv"
REPOS_TO_PREDICT  = SCRIPT_DIR / "repos_to_predict.csv"
DELAY             = 0.8 if not GITHUB_TOKEN else 0.2   # seconds between requests


# ── Load repos from repos_to_predict.csv ─────────────────────────────────────
def load_repos_from_csv(csv_path: Path) -> list[str]:
    """
    Load repo list from repos_to_predict.csv.
    Handles full GitHub URLs like https://github.com/owner/repo
    and plain owner/repo format.
    Returns list of 'owner/repo' strings.
    """
    df = pd.read_csv(csv_path)

    # Find the repo column (could be 'repo', 'repository', 'github_url' etc.)
    repo_col = None
    for col in df.columns:
        if "repo" in col.lower() or "url" in col.lower():
            repo_col = col
            break

    if repo_col is None:
        raise ValueError(f"Could not find repo column in {csv_path}. Columns: {df.columns.tolist()}")

    repos = []
    for raw in df[repo_col].dropna():
        raw = str(raw).strip().rstrip("/")
        # Extract owner/repo from full URL
        if "github.com/" in raw:
            raw = raw.split("github.com/")[-1]
        # Remove trailing .git if present
        raw = raw.replace(".git", "")
        # Normalise to lowercase
        raw = raw.lower()
        if "/" in raw:
            repos.append(raw)

    print(f"   Loaded {len(repos)} repos from {csv_path.name}")
    return repos

# ── Keyword sets ──────────────────────────────────────────────────────────────

# Broad ethereum ecosystem terms
ETHEREUM_KEYWORDS = [
    "ethereum", "eth", "ether", "ethers",
    "smart contract", "dapp", "decentralized",
    "blockchain", "web3", "solidity", "vyper",
    "erc-20", "erc20", "erc-721", "nft",
    "defi", "wallet", "gas", "gwei",
    "transaction", "mainnet", "testnet",
    "metamask", "remix", "hardhat", "foundry",
    "openzeppelin", "truffle", "brownie",
]

# Protocol-level terms — high value for jury (EIPs, consensus, staking)
PROTOCOL_KEYWORDS = [
    "eip", "eip-", "ethereum improvement",
    "beacon chain", "beacon", "validator",
    "consensus", "execution layer", "execution client",
    "consensus client", "proof of stake", "pos",
    "staking", "attestation", "finality",
    "fork choice", "merge", "shapella", "deneb",
    "withdrawals", "slashing", "epoch", "slot",
    "p2p", "libp2p", "devp2p",
    "mempool", "mev", "flashbots",
    "rollup", "layer 2", "l2",
    "zk", "zero knowledge", "zkp", "snark", "stark",
    "circuit", "proving", "verifier",
    "account abstraction", "erc-4337",
]

# Documentation and maturity signals
DOCS_PATTERNS = [
    r"docs\.[a-z]",
    r"readthedocs",
    r"gitbook",
    r"notion\.so",
    r"documentation",
    r"\[docs\]",
    r"api reference",
    r"developer guide",
]

# ── Fetcher ───────────────────────────────────────────────────────────────────

def get_headers() -> dict:
    h = {"User-Agent": "DeepFunding-Research/1.0"}
    if GITHUB_TOKEN:
        h["Authorization"] = f"token {GITHUB_TOKEN}"
    return h


def fetch_readme(owner: str, repo: str) -> str | None:
    """
    Fetch README content via GitHub API.
    Tries multiple branch names and falls back to raw content.
    """
    headers = get_headers()

    # Try GitHub API first (handles any branch/filename automatically)
    api_url = f"https://api.github.com/repos/{owner}/{repo}/readme"
    try:
        r = requests.get(api_url, headers=headers, timeout=15)
        if r.status_code == 200:
            data = r.json()
            content = data.get("content", "")
            encoding = data.get("encoding", "base64")
            if encoding == "base64":
                return base64.b64decode(content).decode("utf-8", errors="replace")
            return content
        elif r.status_code == 404:
            pass  # repo or README not found, try raw
        elif r.status_code == 403:
            remaining = r.headers.get("X-RateLimit-Remaining", "?")
            reset     = r.headers.get("X-RateLimit-Reset", "?")
            print(f"    ⚠️  Rate limited! Remaining={remaining}, Reset={reset}")
            print(f"    Sleeping 60s...")
            time.sleep(60)
            return fetch_readme(owner, repo)  # retry
    except Exception as e:
        print(f"    API error: {e}")

    # Fallback: raw content on common branch/filename combos
    raw_attempts = [
        f"https://raw.githubusercontent.com/{owner}/{repo}/main/README.md",
        f"https://raw.githubusercontent.com/{owner}/{repo}/master/README.md",
        f"https://raw.githubusercontent.com/{owner}/{repo}/main/readme.md",
        f"https://raw.githubusercontent.com/{owner}/{repo}/master/README.rst",
    ]
    for url in raw_attempts:
        try:
            r = requests.get(url, headers=headers, timeout=15)
            if r.status_code == 200:
                return r.text
        except Exception:
            pass

    return None


# ── Signal extraction ─────────────────────────────────────────────────────────

def extract_signals(text: str) -> dict:
    """Extract all signals from README text."""
    lower = text.lower()

    # Basic length signals
    readme_length = len(text)
    readme_lines  = text.count("\n")

    # Ethereum keyword count (unique matches, case-insensitive)
    eth_count = sum(1 for kw in ETHEREUM_KEYWORDS if kw in lower)

    # Protocol keyword count
    proto_count = sum(1 for kw in PROTOCOL_KEYWORDS if kw in lower)

    # Code block count (``` fenced blocks)
    code_blocks = len(re.findall(r"```", text)) // 2

    # Has external docs link
    has_docs = int(any(re.search(p, lower) for p in DOCS_PATTERNS))

    # Has changelog / release notes mention
    has_changelog = int(any(kw in lower for kw in [
        "changelog", "release notes", "what's new", "history", "unreleased"
    ]))

    # Has install instructions (developer tool signal)
    has_install = int(any(kw in lower for kw in [
        "npm install", "pip install", "cargo add", "go get",
        "yarn add", "brew install", "apt install", "installation"
    ]))

    # Has badge (CI, coverage — maturity signal)
    has_badge = int("[![" in text or "![build" in lower or "![ci" in lower)

    return {
        "readme_length"     : readme_length,
        "readme_lines"      : readme_lines,
        "ethereum_keywords" : eth_count,
        "protocol_keywords" : proto_count,
        "code_blocks"       : code_blocks,
        "has_docs_link"     : has_docs,
        "has_changelog"     : has_changelog,
        "has_install"       : has_install,
        "has_badge"         : has_badge,
    }


def empty_signals() -> dict:
    return {
        "readme_length"     : 0,
        "readme_lines"      : 0,
        "ethereum_keywords" : 0,
        "protocol_keywords" : 0,
        "code_blocks"       : 0,
        "has_docs_link"     : 0,
        "has_changelog"     : 0,
        "has_install"       : 0,
        "has_badge"         : 0,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # ── Rate limit info ───────────────────────────────────────────────────────
    if GITHUB_TOKEN:
        print(f"\n🔑 Authenticated — 5000 requests/hour, ~0.2s delay per repo")
    else:
        print(f"\n⚠️  Unauthenticated — 60 requests/hour, ~0.8s delay per repo")
        print(f"   For 98 repos this is fine but slower (~80 seconds total)")
        print(f"   Get a free token at: https://github.com/settings/tokens")
        print(f"   (No special scopes needed for public repos)")

    # ── Load repos from CSV ───────────────────────────────────────────────────
    print(f"\n📂 Loading repos from: {REPOS_TO_PREDICT}")
    if not REPOS_TO_PREDICT.exists():
        print(f"   ERROR: {REPOS_TO_PREDICT} not found!")
        print(f"   Make sure repos_to_predict.csv is in: {SCRIPT_DIR}")
        return

    REPOS = load_repos_from_csv(REPOS_TO_PREDICT)
    if not REPOS:
        print("   ERROR: No repos loaded. Check CSV format.")
        return

    print(f"\nFetching READMEs for {len(REPOS)} repos...\n")
    print(f"{'repo':<50}  {'length':>8}  {'eth_kw':>6}  {'proto_kw':>8}  {'status'}")
    print("-" * 95)

    results = []
    failed  = []

    for i, repo in enumerate(REPOS, 1):
        owner, name = repo.split("/", 1)

        readme_text = fetch_readme(owner, name)

        if readme_text and len(readme_text) > 10:
            signals = extract_signals(readme_text)
            status  = "OK"
        else:
            signals = empty_signals()
            status  = "NOT FOUND"
            failed.append(repo)

        row = {"repo": repo, **signals}
        results.append(row)

        print(
            f"  [{i:2d}/{len(REPOS)}] {repo:<46}  "
            f"{signals['readme_length']:>8,}  "
            f"{signals['ethereum_keywords']:>6}  "
            f"{signals['protocol_keywords']:>8}  "
            f"{status}"
        )

        time.sleep(DELAY if not GITHUB_TOKEN else 0.2)

    # Save
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\n{'='*60}")
    print(f"✅  Saved → {OUTPUT_FILE}")
    print(f"   Total repos    : {len(results)}")
    print(f"   Successful     : {len(results) - len(failed)}")
    print(f"   Failed/missing : {len(failed)}")

    if failed:
        print(f"\n   Failed repos:")
        for r in failed:
            print(f"     - {r}")

    print(f"\n   Top 10 by protocol_keywords:")
    top = df.nlargest(10, "protocol_keywords")[["repo", "protocol_keywords", "ethereum_keywords", "readme_length"]]
    print(top.to_string(index=False))

    print(f"\n   Top 10 by ethereum_keywords:")
    top2 = df.nlargest(10, "ethereum_keywords")[["repo", "ethereum_keywords", "protocol_keywords", "readme_length"]]
    print(top2.to_string(index=False))


if __name__ == "__main__":
    main()
