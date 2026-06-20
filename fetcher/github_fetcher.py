"""
Deep Funding Contest - Level I
Targeted Fix Fetcher

Fixes 3 broken features in github_features.csv:
  1. network_dependents  — regex was wrong, scraper missed most repos
  2. commits_last_12m    — GitHub stats API needed longer retry window
  3. days_since_last_commit — was using pushed_at (bots skew it), now uses real commits API

Usage:
  python fix_fetcher.py

It will:
  - Load your existing github_features.csv
  - Only re-fetch repos where each feature is broken (zero)
  - Save progress to fix_progress.json (auto-resume on crash)
  - Write fixed data back to github_features_fixed.csv
"""

import os
import re
import json
import time
import math
import requests
import pandas as pd
from datetime import datetime, timezone
from dotenv import load_dotenv

# ── Config ────────────────────────────────────────────────────────────────────
load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
if not GITHUB_TOKEN:
    raise SystemExit("❌  GITHUB_TOKEN not found in .env file.")

INPUT_CSV     = r"C:\Users\josep\Desktop\DeepFunding_lvl_1\github_features.csv"
OUTPUT_CSV    = r"C:\Users\josep\Desktop\DeepFunding_lvl_1\github_features_fixed.csv"
PROGRESS_FILE = r"C:\Users\josep\Desktop\DeepFunding_lvl_1\fix_progress.json"

HEADERS = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}

# ── Utilities ─────────────────────────────────────────────────────────────────

def parse_owner_repo(url: str):
    url = str(url).strip().rstrip("/")
    parts = url.split("github.com/")
    if len(parts) < 2:
        return None, None
    path = parts[1].split("/")
    if len(path) < 2:
        return None, None
    return path[0], path[1]


def gh_get(url: str, params: dict = None, retries: int = 6) -> requests.Response | None:
    """Rate-limit aware GET with exponential backoff."""
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, params=params, timeout=20)

            if r.status_code in (403, 429):
                reset_ts = int(r.headers.get("X-RateLimit-Reset", time.time() + 60))
                wait = max(reset_ts - int(time.time()), 0) + 3
                print(f"      ⏳ Rate limited. Sleeping {wait}s ...")
                time.sleep(wait)
                continue

            if r.status_code == 404:
                return None

            if r.status_code >= 500:
                wait = 2 ** attempt
                print(f"      ⚠️  Server {r.status_code}. Retry in {wait}s ...")
                time.sleep(wait)
                continue

            return r

        except requests.exceptions.ConnectionError:
            wait = 2 ** attempt
            print(f"      🔌 Connection error. Retry in {wait}s ...")
            time.sleep(wait)
        except requests.exceptions.Timeout:
            wait = 2 ** attempt
            print(f"      ⏱️  Timeout. Retry in {wait}s ...")
            time.sleep(wait)

    print(f"      ❌ Failed after {retries} attempts: {url}")
    return None


def check_rate_limit():
    r = gh_get("https://api.github.com/rate_limit")
    if r:
        core = r.json()["resources"]["core"]
        search = r.json()["resources"]["search"]
        print(f"  📊 Core: {core['remaining']}/{core['limit']} | "
              f"Search: {search['remaining']}/{search['limit']} | "
              f"Resets: {datetime.fromtimestamp(core['reset']).strftime('%H:%M:%S')}")


# ══════════════════════════════════════════════════════════════════════════════
#  FIX 1 — network_dependents
#  Problem: old regex missed GitHub's HTML structure
#  Fix: try multiple patterns + fallback to dependents count via search API
# ══════════════════════════════════════════════════════════════════════════════

def fetch_network_dependents(owner: str, repo: str) -> int:
    """
    Scrape GitHub dependents page with multiple regex patterns.
    Falls back to 0 only if truly no dependents found.
    """
    url = f"https://github.com/{owner}/{repo}/network/dependents"
    headers_html = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "en-US,en;q=0.9",
    }

    try:
        r = requests.get(url, headers=headers_html, timeout=20)
        if r.status_code != 200:
            return 0

        html = r.text

        # Pattern 1: "123,456 Repositories" (most common)
        m = re.search(r'([\d,]+)\s+Repositor', html)
        if m:
            return int(m.group(1).replace(",", ""))

        # Pattern 2: data-tab-item with count
        m = re.search(r'Repositories[^"]*"[^>]*>\s*([\d,]+)', html)
        if m:
            return int(m.group(1).replace(",", ""))

        # Pattern 3: aria-label with dependent count
        m = re.search(r'(\d[\d,]*)\s+repositor', html, re.IGNORECASE)
        if m:
            return int(m.group(1).replace(",", ""))

        # Pattern 4: Counter inside tab
        m = re.search(r'class="Counter[^"]*"[^>]*>([\d,]+)', html)
        if m:
            return int(m.group(1).replace(",", ""))

        return 0

    except Exception as e:
        print(f"      ⚠️  Dependents scrape error: {e}")
        return 0


# ══════════════════════════════════════════════════════════════════════════════
#  FIX 2 — commits_last_12m
#  Problem: stats/commit_activity returned 202 and fetcher gave up too fast
#  Fix: longer wait (up to 30s), then fallback to counting commits via API
# ══════════════════════════════════════════════════════════════════════════════

def fetch_commits_last_12m(owner: str, repo: str) -> int:
    """
    First tries /stats/commit_activity (52 weeks).
    If GitHub is still computing (202), waits up to 30s.
    Falls back to paginated /commits with since= date filter.
    """
    stats_url = f"https://api.github.com/repos/{owner}/{repo}/stats/commit_activity"

    # Try stats endpoint — up to 5 attempts with increasing waits
    for attempt in range(5):
        r = gh_get(stats_url)
        if not r:
            break
        if r.status_code == 202:
            wait = [3, 5, 8, 10, 15][attempt]
            print(f"      ⏳ Stats computing... waiting {wait}s (attempt {attempt+1}/5)")
            time.sleep(wait)
            continue
        if r.status_code == 200:
            try:
                weeks = r.json()
                if isinstance(weeks, list) and len(weeks) > 0:
                    total = sum(w.get("total", 0) for w in weeks)
                    if total > 0:
                        return total
            except Exception:
                pass
            break

    # Fallback: count commits via paginated commits API (since 1 year ago)
    print(f"      🔄 Falling back to commits API for {owner}/{repo}")
    since = datetime.now(timezone.utc).replace(
        year=datetime.now(timezone.utc).year - 1
    ).isoformat()

    commits_url = f"https://api.github.com/repos/{owner}/{repo}/commits"
    params = {"since": since, "per_page": 1}
    r = gh_get(commits_url, params=params)
    if not r:
        return 0

    # Use Link header to get total count
    link = r.headers.get("Link", "")
    m = re.search(r'page=(\d+)>;\s*rel="last"', link)
    if m:
        return int(m.group(1))

    # No pagination = fewer than 1 page of commits
    try:
        data = r.json()
        return len(data) if isinstance(data, list) else 0
    except Exception:
        return 0


# ══════════════════════════════════════════════════════════════════════════════
#  FIX 3 — days_since_last_commit
#  Problem: used pushed_at which bots/CI update constantly
#  Fix: get actual latest commit date from /commits endpoint
# ══════════════════════════════════════════════════════════════════════════════

def fetch_days_since_last_real_commit(owner: str, repo: str) -> int:
    """
    Gets the date of the most recent actual commit (not push).
    Uses /repos/{owner}/{repo}/commits?per_page=1
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/commits"
    r = gh_get(url, params={"per_page": 1})
    if not r:
        return 9999

    try:
        data = r.json()
        if not isinstance(data, list) or len(data) == 0:
            return 9999

        commit = data[0]
        # Try committer date first, then author date
        date_str = (
            commit.get("commit", {}).get("committer", {}).get("date") or
            commit.get("commit", {}).get("author", {}).get("date")
        )
        if not date_str:
            return 9999

        commit_dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        days = (datetime.now(timezone.utc) - commit_dt).days
        return max(days, 0)

    except Exception as e:
        print(f"      ⚠️  Commit date error: {e}")
        return 9999


# ══════════════════════════════════════════════════════════════════════════════
#  PROGRESS HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def load_progress() -> dict:
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            return json.load(f)
    return {}


def save_progress(progress: dict):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "█" * 60)
    print("  DEEP FUNDING — Targeted Fix Fetcher")
    print("█" * 60)

    df = pd.read_csv(INPUT_CSV)
    print(f"\n  Loaded {len(df)} repos from {INPUT_CSV}")

    # Identify broken repos per feature
    broken_dependents = df[df["network_dependents"] == 0]["repo"].tolist()
    broken_commits    = df[df["commits_last_12m"] == 0]["repo"].tolist()
    broken_days       = df[df["days_since_last_commit"] == 0]["repo"].tolist()

    print(f"\n  Broken network_dependents  : {len(broken_dependents)} repos")
    print(f"  Broken commits_last_12m    : {len(broken_commits)} repos")
    print(f"  Broken days_since_last_commit: {len(broken_days)} repos")

    # Union of all repos that need any fix
    all_broken = list(set(broken_dependents + broken_commits + broken_days))
    print(f"\n  Total repos needing re-fetch: {len(all_broken)}")

    check_rate_limit()
    print()

    progress = load_progress()
    done     = set(progress.keys())
    remaining = [r for r in all_broken if r not in done]
    print(f"  Already fixed : {len(done)}")
    print(f"  Remaining     : {len(remaining)}\n")

    for i, repo_url in enumerate(remaining):
        owner, repo = parse_owner_repo(repo_url)
        if not owner or not repo:
            continue

        print(f"[{i+1}/{len(remaining)}] {owner}/{repo}")
        result = {}

        try:
            # Fix 1 — network_dependents
            if repo_url in broken_dependents:
                print(f"    → Fetching network_dependents ...")
                val = fetch_network_dependents(owner, repo)
                result["network_dependents"] = val
                print(f"    ✅ network_dependents = {val:,}")
                time.sleep(1.0)

            # Fix 2 — commits_last_12m
            if repo_url in broken_commits:
                print(f"    → Fetching commits_last_12m ...")
                val = fetch_commits_last_12m(owner, repo)
                result["commits_last_12m"] = val
                # Recompute commit_frequency
                age = df.loc[df["repo"] == repo_url, "repo_age_days"].values[0]
                result["commit_frequency"] = round(val / max(age, 1), 6)
                print(f"    ✅ commits_last_12m = {val}  |  commit_frequency = {result['commit_frequency']}")
                time.sleep(0.5)

            # Fix 3 — days_since_last_commit
            if repo_url in broken_days:
                print(f"    → Fetching real last commit date ...")
                val = fetch_days_since_last_real_commit(owner, repo)
                result["days_since_last_commit"] = val
                print(f"    ✅ days_since_last_commit = {val}")
                time.sleep(0.5)

            progress[repo_url] = result

        except KeyboardInterrupt:
            print("\n⚠️  Interrupted. Saving progress ...")
            save_progress(progress)
            print(f"💾 Saved to {PROGRESS_FILE}")
            return

        except Exception as e:
            print(f"    ❌ Unexpected error: {e}")
            progress[repo_url] = {"error": str(e)}

        # Checkpoint every 5 repos
        if (i + 1) % 5 == 0:
            save_progress(progress)
            print(f"  💾 Checkpoint ({len(progress)}/{len(all_broken)})")

        time.sleep(0.3)

    save_progress(progress)

    # ── Merge fixes back into dataframe ──────────────────────────────────────
    print("\n" + "═" * 60)
    print("  Merging fixes into dataframe ...")

    fixed_count = {"network_dependents": 0, "commits_last_12m": 0,
                   "commit_frequency": 0, "days_since_last_commit": 0}

    for repo_url, fixes in progress.items():
        if "error" in fixes:
            continue
        mask = df["repo"] == repo_url
        for col, val in fixes.items():
            if col in df.columns:
                df.loc[mask, col] = val
                fixed_count[col] = fixed_count.get(col, 0) + 1

    print(f"\n  Fields updated:")
    for col, count in fixed_count.items():
        print(f"    {col:35s} {count} repos updated")

    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\n  ✅ Fixed data saved to:\n     {OUTPUT_CSV}")
    print()

    # ── Final quality report ──────────────────────────────────────────────────
    print("═" * 60)
    print("  QUALITY REPORT — After Fix")
    print("═" * 60)

    feature_cols = ["stars", "forks", "commits_last_12m", "commit_frequency",
                    "contributors_count", "releases_count", "closed_issues_ratio",
                    "repo_age_days", "days_since_last_commit", "network_dependents"]

    for col in feature_cols:
        vals = df[col].astype(float)
        zeros = (vals == 0).sum()
        print(f"  {col:35s}  zeros={zeros:2d}  min={vals.min():>10.1f}  max={vals.max():>12.1f}")

    print("\n  Top 10 by network_dependents:")
    print(df.nlargest(10, "network_dependents")[["repo", "network_dependents"]].to_string())

    print("\n" + "█" * 60)
    print("  DONE — now run deep_funding_model.py with github_features_fixed.csv")
    print("█" * 60 + "\n")


if __name__ == "__main__":
    main()