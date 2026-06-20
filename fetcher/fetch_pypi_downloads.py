"""
fetch_pypi_downloads.py
Run this LOCALLY on your machine (pypistats.org is blocked in the cloud env).

Usage:
    python fetch_pypi_downloads.py

Output:
    pypi_downloads.csv  — columns: repo, package, monthly_downloads, weekly_downloads

Requirements:
    pip install requests
"""

import requests
import time
import csv
import json
from pathlib import Path

# repo → PyPI package name mapping
PYPI_PACKAGES = {
    "ethereum/web3.py"     : "web3",
    "ethereum/py_ecc"      : "py_ecc",
    "vyperlang/vyper"      : "vyper",
    "apeworx/ape"          : "eth-ape",
    "vyperlang/titanoboa"  : "titanoboa",
}

def fetch_pypistats(package: str, period: str = "recent") -> dict:
    """
    Fetch download stats from pypistats.org.
    period='recent' returns last_day, last_week, last_month.
    """
    url = f"https://pypistats.org/api/packages/{package}/{period}"
    try:
        r = requests.get(url, timeout=10, headers={"User-Agent": "DeepFunding-Research/1.0"})
        if r.status_code == 200:
            data = r.json().get("data", {})
            return {
                "monthly_downloads": data.get("last_month", 0),
                "weekly_downloads" : data.get("last_week",  0),
                "daily_downloads"  : data.get("last_day",   0),
            }
        else:
            print(f"  WARNING: {package} → HTTP {r.status_code}")
            return {"monthly_downloads": 0, "weekly_downloads": 0, "daily_downloads": 0}
    except Exception as e:
        print(f"  ERROR: {package} → {e}")
        return {"monthly_downloads": 0, "weekly_downloads": 0, "daily_downloads": 0}

def main():
    results = []
    print(f"Fetching PyPI download stats for {len(PYPI_PACKAGES)} packages...\n")
    print(f"{'repo':<35}  {'package':<15}  {'monthly':>12}  {'weekly':>10}")
    print("-" * 80)

    for repo, pkg in PYPI_PACKAGES.items():
        stats = fetch_pypistats(pkg)
        time.sleep(0.5)

        row = {"repo": repo, "package": pkg, **stats}
        results.append(row)
        print(f"  {repo:<33}  {pkg:<15}  {stats['monthly_downloads']:>12,}  {stats['weekly_downloads']:>10,}")

    # Save CSV
    out_path = Path("pypi_downloads.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["repo","package","monthly_downloads","weekly_downloads","daily_downloads"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✅ Saved → {out_path.resolve()}")
    print(f"   Rows: {len(results)}")

    with open("pypi_downloads.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
