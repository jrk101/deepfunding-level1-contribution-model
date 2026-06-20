"""
fetch_npm_downloads.py
Run this LOCALLY on your machine (api.npmjs.org is blocked in the cloud env).

Usage:
    python fetch_npm_downloads.py

Output:
    npm_downloads.csv  — columns: repo, package, monthly_downloads, weekly_downloads

Requirements:
    pip install requests
"""

import requests
import time
import csv
import json
from pathlib import Path

# repo → npm package name mapping
NPM_PACKAGES = {
    "ethers-io/ethers.js"                   : "ethers",
    "wevm/viem"                             : "viem",
    "nomicfoundation/hardhat"               : "hardhat",
    "openzeppelin/openzeppelin-contracts"   : "@openzeppelin/contracts",
    "protofire/solhint"                     : "solhint",
    "wighawag/hardhat-deploy"               : "hardhat-deploy",
    "ethereum/js-ethereum-cryptography"     : "@ethereumjs/util",
    "paulmillr/noble-curves"                : "@noble/curves",
    "chainsafe/lodestar"                    : "@chainsafe/lodestar",
    "remix-project-org/remix-project"       : "@remix-project/remixd",
    "evmts/tevm-monorepo"                   : "tevm",
    "dl-solarity/solidity-lib"              : "@solarity/solidity-lib",
    "vectorized/solady"                     : "solady",
    "eth-infinitism/account-abstraction"    : "@account-abstraction/contracts",
    "safe-global/safe-smart-account"        : "@safe-global/safe-contracts",
    "chainsafe/bls"                         : "@chainsafe/bls",
    "ethereum-lists/chains"                 : "@ethereum-lists/chains",
    "scaffold-eth/scaffold-eth-2"           : "create-eth",
}

def fetch_downloads(package: str, period: str = "last-month") -> int:
    """Fetch npm download count for a package over the given period."""
    url = f"https://api.npmjs.org/downloads/point/{period}/{package}"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            return r.json().get("downloads", 0)
        else:
            print(f"  WARNING: {package} → HTTP {r.status_code}")
            return 0
    except Exception as e:
        print(f"  ERROR: {package} → {e}")
        return 0

def main():
    results = []
    print(f"Fetching npm download stats for {len(NPM_PACKAGES)} packages...\n")
    print(f"{'repo':<50}  {'package':<40}  {'monthly':>12}  {'weekly':>10}")
    print("-" * 120)

    for repo, pkg in NPM_PACKAGES.items():
        monthly = fetch_downloads(pkg, "last-month")
        time.sleep(0.3)
        weekly  = fetch_downloads(pkg, "last-week")
        time.sleep(0.3)

        results.append({
            "repo"             : repo,
            "package"          : pkg,
            "monthly_downloads": monthly,
            "weekly_downloads" : weekly,
        })
        print(f"  {repo:<48}  {pkg:<40}  {monthly:>12,}  {weekly:>10,}")

    # Save CSV
    out_path = Path("npm_downloads.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["repo","package","monthly_downloads","weekly_downloads"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✅ Saved → {out_path.resolve()}")
    print(f"   Rows: {len(results)}")

    # Also save JSON for debugging
    with open("npm_downloads.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
