"""
fetch_stackoverflow.py
Run this LOCALLY on your machine (api.stackexchange.com is blocked in the cloud env).

Usage:
    python fetch_stackoverflow.py

Output:
    stackoverflow_counts.csv  — columns: repo, tag, question_count, answered_count, answered_ratio

Requirements:
    pip install requests

Note:
    Stack Exchange API allows 300 requests/day without a key.
    With an API key (free at stackapps.com) you get 10,000/day.
    Set YOUR_API_KEY below if you have one, otherwise leave as empty string.
"""

import requests
import time
import csv
import json
from pathlib import Path

API_KEY = ""  # Optional: get free key at https://stackapps.com/apps/oauth/register

# repo → Stack Overflow tag(s) mapping
# Multiple tags per repo: we sum question counts across all relevant tags
SO_TAGS = {
    "argotorg/solidity"                     : ["solidity"],
    "nomicfoundation/hardhat"               : ["hardhat", "hardhat-waffle"],
    "foundry-rs/foundry"                    : ["foundry", "foundry-forge"],
    "ethers-io/ethers.js"                   : ["ethers.js", "ethers"],
    "wevm/viem"                             : ["viem"],
    "ethereum/web3.py"                      : ["web3py", "web3.py"],
    "ethereum/go-ethereum"                  : ["go-ethereum", "geth"],
    "ethereum/eips"                         : ["eip", "ethereum-eip"],
    "openzeppelin/openzeppelin-contracts"   : ["openzeppelin", "openzeppelin-contracts"],
    "ethereum/consensus-specs"              : ["ethereum-consensus", "proof-of-stake"],
    "vyperlang/vyper"                       : ["vyper"],
    "remix-project-org/remix-project"       : ["remix-solidity", "remix-ide"],
    "safe-global/safe-smart-account"        : ["gnosis-safe", "safe-multisig"],
    "sigp/lighthouse"                       : ["lighthouse-ethereum"],
    "nethermindeth/nethermind"              : ["nethermind"],
    "eth-infinitism/account-abstraction"    : ["erc-4337", "account-abstraction"],
    "paulmillr/noble-curves"                : ["noble-curves"],
    "protofire/solhint"                     : ["solhint"],
    "alloy-rs/alloy"                        : ["alloy-rs"],
    "paradigmxyz/reth"                      : ["reth-ethereum"],
    "wighawag/hardhat-deploy"               : ["hardhat-deploy"],
    "blockscout/blockscout"                 : ["blockscout"],
}

def fetch_tag_stats(tag: str, api_key: str = "") -> dict:
    """Fetch question count for a Stack Overflow tag."""
    params = {
        "site"    : "stackoverflow",
        "inname"  : tag,
        "pagesize": 5,
        "order"   : "desc",
        "sort"    : "popular",
    }
    if api_key:
        params["key"] = api_key

    url = "https://api.stackexchange.com/2.3/tags"
    try:
        r = requests.get(url, params=params, timeout=10)
        if r.status_code == 200:
            items = r.json().get("items", [])
            # Find exact match first, then best partial match
            exact = next((i for i in items if i["name"] == tag), None)
            if exact:
                return {
                    "question_count": exact.get("count", 0),
                    "has_synonyms"  : exact.get("has_synonyms", False),
                }
            elif items:
                # Take best partial match
                return {"question_count": items[0].get("count", 0), "has_synonyms": False}
        return {"question_count": 0, "has_synonyms": False}
    except Exception as e:
        print(f"    ERROR fetching tag '{tag}': {e}")
        return {"question_count": 0, "has_synonyms": False}

def main():
    results = []
    print(f"Fetching Stack Overflow stats for {len(SO_TAGS)} repos...\n")
    print(f"{'repo':<50}  {'tags':<35}  {'total_questions':>15}")
    print("-" * 110)

    for repo, tags in SO_TAGS.items():
        total_questions = 0
        tag_details = []

        for tag in tags:
            stats = fetch_tag_stats(tag, API_KEY)
            total_questions += stats["question_count"]
            tag_details.append(f"{tag}:{stats['question_count']}")
            time.sleep(0.4)  # Respect rate limits

        results.append({
            "repo"            : repo,
            "tags"            : "|".join(tags),
            "tag_details"     : "|".join(tag_details),
            "total_questions" : total_questions,
        })
        print(f"  {repo:<48}  {', '.join(tags):<35}  {total_questions:>15,}")

    # Save CSV
    out_path = Path("stackoverflow_counts.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["repo","tags","tag_details","total_questions"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✅ Saved → {out_path.resolve()}")
    print(f"   Rows: {len(results)}")

    with open("stackoverflow_counts.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
