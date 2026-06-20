"""
fetch_client_market_share.py
Run this LOCALLY on your machine.

Fetches Ethereum client market share from:
  1. Miga Labs API  — consensus clients (lighthouse, prysm, teku, lodestar, nimbus, grandine)
  2. Ethernodes.org — execution clients (geth, erigon, nethermind, besu, reth)

Output:
    client_market_share.csv

Requirements:
    pip install requests pandas
"""

import requests
import time
import json
import pandas as pd
from pathlib import Path

HEADERS = {"User-Agent": "DeepFunding-Research/1.0 (github.com/deepfunding)"}

# ── repo mapping ──────────────────────────────────────────────────────────────
# client name (as returned by API) → github repo
CONSENSUS_REPO_MAP = {
    "lighthouse"  : "sigp/lighthouse",
    "prysm"       : "offchainlabs/prysm",
    "teku"        : "consensys/teku",
    "nimbus"      : "status-im/nimbus-eth2",
    "lodestar"    : "chainsafe/lodestar",
    "grandine"    : "grandinetech/grandine",
    "nimbus-eth2" : "status-im/nimbus-eth2",
}

EXECUTION_REPO_MAP = {
    "geth"        : "ethereum/go-ethereum",
    "go-ethereum" : "ethereum/go-ethereum",
    "erigon"      : "erigontech/erigon",
    "nethermind"  : "nethermindeth/nethermind",
    "besu"        : "hyperledger/besu",
    "reth"        : "paradigmxyz/reth",
    "silkworm"    : "erigontech/silkworm",
}

# ── Source 1: Miga Labs (consensus clients) ───────────────────────────────────
def fetch_migalabs() -> dict:
    """
    Miga Labs crawls the beacon network and reports client distribution.
    Returns dict of {client_name: percentage}
    """
    urls = [
        "https://migalabs.es/api/v1/eth2/nodes",
        "https://api.migalabs.es/v1/beacon/clients",
        "https://migalabs.es/api/v1/beacon/client_distribution",
    ]

    for url in urls:
        try:
            print(f"  Trying: {url}")
            r = requests.get(url, timeout=15, headers=HEADERS)
            print(f"  Status: {r.status_code}")
            if r.status_code == 200:
                data = r.json()
                print(f"  Keys: {list(data.keys()) if isinstance(data, dict) else type(data)}")
                print(f"  Preview: {str(data)[:300]}")
                return {"source": url, "raw": data}
        except Exception as e:
            print(f"  Error: {e}")
        time.sleep(1)

    return {}


# ── Source 2: Ethernodes (execution clients) ──────────────────────────────────
def fetch_ethernodes() -> dict:
    """
    Ethernodes tracks execution layer node distribution.
    """
    urls = [
        "https://ethernodes.org/api/clients",
        "https://www.ethernodes.org/api/v1/clients",
        "https://ethernodes.org/data",
    ]

    for url in urls:
        try:
            print(f"  Trying: {url}")
            r = requests.get(url, timeout=15, headers=HEADERS)
            print(f"  Status: {r.status_code}")
            if r.status_code == 200:
                try:
                    data = r.json()
                    print(f"  Preview: {str(data)[:300]}")
                    return {"source": url, "raw": data}
                except Exception:
                    print(f"  Not JSON, content: {r.text[:200]}")
        except Exception as e:
            print(f"  Error: {e}")
        time.sleep(1)

    return {}


# ── Source 3: supermajority.info ──────────────────────────────────────────────
def fetch_supermajority() -> dict:
    urls = [
        "https://supermajority.info/api/data",
        "https://supermajority.info/data.json",
        "https://supermajority.info",
    ]
    for url in urls:
        try:
            print(f"  Trying: {url}")
            r = requests.get(url, timeout=15, headers=HEADERS)
            print(f"  Status: {r.status_code}")
            if r.status_code == 200:
                try:
                    data = r.json()
                    print(f"  Preview: {str(data)[:300]}")
                    return {"source": url, "raw": data}
                except Exception:
                    print(f"  HTML response, length={len(r.text)}")
        except Exception as e:
            print(f"  Error: {e}")
        time.sleep(1)
    return {}


# ── Source 4: clientdiversity.org ────────────────────────────────────────────
def fetch_clientdiversity() -> dict:
    urls = [
        "https://clientdiversity.org/api/data",
        "https://clientdiversity.org/data.json",
        "https://clientdiversity.org/api/v1/distribution",
    ]
    for url in urls:
        try:
            print(f"  Trying: {url}")
            r = requests.get(url, timeout=15, headers=HEADERS)
            print(f"  Status: {r.status_code}")
            if r.status_code == 200:
                try:
                    data = r.json()
                    print(f"  Preview: {str(data)[:300]}")
                    return {"source": url, "raw": data}
                except Exception:
                    print(f"  HTML, length={len(r.text)}")
        except Exception as e:
            print(f"  Error: {e}")
        time.sleep(1)
    return {}


# ── Parse helpers ─────────────────────────────────────────────────────────────
def parse_percentage_dict(raw) -> dict:
    """
    Try to extract {client: pct} from various response shapes.
    Returns empty dict if parsing fails.
    """
    if isinstance(raw, dict):
        # Shape: {"lighthouse": 35.2, "prysm": 33.1, ...}
        if all(isinstance(v, (int, float)) for v in raw.values()):
            return raw
        # Shape: {"data": {"lighthouse": 35.2, ...}}
        if "data" in raw:
            return parse_percentage_dict(raw["data"])
        # Shape: {"clients": [{"name": "lighthouse", "percentage": 35.2}, ...]}
        for key in ("clients", "results", "nodes", "distribution"):
            if key in raw and isinstance(raw[key], list):
                return parse_list_of_dicts(raw[key])

    if isinstance(raw, list):
        return parse_list_of_dicts(raw)

    return {}


def parse_list_of_dicts(lst: list) -> dict:
    result = {}
    for item in lst:
        if not isinstance(item, dict):
            continue
        name = (item.get("name") or item.get("client") or
                item.get("clientName") or "").lower().strip()
        pct  = (item.get("percentage") or item.get("pct") or
                item.get("share") or item.get("count") or 0)
        if name and pct:
            result[name] = float(pct)
    return result


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    rows = []

    print("\n" + "="*60)
    print("CONSENSUS CLIENTS")
    print("="*60)

    print("\n[1] Trying Miga Labs...")
    miga = fetch_migalabs()
    if miga:
        parsed = parse_percentage_dict(miga.get("raw", {}))
        print(f"  Parsed: {parsed}")
        for client, pct in parsed.items():
            repo = CONSENSUS_REPO_MAP.get(client.lower())
            if repo:
                rows.append({
                    "repo": repo,
                    "client_name": client,
                    "client_market_share_pct": round(float(pct), 2),
                    "client_type": "consensus",
                    "source": miga.get("source", "migalabs"),
                })
        with open("migalabs_raw.json", "w") as f:
            json.dump(miga, f, indent=2, default=str)
        print("  Raw data saved → migalabs_raw.json")

    print("\n[2] Trying clientdiversity.org...")
    cd = fetch_clientdiversity()
    if cd:
        with open("clientdiversity_raw.json", "w") as f:
            json.dump(cd, f, indent=2, default=str)
        print("  Raw data saved → clientdiversity_raw.json")

    print("\n[3] Trying supermajority.info...")
    sm = fetch_supermajority()
    if sm:
        with open("supermajority_raw.json", "w") as f:
            json.dump(sm, f, indent=2, default=str)
        print("  Raw data saved → supermajority_raw.json")

    print("\n" + "="*60)
    print("EXECUTION CLIENTS")
    print("="*60)

    print("\n[4] Trying Ethernodes...")
    eth = fetch_ethernodes()
    if eth:
        parsed = parse_percentage_dict(eth.get("raw", {}))
        print(f"  Parsed: {parsed}")
        for client, pct in parsed.items():
            repo = EXECUTION_REPO_MAP.get(client.lower())
            if repo:
                rows.append({
                    "repo": repo,
                    "client_name": client,
                    "client_market_share_pct": round(float(pct), 2),
                    "client_type": "execution",
                    "source": eth.get("source", "ethernodes"),
                })
        with open("ethernodes_raw.json", "w") as f:
            json.dump(eth, f, indent=2, default=str)
        print("  Raw data saved → ethernodes_raw.json")

    # ── Output ────────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    if rows:
        df = pd.DataFrame(rows)
        df = df.sort_values(["client_type", "client_market_share_pct"], ascending=[True, False])
        df.to_csv("client_market_share.csv", index=False)
        print(f"✅  Saved → client_market_share.csv  ({len(df)} rows)")
        print()
        print(df.to_string(index=False))
    else:
        print("⚠️  No data fetched from any API source.")
        print()
        print("All APIs may require browser-based access or have changed endpoints.")
        print("In that case, manually copy the data from:")
        print("  Consensus: https://clientdiversity.org  (pie chart, top of page)")
        print("  Execution: https://ethernodes.org       (client share section)")
        print()
        print("Then fill in client_market_share_MANUAL.csv below:")

        manual = pd.DataFrame([
            # CONSENSUS — fill these from clientdiversity.org
            {"repo": "sigp/lighthouse",                    "client_name": "lighthouse", "client_market_share_pct": 0.0, "client_type": "consensus"},
            {"repo": "offchainlabs/prysm",                 "client_name": "prysm",      "client_market_share_pct": 0.0, "client_type": "consensus"},
            {"repo": "consensys/teku",                     "client_name": "teku",       "client_market_share_pct": 0.0, "client_type": "consensus"},
            {"repo": "status-im/nimbus-eth2",              "client_name": "nimbus",     "client_market_share_pct": 0.0, "client_type": "consensus"},
            {"repo": "chainsafe/lodestar",                 "client_name": "lodestar",   "client_market_share_pct": 0.0, "client_type": "consensus"},
            {"repo": "grandinetech/grandine",              "client_name": "grandine",   "client_market_share_pct": 0.0, "client_type": "consensus"},
            # EXECUTION — fill these from ethernodes.org
            {"repo": "ethereum/go-ethereum",               "client_name": "geth",       "client_market_share_pct": 0.0, "client_type": "execution"},
            {"repo": "erigontech/erigon",                  "client_name": "erigon",     "client_market_share_pct": 0.0, "client_type": "execution"},
            {"repo": "nethermindeth/nethermind",           "client_name": "nethermind", "client_market_share_pct": 0.0, "client_type": "execution"},
            {"repo": "hyperledger/besu",                   "client_name": "besu",       "client_market_share_pct": 0.0, "client_type": "execution"},
            {"repo": "paradigmxyz/reth",                   "client_name": "reth",       "client_market_share_pct": 0.0, "client_type": "execution"},
        ])
        manual.to_csv("client_market_share_MANUAL.csv", index=False)
        print("  Template saved → client_market_share_MANUAL.csv")
        print("  Fill in the 0.0 values and rename to client_market_share.csv")

if __name__ == "__main__":
    main()
