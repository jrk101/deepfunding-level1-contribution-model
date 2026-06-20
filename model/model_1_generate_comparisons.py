import os, json, time, re, random, threading
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from itertools import combinations
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FutureTimeout
from scipy.optimize import minimize
from openai import OpenAI
from dotenv import load_dotenv

# ── PATHS ─────────────────────────────────────────────────────────
BASE = Path(r"C:DeepFunding_lvl_1")
load_dotenv(BASE / ".env")

LLM_OUT      = BASE / "llm_comparisons_v7.csv"
GITHUB_CSV   = BASE / "github_features_fixed.csv"
NPM_CSV      = BASE / "npm_downloads.csv"
PYPI_CSV     = BASE / "pypi_downloads.csv"
SO_CSV       = BASE / "stackoverflow_counts.csv"
CMS_CSV      = BASE / "client_market_share.csv"
README_CSV   = BASE / "readme_data.csv"
PAIRWISE_CSV = BASE / "pairwise_data.csv"
REPOS_CSV    = BASE / "repos_to_predict.csv"
DEP_JSON     = BASE / "seedReposWithNoTransitiveDependencies.json"

CARGO_DOWNLOADS = {
    "alloy-rs/alloy": 1_643_712, "arkworks-rs/algebra": 10_793_597,
    "lambdaclass/lambdaworks": 463_900, "plonky3/plonky3": 297_255,
    "succinctlabs/sp1": 188_595, "axiom-crypto/snark-verifier": 68_610,
    "supranational/blst": 3_990_315, "offchainlabs/stylus-sdk-rs": 19_196,
}

MODEL_TIMEOUT = 28   # seconds per model call before skipping


MAX_MULTIPLIER = 50

# Change #5 / Improvement #1: raised from 5 → 10
# 98 × 10 = 980 appearances → ~490 comparisons (still manageable)
MIN_APPEARANCES = 10

# ── API CONFIGS ───────────────────────────────────────────────────
# (name, api_type, model_id, min_interval_secs_between_calls)
MODEL_CONFIGS = [
    # Direct APIs — not subject to NVIDIA rate limits
    ("deepseek",       "deepseek", "deepseek-chat",                               0.5),
    ("mistral-large",  "mistral",  "mistral-large-latest",                        0.5),
    # NVIDIA top-3 from evaluation list (probe at startup, fallback order below)
    ("nemotron-ultra", "nvidia",   "nvidia/nemotron-3-ultra-550b-a55b",            1.2),
    ("kimi-k2.6",      "nvidia",   "moonshotai/kimi-k2.6",                         1.2),
    ("nemotron-49b",   "nvidia",   "nvidia/llama-3.3-nemotron-super-49b-v1.5",     1.2),
]

# Fallback NVIDIA models if top-3 are dead
NVIDIA_FALLBACKS = [
    ("llama70b",    "nvidia", "meta/llama-3.3-70b-instruct",                      1.0),
    ("mistral-675", "nvidia", "mistralai/mistral-large-3-675b-instruct-2512",     1.0),
    ("nemotron-v1", "nvidia", "nvidia/llama-3.3-nemotron-super-49b-v1",           1.0),
]

# ── RATE LIMITERS (per API) ───────────────────────────────────────
class RateLimiter:
    """Thread-safe minimum-interval enforcer per API endpoint."""
    def __init__(self, min_interval: float):
        self.min_interval = min_interval
        self._lock = threading.Lock()
        self._last = 0.0

    def wait(self):
        with self._lock:
            gap = time.time() - self._last
            if gap < self.min_interval:
                time.sleep(self.min_interval - gap)
            self._last = time.time()

# One limiter per api_type — shared across threads for the same API
_limiters: dict[str, RateLimiter] = {}

def get_limiter(api_type: str, min_interval: float) -> RateLimiter:
    if api_type not in _limiters:
        _limiters[api_type] = RateLimiter(min_interval)
    return _limiters[api_type]


# ── HELPERS ───────────────────────────────────────────────────────
def norm(url):
    url = str(url).strip().rstrip("/")
    if "github.com/" in url:
        url = url.split("github.com/")[-1]
    return url.lower()

def geomean(values):
    return float(np.exp(np.mean(np.log(np.maximum(values, 1e-6)))))

def safe_save(rows, path):
    tmp = path.with_suffix(".tmp")
    pd.DataFrame(rows).to_csv(tmp, index=False)
    tmp.replace(path)


# ── LOAD DATA ─────────────────────────────────────────────────────
print("Loading data...")
gh      = pd.read_csv(GITHUB_CSV)
npm     = pd.read_csv(NPM_CSV)
pypi    = pd.read_csv(PYPI_CSV)
so      = pd.read_csv(SO_CSV)
cms     = pd.read_csv(CMS_CSV)
rdm     = pd.read_csv(README_CSV)
pairwise = pd.read_csv(PAIRWISE_CSV)
target  = pd.read_csv(REPOS_CSV)

gh["repo_key"]   = gh["repo"].apply(norm)
rdm["repo_key"]  = rdm["repo_url"].apply(norm)
target["repo_key"] = target["repo"].apply(norm)
for df_ in [npm, pypi, so, cms]:
    if "repo" in df_.columns:
        df_["repo"] = df_["repo"].apply(norm)

npm_map  = dict(zip(npm["repo"],  npm["monthly_downloads"]))
pyp_map  = dict(zip(pypi["repo"], pypi["monthly_downloads"]))
so_map   = dict(zip(so["repo"],   so["total_questions"]))
cms_map  = dict(zip(cms["repo"],  cms["client_market_share_pct"]))
readme_map = dict(zip(rdm["repo_key"], rdm["readme_excerpt"]))

import json as jsonlib
with open(DEP_JSON) as f:
    dep_data = jsonlib.load(f)
target_set = set(target["repo_key"])
rev_dep_count = Counter()
for url, deps in dep_data.items():
    for d in deps:
        dn = norm(d)
        if dn in target_set:
            rev_dep_count[dn] += 1

gh_idx = gh.set_index("repo_key")

# All 98 repos we need to rank
repos = sorted(target_set)


# ── HUMAN BT SCORES ───────────────────────────────────────────────
def compute_human_bt(pairwise_df):
    df = pairwise_df.copy()
    df["a"] = df["repo_a"].apply(norm)
    df["b"] = df["repo_b"].apply(norm)
    df = df[df["multiplier"] != 1.1].copy()
    df["log_ratio"]  = np.log(df["multiplier"])
    df["signed_log"] = df.apply(
        lambda r: r["log_ratio"] if r["winner"].lower().strip() == r["a"] else -r["log_ratio"],
        axis=1
    )
    bt_repos = sorted(set(df["a"]) | set(df["b"]))
    ridx  = {r: i for i, r in enumerate(bt_repos)}

    def huber(theta, rows, delta=1.0):
        loss = 0.0
        for a, b, sl in rows:
            d = theta[a] - theta[b] - sl
            loss += delta*(abs(d)-0.5*delta) if abs(d) > delta else 0.5*d**2
        return loss

    rows = [(ridx[r["a"]], ridx[r["b"]], r["signed_log"]) for _, r in df.iterrows()]
    res  = minimize(huber, np.zeros(len(bt_repos)), args=(rows,),
                    method="L-BFGS-B", options={"maxiter": 2000, "ftol": 1e-12})
    return dict(zip(bt_repos, res.x - res.x.mean()))

print("Computing human BT scores...")
human_theta  = compute_human_bt(pairwise)
covered_repos = set(human_theta.keys())

# ── Change #2: Pre-flight coverage audit ─────────────────────────
# Print missing repos before generating anything
missing_from_jury = [r for r in repos if r not in covered_repos]
print(f"\n── Coverage audit ──────────────────────────────────────────")
print(f"  Total repos to rank: {len(repos)}")
print(f"  Covered by jury data: {len(covered_repos & target_set)}")
print(f"  Missing from jury data: {len(missing_from_jury)}")
if missing_from_jury:
    print(f"  Missing repos:")
    for r in missing_from_jury:
        print(f"    {r}")
print()


# ── CALIBRATION EXAMPLES FROM REAL JURY DATA ─────────────────────
def build_calibration_examples(pairwise_df, human_theta, n=12):
    df = pairwise_df.copy()
    df["a"] = df["repo_a"].apply(norm)
    df["b"] = df["repo_b"].apply(norm)
    df = df[df["multiplier"] != 1.1].copy()
    df["winner_n"] = df.apply(lambda r: r["a"] if r["winner"].lower().strip()==r["a"] else r["b"], axis=1)
    df["loser_n"]  = df.apply(lambda r: r["b"] if r["winner"].lower().strip()==r["a"] else r["a"], axis=1)
    df = df.drop_duplicates(subset=["a","b"]).sort_values("multiplier")

    selected, seen = [], set()

    def add(r):
        k = (r["winner_n"], r["loser_n"])
        if k not in seen:
            seen.add(k); selected.append(r)

    # Surprise picks first (lower-star wins) in close range
    for _, r in df[df["multiplier"].between(1.5, 4.0)].iterrows():
        ws = float(gh_idx.loc[r["winner_n"]]["stars"]) if r["winner_n"] in gh_idx.index else 0
        ls = float(gh_idx.loc[r["loser_n"]]["stars"])  if r["loser_n"]  in gh_idx.index else 0
        if ws < ls: add(r)
        if sum(1 for s in selected if s["multiplier"] <= 4) >= 3: break

    # Fill close range if needed
    for _, r in df[df["multiplier"].between(1.5, 4.0)].iterrows():
        add(r)
        if sum(1 for s in selected if s["multiplier"] <= 4) >= 4: break

    # Medium range
    med = df[df["multiplier"].between(5, 30)]
    for _, r in med.iloc[::max(1, len(med)//4)].iterrows():
        add(r)
        if sum(1 for s in selected if 5 <= s["multiplier"] <= 30) >= 4: break

    # High range
    for _, r in df[df["multiplier"].between(30, 200)].iterrows():
        add(r)
        if sum(1 for s in selected if 30 <= s["multiplier"] <= 200) >= 3: break

    # Extreme
    for _, r in df[df["multiplier"] >= 200].tail(2).iterrows():
        add(r)

    lines = []
    for i, r in enumerate(selected[:n], 1):
        wt = human_theta.get(r["winner_n"], 0)
        lt = human_theta.get(r["loser_n"], 0)
        lines.append(
            f"  {i:2d}. WINNER={r['winner_n']}  beats  {r['loser_n']}  "
            f"by {r['multiplier']:.0f}x   "
            f"[jury BT: winner={wt:+.2f}, loser={lt:+.2f}]"
        )
    return "\n".join(lines)

calib_examples = build_calibration_examples(pairwise, human_theta)
print("Calibration examples for prompt:")
print(calib_examples, "\n")


# ── SYSTEM PROMPT (v6: technical importance > popularity) ─────────
# Improvement #3: Explicit instruction that technical importance to Ethereum
# outweighs GitHub popularity; warns against overvaluing stars/downloads
SYSTEM_PROMPT = f"""You are an expert evaluator judging open source repository contributions to Ethereum.

═══ WHAT "CONTRIBUTION" MEANS ═══
Score based on these factors in STRICT priority order:

1. IRREPLACEABILITY (40%) — What would break in Ethereum if this repo disappeared tomorrow?
   HIGH: execution clients, consensus clients, core compilers, BLS/ZK crypto primitives with no alternative
   MEDIUM: widely-adopted dev frameworks, smart contract libraries used in production
   LOW: analytics tools, peripheral utilities, repos with multiple viable alternatives

2. ECOSYSTEM LEVERAGE (30%) — How many other critical repos depend on this one?
   The info card tells you "X contest repos depend on it" — this is a strong signal.
   A library depended on by 10 core repos contributes indirectly to all of them.
   A standalone tool that nothing depends on scores low here regardless of popularity.

3. ADOPTION AT CRITICAL PATHS (20%) — Real usage in Ethereum's critical infrastructure
   Client market share % > monthly downloads (npm/PyPI/Cargo) > GitHub stars
   Stars are the WEAKEST signal — don't lead with them.

4. UNIQUENESS (10%) — Is there a viable substitute?
   Vyper is the ONLY alternative EVM compiler → maximum uniqueness score
   go-ethereum has 4+ alternatives (Besu, Nethermind, Erigon, Reth) → low uniqueness
   A repo that is one of many similar tools scores lower, even if individually good.

⚠️ CRITICAL — TECHNICAL IMPORTANCE OVER POPULARITY:
A repository that Ethereum would struggle to function without should be ranked
MUCH higher than a popular developer tool, even if the popular tool has more
GitHub stars or npm downloads. Stars measure developer interest; irreplaceability
measures what would actually break.

Examples of this principle:
  • BLS cryptographic primitives (e.g. supranational/blst) underpin the entire
    staking and validator ecosystem — they rank above popular analytics dashboards
    even if the dashboard has 10× more stars.
  • Consensus clients (lighthouse, prysm, teku) are less replaceable than popular
    smart contract libraries; there are only 4-5 consensus clients in the world.
  • A new ZK library with 8 dependents among the 98 contest repos is more critical
    than a standalone tool with 10,000 stars but 0 dependents.

═══ MULTIPLIER SCALE — CALIBRATE TO REAL JURY DATA ═══
The REAL jury for this competition uses these multipliers:
  Median: 4x  |  75th percentile: 7.5x  |  90th percentile: 10x  |  Max ever: 100x

Calibrate against REAL jury decisions from this exact competition:
{calib_examples}

Scale reference (match jury distribution):
  1–3x   → Same tier, similar criticality, close call (39% of jury votes are here)
  3–10x  → Clear difference in importance (50% of jury votes are here)
  10–20x → Large gap: e.g. core consensus client vs developer convenience tool
  20–50x → Extreme: e.g. core execution client vs a niche analytics tool
  50x+   → NEVER use. The real jury's absolute maximum is 100x (one vote, geth vs
            swiss-knife, confirmed by two independent jurors). If you reach for 50x+,
            reconsider — you are almost certainly over-inflating.

⚠️ ANTI-INFLATION: The jury median is 4x. If you are giving most pairs 50x or 100x,
   you are calibrated wrong. Most pairs between repos of different importance should
   be 5–20x, not 100x+.

⚠️ ANTI-COMPRESSION: If you find yourself using 1–2x for repos with clearly
   different criticality, reconsider — most pairs with a real difference are 5–15x.

═══ SPECIAL CASES ═══
• Core client vs compiler: Compilers (Vyper, Solidity) can OUTRANK clients if they
  are the only implementation. Clients typically have 3-5 alternatives; a unique
  compiler has none. The jury ranked Vyper #1, above all execution/consensus clients.

• New repo vs established: Don't penalize recent repos for low commit history.
  A new ZK library that 8 other contest repos depend on is CRITICAL regardless of age.

• Niche crypto lib vs popular tool: BLS/ZK primitive libraries that underpin the
  entire staking or rollup ecosystem can outweigh popular dev tools used casually.

═══ RESPONSE FORMAT ═══
Respond ONLY with valid JSON — no other text, no markdown:
{{"choice": 1 or 2, "multiplier": <number 1.0–50.0>, "reasoning": "<comparative sentence: Repo X has [specific stat] vs Repo Y has [specific stat], therefore X wins by Nx because [factor]>"}}

Your reasoning MUST: (1) cite specific numbers from the info cards, (2) name the
winning factor explicitly (irreplaceability/leverage/adoption/uniqueness),
(3) justify the multiplier magnitude.
Multiplier MUST be between 1.0 and 50.0. Values above 50 will be clipped."""


# ── REPO INFO CARD (v6: all features exposed to LLM) ─────────────
# Change #4 / Improvement #4: info card now includes stars, forks, contributors,
# downloads, PageRank, BT score — not just description
def get_info(r):
    g = gh_idx.loc[r] if r in gh_idx.index else None
    def gv(col, d=0):
        return float(g[col]) if g is not None and col in g.index else d

    s   = int(gv("stars")); fk  = int(gv("forks"))
    ct  = int(gv("contributors_count")); cm = int(gv("commits_last_12m"))
    dy  = int(gv("days_since_last_commit", 999))
    mkt = float(cms_map.get(r, 0)); soq = int(so_map.get(r, 0))
    rdc = rev_dep_count.get(r, 0)

    dl = []
    if npm_map.get(r, 0): dl.append(f"npm {int(npm_map[r]):,}/mo")
    if pyp_map.get(r, 0): dl.append(f"PyPI {int(pyp_map[r]):,}/mo")
    if CARGO_DOWNLOADS.get(r, 0): dl.append(f"Cargo {CARGO_DOWNLOADS[r]:,}/mo")

    readme = str(readme_map.get(r, "")).replace("\n", " ").strip()[:280]

    # ── Core info ──────────────────────────────────────────────────
    lines = [
        f"Repository: github.com/{r}",
        f"Description: {readme or 'No description available.'}",
        "",
        # Change #4: explicitly surface each metric on its own line so the
        # LLM sees and uses them (rather than everything buried in one line)
        f"Stars:              {s:,}",
        f"Forks:              {fk:,}",
        f"Contributors:       {ct:,}",
        f"Commits (12 mo):    {cm:,}",
        f"Days since commit:  {dy}",
    ]

    if dl:
        lines.append(f"Package downloads:  {', '.join(dl)}")
    if mkt > 0:
        lines.append(f"Client market share:{mkt:.1f}%  ← strong adoption signal")
    if soq > 0:
        lines.append(f"Stack Overflow:     {soq:,} questions tagged")

    # Improvement #4: reverse deps and BT score as top-line signals
    if rdc >= 5:
        lines.append(f"⚡ CORE LIBRARY: {rdc} of the 98 contest repos directly depend on this")
    elif rdc >= 2:
        lines.append(f"Ecosystem deps:     {rdc} contest repos depend on this")
    elif rdc == 1:
        lines.append(f"Ecosystem deps:     1 contest repo depends on this")
    else:
        lines.append(f"Ecosystem deps:     0 contest repos depend on this (standalone)")

    if r in human_theta:
        rank = sorted(human_theta, key=lambda x: -human_theta[x]).index(r) + 1
        lines.append(
            f"Jury BT rank:       #{rank} of {len(human_theta)} jury-evaluated repos "
            f"(BT score {human_theta[r]:+.2f})"
        )
    else:
        lines.append(f"Jury BT rank:       not yet in jury data (new/uncovered repo)")

    return "\n".join(lines)


def make_user_prompt(repo_a, repo_b):
    return (
        "Which repository contributes MORE to Ethereum's success?\n\n"
        f"── REPOSITORY 1 ──\n{get_info(repo_a)}\n\n"
        f"── REPOSITORY 2 ──\n{get_info(repo_b)}\n\n"
        "Compare using the four factors (irreplaceability, leverage, adoption, uniqueness).\n"
        "Remember: technical importance to Ethereum > GitHub popularity.\n"
        "Cite specific stats. Justify the multiplier magnitude. Respond with JSON only."
    )


# ── BUILD CLIENTS ─────────────────────────────────────────────────
def build_clients():
    nvidia_key   = os.environ.get("NVIDIA_API_KEY", "")
    deepseek_key = os.environ.get("DEEPSEEK_API_KEY", "")
    mistral_key  = os.environ.get("MISTRAL_API_KEY", "")

    clients = {}
    if nvidia_key:
        clients["nvidia"] = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1", api_key=nvidia_key)
    if deepseek_key:
        clients["deepseek"] = OpenAI(
            base_url="https://api.deepseek.com/v1", api_key=deepseek_key)
    if mistral_key:
        clients["mistral"] = OpenAI(
            base_url="https://api.mistral.ai/v1", api_key=mistral_key)

    missing = [t for t in ["nvidia","deepseek","mistral"] if t not in clients]
    if missing:
        print(f"  ⚠️  Missing API keys for: {missing}")
    return clients


# ── PROBE MODELS ──────────────────────────────────────────────────
def probe_models(clients):
    print("Probing model availability...")
    alive = []
    TEST  = '{"choice": 1, "multiplier": 5, "reasoning": "test ok"}'

    all_candidates = MODEL_CONFIGS + NVIDIA_FALLBACKS
    nvidia_alive = 0

    for name, api_type, model_id, min_interval in all_candidates:
        if api_type not in clients:
            print(f"  ⏭️  {name:18s}  no {api_type} key")
            continue
        # Only need 3 NVIDIA models
        if api_type == "nvidia" and nvidia_alive >= 3:
            break
        try:
            resp = clients[api_type].chat.completions.create(
                model=model_id, max_tokens=40, temperature=0.0,
                timeout=15,
                messages=[{"role":"user","content":f"Reply with exactly: {TEST}"}]
            )
            txt = (resp.choices[0].message.content or "").strip()
            if txt:
                print(f"  ✅ {name:18s}  {api_type:10s}  {model_id}")
                alive.append((name, api_type, model_id, min_interval))
                if api_type == "nvidia": nvidia_alive += 1
            else:
                print(f"  ⚠️  {name:18s}  empty response")
        except Exception as e:
            print(f"  ❌ {name:18s}  {str(e)[:65]}")
        time.sleep(0.3)

    print(f"\n  → Ensemble: {len(alive)} models: {[n for n,*_ in alive]}\n")
    assert len(alive) >= 2, "Need at least 2 alive models."
    return alive


# ── SINGLE MODEL CALL ─────────────────────────────────────────────
def call_model(clients, name, api_type, model_id, min_interval,
               repo_a, repo_b, retries=2):
    limiter = get_limiter(api_type, min_interval)
    user_msg = make_user_prompt(repo_a, repo_b)

    for attempt in range(retries):
        limiter.wait()
        try:
            resp = clients[api_type].chat.completions.create(
                model=model_id, max_tokens=350, temperature=0.2,
                timeout=MODEL_TIMEOUT,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg}
                ]
            )
            text = resp.choices[0].message.content
            if text is None:
                text = getattr(resp.choices[0].message, "reasoning_content", None)
            if not text:
                time.sleep(1); continue

            text = re.sub(r"<think>.*?</think>", "", str(text), flags=re.DOTALL).strip()
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"): text = text[4:]
            m = re.search(r"\{[^{}]+\}", text, re.DOTALL)
            if m: text = m.group()

            data       = json.loads(text.strip())
            choice     = int(data["choice"])
            multiplier = max(1.0, min(float(data["multiplier"]), MAX_MULTIPLIER))
            reasoning  = str(data.get("reasoning", ""))
            assert choice in [1, 2]
            return choice, multiplier, reasoning

        except Exception as e:
            if attempt < retries - 1: time.sleep(1.5)

    return None, None, None


# ── PARALLEL ENSEMBLE ─────────────────────────────────────────────
def ensemble_compare(clients, alive_models, repo_a, repo_b):
    results = {}

    def task(name, api_type, model_id, min_interval):
        return name, call_model(clients, name, api_type, model_id,
                                min_interval, repo_a, repo_b)

    with ThreadPoolExecutor(max_workers=len(alive_models)) as ex:
        futs = {ex.submit(task, *m): m[0] for m in alive_models}
        try:
            for fut in as_completed(futs, timeout=MODEL_TIMEOUT + 20):
                try:
                    name, (c, m, r) = fut.result()
                    if c is not None:
                        results[name] = (c, m, r)
                except Exception:
                    pass
        except FutureTimeout:
            pass

    if not results:
        return None, None, None, None, None, None

    choices     = [v[0] for v in results.values()]
    multipliers = [v[1] for v in results.values()]
    n           = len(choices)
    votes_a, votes_b = choices.count(1), choices.count(2)
    winner      = 1 if votes_a >= votes_b else 2
    win_count   = max(votes_a, votes_b)
    win_frac    = win_count / n

    agree_mults = [multipliers[i] for i, c in enumerate(choices) if c == winner]
    final_mult  = geomean(agree_mults)

    if win_count == n:     agreement, weight = "full",     1.0
    elif win_frac >= 0.75: agreement, weight = "strong",   0.85
    elif win_frac >= 0.60: agreement, weight = "majority", 0.6
    else:                  agreement, weight, final_mult = "weak", 0.3, 1.5

    best_reasoning = next((r for _, (c, m, r) in results.items() if c == winner), "")
    vote_summary   = " | ".join(
        f"{name}: {'A' if c==1 else 'B'} {round(m)}x"
        for name, (c, m, r) in results.items()
    )

    return winner, round(final_mult, 1), best_reasoning, agreement, weight, vote_summary



ANCHOR_REPOS = [
    # Execution clients
    "ethereum/go-ethereum",
    "hyperledger/besu",
    "nethermindeth/nethermind",
    # Consensus clients
    "sigp/lighthouse",
    "prysmaticlabs/prysm",
    "consensys/teku",
    # Core tooling / compilers
    "argotorg/solidity",
    "vyperlang/vyper",
    "nomicfoundation/hardhat",
    "foundry-rs/foundry",
    # Smart contract standards
    "openzeppelin/openzeppelin-contracts",
    "ethereum/eips",
    # ZK / cryptographic primitives
    "succinctlabs/sp1",
    "supranational/blst",
    "risc0/risc0-ethereum",
    # Protocol specs & APIs
    "ethereum/consensus-specs",
    "ethereum/execution-apis",
    # JS/TS ecosystem
    "ethers-io/ethers.js",
]

# Validate anchors are in our repo list
for _a in ANCHOR_REPOS:
    if _a not in target_set:
        print(f"  ⚠️  Anchor not in repos_to_predict: {_a}  (will skip)")
ANCHOR_REPOS = [a for a in ANCHOR_REPOS if a in target_set]
print(f"  Using {len(ANCHOR_REPOS)} validated anchors\n")


def build_pairs(repos, existing_rows=None):
    """
    Build the generation queue:

    Phase 1 — Anchor pairs for every missing repo (Change #3)
        For each repo not yet in existing_rows, pair it with every anchor.
        This guarantees connectivity into the BT graph.

    Phase 2 — Full deterministic pair list (Change #1)
        Generate ALL 98*97/2 = 4753 unique pairs via combinations,
        shuffle once, then use as a reservoir.

    The outer MIN_APPEARANCES loop (in main) then draws from this
    reservoir until every repo has >= MIN_APPEARANCES successful comparisons.
    """
    # Change #2: find already-covered repos in existing data
    covered = set()
    if existing_rows:
        for r in existing_rows:
            covered.add(norm(r["repo_a"]))
            covered.add(norm(r["repo_b"]))

    missing = [r for r in repos if r not in covered]
    print(f"  Pre-flight: {len(missing)} repos not yet in any comparison")

    seen = set()
    queue = []

    def add(a, b):
        k = tuple(sorted([a, b]))
        if k not in seen and a != b:
            seen.add(k)
            queue.append((a, b))

    # Phase 1: anchor pairs for every missing repo
    for repo in missing:
        for anchor in ANCHOR_REPOS:
            add(repo, anchor)
    anchor_pairs = len(queue)
    print(f"  Phase 1 — anchor pairs for {len(missing)} missing repos × "
          f"{len(ANCHOR_REPOS)} anchors = {anchor_pairs} pairs")

    # Phase 2: full deterministic pair list as reservoir
    all_pairs = list(combinations(repos, 2))
    random.shuffle(all_pairs)          # one-time shuffle for variety
    for a, b in all_pairs:
        add(a, b)

    print(f"  Phase 2 — {len(queue) - anchor_pairs} additional pairs from full "
          f"combinations reservoir (total unique pool: {len(queue)})\n")
    return queue


# ── MAIN ──────────────────────────────────────────────────────────
def main():
    clients = build_clients()
    alive   = probe_models(clients)

    print("\n=== MODEL 1 v7: Full Coverage, Expanded Anchors, Jury-Calibrated Multipliers ===")
    print(f"  Models:          {', '.join(n for n,*_ in alive)}")
    print(f"  APIs:            {set(t for _,t,*_ in alive)}")
    print(f"  MIN_APPEARANCES: {MIN_APPEARANCES}")
    print(f"  Target pairs:    ~{len(repos) * MIN_APPEARANCES // 2} "
          f"(98 repos × {MIN_APPEARANCES} appearances / 2)\n")

    # Resume support
    if LLM_OUT.exists():
        existing = pd.read_csv(LLM_OUT)
        done = (set(zip(existing["repo_a"].apply(norm), existing["repo_b"].apply(norm))) |
                set(zip(existing["repo_b"].apply(norm), existing["repo_a"].apply(norm))))
        rows = existing.to_dict("records")
        print(f"Resuming: {len(rows)} comparisons already done\n")
    else:
        done, rows = set(), []

    # Change #5 / Bug fix: track appearances from SUCCESSFUL parses only
    # (not from attempts), so the stop condition is meaningful
    live_counts: dict[str, int] = Counter()
    for r in rows:
        live_counts[norm(r["repo_a"])] += 1
        live_counts[norm(r["repo_b"])] += 1

    # Build full pair queue
    pair_queue = build_pairs(repos, rows)
    todo = [(a, b) for a, b in pair_queue if tuple(sorted([a, b])) not in done]
    print(f"  Pairs remaining in queue: {len(todo)}\n")

    counts = {"full": 0, "strong": 0, "majority": 0, "weak": 0}
    start  = time.time()

    for i, (a, b) in enumerate(todo):
        # Change #5: stop when every repo has MIN_APPEARANCES successful comparisons
        still_under = [r for r in repos if live_counts.get(r, 0) < MIN_APPEARANCES]
        if not still_under:
            print(f"\n✅ All {len(repos)} repos have ≥ {MIN_APPEARANCES} appearances. Stopping.")
            break

        # Skip if already done (resume guard)
        key = tuple(sorted([a, b]))
        if key in done:
            continue

        a_short, b_short = a.split("/")[-1], b.split("/")[-1]
        print(f"[{len(rows)+1}] {a_short} vs {b_short}  ", end="", flush=True)

        result = ensemble_compare(clients, alive, a, b)
        done.add(key)   # mark attempted regardless of success

        if result[0] is None:
            print("ALL FAILED — skipping")
            continue

        choice, mult, reason, agreement, weight, vote_summary = result
        winner_repo = a if choice == 1 else b

        # Bug fix: only increment live_counts on SUCCESSFUL parse
        live_counts[a] += 1
        live_counts[b] += 1

        icons = {"full":"✓✓✓✓✓","strong":"✓✓✓✓✗","majority":"✓✓✓✗✗","weak":"✓✓✗✗✗"}
        elapsed = time.time() - start
        rate    = (i + 1) / max(elapsed, 1)
        remaining = len([r for r in repos if live_counts.get(r, 0) < MIN_APPEARANCES])
        eta     = int((remaining * MIN_APPEARANCES / 2) / max(rate, 0.01))
        print(f"{winner_repo.split('/')[-1]} {mult:.0f}x  "
              f"{icons.get(agreement,'?')}  "
              f"[{remaining} repos still under min]  "
              f"ETA ~{eta//60}m{eta%60:02d}s  [{vote_summary[:65]}]")

        counts[agreement] += 1
        rows.append({
            "repo_a": a, "repo_b": b,
            "choice": choice, "multiplier": round(mult, 1),
            "reasoning": reason, "agreement": agreement,
            "weight": weight, "vote_summary": vote_summary,
            "juror": "LLM_ensemble_v7", "parent": "ethereum",
        })
        safe_save(rows, LLM_OUT)

        # Periodic coverage diagnostic (every 50 comparisons)
        if len(rows) % 50 == 0:
            covered_now = sum(1 for r in repos if live_counts.get(r, 0) >= MIN_APPEARANCES)
            under_min   = [(r, live_counts.get(r, 0)) for r in repos
                           if live_counts.get(r, 0) < MIN_APPEARANCES]
            under_min.sort(key=lambda x: x[1])
            print(f"\n  ── Progress @ {len(rows)} rows ──────────────────────────")
            print(f"     Coverage: {covered_now}/{len(repos)} repos at ≥{MIN_APPEARANCES} appearances")
            print(f"     {len(under_min)} repos still under min — lowest 5:")
            for r, cnt in under_min[:5]:
                print(f"       {r}: {cnt}")
            print(f"  ────────────────────────────────────────────────────\n")

    # ── Change #5 / Improvement #5: coverage summary ──────────────
    repos_seen = set()
    for row in rows:
        repos_seen.add(norm(row["repo_a"]))
        repos_seen.add(norm(row["repo_b"]))

    uncovered_final = [r for r in repos if r not in repos_seen]
    print(f"\n── Final coverage ──────────────────────────────────────────")
    print(f"  Coverage: {len(repos_seen & target_set)} / {len(repos)} repos")
    if uncovered_final:
        print(f"  ⚠️  Still uncovered ({len(uncovered_final)} repos):")
        for r in uncovered_final:
            print(f"    {r}")
    else:
        print(f"  ✅ All {len(repos)} repos covered — safe to feed into Bradley-Terry")

    # Appearance distribution
    low_coverage = [(r, live_counts.get(r, 0)) for r in repos
                    if live_counts.get(r, 0) < MIN_APPEARANCES]
    if low_coverage:
        low_coverage.sort(key=lambda x: x[1])
        print(f"\n  Repos below MIN_APPEARANCES={MIN_APPEARANCES}:")
        for r, cnt in low_coverage[:10]:
            print(f"    {r}: {cnt} appearances")
        if len(low_coverage) > 10:
            print(f"    ... and {len(low_coverage)-10} more")

    total = len(rows)
    print(f"\n=== Done: {total} comparisons ===")
    print(f"  Full:{counts['full']}  Strong:{counts['strong']}  "
          f"Majority:{counts['majority']}  Weak:{counts['weak']}")
    if total:
        mults = [r["multiplier"] for r in rows]
        print(f"  Multiplier: min={np.min(mults):.0f}  median={np.median(mults):.0f}  max={np.max(mults):.0f}")
        usable = sum(1 for r in rows if r["agreement"] in ["full","strong","majority"])
        print(f"  Usable (full+strong+majority): {usable}/{total} ({100*usable/total:.0f}%)")
    print(f"\nNext step: run model2_bradley_terry.py with llm_comparisons_v7.csv")

if __name__ == "__main__":
    main()