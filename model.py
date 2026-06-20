import json, re
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import networkx as nx
from scipy.optimize import minimize
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, cross_val_predict

# ── PATHS ─────────────────────────────────────────────────────────
BASE = Path(r"C:\DeepFunding_lvl_1")

PAIRWISE_CSV = BASE / "pairwise_data.csv"
TRAIN_CSV    = BASE / "train.csv"        # FIX 1: previous round juror data
LLM_CSV      = BASE / "llm_comparisons_v7.csv"
LLM_FIX_CSV  = BASE / "llm_comparisons_grok_fix.csv"   # sparse-coverage supplement
REPOS_CSV    = BASE / "repos_to_predict.csv"
GITHUB_CSV   = BASE / "github_features_fixed.csv"
NPM_CSV      = BASE / "npm_downloads.csv"
PYPI_CSV     = BASE / "pypi_downloads.csv"
CMS_CSV      = BASE / "client_market_share.csv"
SO_CSV       = BASE / "stackoverflow_counts.csv"
README_CSV   = BASE / "readme_data.csv"
DEP_JSON     = BASE / "seedReposWithNoTransitiveDependencies.json"

CARGO_DOWNLOADS = {
    "alloy-rs/alloy": 1_643_712, "arkworks-rs/algebra": 10_793_597,
    "lambdaclass/lambdaworks": 463_900, "plonky3/plonky3": 297_255,
    "succinctlabs/sp1": 188_595, "axiom-crypto/snark-verifier": 68_610,
    "supranational/blst": 3_990_315, "offchainlabs/stylus-sdk-rs": 19_196,
}

# ── HYPERPARAMETERS ────────────────────────────────────────────────
TRAIN_WEIGHT    = 0.2    # FIX 1: weight for previous-round juror rows
TRAIN_MULT_CAP  = 100.0  # FIX 2: clip train multipliers (current pairwise already capped)
CONFIDENCE_K    = 10.0   # FIX 3: confidence = obs / (obs + K)
OBS_THRESHOLD   = 4      # FIX 4: min obs to include repo as Ridge training target


# ══════════════════════════════════════════════════════════════════
# FIX 6: ALIAS NORMALIZATION
# Repos that exist under multiple GitHub handles in jury data
# ══════════════════════════════════════════════════════════════════
ALIASES = {
    "offchainlabs/prysm":          "prysmaticlabs/prysm",
    "aestus-relay/mev-boost-relay": "flashbots/mev-boost-relay",
    "ipsilon/evmone":               "ethereum/evmone",
    "ethereum/remix-project":       "remix-project-org/remix-project",
    "hyperledger-web3j/web3j":      "lfdt-web3j/web3j",
}


# ── HELPERS ───────────────────────────────────────────────────────
def norm(url):
    """Normalize a GitHub URL or repo slug to 'owner/repo' lowercase,
    then apply alias resolution so fragmented observations merge."""
    url = str(url).strip().rstrip("/")
    if "github.com/" in url:
        url = url.split("github.com/")[-1]
    url = url.lower()
    return ALIASES.get(url, url)   # FIX 6: alias resolution


def huber_loss(theta, rows, delta=1.0):
    loss = 0.0
    for i, j, sl, w in rows:
        e = theta[i] - theta[j] - sl
        loss += w * (0.5 * e * e if abs(e) <= delta else delta * (abs(e) - 0.5 * delta))
    return loss


# ══════════════════════════════════════════════════════════════════
# STEP 1: LOAD PAIRWISE DATA (current round)
# ══════════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 1: Load human jury data (pairwise + train)")
print("=" * 60)

# ── Current-round pairwise_data.csv ───────────────────────────────
pairwise_raw = pd.read_csv(PAIRWISE_CSV)
print(f"\nCurrent round pairwise_data.csv: {len(pairwise_raw)} rows")
print("Columns:", pairwise_raw.columns.tolist())

pairwise_raw["a"] = pairwise_raw["repo_a"].apply(norm)
pairwise_raw["b"] = pairwise_raw["repo_b"].apply(norm)
pairwise_raw["winner_norm"] = pairwise_raw["winner"].apply(norm)

# Drop 1.1-multiplier bug rows (known data-release bug)
pairwise_clean = pairwise_raw[pairwise_raw["multiplier"] != 1.1].copy()
print(f"  After dropping 1.1-bug rows: {len(pairwise_clean)}")

pairwise_clean["log_ratio"] = np.log(pairwise_clean["multiplier"].clip(1, TRAIN_MULT_CAP))
pairwise_clean["signed_log"] = pairwise_clean.apply(
    lambda r: r["log_ratio"] if r["winner_norm"] == r["a"] else -r["log_ratio"],
    axis=1
)
pairwise_clean["weight"] = 1.0  # full weight for current-round data
pairwise_clean["source"] = "current"

# ── Previous-round train.csv (FIX 1) ──────────────────────────────
train_raw = pd.read_csv(TRAIN_CSV)
print(f"\nPrevious round train.csv: {len(train_raw)} rows")
print("Columns:", train_raw.columns.tolist())
print(f"Train multiplier range: {train_raw['multiplier'].min()} – {train_raw['multiplier'].max()}")

train_raw["a"] = train_raw["repo_a"].apply(norm)
train_raw["b"] = train_raw["repo_b"].apply(norm)

# FIX 2: clip train multipliers to [1, 100] (999x extremes are noise)
train_raw["multiplier_clipped"] = train_raw["multiplier"].clip(1, TRAIN_MULT_CAP)
clipped_count = (train_raw["multiplier"] > TRAIN_MULT_CAP).sum()
print(f"  Clipped {clipped_count} train rows with multiplier > {TRAIN_MULT_CAP}")

# choice=1 means repo_a won, choice=2 means repo_b won
train_raw["winner_norm"] = np.where(
    train_raw["choice"] == 1,
    train_raw["a"],
    train_raw["b"]
)

train_raw["log_ratio"] = np.log(train_raw["multiplier_clipped"])
train_raw["signed_log"] = train_raw.apply(
    lambda r: r["log_ratio"] if r["winner_norm"] == r["a"] else -r["log_ratio"],
    axis=1
)
train_raw["weight"] = TRAIN_WEIGHT  # FIX 1: discounted weight
train_raw["source"] = "train"

print(f"  Train rows after normalization: {len(train_raw)}")

# ── Verify signed_log direction ────────────────────────────────────
print("\n--- Signed log sanity check (current round, first 10 rows) ---")
check_cols = ["a", "b", "winner_norm", "signed_log", "multiplier"]
print(pairwise_clean[[c for c in check_cols if c in pairwise_clean.columns]].head(10).to_string())

pos = (pairwise_clean["signed_log"] > 0).sum()
neg = (pairwise_clean["signed_log"] < 0).sum()
print(f"Positive signed logs: {pos}  |  Negative: {neg}")

# ── Merged dataset for BT ─────────────────────────────────────────
# Select only the columns we need from each
KEEP = ["a", "b", "winner_norm", "signed_log", "weight", "source"]
pairwise_bt = pairwise_clean[KEEP].copy()
train_bt    = train_raw[[c for c in KEEP if c in train_raw.columns]].copy()

merged = pd.concat([pairwise_bt, train_bt], ignore_index=True)
print(f"\nMerged BT dataset: {len(merged)} rows")
print(f"  Current-round: {(merged['source']=='current').sum()}")
print(f"  Train (weighted {TRAIN_WEIGHT}x): {(merged['source']=='train').sum()}")


# ══════════════════════════════════════════════════════════════════
# STEP 2: MERGED HUBER BT
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 2: Merged Huber BT")
print("=" * 60)

# Observation counter from BOTH datasets (each appearance in a pair counts)
obs_counter = Counter()
for _, r in merged.iterrows():
    obs_counter[r["a"]] += 1
    obs_counter[r["b"]] += 1

h_repos = sorted(set(merged["a"]) | set(merged["b"]))
h_ridx  = {r: i for i, r in enumerate(h_repos)}

h_rows = [
    (h_ridx[r["a"]], h_ridx[r["b"]], float(r["signed_log"]), float(r["weight"]))
    for _, r in merged.iterrows()
]

res = minimize(
    huber_loss, np.zeros(len(h_repos)), args=(h_rows,),
    method="L-BFGS-B", options={"maxiter": 5000, "ftol": 1e-13}
)
theta_raw = res.x - res.x.mean()
merged_human_theta = dict(zip(h_repos, theta_raw))

# FIX 3: Confidence-aware theta
# Repos with few observations are shrunk toward 0 before any downstream use
def confidence(repo):
    return obs_counter.get(repo, 0) / (obs_counter.get(repo, 0) + CONFIDENCE_K)

confident_theta = {
    r: merged_human_theta[r] * confidence(r)
    for r in merged_human_theta
}

print(f"\nMerged BT covers {len(merged_human_theta)} repos")

# ── FIX 7: Permanent diagnostics ──────────────────────────────────
print("\n" + "=" * 70)
print("DIAGNOSTIC: SPARSE HIGH-SCORING REPOS (obs <= 3 but high BT)")
print("=" * 70)
sparse_high = [
    (repo, score, obs_counter.get(repo, 0))
    for repo, score in merged_human_theta.items()
    if obs_counter.get(repo, 0) <= 3
]
sparse_high.sort(key=lambda x: -x[1])
for repo, score, obs in sparse_high[:15]:
    conf = confidence(repo)
    print(f"  {repo:55s}  theta={score:+.3f}  obs={obs}  conf={conf:.2f}  conf_theta={score*conf:+.3f}")

print("\n" + "=" * 70)
print("DIAGNOSTIC: BT CONFIDENCE SCORES (TOP 25 MERGED BT)")
print("=" * 70)
print(f"  {'Repo':55s}  raw_theta  obs   conf  conf_theta")
for repo, score in sorted(merged_human_theta.items(), key=lambda x: -x[1])[:25]:
    obs  = obs_counter.get(repo, 0)
    conf = confidence(repo)
    print(f"  {repo:55s}  {score:+.4f}  {obs:3d}  {conf:.3f}  {score*conf:+.4f}")

print("\n" + "=" * 70)
print("DIAGNOSTIC: BIGGEST PAIRWISE vs TRAIN DISAGREEMENTS")
print("=" * 70)
# Only compare repos present in BOTH datasets
pairwise_repos = set(pairwise_bt["a"]) | set(pairwise_bt["b"])
train_repos     = set(train_bt["a"]) | set(train_bt["b"])
overlap_repos   = pairwise_repos & train_repos

# Build separate BT from pairwise only for comparison
p_repos = sorted(pairwise_repos)
p_ridx  = {r: i for i, r in enumerate(p_repos)}
p_rows  = [
    (p_ridx[r["a"]], p_ridx[r["b"]], float(r["signed_log"]), 1.0)
    for _, r in pairwise_bt.iterrows()
    if r["a"] in p_ridx and r["b"] in p_ridx
]
res_p = minimize(
    huber_loss, np.zeros(len(p_repos)), args=(p_rows,),
    method="L-BFGS-B", options={"maxiter": 3000, "ftol": 1e-12}
)
pairwise_only_theta = dict(zip(p_repos, res_p.x - res_p.x.mean()))

# Build separate BT from train only
t_repos = sorted(train_repos)
t_ridx  = {r: i for i, r in enumerate(t_repos)}
t_rows  = [
    (t_ridx[r["a"]], t_ridx[r["b"]], float(r["signed_log"]), 1.0)
    for _, r in train_bt.iterrows()
    if r["a"] in t_ridx and r["b"] in t_ridx
]
res_t = minimize(
    huber_loss, np.zeros(len(t_repos)), args=(t_rows,),
    method="L-BFGS-B", options={"maxiter": 3000, "ftol": 1e-12}
)
train_only_theta = dict(zip(t_repos, res_t.x - res_t.x.mean()))

# Show biggest disagreements
disagreements = []
for r in overlap_repos:
    p_s = pairwise_only_theta.get(r, 0.0)
    t_s = train_only_theta.get(r, 0.0)
    disagreements.append((r, p_s, t_s, abs(p_s - t_s)))

disagreements.sort(key=lambda x: -x[3])
print(f"  {'Repo':55s}  pairwise  train   |diff|  obs")
for repo, p_s, t_s, diff in disagreements[:15]:
    obs = obs_counter.get(repo, 0)
    print(f"  {repo:55s}  {p_s:+.3f}   {t_s:+.3f}  {diff:.3f}  {obs}")


# ══════════════════════════════════════════════════════════════════
# STEP 3: LLM BT — calibrate to MERGED human scale (FIX 5)
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 3: LLM BT calibration to merged human scale")
print("=" * 60)

llm_main = pd.read_csv(LLM_CSV)
llm_main["a"] = llm_main["repo_a"].apply(norm)
llm_main["b"] = llm_main["repo_b"].apply(norm)

# Filter main LLM rows by agreement before computing signed_log
llm_main_f = llm_main[llm_main["agreement"].isin(["full", "strong", "majority"])].copy()
llm_main_f["log_ratio"]  = np.log(llm_main_f["multiplier"].clip(1, 1000))
llm_main_f["signed_log"] = llm_main_f.apply(
    lambda r: r["log_ratio"] if r["choice"] == 1 else -r["log_ratio"], axis=1
)

# ── Load grok fix file and normalise independently ─────────────────
# Grok rows use a 'winner' URL column instead of choice=1/2.
# We resolve signed_log here directly from winner, before any concat,
# so there is no column schema mismatch.
llm_fix_raw = pd.read_csv(LLM_FIX_CSV)
llm_fix_raw["a"] = llm_fix_raw["repo_a"].apply(norm)
llm_fix_raw["b"] = llm_fix_raw["repo_b"].apply(norm)
llm_fix_raw["winner_norm"] = llm_fix_raw["winner"].apply(norm)
llm_fix_raw["log_ratio"]   = np.log(llm_fix_raw["multiplier"].clip(1, 1000))
# signed_log: positive if repo_a won, negative if repo_b won
llm_fix_raw["signed_log"]  = llm_fix_raw.apply(
    lambda r: r["log_ratio"] if r["winner_norm"] == r["a"] else -r["log_ratio"], axis=1
)

# Diagnostics: what did the grok file add?
main_repos = set(llm_main_f["a"]) | set(llm_main_f["b"])
fix_repos  = set(llm_fix_raw["a"]) | set(llm_fix_raw["b"])
new_repos  = fix_repos - main_repos

print(f"\n  Original LLM rows  : {len(llm_main_f)}")
print(f"  Grok rows          : {len(llm_fix_raw)}")
print(f"  Combined LLM rows  : {len(llm_main_f) + len(llm_fix_raw)}")
print(f"  New repos introduced by Grok: {sorted(new_repos) if new_repos else 'none'}")
print(f"  Grok pairs:")
for _, r in llm_fix_raw.iterrows():
    winner_label = r["a"] if r["signed_log"] > 0 else r["b"]
    print(f"    {r['a']:45s} vs {r['b']:45s}  winner={winner_label}  mult={r['multiplier']}  signed_log={r['signed_log']:+.3f}")

# Merge on the minimal common columns — no schema conflict possible
KEEP_LLM = ["a", "b", "signed_log", "multiplier"]
llm = pd.concat(
    [llm_main_f[KEEP_LLM], llm_fix_raw[KEEP_LLM]],
    ignore_index=True
)

# Sanity check: no NaNs in signed_log
nan_count = llm["signed_log"].isna().sum()
if nan_count:
    print(f"  WARNING: {nan_count} NaN values in signed_log after merge — dropping")
    llm = llm.dropna(subset=["signed_log"])

# BT on LLM data for all 98 repos
l_repos = sorted(set(llm["a"]) | set(llm["b"]))
l_ridx  = {r: i for i, r in enumerate(l_repos)}
l_rows  = [
    (l_ridx[r["a"]], l_ridx[r["b"]], float(r["signed_log"]), 1.0)
    for _, r in llm.iterrows()
]
resl = minimize(
    huber_loss, np.zeros(len(l_repos)), args=(l_rows,),
    method="L-BFGS-B", options={"maxiter": 5000, "ftol": 1e-13}
)
llm_theta_raw = dict(zip(l_repos, resl.x - resl.x.mean()))

print("\n" + "=" * 70)
print("RAW LLM BRADLEY-TERRY TOP 25")
print("=" * 70)
for rank, (repo, score) in enumerate(
    sorted(llm_theta_raw.items(), key=lambda x: -x[1])[:25], 1
):
    print(f"{rank:2d}. {repo:55s} {score:+.4f}")


l_repos_before_grok = sorted(set(llm_main_f["a"]) | set(llm_main_f["b"]))
l_ridx_before_grok  = {r: i for i, r in enumerate(l_repos_before_grok)}
l_rows_before_grok  = [
    (l_ridx_before_grok[r["a"]], l_ridx_before_grok[r["b"]], float(r["signed_log"]), 1.0)
    for _, r in llm_main_f.iterrows()
]
res_l_before_grok = minimize(
    huber_loss, np.zeros(len(l_repos_before_grok)), args=(l_rows_before_grok,),
    method="L-BFGS-B", options={"maxiter": 5000, "ftol": 1e-13}
)
llm_theta_raw_before_grok = dict(zip(
    l_repos_before_grok, res_l_before_grok.x - res_l_before_grok.x.mean()
))
print(f"\n  [chart-only] LLM BT before Grok merge computed over {len(l_repos_before_grok)} repos")

# FIX 5: Calibrate LLM to MERGED human BT (not sparse pairwise-only)
# Use confident_theta as anchor targets so sparse repos don't bias the fit
anchors   = [r for r in h_repos if r in llm_theta_raw]
h_anchor  = np.array([confident_theta[r] for r in anchors])   # FIX 5
l_anchor  = np.array([llm_theta_raw[r]   for r in anchors])

from scipy.stats import theilslopes
slope, intercept, _, _ = theilslopes(h_anchor, l_anchor)
calib_corr = np.corrcoef(h_anchor, l_anchor)[0, 1]
print(f"\n  LLM-vs-merged-human correlation on {len(anchors)} anchors: {calib_corr:.3f}")
print(f"  Calibration: human_theta = {slope:.3f} * llm_theta + {intercept:.3f}")

llm_theta_calibrated = {
    r: slope * s + intercept
    for r, s in llm_theta_raw.items()
}
print(f"  LLM BT covers {len(llm_theta_calibrated)} repos")

# Grok integration diagnostics for the three target sparse repos
print("\n" + "=" * 70)
print("GROK INTEGRATION EFFECT — SPARSE REPO DIAGNOSTICS")
print("  (raw_llm_theta and calibrated_llm_theta after Grok merge)")
print("=" * 70)
print(f"  {'Repo':45s}  raw_llm_theta  calib_llm_theta  human_conf_theta  obs")
for repo in ["cyfrin/aderyn", "risc0/risc0-ethereum", "supranational/blst"]:
    raw_l  = llm_theta_raw.get(repo, float("nan"))
    cal_l  = llm_theta_calibrated.get(repo, float("nan"))
    h_conf = confident_theta.get(repo, 0.0)
    obs    = obs_counter.get(repo, 0)
    print(f"  {repo:45s}  {raw_l:+.4f}         {cal_l:+.4f}           {h_conf:+.4f}         {obs}")

print("\n" + "=" * 70)
print("HUMAN (merged+conf) VS LLM (calibrated)")
print("=" * 70)
for repo in [
    "ethereum/go-ethereum", "vyperlang/vyper", "wevm/viem",
    "ethereum/eips", "ethereum/consensus-specs", "sigp/lighthouse",
]:
    print(
        f"{repo:55s}  "
        f"conf_theta={confident_theta.get(repo, 0.0):+.3f}  "
        f"llm_cal={llm_theta_calibrated.get(repo, 0.0):+.3f}"
    )


# ══════════════════════════════════════════════════════════════════
# STEP 4: FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 4: Feature engineering")
print("=" * 60)

target   = pd.read_csv(REPOS_CSV)
target["rk"] = target["repo"].apply(norm)
all_repos = list(target["rk"])
print(f"Target repos: {len(all_repos)}  Unique: {len(set(all_repos))}")
if len(all_repos) != len(set(all_repos)):
    print("WARNING: DUPLICATE REPOS IN REPOS_CSV")

# Load feature tables
gh   = pd.read_csv(GITHUB_CSV);  gh["rk"]  = gh["repo"].apply(norm)
npm  = pd.read_csv(NPM_CSV);     npm["rk"] = npm["repo"].apply(norm)
pypi = pd.read_csv(PYPI_CSV);    pypi["rk"]= pypi["repo"].apply(norm)
cms  = pd.read_csv(CMS_CSV);     cms["rk"] = cms["repo"].apply(norm)
so   = pd.read_csv(SO_CSV);      so["rk"]  = so["repo"].apply(norm)

gh_idx  = gh.set_index("rk")
npm_map = dict(zip(npm["rk"],  npm["monthly_downloads"]))
pyp_map = dict(zip(pypi["rk"], pypi["monthly_downloads"]))
so_map  = dict(zip(so["rk"],   so["total_questions"]))
cms_map = dict(zip(cms["rk"],  cms["client_market_share_pct"]))

# Dependency graph features
with open(DEP_JSON) as f:
    dep_data = json.load(f)

all_set = set(all_repos)
G = nx.DiGraph()
for url, deps in dep_data.items():
    src = norm(url)
    if src not in all_set:
        continue
    for d in deps:
        G.add_edge(src, norm(d))

rev_dep_count = Counter()
for url, deps in dep_data.items():
    for d in deps:
        dn = norm(d)
        if dn in all_set:
            rev_dep_count[dn] += 1

for r in all_repos:
    G.add_node(r)

pr = nx.pagerank(G, alpha=0.85, max_iter=200)
print(f"  Dep graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
print(f"  Isolated nodes: {len(list(nx.isolates(G)))}")


def get_features(r):
    g = gh_idx.loc[r] if r in gh_idx.index else None
    def gv(col, d=0.0):
        try:
            return float(g[col]) if g is not None else d
        except Exception:
            return d

    stars      = max(gv("stars"), 1)
    forks      = max(gv("forks"), 1)
    contribs   = max(gv("contributors_count"), 1)
    commits12m = max(gv("commits_last_12m"), 0)
    commit_freq= max(gv("commit_frequency"), 0)
    releases   = max(gv("releases_count"), 0)
    closed_r   = gv("closed_issues_ratio", 0.5)
    net_deps   = max(gv("network_dependents"), 0)
    days_last  = min(gv("days_since_last_commit", 365), 1825)

    npm_dl   = npm_map.get(r, 0)
    pypi_dl  = pyp_map.get(r, 0)
    cargo_dl = CARGO_DOWNLOADS.get(r, 0)
    so_q     = so_map.get(r, 0)
    mkt      = cms_map.get(r, 0.0)

    rdc          = rev_dep_count.get(r, 0)
    pagerank_val = pr.get(r, 1e-6)
    llm_calib    = llm_theta_calibrated.get(r, 0.0)

    return {
        "log_stars":              np.log1p(stars),
        "log_forks":              np.log1p(forks),
        "log_contributors":       np.log1p(contribs),
        "log_commits12m":         np.log1p(commits12m),
        "commit_frequency":       commit_freq,
        "log_releases":           np.log1p(releases),
        "closed_issues_ratio":    closed_r,
        "log_network_dependents": np.log1p(net_deps),
        "log_days_stale":         np.log1p(days_last),
        "log_npm":                np.log1p(npm_dl),
        "log_pypi":               np.log1p(pypi_dl),
        "log_cargo":              np.log1p(cargo_dl),
        "has_package":            float((npm_dl + pypi_dl + cargo_dl) > 0),
        "client_market_share":    mkt,
        "stackoverflow":          np.log1p(so_q),
        "rev_dep_count":          float(rdc),
        "log_rev_dep":            np.log1p(rdc),
        "pagerank":               np.log1p(pagerank_val * 1e4),
        "llm_bt_calibrated":      llm_calib,
    }


feat_names = list(get_features(all_repos[0]).keys())
print(f"  Features: {len(feat_names)}")
print(f"  {feat_names}")

X_all = np.array([list(get_features(r).values()) for r in all_repos])
repo_to_idx = {r: i for i, r in enumerate(all_repos)}

print("\n--- Feature matrix stats ---")
for i, f in enumerate(feat_names):
    col = X_all[:, i]
    print(f"  {f:30s}  mean={np.mean(col):.3f}  std={np.std(col):.3f}  "
          f"min={np.min(col):.3f}  max={np.max(col):.3f}")


# ══════════════════════════════════════════════════════════════════
# STEP 5: RIDGE REGRESSION — FIX 4: train only on high-confidence repos
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(f"STEP 5: RidgeCV training (obs >= {OBS_THRESHOLD} only)")
print("=" * 60)

# FIX 4: Only use repos with enough observations as Ridge targets.
# This prevents Vyper (obs=3), Viem (obs=2), Cyfrin (obs=1) from
# contaminating the learned mapping from features → BT.
covered_idx = [
    i for i, r in enumerate(all_repos)
    if r in confident_theta and obs_counter.get(r, 0) >= OBS_THRESHOLD
]
covered_repos = [all_repos[i] for i in covered_idx]

excluded_from_ridge = [
    (r, obs_counter.get(r, 0), confident_theta.get(r, 0.0))
    for r in all_repos
    if r in confident_theta and obs_counter.get(r, 0) < OBS_THRESHOLD
]
excluded_from_ridge.sort(key=lambda x: -x[2])

print(f"\n  Ridge training repos (obs >= {OBS_THRESHOLD}): {len(covered_repos)}")
print(f"  Excluded (too sparse): {len(excluded_from_ridge)}")
if excluded_from_ridge:
    print("  --- Excluded sparse repos (would have contaminated Ridge) ---")
    for repo, obs, ct in excluded_from_ridge:
        raw = merged_human_theta.get(repo, 0.0)
        print(f"    {repo:55s}  obs={obs}  raw_bt={raw:+.3f}  conf_theta={ct:+.3f}")

X_train = X_all[covered_idx]
y_train = np.array([confident_theta[r] for r in covered_repos])  # use confident_theta

scaler   = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_all_s   = scaler.transform(X_all)

alphas = np.logspace(-3, 2, 50)   # extended to 100 — previous ceiling of 10 was hit exactly
ridge  = RidgeCV(alphas=alphas, cv=LeaveOneOut(), scoring="neg_mean_absolute_error")
ridge.fit(X_train_s, y_train)

print(f"  Best alpha: {ridge.alpha_:.4f}")
if ridge.alpha_ >= alphas[-1] * 0.95:
    print(f"  WARNING: Ridge alpha at or near ceiling ({alphas[-1]:.1f}) — distribution will be flat. Consider wider alpha range.")
elif ridge.alpha_ <= alphas[0] * 1.05:
    print(f"  WARNING: Ridge alpha at or near floor ({alphas[0]:.4f}) — possible overfitting.")
else:
    print(f"  Ridge alpha comfortably inside range [{alphas[0]:.4f}, {alphas[-1]:.1f}] — good.")

# Alpha curve: show LOO MAE across a sample of alphas to see where the optimum sits
print("\n  Alpha curve (LOO MAE at key alpha values):")
from sklearn.linear_model import Ridge as RidgePlain
from sklearn.model_selection import cross_val_score
sample_alphas = np.logspace(-3, 2, 20)
for a in sample_alphas:
    scores = cross_val_score(
        RidgePlain(alpha=a), X_train_s, y_train,
        cv=LeaveOneOut(), scoring="neg_mean_absolute_error"
    )
    print(f"    alpha={a:8.4f}  LOO MAE={-scores.mean():.4f}"
          f"{'  ← CHOSEN' if abs(a - ridge.alpha_) / ridge.alpha_ < 0.01 else ''}")

loo_preds = cross_val_predict(
    RidgeCV(alphas=[ridge.alpha_]), X_train_s, y_train, cv=LeaveOneOut()
)
loo_mae  = np.mean(np.abs(loo_preds - y_train))
loo_corr = np.corrcoef(loo_preds, y_train)[0, 1]
print(f"  LOO CV MAE: {loo_mae:.4f}")
print(f"  LOO CV Pearson R: {loo_corr:.3f}")

coef_sorted = sorted(zip(feat_names, ridge.coef_), key=lambda x: -abs(x[1]))
print("  Top feature importances:")
for fname, coef in coef_sorted[:10]:
    print(f"    {fname:30s}  {coef:+.4f}")

feature_theta = dict(zip(all_repos, ridge.predict(X_all_s)))

print("\n" + "=" * 70)
print("FEATURE PRIOR TOP 25")
print("=" * 70)
for rank, (repo, score) in enumerate(
    sorted(feature_theta.items(), key=lambda x: -x[1])[:25], 1
):
    print(f"{rank:2d}. {repo:55s} {score:+.4f}")

# Change 6: Feature leakage audit — top 20 by feature_theta only
# Reveals repos being promoted entirely by features, not human signal
print("\n" + "=" * 70)
print("Change 6 — FEATURE LEAKAGE AUDIT: TOP 20 BY FEATURE_THETA ONLY")
print("  Repos high here but low in human BT are being feature-promoted")
print("=" * 70)
print(f"  {'Repo':55s}  feat_theta  conf_theta  obs  gap(feat-conf)")
for rank, (repo, feat) in enumerate(
    sorted(feature_theta.items(), key=lambda x: -x[1])[:20], 1
):
    conf = confident_theta.get(repo, 0.0)
    obs  = obs_counter.get(repo, 0)
    gap  = feat - conf
    flag = "  ← FEATURE-PROMOTED" if gap > 0.5 and obs <= 3 else ""
    print(f"  {rank:2d}. {repo:55s}  {feat:+.4f}      {conf:+.4f}  {obs:3d}  {gap:+.4f}{flag}")

print("\n--- Feature prior vs confident BT for key repos ---")
for repo in [
    "vyperlang/vyper", "wevm/viem", "cyfrin/aderyn",
    "risc0/risc0-ethereum", "ethereum/go-ethereum", "argotorg/solidity",
]:
    raw  = merged_human_theta.get(repo, 0.0)
    conf = confident_theta.get(repo, 0.0)
    feat = feature_theta.get(repo, 0.0)
    obs  = obs_counter.get(repo, 0)
    print(f"  {repo:55s}  obs={obs}  raw_bt={raw:+.3f}  conf_theta={conf:+.3f}  feature={feat:+.3f}")



# ══════════════════════════════════════════════════════════════════
# Change 2 — VARIANCE DIAGNOSTICS
# Tells you if Ridge is flattening the distribution
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Change 2 — VARIANCE DIAGNOSTICS (std of each theta layer)")
print("  Low std = flat distribution = over-compressed weights")
print("=" * 70)
print(f"  merged_human_theta std : {np.std(list(merged_human_theta.values())):.4f}")
print(f"  confident_theta std    : {np.std(list(confident_theta.values())):.4f}")
print(f"  feature_theta std      : {np.std(list(feature_theta.values())):.4f}")
print(f"  (final theta std printed inside compute_final_weights below)")

# ══════════════════════════════════════════════════════════════════
# STEP 6: DYNAMIC SHRINKAGE + FINAL THETA
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 6: Dynamic shrinkage + final weights")
print("=" * 60)

# REPORTING ONLY: cache of per-lambda final theta dicts (pre-softmax),
# keyed by lam. Populated below inside compute_final_weights without
# changing that function's return value or behavior. Used only by the
# double_shrinkage_analysis.png chart at the end of the script.
_FINAL_THETA_CACHE = {}

def compute_final_weights(all_repos, confident_theta, feature_theta,
                          obs_counter, lam, diag=False):
    """
    final_theta[i] = shrink(i) * conf_theta[i] + (1 - shrink(i)) * feature_prior[i]
    shrink(i) = obs_count / (obs_count + lambda)

    Note: conf_theta already embeds confidence = obs/(obs+10), so sparse repos
    get double-shrunk: once by confidence, again by lambda shrinkage.

    lambda=3:  1 obs → 25% BT, 75% prior  |  5 obs → 63%  |  10 obs → 77%
    lambda=5:  1 obs → 17% BT, 83% prior  |  5 obs → 50%  |  10 obs → 67%
    lambda=10: 1 obs → 9%  BT, 91% prior  |  5 obs → 33%  |  10 obs → 50%
    """
    final = {}
    for r in all_repos:
        n    = obs_counter.get(r, 0)
        shrk = n / (n + lam)
        hbt  = confident_theta.get(r, 0.0)
        fpr  = feature_theta.get(r, 0.0)
        final[r] = shrk * hbt + (1.0 - shrk) * fpr

    # REPORTING ONLY: stash this lambda's final theta dict for chart use.
    # Does not affect weight_dict, which is still computed identically below.
    _FINAL_THETA_CACHE[lam] = dict(final)

    thetas = np.array([final[r] for r in all_repos])
    thetas -= thetas.max()
    exp_t   = np.exp(thetas)
    weights = exp_t / exp_t.sum()
    weight_dict = dict(zip(all_repos, weights))

    if diag:
        print(f"\n  λ={lam}: weight distribution")
        sorted_w = sorted(weight_dict.items(), key=lambda x: -x[1])
        print(f"  {'Repo':55s}  weight%  theta   obs  shrk%  source")
        for repo, w in sorted_w[:20]:
            n    = obs_counter.get(repo, 0)
            shrk = n / (n + lam)
            src  = "HUMAN" if n >= OBS_THRESHOLD else ("sparse" if n >= 1 else "BLIND")
            print(f"  {repo:55s}  {100*w:5.2f}%  "
                  f"{final[repo]:+.2f}  {n:3d}  {100*shrk:4.0f}%  {src}")

        print(f"\n  Shrinkage diagnostics λ={lam}")
        for repo in ["ethereum/go-ethereum", "vyperlang/vyper", "wevm/viem",
                     "cyfrin/aderyn", "argotorg/solidity"]:
            n    = obs_counter.get(repo, 0)
            shrk = n / (n + lam)
            print(
                f"  {repo:55s}  obs={n}  shrink={shrk:.3f}  "
                f"conf_theta={confident_theta.get(repo, 0.0):+.3f}  "
                f"feature={feature_theta.get(repo, 0.0):+.3f}  "
                f"final={final.get(repo, 0.0):+.3f}"
            )

        # Change 1: Feature vs human contribution diagnostics for key sparse repos
        print(f"\n  Change 1 — HUMAN vs FEATURE CONTRIBUTION (λ={lam})")
        print(f"  {'Repo':55s}  obs  shrk%  human_contrib  feat_contrib  final_theta")
        for repo in ["vyperlang/vyper", "wevm/viem", "cyfrin/aderyn", "risc0/risc0-ethereum"]:
            n    = obs_counter.get(repo, 0)
            shrk = n / (n + lam)
            hbt  = confident_theta.get(repo, 0.0)
            fpr  = feature_theta.get(repo, 0.0)
            human_contrib   = shrk * hbt
            feature_contrib = (1.0 - shrk) * fpr
            ft              = final.get(repo, 0.0)
            dom = "BT-dominated" if abs(human_contrib) > abs(feature_contrib) else "FEAT-dominated"
            print(
                f"  {repo:55s}  {n:3d}  {100*shrk:4.0f}%  "
                f"{human_contrib:+.4f}         {feature_contrib:+.4f}        "
                f"{ft:+.4f}  [{dom}]"
            )

        # Change 2 (final theta std): printed here so it's inside diag context
        print(f"\n  Change 2 — final_theta std (λ={lam}): {np.std(list(final.values())):.4f}")

        # Change 5: Sparse repo audit — all repos with obs <= 3
        print(f"\n  Change 5 — SPARSE REPO AUDIT (obs <= 3, λ={lam})")
        print(f"  {'Repo':55s}  obs  raw_bt    conf_theta  feat_theta  final_theta")
        sparse_audit = [
            (r, obs_counter.get(r, 0), merged_human_theta.get(r, 0.0),
             confident_theta.get(r, 0.0), feature_theta.get(r, 0.0), final.get(r, 0.0))
            for r in all_repos
            if obs_counter.get(r, 0) <= 3
        ]
        sparse_audit.sort(key=lambda x: -x[5])  # sort by final_theta descending
        for repo, obs, raw_bt, conf_t, feat_t, final_t in sparse_audit:
            flag = "  ← HIGH" if final_t > 0 else ""
            print(
                f"  {repo:55s}  {obs:3d}  {raw_bt:+.4f}    {conf_t:+.4f}      "
                f"{feat_t:+.4f}      {final_t:+.4f}{flag}"
            )

    return weight_dict


print("\nλ=5 (recommended):")
w5  = compute_final_weights(all_repos, confident_theta, feature_theta, obs_counter, lam=5,  diag=True)
w3  = compute_final_weights(all_repos, confident_theta, feature_theta, obs_counter, lam=3)
w10 = compute_final_weights(all_repos, confident_theta, feature_theta, obs_counter, lam=10)

w_ens = {r: (w3[r] + w5[r] + w10[r]) / 3.0 for r in all_repos}
total = sum(w_ens.values())
w_ens = {r: v / total for r, v in w_ens.items()}


# ══════════════════════════════════════════════════════════════════
# Change 2 — NO-CONFIDENCE-SHRINKAGE EXPERIMENT
# Current pipeline: conf_theta = raw_theta * obs/(obs+10)  (pre-shrunk)
#                   final = shrink(lam) * conf_theta + (1-shrink) * feature
# This experiment:  raw_theta passed directly, no pre-shrinkage
#                   final = shrink(lam) * raw_theta + (1-shrink) * feature
# Purpose: isolate whether double-shrinkage is over-compressing sparse repos.
# Outputs: submission_no_conf_lam5.csv — DO NOT submit without evaluating first.
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Change 2 — NO-CONFIDENCE-SHRINKAGE EXPERIMENT (raw_theta, λ=5)")
print("  Uses raw merged_human_theta directly — no obs/(obs+10) pre-shrinkage")
print("  Compare spread vs main ensemble to decide if double-shrinkage hurts")
print("=" * 70)

w_no_conf = compute_final_weights(
    all_repos, merged_human_theta, feature_theta, obs_counter, lam=5, diag=False
)
ws_nc = np.array([w_no_conf[r] for r in all_repos])
ws_main = np.array([w5[r] for r in all_repos])
print(f"  No-conf λ=5:  max={100*ws_nc.max():.2f}%  min={100*ws_nc.min():.4f}%  "
      f"ratio={ws_nc.max()/ws_nc.min():.1f}x  std_theta={np.std([w_no_conf[r] for r in all_repos]):.4f}")
print(f"  Main   λ=5:  max={100*ws_main.max():.2f}%  min={100*ws_main.min():.4f}%  "
      f"ratio={ws_main.max()/ws_main.min():.1f}x  std_theta={np.std([w5[r] for r in all_repos]):.4f}")

print("\n  Key sparse repos — no-conf vs main:")
print(f"  {'Repo':45s}  no_conf_w%  main_w%   delta")
for repo in ["cyfrin/aderyn", "risc0/risc0-ethereum", "supranational/blst",
             "vyperlang/vyper", "wevm/viem", "ethereum/go-ethereum"]:
    nc = 100 * w_no_conf.get(repo, 0)
    m  = 100 * w5.get(repo, 0)
    print(f"  {repo:45s}  {nc:6.3f}%     {m:6.3f}%   {nc-m:+.3f}%")


# ══════════════════════════════════════════════════════════════════
# Change 3 — TEMPERATURE SCALING EXPERIMENT
# Applies a sharpening multiplier T to final thetas before softmax.
# T=1.0 = no change (identical to current ensemble)
# T=1.5 = moderate sharpening
# T=2.0 = strong sharpening
# Purpose: test whether compressed distribution hurts MAE vs jury.
# Outputs: submission_temp15.csv, submission_temp20.csv
# DO NOT submit without comparing leaderboard scores first.
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Change 3 — TEMPERATURE SCALING EXPERIMENT (T=1.5, T=2.0)")
print("  Applied to ensemble thetas — no model retraining")
print("=" * 70)

def apply_temperature(weight_dict_base, all_repos, T):
    """Re-derive weights from the ensemble by reversing softmax, scaling, re-applying."""
    # Recover thetas from weights: theta_i = log(w_i) + const (const cancels in softmax)
    ws = np.array([weight_dict_base[r] for r in all_repos])
    log_w = np.log(ws)
    log_w -= log_w.mean()         # centre
    scaled = log_w * T            # temperature scaling
    scaled -= scaled.max()
    exp_s = np.exp(scaled)
    new_ws = exp_s / exp_s.sum()
    return dict(zip(all_repos, new_ws))

w_temp15 = apply_temperature(w_ens, all_repos, T=1.5)
w_temp20 = apply_temperature(w_ens, all_repos, T=2.0)

for label, wd in [("T=1.0 (ensemble)", w_ens), ("T=1.5", w_temp15), ("T=2.0", w_temp20)]:
    ws = np.array([wd[r] for r in all_repos])
    print(f"  {label:20s}  max={100*ws.max():.2f}%  min={100*ws.min():.4f}%  "
          f"ratio={ws.max()/ws.min():.1f}x")


# ══════════════════════════════════════════════════════════════════
# Change 1/2/3 — NO-CONFIDENCE ENSEMBLE (full λ=3/5/10) + T=1.15 VARIANT
# Extends the "no-conf" experiment above (which only built λ=5) into a
# full ensemble mirroring how w3/w5/w10 → w_ens is built, but blending
# merged_human_theta directly (no obs/(obs+10) pre-shrinkage) through the
# SAME compute_final_weights() function — λ shrinkage, feature prior,
# Ridge, Grok integration, calibration, and alias normalization are all
# untouched. Only the "hbt" input to the final blend changes.
# Outputs: submission_no_conf_ensemble.csv, submission_no_conf_temp115.csv
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Change 1/2/3 — NO-CONFIDENCE ENSEMBLE + T=1.15 VARIANT")
print("  hbt = merged_human_theta.get(r, 0.0)  (confidence pre-shrinkage removed)")
print("  Lambda shrinkage, feature prior, Ridge, Grok, calibration all unchanged")
print("=" * 70)

# w_no_conf (λ=5) was already computed in the experiment block above.
# Add λ=3 and λ=10 so the no-conf variant gets a real 3-way ensemble,
# exactly mirroring how w3/w5/w10 are combined into w_ens.
w_no_conf_3 = compute_final_weights(
    all_repos, merged_human_theta, feature_theta, obs_counter, lam=3, diag=False
)
w_no_conf_10 = compute_final_weights(
    all_repos, merged_human_theta, feature_theta, obs_counter, lam=10, diag=False
)

w_no_conf_ens = {
    r: (w_no_conf_3[r] + w_no_conf[r] + w_no_conf_10[r]) / 3.0
    for r in all_repos
}
_nc_total = sum(w_no_conf_ens.values())
w_no_conf_ens = {r: v / _nc_total for r, v in w_no_conf_ens.items()}

print(f"\n  No-conf ensemble built from λ=3, λ=5 (reused), λ=10 — {len(w_no_conf_ens)} repos")

# T=1.15 sharpening applied to the no-conf ensemble (mild — per spec, NOT 1.5/2.0).
# Equivalent to: thetas *= 1.15; weights = softmax(thetas).
# Reuses apply_temperature(), which recovers centred thetas from weights via
# log(w) (valid since log(softmax(theta)) == theta - const), so this matches
# scaling the underlying final_theta values directly before softmax.
w_no_conf_temp115 = apply_temperature(w_no_conf_ens, all_repos, T=1.15)

_ws_nc_ens  = np.array([w_no_conf_ens[r] for r in all_repos])
_ws_nc_t115 = np.array([w_no_conf_temp115[r] for r in all_repos])
print(f"  no_conf_ens           max={100*_ws_nc_ens.max():.2f}%  min={100*_ws_nc_ens.min():.4f}%  "
      f"ratio={_ws_nc_ens.max()/_ws_nc_ens.min():.1f}x")
print(f"  no_conf_ens (T=1.15)  max={100*_ws_nc_t115.max():.2f}%  min={100*_ws_nc_t115.min():.4f}%  "
      f"ratio={_ws_nc_t115.max()/_ws_nc_t115.min():.1f}x")


# ══════════════════════════════════════════════════════════════════
# Change 5 — CURRENT ENSEMBLE vs NO-CONFIDENCE ENSEMBLE (side-by-side)
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Change 5 — CURRENT ENSEMBLE vs NO-CONFIDENCE ENSEMBLE")
print("=" * 70)

def _spread_stats(weight_dict, all_repos):
    ws = np.array([weight_dict[r] for r in all_repos])
    return ws.max(), ws.min(), ws.max() / ws.min()

_cur_max, _cur_min, _cur_ratio = _spread_stats(w_ens, all_repos)
_nc_max,  _nc_min,  _nc_ratio  = _spread_stats(w_no_conf_ens, all_repos)

print(f"\n  {'Version':28s}  max%       min%         spread_ratio")
print(f"  {'submission_ensemble':28s}  {100*_cur_max:6.2f}%    {100*_cur_min:.4f}%      {_cur_ratio:6.1f}x")
print(f"  {'submission_no_conf_ens':28s}  {100*_nc_max:6.2f}%    {100*_nc_min:.4f}%      {_nc_ratio:6.1f}x")

print(f"\n  Top 10 — submission_ensemble (current):")
for rank, (r, w) in enumerate(sorted(w_ens.items(), key=lambda x: -x[1])[:10], 1):
    print(f"    #{rank:2d}  {r:55s}  {100*w:5.2f}%")

print(f"\n  Top 10 — submission_no_conf_ensemble:")
for rank, (r, w) in enumerate(sorted(w_no_conf_ens.items(), key=lambda x: -x[1])[:10], 1):
    print(f"    #{rank:2d}  {r:55s}  {100*w:5.2f}%")

print(f"\n  Key sparse-repo weights, current vs no-conf:")
print(f"  {'Repo':45s}  current%   no_conf%   delta")
for repo in ["cyfrin/aderyn", "risc0/risc0-ethereum", "vyperlang/vyper", "wevm/viem"]:
    _cw = 100 * w_ens.get(repo, 0.0)
    _nw = 100 * w_no_conf_ens.get(repo, 0.0)
    print(f"  {repo:45s}  {_cw:6.3f}%    {_nw:6.3f}%    {_nw-_cw:+.3f}%")


# ══════════════════════════════════════════════════════════════════
# STEP 7: GENERATE SUBMISSIONS
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 7: Writing submissions")
print("=" * 60)


def make_submission(weight_dict, target_df, path):
    rows = []
    for _, row in target_df.iterrows():
        rk = norm(row["repo"])
        rows.append({
            "repo":   row["repo"],
            "parent": row["parent"],
            "weight": round(weight_dict[rk], 8),
        })
    df    = pd.DataFrame(rows)
    total = df["weight"].sum()
    df["weight"] = df["weight"] / total
    df.to_csv(path, index=False)
    print(f"  {path.name}  sum={df['weight'].sum():.6f}  "
          f"min={100*df['weight'].min():.4f}%  max={100*df['weight'].max():.2f}%")
    return df


target_df = pd.read_csv(REPOS_CSV)
target_df["rk"] = target_df["repo"].apply(norm)

make_submission(w3,    target_df, BASE / "submission_lambda3.csv")
make_submission(w5,    target_df, BASE / "submission_lambda5.csv")
make_submission(w10,   target_df, BASE / "submission_lambda10.csv")
make_submission(w_ens, target_df, BASE / "submission_ensemble.csv")

# v5: no-confidence variant (Change 2) + T=1.15 sharpened no-conf variant (Change 3)
make_submission(w_no_conf_ens,     target_df, BASE / "submission_no_conf_ensemble.csv")
make_submission(w_no_conf_temp115, target_df, BASE / "submission_no_conf_temp115.csv")


# ══════════════════════════════════════════════════════════════════
# DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("DIAGNOSTICS")
print("=" * 60)

print(f"\nLOO CV performance on {len(covered_repos)} covered repos (obs >= {OBS_THRESHOLD}):")
print(f"  MAE: {loo_mae:.4f}  |  Pearson R: {loo_corr:.3f}")

sorted_ens = sorted(w_ens.items(), key=lambda x: -x[1])

print("\n" + "=" * 70)
print("FINAL TOP 25 (ensemble)")
print("=" * 70)
for rank, (repo, w) in enumerate(sorted_ens[:25], 1):
    obs  = obs_counter.get(repo, 0)
    conf = confidence(repo)
    tag  = "HUMAN" if obs >= OBS_THRESHOLD else ("sparse" if obs >= 1 else "BLIND")
    print(f"  #{rank:2d}  {repo:55s}  {100*w:5.2f}%  obs={obs}  conf={conf:.2f}  [{tag}]")

print("\n" + "=" * 70)
print("FULL TOP 30 WITH COVERAGE DETAIL (ensemble)")
print("=" * 70)
print(f"  {'#':3}  {'Repo':55s}  weight%   obs  conf  coverage")
for rank, (r, w) in enumerate(sorted_ens[:30], 1):
    n    = obs_counter.get(r, 0)
    conf = confidence(r)
    tag  = "HUMAN" if n >= OBS_THRESHOLD else ("sparse" if n >= 1 else "BLIND")
    print(f"  #{rank:2d}  {r:55s}  {100*w:5.2f}%  {n:3d}  {conf:.2f}  {tag}")

print("\nBlind repo weights (sorted):")
blind_weights = {r: w_ens[r] for r in all_repos if obs_counter.get(r, 0) == 0}
for r, w in sorted(blind_weights.items(), key=lambda x: -x[1]):
    fpr = feature_theta.get(r, 0.0)
    llm = llm_theta_calibrated.get(r, 0.0)
    print(f"  {r:55s}  {100*w:5.2f}%  feat={fpr:+.3f}  llm_cal={llm:+.3f}")

print("\nLambda sensitivity:")
for lam, wdict in [(3, w3), (5, w5), (10, w10), ("ens", w_ens)]:
    ws = np.array([wdict[r] for r in all_repos])
    print(f"  λ={lam}: max_weight={100*ws.max():.2f}%  "
          f"min_weight={100*ws.min():.4f}%  "
          f"top5_sum={100*np.sort(ws)[::-1][:5].sum():.1f}%")

# FIX 7: Permanent diagnostic — missing LLM repos
print("\nMISSING LLM REPOS")
missing_llm = [r for r in all_repos if r not in llm_theta_calibrated]
print(f"  Count: {len(missing_llm)}")
for r in missing_llm[:20]:
    print(f"  {r}")

# FIX 7: Permanent diagnostic — obs distribution
print("\nOBS COUNT DISTRIBUTION (merged pairwise + train)")
for n in sorted(set(obs_counter.values())):
    count = sum(1 for v in obs_counter.values() if v == n)
    print(f"  obs={n:3d}: {count} repos")

# FIX 7: Permanent diagnostic — top repos raw vs confident BT
print("\nTOP 20 RAW BT vs CONFIDENCE-ADJUSTED BT")
print(f"  {'Repo':55s}  raw_bt   obs   conf  conf_bt")
for repo, raw in sorted(merged_human_theta.items(), key=lambda x: -x[1])[:20]:
    obs  = obs_counter.get(repo, 0)
    conf = confidence(repo)
    cbt  = confident_theta.get(repo, 0.0)
    print(f"  {repo:55s}  {raw:+.3f}  {obs:3d}  {conf:.3f}  {cbt:+.3f}")

# Write diagnostics to file
diag_path = BASE / "model2_diagnostics.txt"
with open(diag_path, "w") as f:
    f.write(f"Model v2 Diagnostics\n{'='*60}\n\n")
    f.write(f"Merged human BT repos: {len(merged_human_theta)}\n")
    f.write(f"  Current-round pairs: {(merged['source']=='current').sum()}\n")
    f.write(f"  Train pairs (weight={TRAIN_WEIGHT}): {(merged['source']=='train').sum()}\n")
    f.write(f"LLM calibration: slope={slope:.3f}, intercept={intercept:.3f}, R={calib_corr:.3f}\n")
    f.write(f"Ridge OBS_THRESHOLD: {OBS_THRESHOLD}\n")
    f.write(f"Ridge training repos: {len(covered_repos)}\n")
    f.write(f"Ridge best alpha: {ridge.alpha_:.4f}\n")
    f.write(f"LOO CV MAE: {loo_mae:.4f}  Pearson R: {loo_corr:.3f}\n\n")
    f.write("Feature importances:\n")
    for fname, coef in coef_sorted:
        f.write(f"  {fname:30s}  {coef:+.4f}\n")
    f.write(f"\nSparse repos excluded from Ridge (obs < {OBS_THRESHOLD}):\n")
    for repo, obs, ct in excluded_from_ridge:
        f.write(f"  {repo:55s}  obs={obs}  conf_theta={ct:+.3f}\n")
    f.write("\nFull ranking (ensemble):\n")
    for rank, (r, w) in enumerate(sorted_ens, 1):
        n   = obs_counter.get(r, 0)
        tag = "human" if n >= OBS_THRESHOLD else ("sparse" if n >= 1 else "blind")
        f.write(f"  #{rank:3d}  {r:55s}  {100*w:5.3f}%  {n}obs  {tag}\n")

print(f"\nDiagnostics written to {diag_path}")

# ══════════════════════════════════════════════════════════════════
# Change 6 — FINAL RECOMMENDATION SUMMARY
# Side-by-side comparison of the three candidate submissions so they
# can be evaluated against the jury/leaderboard before picking one.
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Change 6 — RECOMMENDATION SUMMARY")
print("=" * 70)
_candidates = [
    ("submission_ensemble.csv",         w_ens),
    ("submission_no_conf_ensemble.csv", w_no_conf_ens),
    ("submission_no_conf_temp115.csv",  w_no_conf_temp115),
]
print(f"  {'File':35s}  max%       min%         spread_ratio")
for name, wd in _candidates:
    ws = np.array([wd[r] for r in all_repos])
    print(f"  {name:35s}  {100*ws.max():6.2f}%    {100*ws.min():.4f}%      {ws.max()/ws.min():7.1f}x")

print("\n  Notes:")
print("  - submission_ensemble.csv         : main pipeline (confidence pre-shrinkage + λ shrinkage)")
print("  - submission_no_conf_ensemble.csv : confidence pre-shrinkage removed (Change 1/2)")
print("  - submission_no_conf_temp115.csv  : no-conf ensemble + mild T=1.15 sharpening (Change 3)")
print("  Compare leaderboard / jury MAE across all three before finalizing a submission.")

print("\n✅ Done. Main submission: submission_ensemble.csv  "
      "(v5 variants above are diagnostic — compare before swapping)")




# ── Debug CSV exports ────────────────────────────────────────────
print("\n" + "=" * 60)
print("REPORTING: writing debug CSVs")
print("=" * 60)

_final_theta_lam5 = _FINAL_THETA_CACHE.get(5, {})  # lam=5 is the recommended setting

debug_raw_bt = pd.DataFrame([
    {"repo": r, "raw_bt_theta": merged_human_theta.get(r, 0.0), "obs_count": obs_counter.get(r, 0)}
    for r in all_repos
]).sort_values("raw_bt_theta", ascending=False)
debug_raw_bt.to_csv(BASE / "debug_raw_bt.csv", index=False)

debug_feature_prior = pd.DataFrame([
    {"repo": r, "feature_prior_theta": feature_theta.get(r, 0.0)}
    for r in all_repos
]).sort_values("feature_prior_theta", ascending=False)
debug_feature_prior.to_csv(BASE / "debug_feature_prior.csv", index=False)

debug_conf_theta = pd.DataFrame([
    {"repo": r, "confidence_adjusted_theta": confident_theta.get(r, 0.0)}
    for r in all_repos
]).sort_values("confidence_adjusted_theta", ascending=False)
debug_conf_theta.to_csv(BASE / "debug_conf_theta.csv", index=False)

debug_final_theta = pd.DataFrame([
    {
        "repo": r,
        "final_theta_lambda5": _final_theta_lam5.get(r, 0.0),
        "final_weight_ensemble": w_ens.get(r, 0.0),
    }
    for r in all_repos
]).sort_values("final_weight_ensemble", ascending=False)
debug_final_theta.to_csv(BASE / "debug_final_theta.csv", index=False)

debug_repo_breakdown = pd.DataFrame([
    {
        "repo": r,
        "obs_count": obs_counter.get(r, 0),
        "raw_bt_theta": merged_human_theta.get(r, 0.0),
        "confidence_adjusted_theta": confident_theta.get(r, 0.0),
        "feature_prior_theta": feature_theta.get(r, 0.0),
        "llm_theta_calibrated": llm_theta_calibrated.get(r, 0.0),
        "final_theta_lambda5": _final_theta_lam5.get(r, 0.0),
        "final_weight_ensemble": w_ens.get(r, 0.0),
    }
    for r in all_repos
]).sort_values("final_weight_ensemble", ascending=False)
debug_repo_breakdown.to_csv(BASE / "debug_repo_breakdown.csv", index=False)

print("  debug_raw_bt.csv")
print("  debug_feature_prior.csv")
print("  debug_conf_theta.csv")
print("  debug_final_theta.csv")
print("  debug_repo_breakdown.csv")

# ── Chart generation ──────────────────────────────────────────────
# Wrapped in try/except so that if matplotlib is missing or a chart
# fails for any reason, every submission file and diagnostic written
# above is still complete and unaffected.
try:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(BASE))
    from chart_generator import generate_all_charts

    CHARTS_DIR = BASE / "charts"

    chart_data = {
        # Chart 1
        "weights": w_ens,
        "all_repos": all_repos,
        # Chart 2
        "human_theta": confident_theta,
        "llm_theta_calibrated": llm_theta_calibrated,
        "anchors": anchors,
        "correlation": calib_corr,
        # Chart 3
        "feature_theta": feature_theta,
        # Chart 4
        "feat_names": feat_names,
        "coefs": ridge.coef_,
        # Chart 5 + 6
        "obs_counter": obs_counter,
        "human_theta_raw": merged_human_theta,
        "highlight_repos": [
            "vyperlang/vyper", "wevm/viem", "cyfrin/aderyn",
            "risc0/risc0-ethereum", "supranational/blst",
            "ethereum/go-ethereum", "argotorg/solidity",
        ],
        # Chart 7
        "repos": [
            "ethereum/consensus-specs", "vyperlang/vyper", "wevm/viem",
            "cyfrin/aderyn", "risc0/risc0-ethereum",
        ],
        "raw_bt": merged_human_theta,
        "conf_theta": confident_theta,
        "final_theta": _final_theta_lam5,
        # Chart 8
        "weights_original": w_ens,
        "weights_no_conf": w_no_conf_ens,
        "weights_no_conf_temp": w_no_conf_temp115,
        # Bonus chart
        "pairwise_only_theta": pairwise_only_theta,
        "train_only_theta": train_only_theta,
        "merged_human_theta": merged_human_theta,
        "llm_theta_raw_before_grok": llm_theta_raw_before_grok,
        "llm_theta_raw": llm_theta_raw,
        "vyper_repo": "vyperlang/vyper",
        "viem_repo": "wevm/viem",
        "cyfrin_repo": "cyfrin/aderyn",
        "risc0_repo": "risc0/risc0-ethereum",
    }

    generate_all_charts(chart_data, CHARTS_DIR)
except Exception as e:
    print(f"\nWARNING: chart generation failed ({e}). All submission files "
          f"and diagnostics above are unaffected.")