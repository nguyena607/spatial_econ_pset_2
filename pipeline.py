"""
pipeline.py
===========
Data pipeline: labor market effects of immigration enforcement (post-2017).

Outputs
-------
state_qcew_annual.csv      BLS QCEW: state × year employment & wages (2012-2024)
state_migration_annual.csv IRS SOI: bilateral state-to-state flows (2012-2023)
treated_states.csv         High-exposure treatment indicator
state_panel.csv            Main analysis panel
mu_minus1.csv              Baseline migration matrix (avg 2012-2016, row-stochastic)

Sources
-------
QCEW : https://data.bls.gov/cew/data/files/{year}/csv/{year}_annual_singlefile.zip
IRS  : https://www.irs.gov/pub/irs-soi/{y1}{y2}stmigr.xls(x)
"""

import io, os, re, time, zipfile, warnings
import requests
import numpy  as np
import pandas as pd

warnings.filterwarnings("ignore")

CACHE = "pipeline_cache"
os.makedirs(CACHE, exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

# ── State reference table ─────────────────────────────────────────────────────
STATE_REF = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA",
    "08": "CO", "09": "CT", "10": "DE", "11": "DC", "12": "FL",
    "13": "GA", "15": "HI", "16": "ID", "17": "IL", "18": "IN",
    "19": "IA", "20": "KS", "21": "KY", "22": "LA", "23": "ME",
    "24": "MD", "25": "MA", "26": "MI", "27": "MN", "28": "MS",
    "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH",
    "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND",
    "39": "OH", "40": "OK", "41": "OR", "42": "PA", "44": "RI",
    "45": "SC", "46": "SD", "47": "TN", "48": "TX", "49": "UT",
    "50": "VT", "51": "VA", "53": "WA", "54": "WV", "55": "WI",
    "56": "WY",
}
# state_fips (2-char) → abbreviation
VALID_FIPS   = set(STATE_REF.keys())
# 5-char area fips used in QCEW (e.g. "06000")
QCEW_FIPS    = {f + "000" for f in VALID_FIPS}
FIPS5_TO_ABB = {f + "000": a for f, a in STATE_REF.items()}
ABB_TO_FIPS2 = {v: k for k, v in STATE_REF.items()}

QCEW_YEARS = list(range(2012, 2025))   # 2012–2024
MIG_YEARS  = list(range(2012, 2024))   # migration file covers year → year+1


# ══════════════════════════════════════════════════════════════════════════════
# 1. BLS QCEW — state × year employment & wages
# ══════════════════════════════════════════════════════════════════════════════

def _qcew_url(year):
    return (f"https://data.bls.gov/cew/data/files/{year}/csv/"
            f"{year}_annual_singlefile.zip")

def fetch_qcew_year(year):
    """Download one year's bulk QCEW zip and return filtered state-level rows."""
    cache_path = os.path.join(CACHE, f"qcew_{year}.parquet")
    if os.path.exists(cache_path):
        return pd.read_parquet(cache_path)

    url = _qcew_url(year)
    print(f"  QCEW {year}: fetching {url} ...", end=" ", flush=True)
    try:
        r = requests.get(url, headers=HEADERS, timeout=120)
        r.raise_for_status()
    except Exception as e:
        print(f"FAILED ({e})")
        return None
    print(f"OK ({len(r.content)//1_048_576} MB)", flush=True)

    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        fname = next(n for n in z.namelist() if n.endswith(".csv"))
        with z.open(fname) as f:
            df = pd.read_csv(f, dtype=str)

    df.columns = df.columns.str.strip()
    for c in df.columns:
        df[c] = df[c].str.strip()

    # Keep: state-level total, all ownerships (own_code=0, agglvl=50, naics=10)
    mask = (
        df["area_fips"].isin(QCEW_FIPS) &
        (df["own_code"] == "0") &
        (df["industry_code"] == "10") &
        (df["agglvl_code"] == "50")
    )
    df = df[mask].copy()

    df["year"]        = int(year)
    df["state_fips"]  = df["area_fips"].str[:2]
    df["state_abbr"]  = df["area_fips"].map(FIPS5_TO_ABB)

    num_cols = {
        "annual_avg_emplvl":   "employment",
        "annual_avg_wkly_wage":"avg_weekly_wage",
    }
    for src, dst in num_cols.items():
        if src in df.columns:
            df[dst] = pd.to_numeric(df[src].str.replace(",", ""), errors="coerce")

    df = df[["year","state_fips","state_abbr","employment","avg_weekly_wage"]].dropna()
    df.to_parquet(cache_path, index=False)
    return df


def build_qcew():
    print("\n=== 1. Downloading QCEW ===")
    chunks = []
    for y in QCEW_YEARS:
        df = fetch_qcew_year(y)
        if df is not None:
            chunks.append(df)
        time.sleep(0.3)

    qcew = pd.concat(chunks, ignore_index=True).sort_values(["state_fips","year"])
    qcew.to_csv("state_qcew_annual.csv", index=False)
    print(f"\n  -> state_qcew_annual.csv  ({qcew.shape[0]} rows, "
          f"{qcew['state_fips'].nunique()} states, "
          f"years {qcew['year'].min()}–{qcew['year'].max()})")
    return qcew


# ══════════════════════════════════════════════════════════════════════════════
# 2. IRS SOI — state-to-state migration flows
#
# Correct URL (confirmed): stateoutflow{y1}{y2}.csv
# e.g. 2021-2022 → stateoutflow2122.csv
#
# CSV columns:
#   y1_statefips  origin state 2-digit FIPS
#   y2_statefips  destination FIPS (96=US+Foreign total, 97=US total/same-state, 98=Foreign)
#   y2_state      dest abbreviation
#   y2_state_name dest description
#   n1            number of returns  ← our migration flow measure
#   n2            number of exemptions (individuals)
#   AGI           adjusted gross income (thousands)
# ══════════════════════════════════════════════════════════════════════════════

def _mig_url(year):
    y1 = str(year)[2:]
    y2 = str(year + 1)[2:]
    return f"https://www.irs.gov/pub/irs-soi/stateoutflow{y1}{y2}.csv"


def fetch_migration_year(year):
    """Download IRS SOI state outflow CSV for year → year+1."""
    cache_path = os.path.join(CACHE, f"mig_{year}.parquet")
    if os.path.exists(cache_path):
        return pd.read_parquet(cache_path)

    url = _mig_url(year)
    print(f"  IRS {year}-{year+1}: {url.split('/')[-1]} ...", end=" ", flush=True)
    try:
        r = requests.get(url, headers=HEADERS, timeout=60)
        r.raise_for_status()
    except Exception as e:
        print(f"FAILED ({e})")
        return None

    df = pd.read_csv(io.StringIO(r.text), dtype=str)
    df.columns = df.columns.str.strip()

    # Keep only genuine state-to-state moves:
    #   y1_statefips ∈ valid FIPS, y2_statefips ∈ valid FIPS (not 96/97/98 aggregates)
    # Also keep same-state rows (y2_statefips == 97 AND "Same State" in name) for diagonal
    df["y1"] = df["y1_statefips"].str.strip().str.zfill(2)
    df["y2"] = df["y2_statefips"].str.strip().str.zfill(2)

    movers = df[df["y1"].isin(VALID_FIPS) & df["y2"].isin(VALID_FIPS)].copy()

    # Same-state stayers: y2_statefips = "97" + "Same State" in name
    stayers = df[
        df["y1"].isin(VALID_FIPS) &
        (df["y2_statefips"].str.strip() == "97") &
        (df["y2_state_name"].str.contains("Same State", case=False, na=False))
    ].copy()
    stayers["y2"] = stayers["y1"]   # destination = origin

    combined = pd.concat([movers, stayers], ignore_index=True)
    combined["n_returns"] = pd.to_numeric(
        combined["n1"].str.replace(",",""), errors="coerce"
    )
    combined = combined.dropna(subset=["n_returns"])
    combined = combined[combined["n_returns"] > 0]

    out = combined[["y1","y2","n_returns"]].rename(
        columns={"y1":"origin_fips","y2":"dest_fips"}
    ).copy()
    out["year_start"] = year

    print(f"OK  ({len(movers)} moves, {len(stayers)} stayer rows)")
    out.to_parquet(cache_path, index=False)
    return out


def build_migration():
    print("\n=== 2. Downloading IRS SOI Migration ===")
    chunks, failed = [], []
    for y in MIG_YEARS:
        df = fetch_migration_year(y)
        if df is not None:
            chunks.append(df)
        else:
            failed.append(y)
        time.sleep(0.4)

    if not chunks:
        print("\n  !! All migration downloads failed. Check internet connection.")
        return None

    mig = pd.concat(chunks, ignore_index=True)
    mig["origin_abbr"] = mig["origin_fips"].map(STATE_REF)
    mig["dest_abbr"]   = mig["dest_fips"].map(STATE_REF)
    mig = mig.dropna(subset=["origin_abbr","dest_abbr"])
    mig.to_csv("state_migration_annual.csv", index=False)

    print(f"\n  -> state_migration_annual.csv  ({mig.shape[0]} rows, "
          f"years {sorted(mig['year_start'].unique())})")
    if failed:
        print(f"  NOTE: missing years {failed}")
    return mig


# ══════════════════════════════════════════════════════════════════════════════
# 3. Treated states (high immigration-enforcement exposure)
# ══════════════════════════════════════════════════════════════════════════════

def build_treated():
    print("\n=== 3. Building treated_states.csv ===")
    # High-exposure: large undocumented immigrant populations + active enforcement
    # Sources: Pew Research Center (2016 unauthorized pop estimates),
    #          DHS deportation statistics, 287(g) program participation
    HIGH = {"TX", "AZ", "NM", "CA", "FL", "NY"}

    rows = []
    for fips2, abbr in STATE_REF.items():
        rows.append({
            "state_fips":    fips2,
            "state_abbr":    abbr,
            "high_exposure": int(abbr in HIGH),
        })

    treated = pd.DataFrame(rows).sort_values("state_fips")
    treated.to_csv("treated_states.csv", index=False)
    print(f"  -> treated_states.csv  (high_exposure=1: {HIGH})")
    return treated


# ══════════════════════════════════════════════════════════════════════════════
# 4. Main panel: state × year
# ══════════════════════════════════════════════════════════════════════════════

def build_panel(qcew, treated):
    print("\n=== 4. Building state_panel.csv ===")
    panel = qcew.merge(treated[["state_fips","state_abbr","high_exposure"]],
                       on=["state_fips","state_abbr"], how="left")

    # Wage growth: w_hat = w_t / w_{t-1}
    panel = panel.sort_values(["state_fips","year"])
    panel["w_hat"] = (panel
                      .groupby("state_fips")["avg_weekly_wage"]
                      .pct_change() + 1)

    # Treatment variables
    panel["post"]  = (panel["year"] >= 2017).astype(int)
    panel["shock"] = panel["high_exposure"] * panel["post"]

    panel.to_csv("state_panel.csv", index=False)
    print(f"  -> state_panel.csv  ({panel.shape[0]} rows, "
          f"{panel['state_fips'].nunique()} states)")
    return panel


# ══════════════════════════════════════════════════════════════════════════════
# 5. Baseline migration matrix mu_minus1 (avg 2012-2016, row-stochastic)
# ══════════════════════════════════════════════════════════════════════════════

def build_mu(mig):
    print("\n=== 5. Building mu_minus1.csv ===")
    if mig is None:
        print("  Skipped — no migration data available.")
        return None

    states = sorted(STATE_REF.keys())   # 51 2-digit FIPS
    n      = len(states)
    idx    = {s: i for i, s in enumerate(states)}

    # Average flows over the pre-enforcement baseline (2012–2016)
    base = mig[mig["year_start"].between(2012, 2016)].copy()
    agg  = (base.groupby(["origin_fips","dest_fips"])["n_returns"]
                .mean().reset_index())

    mu = np.zeros((n, n))
    for _, row in agg.iterrows():
        o, d = row["origin_fips"], row["dest_fips"]
        if o in idx and d in idx:
            mu[idx[o], idx[d]] = row["n_returns"]

    # Diagonal (stayers) comes directly from the IRS "Same State" rows.
    # For any state still missing a diagonal (no stayer data), set to a
    # large number so ~98% stay probability after normalization.
    for i in range(n):
        if mu[i, i] == 0:
            off_diag = mu[i, :].sum()
            mu[i, i] = off_diag * (0.98 / 0.02) if off_diag > 0 else 1.0

    # Row-normalize to get probability matrix
    mu = mu / mu.sum(axis=1, keepdims=True)

    mu_df = pd.DataFrame(mu, index=states, columns=states)
    mu_df.index.name   = "origin_fips"
    mu_df.columns.name = "dest_fips"
    mu_df.to_csv("mu_minus1.csv")
    print(f"  -> mu_minus1.csv  ({n}×{n} matrix)")
    print(f"     avg stay prob = {np.diag(mu).mean():.4f}")
    print(f"     avg move prob = {(1 - np.diag(mu)).mean():.4f}")
    return mu_df


# ══════════════════════════════════════════════════════════════════════════════
# 6. QA checks
# ══════════════════════════════════════════════════════════════════════════════

def qa_checks(qcew, mig, panel, mu):
    print("\n=== 6. QA Checks ===")
    ok = True

    # QCEW
    exp_rows = len(QCEW_YEARS) * len(VALID_FIPS)
    print(f"\n  QCEW")
    print(f"    rows: {len(qcew)} (expected ≈ {exp_rows})")
    miss = qcew[["employment","avg_weekly_wage"]].isna().sum()
    print(f"    missing employment:      {miss['employment']}")
    print(f"    missing avg_weekly_wage: {miss['avg_weekly_wage']}")
    print(f"    employment range: {qcew['employment'].min():,.0f} – {qcew['employment'].max():,.0f}")
    print(f"    wage range: ${qcew['avg_weekly_wage'].min():,.0f} – ${qcew['avg_weekly_wage'].max():,.0f}/wk")
    if miss.sum() > 20:
        print("    WARN: high missing count in QCEW")
        ok = False

    # Migration
    if mig is not None:
        print(f"\n  Migration")
        print(f"    rows: {len(mig)}")
        yrs = sorted(mig['year_start'].unique())
        print(f"    years covered: {yrs}")
        if len(yrs) < 4:
            print("    WARN: fewer than 4 years of migration data — mu may be unreliable")
            ok = False
        # Check row sums of mu are ≈ 1
        if mu is not None:
            row_sums = mu.values.sum(axis=1)
            print(f"\n  mu_minus1")
            print(f"    row sum min/max: {row_sums.min():.6f} / {row_sums.max():.6f}  (should be 1.0)")
            if not np.allclose(row_sums, 1.0, atol=1e-6):
                print("    FAIL: mu is not row-stochastic!")
                ok = False
            else:
                print("    row-stochastic: OK")

    # Panel
    print(f"\n  Panel")
    print(f"    shape: {panel.shape}")
    print(f"    years: {sorted(panel['year'].unique())}")
    print(f"    states: {panel['state_fips'].nunique()}")
    print(f"    high_exposure states: {panel[panel['high_exposure']==1]['state_abbr'].unique().tolist()}")
    print(f"    w_hat missing (first year per state expected): {panel['w_hat'].isna().sum()}")
    shock_check = panel[panel["year"] < 2017]["shock"].sum()
    if shock_check != 0:
        print("    FAIL: shock != 0 in pre-2017 years")
        ok = False
    else:
        print("    pre-2017 shock = 0: OK")

    print(f"\n{'=== ALL CHECKS PASSED ===' if ok else '=== SOME CHECKS FAILED — review above ==='}")
    return ok


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    qcew    = build_qcew()
    mig     = build_migration()
    treated = build_treated()
    panel   = build_panel(qcew, treated)
    mu      = build_mu(mig)
    qa_checks(qcew, mig, panel, mu)

    print("\nDone. Output files:")
    for f in ["state_qcew_annual.csv","state_migration_annual.csv",
              "treated_states.csv","state_panel.csv","mu_minus1.csv"]:
        size = os.path.getsize(f) // 1024 if os.path.exists(f) else 0
        print(f"  {f:35s}  {size:>6} KB")
