"""Shared loaders for the EDA scripts in the group folders (training period 2021-22 unless a script says otherwise).

Station PM2.5 comes from the C data cache (work/c_universe.npz: Seoul Jung-gu + 650 stations, hourly, flag != 0 -> NaN).
It is built once from data_v5 by C_graph/cdata.py if missing.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "C_graph"))
import cdata  # noqa: E402

EDA = REPO / "EDA"
FIG = TAB = None            # set per script by setup(__file__): <group folder>/figures, <group folder>/tables
CLUSTERS = EDA / "05_transport_pathways_and_events" / "tables" / "e5_traj_clusters_2021_2023.csv"


def setup(script):
    """write figures/tables next to the calling script (its EDA group folder)."""
    global FIG, TAB
    d = Path(script).resolve().parent
    FIG, TAB = d / "figures", d / "tables"
    FIG.mkdir(exist_ok=True)
    TAB.mkdir(exist_ok=True)


TARGET = cdata.TARGET
# region priority: the first match wins (JP_WEST25 overlaps JP)
REGION_ORDER = ["KR_CAPITAL", "KR_GANGWON", "KR_CHUNGCHEONG", "KR_HONAM", "KR_DAEGYEONG", "KR_DONGNAM", "KR_JEJU",
                "CN_BTH", "CN_SHANDONG", "CN_LIAONING", "CN_JILIN", "CN_HEILONGJIANG", "CN_INNER_MONGOLIA", "CN_EAST",
                "JP", "JP_WEST25", "JP_EAST", "MN"]
plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False})


def root():
    return cdata.data_root()


def universe():
    """(DataFrame hourly KST x station key PM2.5 [2020-12-29 .. 2024-12-31], station table with lon/lat/country/region)."""
    f = cdata.WORK / "c_universe.npz"
    if not f.exists():
        cdata.WORK.mkdir(exist_ok=True)
        cdata.build_universe(root())
    z = np.load(f)
    pm = pd.DataFrame(z["pm25"], index=cdata.HOURS, columns=z["keys"])
    _, regions = cdata.conditions(root())
    reg = {}
    for r in REGION_ORDER:
        for k in regions[r]:
            reg.setdefault(k, r)
    st = pd.DataFrame({"key": z["keys"], "lon": z["lon"], "lat": z["lat"]})
    st["country"] = st.key.str[:2]
    st["region"] = st.key.map(reg).fillna("TARGET")
    return pm, st.set_index("key")


def train(df):
    return df.loc["2021-01-01":"2022-12-31"]


def asos():
    g, names = cdata.asos(root())
    return pd.DataFrame(g, index=cdata.HOURS, columns=names)


def season(idx):
    m = idx.month
    return np.select([m.isin([12, 1, 2]), m.isin([3, 4, 5]), m.isin([6, 7, 8])], ["DJF", "MAM", "JJA"], "SON")


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
    print("fig", name)


def table(df, name, index=True):
    df.to_csv(TAB / f"{name}.csv", index=index, encoding="utf-8-sig")
    print("table", name, df.shape)


def clusters():
    """E5 route cluster of the trajectory arriving at each 3-hourly time (2021-23; run 05 first)."""
    s = pd.read_csv(CLUSTERS, index_col=0).cluster
    s.index = pd.DatetimeIndex(pd.to_datetime(s.index, utc=True)).tz_convert("Asia/Seoul")
    return s


def region_means(pm, st, log=True):
    """hourly mean PM2.5 per region (log1p of each station first when log=True)."""
    x = np.log1p(pm) if log else pm
    return x.T.groupby(st.region.reindex(x.columns)).mean().T
