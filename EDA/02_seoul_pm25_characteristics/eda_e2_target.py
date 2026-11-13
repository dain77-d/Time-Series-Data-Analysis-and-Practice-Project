"""E2 the target series: Seoul Jung-gu hourly PM2.5, training 2021-22 only.

Outputs: figures/e2_timeseries.png, e2_distribution.png, e2_cycles.png, e2_stl.png, e2_acf_pacf.png,
         e2_change_by_horizon.png, e2_episodes.png, e2_holidays.png
         tables/e2_summary.csv, e2_stationarity.csv, e2_change_by_horizon.csv, e2_episodes_by_month.csv, e2_holidays.csv
"""
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import MSTL
from statsmodels.tsa.stattools import adfuller, kpss

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as c  # noqa: E402

c.setup(__file__)

HOLIDAYS = {"Seollal 2021": "2021-02-12", "Chuseok 2021": "2021-09-21", "Seollal 2022": "2022-02-01", "Chuseok 2022": "2022-09-10"}


def main():
    pm, _ = c.universe()
    y = c.train(pm[c.TARGET]).astype(float)
    ly = np.log1p(y)
    d = y.resample("D").mean()

    c.table(y.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).to_frame("pm25").assign(
        log1p=ly.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95, 0.99])).round(3), "e2_summary")

    fig, ax = c.plt.subplots(2, 1, figsize=(13, 6), sharex=True)
    ax[0].plot(y.index, y, lw=0.3, color="#888")
    ax[0].plot(d.index, d, lw=1.2, color="#c0392b", label="daily mean")
    ax[0].axhline(35, ls="--", c="k", lw=0.8, label="35 ug/m3 (Korean 'bad' 24 h standard)")
    ax[0].set_ylabel("PM2.5 (ug/m3)")
    ax[0].legend(fontsize=8)
    ax[0].set_title("Seoul Jung-gu PM2.5, 2021-22 (hourly and daily mean)")
    m = y.rolling(24 * 30, center=True, min_periods=24 * 10).mean()
    ax[1].plot(m.index, m, c="#2c3e50")
    ax[1].set_ylabel("30-day rolling mean")
    c.save(fig, "e2_timeseries")

    fig, ax = c.plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].hist(y.dropna(), bins=120, color="#7f8c8d")
    ax[0].set_title(f"raw: skew {y.skew():.2f}")
    ax[0].set_xlabel("PM2.5")
    ax[1].hist(ly.dropna(), bins=80, color="#2980b9")
    ax[1].set_title(f"log1p: skew {ly.skew():.2f}")
    ax[1].set_xlabel("log1p PM2.5")
    fig.suptitle("Distribution of hourly PM2.5 (models use log1p)")
    c.save(fig, "e2_distribution")

    df = pd.DataFrame({"y": y, "hour": y.index.hour, "dow": y.index.dayofweek, "month": y.index.month, "season": c.season(y.index)})
    fig, ax = c.plt.subplots(1, 3, figsize=(15, 4))
    for s, g in df.groupby("season"):
        ax[0].plot(g.groupby("hour").y.mean(), marker=".", label=s)
    ax[0].set_title("Hour of day (by season)")
    ax[0].set_xlabel("hour KST")
    ax[0].legend(fontsize=8)
    ax[1].bar(range(7), df.groupby("dow").y.mean(), color="#16a085")
    ax[1].set_xticks(range(7), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    ax[1].set_title("Day of week")
    df.boxplot(column="y", by="month", ax=ax[2], showfliers=False)
    ax[2].set_title("Month")
    ax[2].set_xlabel("")
    fig.suptitle("Seoul PM2.5 cycles (mean ug/m3)")
    c.save(fig, "e2_cycles")

    f = ly.interpolate(limit=6).dropna()      # STL needs a gap-free series; 1.2 % missing, short gaps only
    f = f.asfreq("h").interpolate()
    r = MSTL(f, periods=(24, 24 * 7), stl_kwargs={"seasonal_deg": 0}).fit()
    fig, ax = c.plt.subplots(5, 1, figsize=(13, 10), sharex=True)
    for a, (name, s) in zip(ax, [("log1p PM2.5", f), ("trend", r.trend), ("daily (24 h)", r.seasonal.iloc[:, 0]),
                                 ("weekly (168 h)", r.seasonal.iloc[:, 1]), ("residual", r.resid)]):
        a.plot(s.index, s, lw=0.4)
        a.set_ylabel(name, fontsize=8)
    var = {k: float(np.var(v)) for k, v in [("trend", r.trend), ("daily", r.seasonal.iloc[:, 0]), ("weekly", r.seasonal.iloc[:, 1]), ("resid", r.resid)]}
    tot = sum(var.values())
    ax[0].set_title("MSTL decomposition (log1p): variance share " + ", ".join(f"{k} {100 * v / tot:.0f}%" for k, v in var.items()))
    c.save(fig, "e2_stl")

    fig, ax = c.plt.subplots(1, 2, figsize=(13, 3.8))
    plot_acf(f, lags=72, ax=ax[0], markersize=2)
    plot_pacf(f, lags=72, ax=ax[1], method="ywm", markersize=2)
    for a in ax:
        for h in (3, 6, 12, 24):
            a.axvline(h, c="r", lw=0.5, ls=":")
    ax[0].set_title("ACF of log1p PM2.5 (red: forecast horizons)")
    c.save(fig, "e2_acf_pacf")

    rows = []
    for name, s in (("log1p hourly", f), ("daily mean", np.log1p(d.dropna()))):
        adf = adfuller(s, autolag="AIC")
        kp = kpss(s, regression="c", nlags="auto")
        rows.append({"series": name, "ADF_stat": adf[0], "ADF_p": adf[1], "KPSS_stat": kp[0], "KPSS_p": kp[1]})
    c.table(pd.DataFrame(rows).round(4), "e2_stationarity", index=False)

    ch = []
    fig, ax = c.plt.subplots(1, 4, figsize=(15, 3.5), sharey=True)
    for a, h in zip(ax, (3, 6, 12, 24)):
        dlt = (y.shift(-h) - y).dropna()
        ch.append({"horizon": h, "persistence_RMSE_train": np.sqrt((dlt ** 2).mean()), "mean_abs_change": dlt.abs().mean(),
                   "share_change_gt_10": (dlt.abs() > 10).mean(), "corr_y_t_y_t+h": y.corr(y.shift(-h))})
        a.hist(dlt.clip(-60, 60), bins=80, color="#8e44ad")
        a.set_title(f"y(t+{h}) - y(t)")
    fig.suptitle("How much Seoul PM2.5 changes over each forecast horizon (2021-22)")
    c.save(fig, "e2_change_by_horizon")
    c.table(pd.DataFrame(ch).round(3), "e2_change_by_horizon", index=False)

    ep = []
    for thr in (35, 75):
        hi = (y > thr).astype(int)
        run_id = (hi.diff() != 0).cumsum()
        lens = hi.groupby(run_id).agg(["first", "size"])
        lens = lens[lens["first"] == 1]["size"]
        ep.append((thr, lens))
    by_m = pd.DataFrame({f">{t}": (y > t).groupby(y.index.month).mean() * 100 for t in (35, 75)}).round(2)
    c.table(by_m, "e2_episodes_by_month")
    fig, ax = c.plt.subplots(1, 2, figsize=(12, 3.8))
    by_m.plot.bar(ax=ax[0])
    ax[0].set_ylabel("% of hours")
    ax[0].set_title("Hours above threshold, by month")
    for thr, lens in ep:
        ax[1].hist(lens, bins=np.arange(1, 80, 3), alpha=0.6, label=f">{thr}: {len(lens)} episodes, median {lens.median():.0f} h")
    ax[1].set_xlabel("episode length (hours)")
    ax[1].legend(fontsize=8)
    ax[1].set_title("How long high-PM episodes last")
    c.save(fig, "e2_episodes")

    hol = []
    fig, ax = c.plt.subplots(figsize=(10, 4))
    for name, day in HOLIDAYS.items():
        t0 = pd.Timestamp(day, tz="Asia/Seoul")
        w = y[t0 - pd.Timedelta(days=4):t0 + pd.Timedelta(days=5)]
        ax.plot((w.index - t0) / pd.Timedelta(hours=1), w.values, lw=1, label=name)
        hol.append({"holiday": name, "mean_-4..-1d": y[t0 - pd.Timedelta(days=4):t0 - pd.Timedelta(hours=1)].mean(),
                    "mean_0..+2d": y[t0:t0 + pd.Timedelta(days=3)].mean()})
    ax.axvline(0, c="k", lw=0.8)
    ax.set_xlabel("hours from holiday midnight (KST)")
    ax.set_ylabel("PM2.5")
    ax.legend(fontsize=8)
    ax.set_title("Seoul PM2.5 around Seollal and Chuseok")
    c.save(fig, "e2_holidays")
    c.table(pd.DataFrame(hol).round(2), "e2_holidays", index=False)


if __name__ == "__main__":
    main()
