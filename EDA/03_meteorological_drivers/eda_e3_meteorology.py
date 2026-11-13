"""E3 meteorology and Seoul PM2.5 (ASOS Seoul 108 / Baengnyeong 102, training 2021-22).

- wind roses of mean PM2.5 and frequency (Seoul, Baengnyeong)
- PM2.5 against temperature, humidity, pressure, wind speed by season
- rain washout composite around rain onset
- calm vs windy hours
- lagged correlation of weather now with PM2.5 h hours later
- Baengnyeong wind now vs Seoul PM2.5 change over the next 12 h
ASOS precipitation blanks are mostly dry hours (E1: 88-89 % blank); here blank = 0 mm.
Outputs: figures/e3_*.png, tables/e3_*.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as c  # noqa: E402

c.setup(__file__)
SECT = 16


def wind(a, st):
    u, v = a[f"asos{st}_wind_u"], a[f"asos{st}_wind_v"]
    spd = np.hypot(u, v)
    d = (np.degrees(np.arctan2(-u, -v)) + 360) % 360          # direction the wind blows FROM
    sec = ((d + 360 / SECT / 2) // (360 / SECT)) % SECT
    return spd, sec.where(spd >= 0.5)                        # calm (< 0.5 m/s) has no sector


def rose(ax, sec, val, title, cmap="magma_r"):
    th = np.deg2rad(np.arange(SECT) * 360 / SECT)
    g = pd.DataFrame({"s": sec, "v": val}).dropna().groupby("s").v.agg(["mean", "size"]).reindex(range(SECT))
    bars = ax.bar(th, g["size"] / g["size"].sum() * 100, width=2 * np.pi / SECT * 0.9,
                  color=c.plt.get_cmap(cmap)((g["mean"] - 5) / 30), edgecolor="k", lw=0.3)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_title(title, fontsize=9)
    ax.set_yticklabels([])
    for t, (m, n) in zip(th, g.itertuples(index=False)):
        if np.isfinite(m):
            ax.text(t, n / g["size"].sum() * 100 + 1, f"{m:.0f}", fontsize=6, ha="center")
    return bars


def main():
    pm, _ = c.universe()
    y = c.train(pm[c.TARGET]).astype(float)
    a = c.train(c.asos())
    a["asos108_precip"] = a.asos108_precip.fillna(0)
    a["asos102_precip"] = a.asos102_precip.fillna(0)
    spd108, sec108 = wind(a, 108)
    spd102, sec102 = wind(a, 102)
    sea = pd.Series(c.season(y.index), index=y.index)

    # 1 wind roses
    fig = c.plt.figure(figsize=(14, 7))
    for i, (sec, name) in enumerate([(sec108, "Seoul 108"), (sec102, "Baengnyeong 102")]):
        ax = fig.add_subplot(2, 3, 1 + 3 * i, projection="polar")
        rose(ax, sec, y, f"{name}: all seasons\n(bar = % of hours, colour/number = mean Seoul PM2.5)")
        for j, s in enumerate(["DJF", "JJA"]):
            ax = fig.add_subplot(2, 3, 2 + j + 3 * i, projection="polar")
            m = sea.eq(s)
            rose(ax, sec[m], y[m], f"{name}: {s}")
    fig.suptitle("Wind direction (from) and Seoul PM2.5, 2021-22")
    c.save(fig, "e3_wind_roses")
    tab = pd.DataFrame({"sector_deg": np.arange(SECT) * 22.5})
    for sec, name in [(sec108, "seoul108"), (sec102, "baengnyeong102")]:
        g = pd.DataFrame({"s": sec, "v": y}).dropna().groupby("s").v.agg(["mean", "size"]).reindex(range(SECT))
        tab[f"{name}_mean_pm"], tab[f"{name}_share"] = g["mean"].to_numpy(), (g["size"] / g["size"].sum()).to_numpy()
    c.table(tab.round(3), "e3_wind_sector_pm", index=False)

    # 2 PM against weather variables, by season
    vars_ = {"temperature (C)": a.asos108_temperature, "relative humidity (%)": a.asos108_rh,
             "surface pressure (hPa)": a.asos108_pressure, "wind speed (m/s)": spd108}
    fig, axs = c.plt.subplots(1, 4, figsize=(17, 4))
    for ax, (name, x) in zip(axs, vars_.items()):
        for s in ["DJF", "MAM", "JJA", "SON"]:
            m = sea.eq(s)
            b = pd.qcut(x[m], 10, duplicates="drop")
            g = y[m].groupby(b, observed=True).mean()
            ax.plot([i.mid for i in g.index], g.values, marker="o", ms=3, label=s)
        ax.set_xlabel(name)
    axs[0].set_ylabel("mean Seoul PM2.5 (deciles of x)")
    axs[0].legend(fontsize=8)
    fig.suptitle("Seoul PM2.5 against local weather, by season (2021-22)")
    c.save(fig, "e3_pm_vs_weather")
    corr = pd.DataFrame({s: {k: y[sea.eq(s)].corr(v[sea.eq(s)], method="spearman") for k, v in vars_.items()}
                         for s in ["DJF", "MAM", "JJA", "SON"]})
    c.table(corr.round(3), "e3_weather_spearman_by_season")

    # 3 rain washout: onset = first hour with >= 0.5 mm after >= 6 dry hours
    r = a.asos108_precip
    wet = r >= 0.5
    onset = wet & (~wet.shift(1, fill_value=False).rolling(6, min_periods=6).max().fillna(1).astype(bool))
    rows = []
    for t0 in r.index[onset]:
        w = y[t0 - pd.Timedelta(hours=12):t0 + pd.Timedelta(hours=24)]
        if len(w) < 30 or w.isna().mean() > 0.2:
            continue
        tot = r[t0:t0 + pd.Timedelta(hours=12)].sum()
        base = y[t0 - pd.Timedelta(hours=3):t0 - pd.Timedelta(hours=1)].mean()
        rel = (w - base).set_axis(((w.index - t0) / pd.Timedelta(hours=1)).astype(int))
        rows.append((tot, rel))
    fig, ax = c.plt.subplots(figsize=(9, 4))
    summ = []
    for lo, hi, col in [(0, 2, "#85c1e9"), (2, 10, "#2e86c1"), (10, 1e9, "#1b4f72")]:
        sel = [s for tot, s in rows if lo <= tot < hi]
        if not sel:
            continue
        m = pd.concat(sel, axis=1).mean(1)
        ax.plot(m.index, m.values, c=col, label=f"12 h rain {lo}-{hi if hi < 1e8 else '+'} mm (n={len(sel)})")
        summ.append({"rain_12h_mm": f"{lo}-{hi}" if hi < 1e8 else f"{lo}+","events": len(sel), "change_at_+12h": m.get(12), "change_at_+24h": m.get(24)})
    ax.axvline(0, c="k", lw=0.8)
    ax.axhline(0, c="grey", lw=0.6)
    ax.set_xlabel("hours from rain onset (Seoul 108)")
    ax.set_ylabel("PM2.5 minus pre-rain level")
    ax.legend(fontsize=8)
    ax.set_title("Rain washout: Seoul PM2.5 around rain onset (2021-22)")
    c.save(fig, "e3_rain_washout")
    c.table(pd.DataFrame(summ).round(2), "e3_rain_washout", index=False)

    # 4 calm vs windy
    calm = pd.cut(spd108, [-0.1, 0.5, 1.5, 3, 5, 30], labels=["calm <0.5", "0.5-1.5", "1.5-3", "3-5", ">5"])
    t = y.groupby([sea, calm], observed=True).mean().unstack()
    c.table(t.round(2), "e3_pm_by_wind_speed")
    fig, ax = c.plt.subplots(figsize=(8, 4))
    t.reindex(["DJF", "MAM", "JJA", "SON"]).plot.bar(ax=ax, colormap="viridis")
    ax.set_ylabel("mean Seoul PM2.5")
    ax.set_title("Weak wind, high PM2.5 (Seoul 108 wind speed, m/s)")
    ax.tick_params(axis="x", rotation=0)
    c.save(fig, "e3_pm_by_wind_speed")

    # 5 lagged correlation: weather at t vs log PM at t+h
    ly = np.log1p(y)
    X = {"T 108": a.asos108_temperature, "RH 108": a.asos108_rh, "P 108": a.asos108_pressure, "rain 108": r,
         "u 108 (west+)": a.asos108_wind_u, "v 108 (south+)": a.asos108_wind_v, "speed 108": spd108,
         "u 102 (west+)": a.asos102_wind_u, "v 102 (south+)": a.asos102_wind_v, "speed 102": spd102,
         "P 102 - P 108": a.asos102_pressure - a.asos108_pressure}
    lags = range(0, 25, 3)
    lc = pd.DataFrame({h: {k: x.corr(ly.shift(-h), method="spearman") for k, x in X.items()} for h in lags})
    c.table(lc.round(3), "e3_lagged_spearman")
    fig, ax = c.plt.subplots(figsize=(9, 5))
    im = ax.imshow(lc.to_numpy(float), cmap="RdBu_r", vmin=-0.4, vmax=0.4, aspect="auto")
    ax.set_xticks(range(len(lags)), [f"+{h} h" for h in lags])
    ax.set_yticks(range(len(lc)), lc.index)
    for (i, j), v in np.ndenumerate(lc.to_numpy(float)):
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7)
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="Spearman")
    ax.set_title("Weather now vs Seoul log PM2.5 h hours later (2021-22)")
    c.save(fig, "e3_lagged_correlation")

    # 6 Baengnyeong wind now -> Seoul change over the next 12 h
    d12 = y.shift(-12) - y
    g = pd.DataFrame({"s": sec102, "d": d12, "spd": spd102}).dropna()
    g = g[g.spd >= 2].groupby("s").d.agg(["mean", "size"]).reindex(range(SECT))
    c.table(g.assign(sector_deg=np.arange(SECT) * 22.5).round(3), "e3_baengnyeong_wind_next12h")
    fig = c.plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(projection="polar")
    th = np.deg2rad(np.arange(SECT) * 22.5)
    ax.bar(th, g["mean"].clip(lower=0), width=0.35, color="#c0392b", label="Seoul rises")
    ax.bar(th, (-g["mean"]).clip(lower=0), width=0.35, color="#2980b9", label="Seoul falls")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("Baengnyeong wind direction (>= 2 m/s) now\nvs mean Seoul PM2.5 change over the next 12 h (ug/m3)", fontsize=9)
    c.save(fig, "e3_baengnyeong_wind_next12h")


if __name__ == "__main__":
    main()
