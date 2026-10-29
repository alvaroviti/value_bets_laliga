"""Genera CSV sintéticos con el mismo formato que football-data.co.uk para probar
la app sin conexión:  python tests/make_synthetic.py <carpeta>  y luego
DATA_DIR=<carpeta> streamlit run app.py"""
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson

rng = np.random.default_rng(7)
out = Path(sys.argv[1] if len(sys.argv) > 1 else "synthetic")
out.mkdir(parents=True, exist_ok=True)

POOL = ["Real Madrid", "Barcelona", "Ath Madrid", "Ath Bilbao", "Villarreal", "Betis", "Sociedad",
        "Sevilla", "Valencia", "Girona", "Celta", "Osasuna", "Getafe", "Mallorca", "Vallecano",
        "Alaves", "Espanol", "Elche", "Levante", "Oviedo", "Las Palmas", "Leganes", "Valladolid",
        "Cadiz", "Almeria", "Granada"]
att = {t: rng.normal(0, 0.25) for t in POOL}
dfn = {t: rng.normal(0, 0.2) for t in POOL}
for t, a in [("Real Madrid", .5), ("Barcelona", .55), ("Ath Madrid", .3)]:
    att[t] = a; dfn[t] = -.35
c_att = {t: rng.normal(0, .12) + .5 * att[t] for t in POOL}
c_def = {t: rng.normal(0, .1) for t in POOL}
k_att = {t: rng.normal(0, .15) for t in POOL}
k_def = {t: rng.normal(0, .1) for t in POOL}
k_att["Getafe"] = .35; k_def["Getafe"] = .25  # el Getafe ve muchas tarjetas


def odds_from(p, margin, noise=0.03):
    p = np.clip(np.array(p) * np.exp(rng.normal(0, noise, len(p))), 0.01, None)
    p = p / p.sum() * (1 + margin)
    return np.round(1 / p, 2)


def match_row(h, a, d, played=True):
    lh = np.exp(0.12 + 0.25 + att[h] + dfn[a]); la = np.exp(0.12 + att[a] + dfn[h])
    g = np.arange(11)
    m = np.outer(poisson.pmf(g, lh), poisson.pmf(g, la))
    p1, px, p2 = np.tril(m, -1).sum(), np.trace(m), np.triu(m, 1).sum()
    tot = np.add.outer(g, g); po = m[tot > 2.5].sum()
    r = {"Div": "SP1", "Date": d.strftime("%d/%m/%Y"), "Time": "21:00", "HomeTeam": h, "AwayTeam": a}
    for pre, mg, nz in (("B365", .055, .04), ("PS", .025, .02), ("Avg", .06, .03), ("Max", -.005, .03)):
        o = odds_from([p1, px, p2], mg, nz)
        r[f"{pre}H"], r[f"{pre}D"], r[f"{pre}A"] = o
        oo = odds_from([po, 1 - po], mg, nz)
        k = "P" if pre == "PS" else pre
        r[f"{k}>2.5"], r[f"{k}<2.5"] = oo
        oc = odds_from([p1, px, p2], mg, nz / 2)
        kc = "PSC" if pre == "PS" else pre + "C"
        r[f"{kc}H"], r[f"{kc}D"], r[f"{kc}A"] = oc
        ooc = odds_from([po, 1 - po], mg, nz / 2)
        kk = "PC" if pre == "PS" else pre + "C"
        r[f"{kk}>2.5"], r[f"{kk}<2.5"] = ooc
    if played:
        hg, ag = rng.poisson(lh), rng.poisson(la)
        r.update(FTHG=hg, FTAG=ag, FTR="H" if hg > ag else ("D" if hg == ag else "A"),
                 HC=rng.negative_binomial(10, 10 / (10 + np.exp(1.65 + .1 + c_att[h] + c_def[a]))),
                 AC=rng.negative_binomial(10, 10 / (10 + np.exp(1.65 + c_att[a] + c_def[h]))),
                 HY=rng.poisson(np.exp(0.8 - .05 + k_att[h] + k_def[a])),
                 AY=rng.poisson(np.exp(0.8 + .05 + k_att[a] + k_def[h])),
                 HR=rng.binomial(1, .07), AR=rng.binomial(1, .08))
        r["HS"], r["AS"] = rng.poisson(13), rng.poisson(10)
    return r


teams_by_season = {}
cur = list(rng.choice(POOL, 20, replace=False))
for y in range(2021, 2027):
    teams_by_season[y] = cur
    rest = [t for t in POOL if t not in cur]
    cur = cur[:17] + list(rng.choice(rest, 3, replace=False))
    rng.shuffle(cur)

for y, teams in teams_by_season.items():
    pairs = [(h, a) for h in teams for a in teams if h != a]
    rng.shuffle(pairs)
    start = date(y, 8, 15)
    end = date(2026, 10, 27) if y == 2026 else date(y + 1, 5, 25)
    rows = []
    n = len(pairs) if y < 2026 else 100
    for i, (h, a) in enumerate(pairs[:n]):
        d = start + timedelta(days=int(i / n * (end - start).days))
        rows.append(match_row(h, a, d))
    pd.DataFrame(rows).to_csv(out / f"SP1_{y % 100:02d}{(y + 1) % 100:02d}.csv", index=False)

teams = teams_by_season[2026]
fx = [match_row(teams[2 * i], teams[2 * i + 1], date(2026, 10, 31) + timedelta(days=i // 4), played=False) for i in range(10)]
pd.DataFrame(fx).to_csv(out / "fixtures.csv", index=False)
print("ok", out)
