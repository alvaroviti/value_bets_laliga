"""Backtest walk-forward: cada semana se reentrena solo con datos anteriores
y se apuesta con las cuotas históricas que hubo de verdad."""
from __future__ import annotations

import numpy as np
import pandas as pd

from model import XI_DEFAULT, expected_vec, fit_cards, fit_corners, fit_goals, goal_markets
from value import implied_no_vig

ODDS_SETS = {
    "Bet365 apertura": {"1": "B365H", "X": "B365D", "2": "B365A", "O": "B365>2.5", "U": "B365<2.5"},
    "Mejor cuota apertura": {"1": "MaxH", "X": "MaxD", "2": "MaxA", "O": "Max>2.5", "U": "Max<2.5"},
    "Bet365 cierre": {"1": "B365CH", "X": "B365CD", "2": "B365CA", "O": "B365C>2.5", "U": "B365C<2.5"},
    "Pinnacle cierre": {"1": "PSCH", "X": "PSCD", "2": "PSCA", "O": "PC>2.5", "U": "PC<2.5"},
}
# Referencia de mercado (sin margen) para mezclar con el modelo
REF_SETS = {
    "Bet365 apertura": (("PSH", "PSD", "PSA"), ("P>2.5", "P<2.5")),
    "Mejor cuota apertura": (("PSH", "PSD", "PSA"), ("P>2.5", "P<2.5")),
    "Bet365 cierre": (("PSCH", "PSCD", "PSCA"), ("PC>2.5", "PC<2.5")),
    "Pinnacle cierre": (("PSCH", "PSCD", "PSCA"), ("PC>2.5", "PC<2.5")),
}


def walk_forward_predictions(hist: pd.DataFrame, test_seasons: list[int], xi: float = XI_DEFAULT) -> pd.DataFrame:
    """Probabilidades del modelo para cada partido de las temporadas de test,
    entrenando solo con lo anterior a cada semana."""
    out = []
    test = hist[hist["SeasonStart"].isin(test_seasons)].copy()
    test["week"] = test["Date"].dt.to_period("W-MON")
    for week, g in test.groupby("week"):
        cutoff = g["Date"].min() - pd.Timedelta(days=1)
        train = hist[hist["Date"] <= cutoff]
        train = train[train["Date"] >= cutoff - pd.Timedelta(days=4 * 365)]
        if len(train) < 300:
            continue
        gm = fit_goals(train, ref_date=cutoff, xi=xi)
        cm = fit_corners(train, ref_date=cutoff, xi=xi)
        km = fit_cards(train, ref_date=cutoff, xi=xi)
        clh, cla = expected_vec(cm.rates, g.HomeTeam, g.AwayTeam)
        klh, kla = expected_vec(km.rates, g.HomeTeam, g.AwayTeam)
        for j, (_, r) in enumerate(g.iterrows()):
            p = goal_markets(gm.score_matrix(r.HomeTeam, r.AwayTeam))
            _, _, pt_c = cm.distributions(r.HomeTeam, r.AwayTeam)
            _, _, pt_k = km.distributions(r.HomeTeam, r.AwayTeam)
            rec = r.to_dict()
            rec.update({
                "p1": p["1"], "pX": p["X"], "p2": p["2"],
                "pO": p["Más de 2.5 goles"], "pU": p["Menos de 2.5 goles"],
                "exp_corners": clh[j] + cla[j], "exp_cards": klh[j] + kla[j],
                "pC_over9.5": float(pt_c[10:].sum()), "pK_over4.5": float(pt_k[5:].sum()),
            })
            out.append(rec)
    return pd.DataFrame(out)


def simulate_bets(pred: pd.DataFrame, odds_set: str, min_ev: float, model_weight: float,
                  max_odds: float = 6.0, staking: str = "Plana", kelly_fraction: float = 0.25,
                  markets=("1X2", "Goles 2.5")) -> pd.DataFrame:
    cols = ODDS_SETS[odds_set]
    ref1x2, refou = REF_SETS[odds_set]
    bets = []
    for _, r in pred.iterrows():
        res = "1" if r.FTHG > r.FTAG else ("X" if r.FTHG == r.FTAG else "2")
        over = (r.FTHG + r.FTAG) > 2.5
        cand = []
        if "1X2" in markets:
            mp = implied_no_vig([r.get(c) for c in ref1x2])
            for i, s in enumerate(("1", "X", "2")):
                cand.append(("1X2", s, r[f"p{s}"], mp[i] if mp else None, r.get(cols[s]), res == s))
        if "Goles 2.5" in markets:
            mp = implied_no_vig([r.get(c) for c in refou])
            cand.append(("Goles 2.5", "Más de 2.5", r.pO, mp[0] if mp else None, r.get(cols["O"]), over))
            cand.append(("Goles 2.5", "Menos de 2.5", r.pU, mp[1] if mp else None, r.get(cols["U"]), not over))
        for market, sel, pmod, pmkt, odds, won in cand:
            if odds is None or not np.isfinite(odds) or odds <= 1 or odds > max_odds:
                continue
            p = model_weight * pmod + (1 - model_weight) * pmkt if pmkt is not None else pmod
            e = p * odds - 1
            if e < min_ev:
                continue
            if staking == "Plana":
                stake = 1.0
            else:
                stake = max(0.0, min((p * odds - 1) / (odds - 1) * kelly_fraction, 0.05)) * 100
            profit = stake * (odds - 1) if won else -stake
            bets.append({"Fecha": r.Date, "Temporada": r.Season, "Partido": f"{r.HomeTeam} - {r.AwayTeam}",
                         "Mercado": market, "Apuesta": sel, "Cuota": odds, "P. usada": p, "EV": e,
                         "Stake": stake, "Ganada": bool(won), "Beneficio": profit})
    b = pd.DataFrame(bets)
    if not b.empty:
        b = b.sort_values("Fecha").reset_index(drop=True)
        b["Acumulado"] = b["Beneficio"].cumsum()
    return b


def calibration(pred: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Probabilidad prevista vs frecuencia real por tramos."""
    out = {}
    specs = {
        "Local gana": ("p1", pred.FTHG > pred.FTAG),
        "Más de 2.5 goles": ("pO", (pred.FTHG + pred.FTAG) > 2.5),
        "Más de 9.5 córners": ("pC_over9.5", (pred.HC + pred.AC) > 9.5),
        "Más de 4.5 tarjetas": ("pK_over4.5", (pred.HY + pred.AY + pred.HR.fillna(0) + pred.AR.fillna(0)) > 4.5),
    }
    for name, (col, y) in specs.items():
        d = pd.DataFrame({"p": pred[col], "y": y.astype(float)}).dropna()
        if d.empty:
            continue
        d["tramo"] = pd.cut(d["p"], bins=np.linspace(0, 1, 11), include_lowest=True)
        t = d.groupby("tramo", observed=True).agg(prevista=("p", "mean"), real=("y", "mean"), n=("y", "size")).reset_index()
        base = d["y"].mean()
        brier_model = float(((d.p - d.y) ** 2).mean())
        brier_base = float(((base - d.y) ** 2).mean())
        out[name] = (t[t.n >= 10], brier_model, brier_base)
    return out


def log_loss_1x2(pred: pd.DataFrame, ref_cols=("PSCH", "PSCD", "PSCA")) -> dict:
    res = np.where(pred.FTHG > pred.FTAG, 0, np.where(pred.FTHG == pred.FTAG, 1, 2))
    pm = pred[["p1", "pX", "p2"]].to_numpy()
    ll_model = -np.mean(np.log(np.clip(pm[np.arange(len(pm)), res], 1e-9, 1)))
    mk = [implied_no_vig([r[c] for c in ref_cols]) for _, r in pred.iterrows()]
    ok = np.array([m is not None for m in mk])
    out = {"modelo": float(ll_model)}
    if ok.any():
        mm = np.array([m for m in mk if m is not None])
        out["mercado (Pinnacle cierre)"] = float(-np.mean(np.log(mm[np.arange(len(mm)), res[ok]])))
        out["modelo (mismos partidos)"] = float(-np.mean(np.log(np.clip(pm[ok][np.arange(ok.sum()), res[ok]], 1e-9, 1))))
    return out
