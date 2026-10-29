"""Cálculo de valor: probabilidad implícita sin margen, EV, Kelly y combinadas."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def implied_no_vig(odds: list[float]) -> list[float] | None:
    """Quita el margen de la casa (método proporcional)."""
    if any(o is None or not np.isfinite(o) or o <= 1 for o in odds):
        return None
    inv = [1 / o for o in odds]
    s = sum(inv)
    return [x / s for x in inv]


def overround(odds: list[float]) -> float | None:
    if any(o is None or not np.isfinite(o) or o <= 1 for o in odds):
        return None
    return sum(1 / o for o in odds) - 1


def ev(p: float, odds: float) -> float:
    return p * odds - 1


def kelly(p: float, odds: float, fraction: float = 0.25, cap: float = 0.05) -> float:
    if odds <= 1:
        return 0.0
    k = (p * odds - 1) / (odds - 1)
    return float(np.clip(k * fraction, 0, cap))


def fair_odds(p: float) -> float:
    return 1 / p if p > 0 else math.inf


def _first(row, cols):
    for c in cols:
        v = row.get(c)
        if v is not None and pd.notna(v) and v > 1:
            return float(v), c
    return None, None


# Selecciones con cuotas en fixtures.csv: (mercado, selección, clave en el modelo,
#   columnas de "mejor cuota", grupo para quitar margen)
FIX_MARKETS = [
    ("1X2", "1", "1", ["MaxH", "B365H", "AvgH"], ("H", "D", "A")),
    ("1X2", "X", "X", ["MaxD", "B365D", "AvgD"], ("H", "D", "A")),
    ("1X2", "2", "2", ["MaxA", "B365A", "AvgA"], ("H", "D", "A")),
    ("Goles 2.5", "Más de 2.5", "Más de 2.5 goles", ["Max>2.5", "B365>2.5", "Avg>2.5"], (">2.5", "<2.5")),
    ("Goles 2.5", "Menos de 2.5", "Menos de 2.5 goles", ["Max<2.5", "B365<2.5", "Avg<2.5"], (">2.5", "<2.5")),
]

SOURCE_LABEL = {"Max": "mejor casa", "B365": "Bet365", "Avg": "media", "PS": "Pinnacle", "P": "Pinnacle"}


def _source(col: str) -> str:
    for k, v in SOURCE_LABEL.items():
        if col.startswith(k):
            return v
    return col


def market_probs_from_row(row, group) -> list[float] | None:
    """Probabilidad de mercado sin margen: Pinnacle si existe, si no la media."""
    if group == ("H", "D", "A"):
        for pref in ("PS", "Avg", "B365"):
            p = implied_no_vig([row.get(f"{pref}{s}") for s in group])
            if p:
                return p
    else:
        for pref in ("P", "Avg", "B365"):
            p = implied_no_vig([row.get(f"{pref}{s}") for s in group])
            if p:
                return p
    return None


def value_table(fixtures: pd.DataFrame, probs_by_match: dict, model_weight: float,
                kelly_fraction: float, bankroll: float, extra_odds: dict | None = None) -> pd.DataFrame:
    """Tabla de todas las selecciones con cuota disponible y su valor esperado.

    probs_by_match: {(home, away): {clave_mercado: prob_modelo}}
    extra_odds: {(home, away): {clave_mercado: (cuota, casa)}} (p. ej. The Odds API)
    """
    rows = []
    for _, r in fixtures.iterrows():
        key = (r.HomeTeam, r.AwayTeam)
        pm = probs_by_match.get(key)
        if pm is None:
            continue
        for market, sel, mkey, cols, group in FIX_MARKETS:
            odds, col = _first(r, cols)
            src = _source(col) if col else None
            if extra_odds and key in extra_odds and mkey in extra_odds[key]:
                o2, s2 = extra_odds[key][mkey]
                if odds is None or o2 > odds:
                    odds, src = o2, s2
            if odds is None:
                continue
            mp = market_probs_from_row(r, group)
            i = group.index({"1": "H", "X": "D", "2": "A"}.get(sel, ">2.5" if "Más" in sel else "<2.5"))
            p_mkt = mp[i] if mp else None
            p_mod = pm[mkey]
            p = model_weight * p_mod + (1 - model_weight) * p_mkt if p_mkt is not None else p_mod
            e = ev(p, odds)
            k = kelly(p, odds, kelly_fraction)
            rows.append({
                "Fecha": r.Date, "Hora": r.get("Time", ""), "Partido": f"{r.HomeTeam} - {r.AwayTeam}",
                "Local": r.HomeTeam, "Visitante": r.AwayTeam,
                "Mercado": market, "Apuesta": sel, "Cuota": odds, "Casa": src,
                "P. modelo": p_mod, "P. mercado": p_mkt, "P. usada": p,
                "Cuota justa": fair_odds(p), "EV": e, "Stake": k * bankroll,
            })
    return pd.DataFrame(rows)


def combo(legs: list[tuple[float, float]]) -> dict:
    """legs = [(prob, cuota)] asumiendo independencia entre patas."""
    p = float(np.prod([l[0] for l in legs]))
    o = float(np.prod([l[1] for l in legs]))
    return {"prob": p, "cuota": o, "ev": p * o - 1, "cuota_justa": fair_odds(p)}
