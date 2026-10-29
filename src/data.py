"""Carga de datos de La Liga (Primera División) desde football-data.co.uk.

- Histórico por temporada: https://www.football-data.co.uk/mmz4281/{AABB}/SP1.csv
- Próximos partidos con cuotas: https://www.football-data.co.uk/fixtures.csv

Si existe la variable de entorno DATA_DIR, se leen los CSV de esa carpeta
(útil para trabajar sin conexión o para tests).
"""
from __future__ import annotations

import io
import os
from datetime import date

import pandas as pd
import requests

BASE = "https://www.football-data.co.uk"
DIV = "SP1"
HEADERS = {"User-Agent": "Mozilla/5.0 (apuestas-laliga)"}

# Columnas que usamos (si alguna falta en un CSV se rellena con NaN)
NUM_COLS = [
    "FTHG", "FTAG", "HTHG", "HTAG", "HS", "AS", "HST", "AST", "HF", "AF",
    "HC", "AC", "HY", "AY", "HR", "AR",
]
ODDS_COLS = [
    # Cuotas pre-partido (apertura)
    "B365H", "B365D", "B365A", "PSH", "PSD", "PSA", "MaxH", "MaxD", "MaxA",
    "AvgH", "AvgD", "AvgA", "B365>2.5", "B365<2.5", "P>2.5", "P<2.5",
    "Max>2.5", "Max<2.5", "Avg>2.5", "Avg<2.5",
    # Cuotas de cierre
    "B365CH", "B365CD", "B365CA", "PSCH", "PSCD", "PSCA", "MaxCH", "MaxCD",
    "MaxCA", "AvgCH", "AvgCD", "AvgCA", "B365C>2.5", "B365C<2.5", "PC>2.5",
    "PC<2.5", "MaxC>2.5", "MaxC<2.5", "AvgC>2.5", "AvgC<2.5",
]


def season_code(start_year: int) -> str:
    """2026 -> '2627'."""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def current_season_start(today: date | None = None) -> int:
    today = today or date.today()
    return today.year if today.month >= 7 else today.year - 1


def _read_csv_bytes(content: bytes) -> pd.DataFrame:
    for enc in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(content), encoding=enc, on_bad_lines="skip")
        except UnicodeDecodeError:
            continue
    return pd.read_csv(io.BytesIO(content), encoding="latin-1", on_bad_lines="skip")


def _fetch(url: str, local_name: str) -> pd.DataFrame | None:
    data_dir = os.environ.get("DATA_DIR")
    if data_dir:
        path = os.path.join(data_dir, local_name)
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            return _read_csv_bytes(f.read())
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        if r.status_code != 200 or len(r.content) < 200:
            return None
        return _read_csv_bytes(r.content)
    except requests.RequestException:
        return None


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    df = df.dropna(subset=["HomeTeam", "AwayTeam"])
    df["HomeTeam"] = df["HomeTeam"].astype(str).str.strip()
    df["AwayTeam"] = df["AwayTeam"].astype(str).str.strip()
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    for c in NUM_COLS + ODDS_COLS:
        if c not in df.columns:
            df[c] = pd.NA
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "Time" not in df.columns:
        df["Time"] = ""
    return df


def load_season(start_year: int) -> pd.DataFrame | None:
    code = season_code(start_year)
    df = _fetch(f"{BASE}/mmz4281/{code}/{DIV}.csv", f"SP1_{code}.csv")
    if df is None or df.empty:
        return None
    df = _normalize(df)
    df = df.dropna(subset=["FTHG", "FTAG"])  # solo partidos jugados
    df["Season"] = f"{start_year}/{str(start_year + 1)[-2:]}"
    df["SeasonStart"] = start_year
    return df


def load_history(n_seasons: int = 5, today: date | None = None) -> pd.DataFrame:
    cur = current_season_start(today)
    frames = []
    for y in range(cur - n_seasons + 1, cur + 1):
        d = load_season(y)
        if d is not None and len(d):
            frames.append(d)
    if not frames:
        raise RuntimeError(
            "No se han podido descargar datos de football-data.co.uk. "
            "Revisa la conexión o define DATA_DIR con los CSV en local."
        )
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def load_fixtures() -> pd.DataFrame:
    """Próximos partidos de Primera con cuotas (football-data fixtures.csv)."""
    df = _fetch(f"{BASE}/fixtures.csv", "fixtures.csv")
    if df is None or df.empty:
        return pd.DataFrame()
    df.columns = [c.strip() for c in df.columns]
    if "Div" in df.columns:
        df = df[df["Div"].astype(str).str.strip() == DIV]
    if df.empty:
        return df
    df = _normalize(df)
    return df.sort_values(["Date", "Time"]).reset_index(drop=True)


def current_teams(hist: pd.DataFrame) -> list[str]:
    cur = hist["SeasonStart"].max()
    d = hist[hist["SeasonStart"] == cur]
    return sorted(set(d["HomeTeam"]) | set(d["AwayTeam"]))


def standings(hist: pd.DataFrame, season_start: int | None = None) -> pd.DataFrame:
    season_start = season_start or hist["SeasonStart"].max()
    d = hist[hist["SeasonStart"] == season_start]
    rows = {}
    for _, m in d.iterrows():
        for team, gf, ga, side in (
            (m.HomeTeam, m.FTHG, m.FTAG, "H"),
            (m.AwayTeam, m.FTAG, m.FTHG, "A"),
        ):
            r = rows.setdefault(team, dict(Equipo=team, PJ=0, G=0, E=0, P=0, GF=0, GC=0, Pts=0, forma=[]))
            r["PJ"] += 1
            r["GF"] += gf
            r["GC"] += ga
            if gf > ga:
                r["G"] += 1; r["Pts"] += 3; r["forma"].append("V")
            elif gf == ga:
                r["E"] += 1; r["Pts"] += 1; r["forma"].append("E")
            else:
                r["P"] += 1; r["forma"].append("D")
    t = pd.DataFrame(rows.values())
    if t.empty:
        return t
    t["DG"] = t["GF"] - t["GC"]
    t["Últimos 5"] = t["forma"].apply(lambda f: " ".join(f[-5:]))
    t = t.drop(columns="forma")
    t = t.sort_values(["Pts", "DG", "GF"], ascending=False).reset_index(drop=True)
    t.index = t.index + 1
    t[["GF", "GC", "DG"]] = t[["GF", "GC", "DG"]].astype(int)
    return t
