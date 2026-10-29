"""Cuotas en directo opcionales con The Odds API (https://the-odds-api.com).

Plan gratuito: 500 créditos/mes. Cada llamada cuesta (nº mercados x nº regiones),
así que con h2h + totals en la región 'eu' son 2 créditos por refresco.
La clave se pone en .streamlit/secrets.toml como ODDS_API_KEY = "...".
"""
from __future__ import annotations

import difflib
import unicodedata

import pandas as pd
import requests

URL = "https://api.the-odds-api.com/v4/sports/soccer_spain_la_liga/odds"

# Nombres de The Odds API -> nombres de football-data.co.uk
ALIASES = {
    "atletico madrid": "Ath Madrid", "atlético madrid": "Ath Madrid",
    "athletic bilbao": "Ath Bilbao", "athletic club": "Ath Bilbao",
    "real betis": "Betis", "celta vigo": "Celta", "rc celta": "Celta",
    "real sociedad": "Sociedad", "rayo vallecano": "Vallecano",
    "espanyol": "Espanol", "rcd espanyol": "Espanol", "alaves": "Alaves",
    "deportivo alaves": "Alaves", "real oviedo": "Oviedo", "elche cf": "Elche",
    "levante ud": "Levante", "rcd mallorca": "Mallorca", "ud las palmas": "Las Palmas",
    "cd leganes": "Leganes", "real valladolid": "Valladolid", "girona fc": "Girona",
    "ca osasuna": "Osasuna", "sevilla fc": "Sevilla", "valencia cf": "Valencia",
    "villarreal cf": "Villarreal", "fc barcelona": "Barcelona", "getafe cf": "Getafe",
    "racing santander": "Santander", "deportivo la coruna": "La Coruna",
    "sd eibar": "Eibar", "granada cf": "Granada", "ud almeria": "Almeria", "cadiz cf": "Cadiz",
}


def _plain(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return s.lower().strip()


def map_team(name: str, known: list[str]) -> str:
    p = _plain(name)
    if p in ALIASES:
        return ALIASES[p]
    for k in known:
        if _plain(k) == p:
            return k
    m = difflib.get_close_matches(p, [_plain(k) for k in known], n=1, cutoff=0.6)
    if m:
        return next(k for k in known if _plain(k) == m[0])
    return name


def fetch_odds(api_key: str, known_teams: list[str]):
    """Devuelve (fixtures_df, best_odds, creditos_restantes)."""
    r = requests.get(URL, params={
        "apiKey": api_key, "regions": "eu", "markets": "h2h,totals",
        "oddsFormat": "decimal", "dateFormat": "iso",
    }, timeout=20)
    r.raise_for_status()
    remaining = r.headers.get("x-requests-remaining")
    fixtures, best = [], {}
    for ev in r.json():
        h = map_team(ev["home_team"], known_teams)
        a = map_team(ev["away_team"], known_teams)
        t = pd.Timestamp(ev["commence_time"]).tz_convert("Europe/Madrid")
        fixtures.append({"Date": t.tz_localize(None).normalize(), "Time": t.strftime("%H:%M"),
                         "HomeTeam": h, "AwayTeam": a})
        b = best.setdefault((h, a), {})

        def upd(key, price, book):
            if key not in b or price > b[key][0]:
                b[key] = (float(price), book)

        for bk in ev.get("bookmakers", []):
            for mk in bk.get("markets", []):
                for o in mk.get("outcomes", []):
                    if mk["key"] == "h2h":
                        if o["name"] == ev["home_team"]:
                            upd("1", o["price"], bk["title"])
                        elif o["name"] == ev["away_team"]:
                            upd("2", o["price"], bk["title"])
                        else:
                            upd("X", o["price"], bk["title"])
                    elif mk["key"] == "totals" and float(o.get("point", 0)) == 2.5:
                        k = "Más de 2.5 goles" if o["name"] == "Over" else "Menos de 2.5 goles"
                        upd(k, o["price"], bk["title"])
    return pd.DataFrame(fixtures), best, remaining
