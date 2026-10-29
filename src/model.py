"""Modelos estadísticos.

1) Goles: Poisson bivariante tipo Dixon-Coles.
   log(lambda_local)    = mu + ventaja_local + ataque_local + defensa_visitante
   log(lambda_visitante) = mu + ataque_visitante + defensa_local
   - Ponderación temporal exponencial (los partidos recientes pesan más).
   - Regularización L2 (encoge a los equipos con pocos datos hacia la media).
   - Ajuste rho de Dixon-Coles para los marcadores bajos (0-0, 1-0, 0-1, 1-1).

2) Córners y tarjetas: el mismo GLM multiplicativo (Poisson con log-link)
   sobre los conteos de cada equipo, y distribución binomial negativa para
   capturar la sobredispersión al calcular probabilidades de over/under.
   - Córners: "ataque" = córners que genera, "defensa" = córners que concede.
   - Tarjetas: "ataque" = tarjetas que recibe, "defensa" = tarjetas que provoca
     en el rival.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import nbinom, poisson

XI_DEFAULT = 0.0019  # decaimiento por día -> vida media ~ 1 año
MAX_GOALS = 10


# ----------------------------------------------------------------------------
# GLM multiplicativo genérico (sirve para goles, córners y tarjetas)
# ----------------------------------------------------------------------------
@dataclass
class TeamRates:
    teams: list[str]
    mu: float
    home: float
    att: np.ndarray
    dfn: np.ndarray
    dispersion: float = 1.0  # varianza / media (1 = Poisson)
    idx: dict = field(default_factory=dict)

    def __post_init__(self):
        self.idx = {t: i for i, t in enumerate(self.teams)}

    def _params(self, team: str) -> tuple[float, float]:
        i = self.idx.get(team)
        if i is not None:
            return self.att[i], self.dfn[i]
        # Equipo sin histórico (recién ascendido): perfil de equipo débil
        return float(np.percentile(self.att, 25)), float(np.percentile(self.dfn, 75))

    def expected(self, home_team: str, away_team: str) -> tuple[float, float]:
        ah, dh = self._params(home_team)
        aa, da = self._params(away_team)
        lam_h = np.exp(self.mu + self.home + ah + da)
        lam_a = np.exp(self.mu + aa + dh)
        return float(lam_h), float(lam_a)

    def table(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"Equipo": self.teams, "ataque": np.exp(self.att), "defensa": np.exp(self.dfn)}
        )


def expected_vec(rates: TeamRates, homes, aways) -> tuple[np.ndarray, np.ndarray]:
    """Versión vectorizada de TeamRates.expected."""
    fa, fd = float(np.percentile(rates.att, 25)), float(np.percentile(rates.dfn, 75))
    hi = np.array([rates.idx.get(t, -1) for t in homes])
    ai = np.array([rates.idx.get(t, -1) for t in aways])
    att_h = np.where(hi >= 0, rates.att[hi], fa); dfn_h = np.where(hi >= 0, rates.dfn[hi], fd)
    att_a = np.where(ai >= 0, rates.att[ai], fa); dfn_a = np.where(ai >= 0, rates.dfn[ai], fd)
    lh = np.exp(rates.mu + rates.home + att_h + dfn_a)
    la = np.exp(rates.mu + att_a + dfn_h)
    return lh, la


def time_weights(dates: pd.Series, ref_date: pd.Timestamp, xi: float) -> np.ndarray:
    days = (ref_date - dates).dt.days.clip(lower=0).to_numpy(dtype=float)
    return np.exp(-xi * days)


def fit_rates(
    df: pd.DataFrame,
    y_home: str,
    y_away: str,
    ref_date: pd.Timestamp | None = None,
    xi: float = XI_DEFAULT,
    l2: float = 2.0,
) -> TeamRates:
    """Ajusta el GLM con gradiente analítico (rápido: < 0,1 s)."""
    d = df.dropna(subset=[y_home, y_away])
    ref_date = ref_date or d["Date"].max()
    d = d[d["Date"] <= ref_date]
    teams = sorted(set(d["HomeTeam"]) | set(d["AwayTeam"]))
    n = len(teams)
    idx = {t: i for i, t in enumerate(teams)}
    hi = d["HomeTeam"].map(idx).to_numpy()
    ai = d["AwayTeam"].map(idx).to_numpy()
    yh = d[y_home].to_numpy(dtype=float)
    ya = d[y_away].to_numpy(dtype=float)
    w = time_weights(d["Date"], ref_date, xi)

    def unpack(p):
        return p[0], p[1], p[2 : 2 + n], p[2 + n :]

    def f(p):
        mu, home, att, dfn = unpack(p)
        eta_h = mu + home + att[hi] + dfn[ai]
        eta_a = mu + att[ai] + dfn[hi]
        lh, la = np.exp(eta_h), np.exp(eta_a)
        nll = np.sum(w * (lh - yh * eta_h)) + np.sum(w * (la - ya * eta_a))
        nll += l2 * (np.sum(att**2) + np.sum(dfn**2))
        rh, ra = w * (lh - yh), w * (la - ya)
        g = np.zeros_like(p)
        g[0] = rh.sum() + ra.sum()
        g[1] = rh.sum()
        g_att = np.bincount(hi, rh, n) + np.bincount(ai, ra, n) + 2 * l2 * att
        g_dfn = np.bincount(ai, rh, n) + np.bincount(hi, ra, n) + 2 * l2 * dfn
        g[2 : 2 + n] = g_att
        g[2 + n :] = g_dfn
        return nll, g

    mean_y = max(np.average(np.r_[yh, ya], weights=np.r_[w, w]), 0.1)
    p0 = np.zeros(2 + 2 * n)
    p0[0] = np.log(mean_y)
    res = minimize(f, p0, jac=True, method="L-BFGS-B")
    mu, home, att, dfn = unpack(res.x)

    # Sobredispersión (Pearson) para conteos tipo córners/tarjetas
    lh = np.exp(mu + home + att[hi] + dfn[ai])
    la = np.exp(mu + att[ai] + dfn[hi])
    resid = np.r_[(yh - lh) ** 2 / lh, (ya - la) ** 2 / la]
    ww = np.r_[w, w]
    disp = float(np.sum(ww * resid) / max(np.sum(ww) - (2 + 2 * n), 1))
    disp = float(np.clip(disp, 1.0, 3.0))
    return TeamRates(teams, float(mu), float(home), att.copy(), dfn.copy(), disp)


# ----------------------------------------------------------------------------
# Goles (Dixon-Coles)
# ----------------------------------------------------------------------------
def _tau_matrix(lh: float, la: float, rho: float, size: int) -> np.ndarray:
    t = np.ones((size, size))
    t[0, 0] = 1 - lh * la * rho
    t[0, 1] = 1 + lh * rho
    t[1, 0] = 1 + la * rho
    t[1, 1] = 1 - rho
    return t


@dataclass
class GoalsModel:
    rates: TeamRates
    rho: float

    def score_matrix(self, home: str, away: str) -> np.ndarray:
        lh, la = self.rates.expected(home, away)
        g = np.arange(MAX_GOALS + 1)
        m = np.outer(poisson.pmf(g, lh), poisson.pmf(g, la))
        m *= _tau_matrix(lh, la, self.rho, MAX_GOALS + 1)
        m = np.clip(m, 0, None)
        return m / m.sum()

    def expected(self, home: str, away: str) -> tuple[float, float]:
        return self.rates.expected(home, away)


def fit_goals(df: pd.DataFrame, ref_date=None, xi=XI_DEFAULT, l2=2.0) -> GoalsModel:
    rates = fit_rates(df, "FTHG", "FTAG", ref_date, xi, l2)
    d = df.dropna(subset=["FTHG", "FTAG"])
    ref = ref_date or d["Date"].max()
    d = d[d["Date"] <= ref]
    lh, la = expected_vec(rates, d.HomeTeam, d.AwayTeam)
    x, y = d["FTHG"].to_numpy(), d["FTAG"].to_numpy()
    w = time_weights(d["Date"], ref, xi)

    def nll(rho):
        t = np.ones_like(lh)
        m00 = (x == 0) & (y == 0); t[m00] = 1 - lh[m00] * la[m00] * rho
        m01 = (x == 0) & (y == 1); t[m01] = 1 + lh[m01] * rho
        m10 = (x == 1) & (y == 0); t[m10] = 1 + la[m10] * rho
        m11 = (x == 1) & (y == 1); t[m11] = 1 - rho
        t = np.clip(t, 1e-9, None)
        return -np.sum(w * np.log(t))

    rho = minimize_scalar(nll, bounds=(-0.2, 0.2), method="bounded").x
    return GoalsModel(rates, float(rho))


def goal_markets(m: np.ndarray) -> dict[str, float]:
    """Probabilidades de los mercados de goles a partir de la matriz de marcadores."""
    n = m.shape[0]
    hg, ag = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    tot = hg + ag
    p = {
        "1": m[hg > ag].sum(),
        "X": m[hg == ag].sum(),
        "2": m[hg < ag].sum(),
    }
    p["1X"] = p["1"] + p["X"]
    p["X2"] = p["X"] + p["2"]
    p["12"] = p["1"] + p["2"]
    for line in (0.5, 1.5, 2.5, 3.5, 4.5):
        p[f"Más de {line} goles"] = m[tot > line].sum()
        p[f"Menos de {line} goles"] = m[tot < line].sum()
    p["Ambos marcan: Sí"] = m[(hg > 0) & (ag > 0)].sum()
    p["Ambos marcan: No"] = 1 - p["Ambos marcan: Sí"]
    for line in (0.5, 1.5, 2.5):
        p[f"Local más de {line}"] = m[hg > line].sum()
        p[f"Visitante más de {line}"] = m[ag > line].sum()
    p["Local portería a cero"] = m[ag == 0].sum()
    p["Visitante portería a cero"] = m[hg == 0].sum()
    return {k: float(v) for k, v in p.items()}


def top_scores(m: np.ndarray, k: int = 5) -> list[tuple[str, float]]:
    flat = [(f"{i}-{j}", m[i, j]) for i in range(m.shape[0]) for j in range(m.shape[1])]
    return sorted(flat, key=lambda x: -x[1])[:k]


# ----------------------------------------------------------------------------
# Córners y tarjetas (binomial negativa)
# ----------------------------------------------------------------------------
def nb_pmf(mean: float, disp: float, kmax: int) -> np.ndarray:
    k = np.arange(kmax + 1)
    if disp <= 1.0001:
        return poisson.pmf(k, mean)
    # var = disp * mean  ->  n = mean / (disp - 1), p = 1 / disp
    n = mean / (disp - 1)
    return nbinom.pmf(k, n, 1 / disp)


@dataclass
class CountModel:
    rates: TeamRates
    name: str  # "córners" o "tarjetas"

    def expected(self, home: str, away: str, factor: float = 1.0) -> tuple[float, float]:
        lh, la = self.rates.expected(home, away)
        return lh * factor, la * factor

    def distributions(self, home: str, away: str, factor: float = 1.0, kmax: int = 30):
        lh, la = self.expected(home, away, factor)
        ph = nb_pmf(lh, self.rates.dispersion, kmax)
        pa = nb_pmf(la, self.rates.dispersion, kmax)
        pt = np.convolve(ph, pa)[: 2 * kmax + 1]
        return ph, pa, pt

    def markets(self, home: str, away: str, factor: float = 1.0, lines_total=None, lines_team=None):
        ph, pa, pt = self.distributions(home, away, factor)
        lh, la = self.expected(home, away, factor)
        if lines_total is None:
            c = lh + la
            base = np.floor(c) + 0.5
            lines_total = [base + d for d in (-2, -1, 0, 1, 2) if base + d > 0]
        if lines_team is None:
            lines_team = sorted({np.floor(lh) + 0.5, np.floor(la) + 0.5, max(np.floor(lh) - 0.5, 0.5), max(np.floor(la) - 0.5, 0.5)})
        rows = []
        k_t = np.arange(len(pt))
        k_s = np.arange(len(ph))
        for L in lines_total:
            p_over = float(pt[k_t > L].sum())
            rows.append(("Total", L, p_over, 1 - p_over))
        for L in lines_team:
            p_over = float(ph[k_s > L].sum())
            rows.append(("Local", L, p_over, 1 - p_over))
            p_over = float(pa[k_s > L].sum())
            rows.append(("Visitante", L, p_over, 1 - p_over))
        p_home_more = float(sum(ph[i] * pa[:i].sum() for i in range(len(ph))))
        p_equal = float(np.sum(ph * pa))
        return rows, {"Local más": p_home_more, "Igual": p_equal, "Visitante más": 1 - p_home_more - p_equal}


def fit_corners(df, ref_date=None, xi=XI_DEFAULT) -> CountModel:
    return CountModel(fit_rates(df, "HC", "AC", ref_date, xi, l2=4.0), "córners")


def fit_cards(df, ref_date=None, xi=XI_DEFAULT) -> CountModel:
    d = df.copy()
    d["H_CARDS"] = d["HY"] + d["HR"].fillna(0)
    d["A_CARDS"] = d["AY"] + d["AR"].fillna(0)
    return CountModel(fit_rates(d, "H_CARDS", "A_CARDS", ref_date, xi, l2=4.0), "tarjetas")
