"""Value bets La Liga — dashboard Streamlit.

Ejecutar en local:  streamlit run app.py
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import backtest as bt
from src.data import current_teams, load_fixtures, load_history, standings
from src.model import fit_cards, fit_corners, fit_goals, goal_markets, top_scores
from src.odds_api import fetch_odds
from src.value import combo, ev, fair_odds, kelly, value_table

st.set_page_config(page_title="Value bets La Liga", page_icon="⚽", layout="wide")

BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#8a8984"
GOOD, BAD = "#0ca30c", "#d03b3b"


def fig_style(fig, height=340):
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="#fcfcfb",
        paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#52514e"), hovermode="x unified",
        legend=dict(orientation="h", y=1.08, x=0),
    )
    fig.update_xaxes(showgrid=False, linecolor="#d8d7d2")
    fig.update_yaxes(gridcolor="#ecebe7", zerolinecolor="#c9c8c2")
    return fig


# ----------------------------------------------------------------------------
# Carga y modelos (cacheados)
# ----------------------------------------------------------------------------
@st.cache_data(ttl=6 * 3600, show_spinner="Descargando histórico de Primera…")
def get_history(n):
    return load_history(n)


@st.cache_data(ttl=3600, show_spinner="Descargando próximos partidos…")
def get_fixtures():
    return load_fixtures()


@st.cache_data(ttl=1800, show_spinner="Consultando cuotas en directo…")
def get_live_odds(key, teams):
    return fetch_odds(key, list(teams))


@st.cache_resource(show_spinner="Entrenando modelos…")
def get_models(hist_hash, _hist, xi):
    return fit_goals(_hist, xi=xi), fit_corners(_hist, xi=xi), fit_cards(_hist, xi=xi)


@st.cache_data(show_spinner="Backtest walk-forward (reentrena cada semana)…")
def get_backtest(hist_hash, _hist, seasons, xi):
    return bt.walk_forward_predictions(_hist, list(seasons), xi=xi)


# ----------------------------------------------------------------------------
# Barra lateral
# ----------------------------------------------------------------------------
st.sidebar.title("Ajustes")
model_w = st.sidebar.slider(
    "Peso del modelo frente al mercado", 0.0, 1.0, 0.5, 0.05,
    help="1 = solo el modelo; 0 = solo la probabilidad sin margen de Pinnacle/media. "
         "Mezclar reduce falsos positivos: el mercado de Primera es muy eficiente.")
min_ev = st.sidebar.slider("EV mínimo para marcar valor", 0.0, 0.20, 0.04, 0.01, format="%.2f")
max_odds = st.sidebar.slider("Cuota máxima", 1.5, 15.0, 6.0, 0.5)
bankroll = st.sidebar.number_input("Bankroll (€)", 10.0, 100000.0, 200.0, 10.0)
kelly_f = st.sidebar.slider("Fracción de Kelly", 0.05, 1.0, 0.25, 0.05,
                            help="Kelly completo es muy agresivo; 1/4 es lo habitual. Stake máx. 5 % del bankroll.")
half_life = st.sidebar.select_slider("Memoria del modelo (vida media)", [90, 180, 270, 365, 540, 730], 365,
                                     format_func=lambda d: f"{d} días")
xi = math.log(2) / half_life
n_seasons = st.sidebar.slider("Temporadas de histórico", 3, 8, 5)
if st.sidebar.button("Refrescar datos"):
    st.cache_data.clear(); st.cache_resource.clear(); st.rerun()

try:
    odds_key = st.secrets["ODDS_API_KEY"]
except Exception:
    odds_key = ""

st.sidebar.caption(
    "Datos: football-data.co.uk (resultados, córners, tarjetas y cuotas). "
    + ("Cuotas en directo: The Odds API activa." if odds_key else "Añade ODDS_API_KEY en secrets para cuotas en directo.")
)

# ----------------------------------------------------------------------------
# Datos
# ----------------------------------------------------------------------------
try:
    hist = get_history(n_seasons)
except RuntimeError as e:
    st.error(str(e)); st.stop()

hist_hash = (len(hist), str(hist["Date"].max()), n_seasons)
gm, cm, km = get_models(hist_hash, hist, xi)
teams = current_teams(hist)
fixtures = get_fixtures()

extra_odds, credits = None, None
if odds_key:
    try:
        live_fx, extra_odds, credits = get_live_odds(odds_key, tuple(teams))
        if not live_fx.empty:
            known = set(zip(fixtures.HomeTeam, fixtures.AwayTeam)) if not fixtures.empty else set()
            add = live_fx[[(h, a) not in known for h, a in zip(live_fx.HomeTeam, live_fx.AwayTeam)]]
            fixtures = pd.concat([fixtures, add], ignore_index=True)
    except Exception as e:  # noqa: BLE001
        st.sidebar.warning(f"The Odds API: {e}")
if credits:
    st.sidebar.caption(f"Créditos restantes The Odds API: {credits}")

if not fixtures.empty:
    fixtures = fixtures[fixtures["Date"] >= pd.Timestamp.today().normalize() - pd.Timedelta(days=1)]
    fixtures = fixtures.sort_values(["Date", "Time"]).reset_index(drop=True)

probs = {}
for _, r in fixtures.iterrows():
    probs[(r.HomeTeam, r.AwayTeam)] = goal_markets(gm.score_matrix(r.HomeTeam, r.AwayTeam))

vt = value_table(fixtures, probs, model_w, kelly_f, bankroll, extra_odds) if len(fixtures) else pd.DataFrame()
if not vt.empty:
    vt = vt[vt["Cuota"] <= max_odds]

if "legs" not in st.session_state:
    st.session_state.legs = []  # apuestas añadidas a mano (córners/tarjetas)

# ----------------------------------------------------------------------------
st.title("Value bets · Primera División")
last = hist["Date"].max().strftime("%d/%m/%Y")
st.caption(f"Histórico hasta el {last} · {len(hist)} partidos · {len(fixtures)} próximos partidos")

tabs = st.tabs(["Infravaloradas", "Partido a partido", "Combinadas", "Equipos", "Clasificación", "Backtest", "Cómo funciona"])

DIAS = ["lun", "mar", "mié", "jue", "vie", "sáb", "dom"]
PCT = st.column_config.NumberColumn(format="percent")


def show_value_df(df, key):
    cols = ["Fecha", "Hora", "Partido", "Mercado", "Apuesta", "Cuota", "Casa", "Cuota justa",
            "P. modelo", "P. mercado", "EV", "Stake"]
    d = df[cols].copy()
    d["Fecha"] = d["Fecha"].apply(lambda x: f"{DIAS[x.weekday()]} {x:%d/%m}")
    st.dataframe(
        d, hide_index=True, width="stretch", key=key,
        column_config={
            "P. modelo": PCT, "P. mercado": PCT, "EV": st.column_config.NumberColumn(format="percent"),
            "Cuota": st.column_config.NumberColumn(format="%.2f"),
            "Cuota justa": st.column_config.NumberColumn(format="%.2f"),
            "Stake": st.column_config.NumberColumn("Stake (€)", format="%.2f"),
        })


# --- 1. Infravaloradas -------------------------------------------------------
with tabs[0]:
    if fixtures.empty:
        st.info("No hay próximos partidos publicados todavía en fixtures.csv. "
                "Suele actualizarse a mitad de semana; con ODDS_API_KEY se cargan antes.")
    elif vt.empty:
        st.info("No hay cuotas disponibles para los próximos partidos.")
    else:
        vb = vt[vt["EV"] >= min_ev].sort_values("EV", ascending=False)
        c1, c2, c3 = st.columns(3)
        c1.metric("Apuestas con valor", len(vb))
        c2.metric("Mejor EV", f"{vb['EV'].max():.1%}" if len(vb) else "—")
        c3.metric("Stake total sugerido", f"{vb['Stake'].sum():.2f} €")
        st.subheader("Cuotas que pagan más de lo que dice la estadística")
        if len(vb):
            show_value_df(vb, "vb")
            st.caption("EV = probabilidad usada × cuota − 1. Cuota justa = 1 / probabilidad usada. "
                       "Stake = Kelly fraccional sobre tu bankroll. Córners y tarjetas: pestaña Partido a partido.")
        else:
            st.success("Ninguna selección supera el EV mínimo esta jornada. No apostar también es una decisión válida.")
        with st.expander("Todas las selecciones de la jornada"):
            show_value_df(vt.sort_values(["Fecha", "Partido"]), "all")

# --- 2. Partido a partido ----------------------------------------------------
with tabs[1]:
    options = [f"{r.HomeTeam} - {r.AwayTeam}" for _, r in fixtures.iterrows()]
    mode = st.radio("Partido", ["Próximos", "Cualquier cruce"], horizontal=True, label_visibility="collapsed")
    if mode == "Próximos" and options:
        sel = st.selectbox("Partido", options)
        home, away = sel.split(" - ", 1)
    else:
        c1, c2 = st.columns(2)
        home = c1.selectbox("Local", teams, index=0)
        away = c2.selectbox("Visitante", teams, index=1)
    if home == away:
        st.warning("Elige dos equipos distintos.")
        away = next(t for t in teams if t != home)

    m = gm.score_matrix(home, away)
    gp = goal_markets(m)
    lh, la = gm.expected(home, away)
    st.subheader(f"{home} vs {away}")
    c = st.columns(5)
    c[0].metric("Goles esperados", f"{lh:.2f} - {la:.2f}")
    for col, (lab, k) in zip(c[1:], [("1", "1"), ("X", "X"), ("2", "2"), ("Más de 2.5", "Más de 2.5 goles")]):
        col.metric(f"{lab} · cuota justa {fair_odds(gp[k]):.2f}", f"{gp[k]:.0%}")

    def market_editor(rows, key, title):
        """Tabla con cuota justa y columna editable para la cuota de tu casa."""
        df = pd.DataFrame(rows, columns=["Mercado", "Probabilidad"])
        df["Cuota justa"] = 1 / df["Probabilidad"]
        df["Cuota casa"] = np.nan
        st.markdown(f"**{title}**")
        ed = st.data_editor(
            df, hide_index=True, width="stretch", key=key, disabled=["Mercado", "Probabilidad", "Cuota justa"],
            column_config={"Probabilidad": PCT, "Cuota justa": st.column_config.NumberColumn(format="%.2f"),
                           "Cuota casa": st.column_config.NumberColumn("Cuota de tu casa", min_value=1.01, format="%.2f")})
        ed["EV"] = ed["Probabilidad"] * ed["Cuota casa"] - 1
        hits = ed.dropna(subset=["Cuota casa"])
        for _, h in hits.iterrows():
            stake = kelly(h.Probabilidad, h["Cuota casa"], kelly_f) * bankroll
            col_a, col_b = st.columns([4, 1])
            tag = "VALOR" if h.EV >= min_ev else "sin valor"
            col_a.write(f"{h.Mercado} @ {h['Cuota casa']:.2f} → EV **{h.EV:+.1%}** ({tag})"
                        + (f" · stake {stake:.2f} €" if h.EV > 0 else ""))
            if col_b.button("A combinada", key=f"{key}_{h.Mercado}"):
                st.session_state.legs.append({"Partido": f"{home} - {away}", "Apuesta": h.Mercado,
                                              "Prob": float(h.Probabilidad), "Cuota": float(h["Cuota casa"])})
                st.toast("Añadida a la combinada")

    left, right = st.columns(2)
    with left:
        rows = [(k, v) for k, v in gp.items() if k not in ("1", "X", "2")]
        rows = [("1", gp["1"]), ("X", gp["X"]), ("2", gp["2"])] + rows
        market_editor(rows, f"g_{home}_{away}", "Goles")
        st.markdown("**Marcadores más probables**")
        st.write(" · ".join(f"{s} ({p:.0%})" for s, p in top_scores(m, 6)))

    with right:
        ch, ca = cm.expected(home, away)
        st.markdown(f"**Córners esperados:** {ch:.1f} - {ca:.1f} (total {ch + ca:.1f})")
        rows_c, cmp_c = cm.markets(home, away)
        rc = []
        for side, L, po, pu in rows_c:
            lab = {"Total": "Córners totales", "Local": f"Córners {home}", "Visitante": f"Córners {away}"}[side]
            rc += [(f"{lab} más de {L}", po), (f"{lab} menos de {L}", pu)]
        rc += [(f"Más córners: {home}", cmp_c["Local más"]), (f"Más córners: {away}", cmp_c["Visitante más"])]
        market_editor(rc, f"c_{home}_{away}", "Córners")

        ref_f = st.slider("Árbitro: factor de tarjetas", 0.7, 1.4, 1.0, 0.05,
                          help="football-data no publica el árbitro en LaLiga. Si el designado es muy tarjetero "
                               "(p. ej. saca un 20 % más que la media), pon 1.20.")
        kh, ka = km.expected(home, away, ref_f)
        st.markdown(f"**Tarjetas esperadas** (amarillas + rojas): {kh:.1f} - {ka:.1f} (total {kh + ka:.1f})")
        rows_k, cmp_k = km.markets(home, away, ref_f)
        rk = []
        for side, L, po, pu in rows_k:
            lab = {"Total": "Tarjetas totales", "Local": f"Tarjetas {home}", "Visitante": f"Tarjetas {away}"}[side]
            rk += [(f"{lab} más de {L}", po), (f"{lab} menos de {L}", pu)]
        rk += [(f"Más tarjetas: {home}", cmp_k["Local más"]), (f"Más tarjetas: {away}", cmp_k["Visitante más"])]
        market_editor(rk, f"k_{home}_{away}_{ref_f}", "Tarjetas")

    st.markdown("**Forma reciente (últimos 6 partidos de cada equipo)**")
    f1, f2 = st.columns(2)
    for col, t in ((f1, home), (f2, away)):
        d = hist[(hist.HomeTeam == t) | (hist.AwayTeam == t)].tail(6).copy()
        d["Res"] = d.apply(lambda r: f"{int(r.FTHG)}-{int(r.FTAG)}", axis=1)
        d["Córners"] = d.apply(lambda r: f"{r.HC:.0f}-{r.AC:.0f}" if pd.notna(r.HC) else "", axis=1)
        d["Tarjetas"] = d.apply(lambda r: f"{r.HY + np.nan_to_num(r.HR):.0f}-{r.AY + np.nan_to_num(r.AR):.0f}" if pd.notna(r.HY) else "", axis=1)
        d["Fecha"] = d.Date.dt.strftime("%d/%m/%y")
        col.caption(t)
        col.dataframe(d[["Fecha", "HomeTeam", "AwayTeam", "Res", "Córners", "Tarjetas"]].iloc[::-1]
                      .rename(columns={"HomeTeam": "Local", "AwayTeam": "Visitante"}),
                      hide_index=True, width="stretch")

# --- 3. Combinadas -----------------------------------------------------------
with tabs[2]:
    st.subheader("Combinadas")
    st.caption("Las combinadas multiplican también el margen de la casa: solo tienen sentido si **cada pata** tiene valor. "
               "Se asume independencia entre patas, por eso no se permite repetir partido.")
    pool = []
    if not vt.empty:
        for _, r in vt.iterrows():
            pool.append({"Partido": r.Partido, "Apuesta": f"{r.Mercado}: {r.Apuesta}", "Prob": r["P. usada"], "Cuota": r.Cuota})
    pool += st.session_state.legs
    if not pool:
        st.info("No hay selecciones disponibles. Añade córners o tarjetas desde Partido a partido.")
    else:
        pdf = pd.DataFrame(pool)
        pdf["EV"] = pdf.Prob * pdf.Cuota - 1
        pdf["label"] = pdf.apply(lambda r: f"{r.Partido} · {r.Apuesta} @ {r.Cuota:.2f} (EV {r.EV:+.1%})", axis=1)

        st.markdown("**Sugeridas automáticamente** (solo patas con EV ≥ mínimo, partidos distintos)")
        s1, s2 = st.columns(2)
        min_p = s1.slider("Probabilidad mínima de acierto de la combinada", 0.05, 0.6, 0.15, 0.05, format="%.2f")
        order = s2.radio("Ordenar por", ["EV", "Prob. acierto"], horizontal=True)
        good = pdf[pdf.EV >= min_ev].sort_values("EV", ascending=False).head(12)
        sug = []
        for n in (2, 3):
            for comb in itertools.combinations(good.index, n):
                legs = pdf.loc[list(comb)]
                if legs.Partido.nunique() < n:
                    continue
                c = combo(list(zip(legs.Prob, legs.Cuota)))
                if c["prob"] < min_p:
                    continue
                sug.append({"Patas": " + ".join(legs.label.str.split(" \\(EV").str[0]), "Nº": n,
                            "Cuota": c["cuota"], "Prob. acierto": c["prob"], "EV": c["ev"]})
        if sug:
            sdf = pd.DataFrame(sug).sort_values([order], ascending=False).head(10)
            st.dataframe(sdf, hide_index=True, width="stretch",
                         column_config={"Prob. acierto": PCT, "EV": PCT, "Cuota": st.column_config.NumberColumn(format="%.2f")})
        else:
            st.write("No hay combinadas que cumplan los filtros.")

        st.markdown("**Construye la tuya**")
        chosen = st.multiselect("Patas", pdf.label.tolist())
        if chosen:
            legs = pdf[pdf.label.isin(chosen)]
            if legs.Partido.duplicated().any():
                st.warning("Hay dos patas del mismo partido: están correlacionadas y la probabilidad real no es el producto.")
            c = combo(list(zip(legs.Prob, legs.Cuota)))
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Cuota combinada", f"{c['cuota']:.2f}")
            k2.metric("Prob. de acierto", f"{c['prob']:.1%}")
            k3.metric("Cuota justa", f"{c['cuota_justa']:.2f}")
            k4.metric("EV", f"{c['ev']:+.1%}")
            st.caption(f"Stake Kelly sugerido: {kelly(c['prob'], c['cuota'], kelly_f) * bankroll:.2f} €")
        if st.session_state.legs and st.button("Vaciar patas añadidas a mano"):
            st.session_state.legs = []; st.rerun()

# --- 4. Equipos --------------------------------------------------------------
with tabs[3]:
    seasons = sorted(hist.SeasonStart.unique(), reverse=True)
    s_sel = st.selectbox("Temporada", seasons, format_func=lambda y: f"{y}/{str(y + 1)[-2:]}")
    d = hist[hist.SeasonStart == s_sel]
    recs = []
    for t in sorted(set(d.HomeTeam) | set(d.AwayTeam)):
        h, a = d[d.HomeTeam == t], d[d.AwayTeam == t]
        n = len(h) + len(a)
        tot = pd.concat([h.FTHG + h.FTAG, a.FTHG + a.FTAG])
        recs.append({
            "Equipo": t, "PJ": n,
            "Goles a favor": (h.FTHG.sum() + a.FTAG.sum()) / n, "Goles en contra": (h.FTAG.sum() + a.FTHG.sum()) / n,
            "% Más de 2.5": (tot > 2.5).mean(),
            "% Ambos marcan": pd.concat([(h.FTHG > 0) & (h.FTAG > 0), (a.FTHG > 0) & (a.FTAG > 0)]).mean(),
            "Córners a favor": (h.HC.sum() + a.AC.sum()) / n, "Córners en contra": (h.AC.sum() + a.HC.sum()) / n,
            "Tarjetas recibidas": (h.HY.sum() + h.HR.sum() + a.AY.sum() + a.AR.sum()) / n,
            "Tarjetas provocadas": (h.AY.sum() + h.AR.sum() + a.HY.sum() + a.HR.sum()) / n,
            "Tarjetas en sus partidos": (h.HY.sum() + h.AY.sum() + h.HR.sum() + h.AR.sum() + a.HY.sum() + a.AY.sum() + a.HR.sum() + a.AR.sum()) / n,
            "% Derrotas": ((h.FTHG < h.FTAG).sum() + (a.FTAG < a.FTHG).sum()) / n,
        })
    tdf = pd.DataFrame(recs)
    metric = st.selectbox("Ordenar y comparar por", [c for c in tdf.columns if c not in ("Equipo", "PJ")], index=6)
    tdf = tdf.sort_values(metric, ascending=False)
    fig = go.Figure(go.Bar(x=tdf[metric], y=tdf.Equipo, orientation="h", marker_color=BLUE,
                           hovertemplate="%{y}: %{x:.2f}<extra></extra>"))
    league_avg = tdf[metric].mean()
    fig.add_vline(x=league_avg, line_dash="dot", line_color=GREY, annotation_text="media liga", annotation_font_color=GREY)
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(hovermode="closest")
    st.plotly_chart(fig_style(fig, 28 * len(tdf) + 60), width="stretch")
    fmt = {c: st.column_config.NumberColumn(format="%.2f") for c in tdf.columns if c not in ("Equipo", "PJ")}
    for c in ("% Más de 2.5", "% Ambos marcan", "% Derrotas"):
        fmt[c] = PCT
    st.dataframe(tdf, hide_index=True, width="stretch", column_config=fmt)

    st.markdown("**Ratings del modelo (1 = media de la liga)**")
    r = gm.rates.table().rename(columns={"ataque": "Ataque goles", "defensa": "Goles que concede"})
    r = r.merge(cm.rates.table().rename(columns={"ataque": "Córners que genera", "defensa": "Córners que concede"}), on="Equipo")
    r = r.merge(km.rates.table().rename(columns={"ataque": "Tarjetas que recibe", "defensa": "Tarjetas que provoca"}), on="Equipo")
    r = r[r.Equipo.isin(teams)].sort_values("Ataque goles", ascending=False)
    st.dataframe(r, hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%.2f") for c in r.columns if c != "Equipo"})

# --- 5. Clasificación --------------------------------------------------------
with tabs[4]:
    st.subheader("Clasificación actual")
    st.dataframe(standings(hist), width="stretch")

# --- 6. Backtest -------------------------------------------------------------
with tabs[5]:
    st.subheader("¿Habría ganado dinero el modelo?")
    st.caption("Walk-forward: cada semana se reentrena solo con partidos anteriores y se apuesta con las cuotas que "
               "había de verdad. Córners y tarjetas no tienen cuotas históricas gratis, así que de esos se mide la calibración.")
    past = sorted([s for s in hist.SeasonStart.unique() if s < hist.SeasonStart.max()], reverse=True)
    c1, c2, c3 = st.columns(3)
    test_seasons = c1.multiselect("Temporadas de test", past, default=past[:2],
                                  format_func=lambda y: f"{y}/{str(y + 1)[-2:]}")
    odds_set = c2.selectbox("Cuotas a las que se apuesta", list(bt.ODDS_SETS), index=0)
    staking = c3.radio("Stake", ["Plana", "Kelly"], horizontal=True)
    markets = st.multiselect("Mercados", ["1X2", "Goles 2.5"], default=["1X2", "Goles 2.5"])
    if test_seasons and st.button("Ejecutar backtest", type="primary"):
        st.session_state.bt_run = (tuple(sorted(test_seasons)),)
    if st.session_state.get("bt_run") and test_seasons:
        pred = get_backtest(hist_hash, hist, tuple(sorted(test_seasons)), xi)
        bets = bt.simulate_bets(pred, odds_set, min_ev, model_w, max_odds, staking, kelly_f, tuple(markets))
        if bets.empty:
            st.info("Ninguna apuesta cumple los filtros.")
        else:
            roi = bets.Beneficio.sum() / bets.Stake.sum()
            k = st.columns(4)
            k[0].metric("Apuestas", len(bets))
            k[1].metric("Acierto", f"{bets.Ganada.mean():.1%}")
            k[2].metric("Beneficio", f"{bets.Beneficio.sum():+.1f} u")
            k[3].metric("ROI (yield)", f"{roi:+.1%}")
            color = GOOD if bets.Acumulado.iloc[-1] >= 0 else BAD
            fig = go.Figure(go.Scatter(x=bets.Fecha, y=bets.Acumulado, mode="lines", line=dict(color=color, width=2),
                                       name="Beneficio acumulado", hovertemplate="%{x|%d/%m/%y}: %{y:+.1f} u<extra></extra>"))
            fig.add_hline(y=0, line_color=GREY, line_width=1)
            st.plotly_chart(fig_style(fig), width="stretch")
            by = bets.groupby(["Mercado", "Apuesta"]).agg(N=("Beneficio", "size"), Acierto=("Ganada", "mean"),
                                                          Cuota_media=("Cuota", "mean"), Beneficio=("Beneficio", "sum"),
                                                          Stake=("Stake", "sum")).reset_index()
            by["ROI"] = by.Beneficio / by.Stake
            st.dataframe(by.drop(columns="Stake"), hide_index=True, width="stretch",
                         column_config={"Acierto": PCT, "ROI": PCT, "Cuota_media": st.column_config.NumberColumn("Cuota media", format="%.2f"),
                                        "Beneficio": st.column_config.NumberColumn(format="%+.1f")})
            ll = bt.log_loss_1x2(pred)
            st.caption("Log-loss 1X2 (menor es mejor): " + " · ".join(f"{k} {v:.4f}" for k, v in ll.items())
                       + ". Si el mercado gana al modelo, conviene subir el peso del mercado.")
            with st.expander("Ver todas las apuestas"):
                st.dataframe(bets, hide_index=True, width="stretch")

        st.markdown("**Calibración: probabilidad prevista vs frecuencia real**")
        cal = bt.calibration(pred)
        cols = st.columns(2)
        for i, (name, (t, bm, bb)) in enumerate(cal.items()):
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=GREY, dash="dot", width=1),
                                     name="Perfecta", hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=t.prevista, y=t.real, mode="lines+markers", line=dict(color=BLUE, width=2),
                                     marker=dict(size=9, line=dict(color="#fcfcfb", width=2)), name="Modelo",
                                     customdata=t.n, hovertemplate="prevista %{x:.0%} · real %{y:.0%} · n=%{customdata}<extra></extra>"))
            fig.update_layout(title=dict(text=f"{name} · Brier {bm:.4f} (base {bb:.4f})", font=dict(size=13)), hovermode="closest")
            fig.update_xaxes(range=[0, 1], tickformat=".0%"); fig.update_yaxes(range=[0, 1], tickformat=".0%")
            cols[i % 2].plotly_chart(fig_style(fig, 300), width="stretch")
        st.caption("Brier menor que la base (media de la liga) = el modelo aporta información.")

# --- 7. Cómo funciona --------------------------------------------------------
with tabs[6]:
    st.markdown(
        """
**Goles (1X2, over/under, ambos marcan):** modelo Poisson tipo Dixon-Coles con fuerza de ataque y defensa por equipo,
ventaja de jugar en casa, ponderación temporal (lo reciente pesa más) y regularización para equipos con pocos datos.

**Córners y tarjetas:** mismo modelo multiplicativo sobre los conteos de cada equipo (lo que genera/recibe y lo que
concede/provoca al rival), con distribución binomial negativa para la dispersión real de estos mercados.

**Valor (EV):** se mezcla la probabilidad del modelo con la del mercado sin margen (Pinnacle o media) según el peso
de la barra lateral, y se compara con la mejor cuota disponible. EV = p × cuota − 1. Una apuesta está
*infravalorada* cuando la cuota es mayor que la cuota justa (1/p).

**Stake:** Kelly fraccional, con tope del 5 % del bankroll por apuesta.

**Límites a tener en cuenta**
- El 1X2 de Primera es de los mercados más eficientes del mundo: revisa el backtest antes de fiarte del modelo.
- Las cuotas de córners y tarjetas no están en fuentes gratuitas: escribe la cuota de tu casa en *Partido a partido*.
- No hay datos de árbitro, alineaciones ni lesiones: ajústalo tú (factor de árbitro) o descarta partidos con bajas clave.
- Ningún modelo garantiza beneficios. Juega con un bankroll que puedas permitirte perder.
        """
    )
