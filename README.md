# Value bets · Primera División

Dashboard en Streamlit que busca cuotas de LaLiga que pagan más de lo que dicen las estadísticas: 1X2, goles, córners, tarjetas y combinadas.

## Pestañas

| Pestaña | Qué hace |
|---|---|
| Infravaloradas | Apuestas de la próxima jornada con EV positivo (1X2 y más/menos de 2.5 goles), con la cuota justa y el stake Kelly |
| Partido a partido | Probabilidades y cuotas justas de goles, córners y tarjetas. Escribe la cuota de tu casa y calcula el EV. Incluye un factor de árbitro |
| Combinadas | Combinadas sugeridas solo con patas que tienen valor, más un constructor manual |
| Equipos | Ranking por métrica: tarjetas recibidas o provocadas, córners, % de más de 2.5, derrotas… También los ratings del modelo |
| Clasificación | Tabla de la temporada actual con los últimos 5 resultados |
| Backtest | Simulación walk-forward con las cuotas reales de temporadas pasadas: ROI, curva de beneficio y calibración |

## Datos

- **football-data.co.uk** (gratis): resultados, tiros, córners, tarjetas y cuotas de apertura y cierre de cada partido desde hace años. `fixtures.csv` trae los próximos partidos con cuotas. Se suele actualizar a mitad de semana.
- **The Odds API** (opcional, 500 créditos/mes gratis): cuotas en directo de 1X2 y más/menos 2.5 de varias casas europeas. Cada refresco gasta 2 créditos y la app lo cachea 30 minutos.

## Despliegue (Streamlit Community Cloud)

1. Crea un repo privado, p. ej. `alvaroviti/value_bets_laliga`, y sube esta carpeta.
2. En share.streamlit.io: New app → el repo → `app.py`.
3. Opcional: en Settings → Secrets añade `ODDS_API_KEY = "..."`.

## En local

```bash
pip install -r requirements.txt
streamlit run app.py
# Sin conexión, con datos sintéticos:
python tests/make_synthetic.py synthetic && DATA_DIR=synthetic streamlit run app.py
```

## Modelo

- **Goles:** Poisson Dixon-Coles con ataque y defensa por equipo, ventaja de campo, decaimiento temporal y regularización L2.
- **Córners y tarjetas:** GLM multiplicativo con binomial negativa para la sobredispersión.
- **Valor:** la probabilidad se mezcla con la del mercado sin margen (Pinnacle). Con el peso configurable se evitan falsos positivos.

> Ningún modelo garantiza beneficios. Mira el backtest antes de apostar y usa solo un bankroll que puedas perder.
