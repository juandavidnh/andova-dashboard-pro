# ANDOVA — Streamlit Cloud Fix (Yahoo REST, feriados y rebalanceo al cierre)

- Yahoo REST (v8) para históricos; **no usa yfinance** (evita 'database is locked').
- Feriados/fines de semana: usa `marketState` de Yahoo (`CLOSED`) para congelar el valor en el **cierre del último día hábil**.
- Live solo si `marketState` ∈ {REGULAR, POST}.
- Rebalanceo **al cierre** del 15 y del **último hábil**; las `shares` guardadas son post-rebalanceo del mismo día.
- `weights.csv`: pesos JFR. `sectors.csv`: mapeo prellenado.

## Cómo correr
```bash
pip install -r requirements.txt
streamlit run app.py
```
