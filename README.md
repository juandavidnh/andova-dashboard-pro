# ANDOVA — Feriados fijos + Rebalanceo al cierre (Yahoo prioritario)

- Feriados/fines de semana: usa `marketState` de Yahoo (`CLOSED`) para congelar el valor en el **cierre del último día hábil**.
- Live solo si `marketState` ∈ {REGULAR, POST}.
- Rebalanceo **al cierre** del 15 y del último hábil; las `shares` guardadas son **post-rebalanceo del mismo día**.
- Yahoo prioritario para históricos (v8) e intradía (v7/v8).
- `weights.csv`: pesos JFR (de tu imagen). `sectors.csv`: mapeo prellenado para exponer por sector.

## Uso
```bash
pip install -r requirements.txt
streamlit run app.py
```
