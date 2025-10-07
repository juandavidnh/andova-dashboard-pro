
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_market_calendars as mcal
import plotly.express as px
import json
from pathlib import Path

st.set_page_config(page_title="Índice Portafolio — Recomp + Backtrack + QQQ", layout="wide")
DATA_FILE = Path("data/events.json")

# ---------- Utilidades ----------
nyse = mcal.get_calendar("XNYS")
def next_business_day(d: pd.Timestamp) -> pd.Timestamp:
    d = pd.Timestamp(d).tz_localize(None).normalize()
    sched = nyse.schedule(start_date=d - pd.Timedelta(days=10), end_date=d + pd.Timedelta(days=10))
    if sched.empty:
        return pd.tseries.offsets.BDay().rollforward(d)
    opens = sched.index
    after = opens[opens >= d]
    if len(after)==0:
        return pd.tseries.offsets.BDay().rollforward(d)
    return pd.Timestamp(after[0]).normalize()

ANCHOR_DATE = next_business_day(pd.Timestamp("2025-08-01"))

def load_events():
    events = {}
    if DATA_FILE.exists():
        try:
            raw = json.loads(DATA_FILE.read_text())
            for k, w in raw.items():
                events[pd.Timestamp(k)] = pd.Series(w, dtype=float)
        except Exception:
            pass
    return events

def save_events(events: dict):
    serial = {pd.Timestamp(k).strftime("%Y-%m-%d"): ({} if v is None else {t: float(x) for t, x in pd.Series(v).dropna().items()}) for k, v in events.items()}
    DATA_FILE.parent.mkdir(exist_ok=True)
    DATA_FILE.write_text(json.dumps(serial, indent=2))

initial_weights = pd.Series(json.loads('{"NVDA": 0.0983480176211454, "MSFT": 0.0924008810572687, "AVGO": 0.0638399412628487, "TSM": 0.0470264317180617, "AAPL": 0.0523494860499266, "AMZN": 0.0394273127753304, "META": 0.0281938325991189, "NFLX": 0.0209985315712188, "TSLA": 0.0190895741556534, "GOOGL": 0.0190161527165932, "GOOG": 0.0179148311306902, "PLTR": 0.0148311306901615, "AMD": 0.0118208516886931, "INTU": 0.0102790014684288, "QCOM": 0.0102790014684288, "ADBE": 0.0102790014684288, "MELI": 0.0102790014684288, "MU": 0.0102790014684288, "PANW": 0.0102790014684288, "SNPS": 0.0102790014684288, "CRWD": 0.0102790014684288, "CEG": 0.0102790014684288, "MRVL": 0.0102790014684288, "ASML": 0.0102790014684288, "ZS": 0.0102790014684288, "DDOG": 0.0102790014684288, "XEL": 0.0102790014684288, "ON": 0.0102790014684288, "ARM": 0.0102790014684288, "MDB": 0.0102790014684288, "AMBA": 0.01, "MBLY": 0.01, "OUST": 0.01, "QS": 0.015, "SLDP": 0.01, "ACHR": 0.01, "EH": 0.01, "JOBY": 0.0, "TLX": 0.01, "VKTX": 0.01, "DNA": 0.01, "GRAL": 0.01, "TEM": 0.01, "SDGR": 0.01, "S": 0.015, "BE": 0.0125, "FLNC": 0.0125, "STEM": 0.01, "ALAB": 0.015, "CRDO": 0.015, "IONQ": 0.015, "QBTS": 0.015, "RGTI": 0.01, "HIMS": 0.01, "SYM": 0.015, "LEGN": 0.01, "GTLB": 0.01}')).astype(float)
initial_weights = initial_weights[initial_weights>0]
initial_weights = initial_weights/initial_weights.sum()

if "events" not in st.session_state:
    st.session_state.events = load_events()
# siempre garantizamos el ancla
st.session_state.events[ANCHOR_DATE] = initial_weights
save_events(st.session_state.events)

def fetch_prices(tickers, start, end):
    if isinstance(tickers, (list, tuple, set)):
        tickers = list(dict.fromkeys([t.strip().upper() for t in tickers if str(t).strip() != ""]))
    else:
        tickers = [tickers]
    if not tickers:
        return pd.DataFrame()
    df = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame()
    df = df.sort_index().ffill().dropna(how="all")
    if len(df.columns) > 0:
        empty_cols = [c for c in df.columns if df[c].dropna().empty]
        df = df.drop(columns=empty_cols, errors="ignore")
    return df

def simulate_with_recompositions(prices: pd.DataFrame, ordered_events, start_value=1000.0):
    if prices.empty:
        return pd.Series(dtype=float), pd.DataFrame(), pd.DataFrame()
    prices = prices.sort_index()

    # asegurar columnas
    all_tk = set(prices.columns)
    for _, w in ordered_events:
        all_tk.update(list(w.index))
    prices = prices.reindex(columns=sorted(all_tk)).ffill()

    events = sorted(ordered_events, key=lambda x: x[0])
    # clip primer evento al primer dato
    if events[0][0] < prices.index[0]:
        events[0] = (prices.index[0], events[0][1])

    pv_parts, shares_parts = [], []
    pv_prev_end, start_val = None, start_value

    for i, (dt_recomp, w) in enumerate(events):
        start_dt = dt_recomp
        end_dt = prices.index[-1] if i == len(events)-1 else events[i+1][0] - pd.Timedelta(days=1)
        segment = prices.loc[(prices.index >= start_dt) & (prices.index <= end_dt)].copy()
        if segment.empty: continue
        w_seg = w.reindex(segment.columns).fillna(0.0)
        if w_seg.sum() == 0: continue
        w_seg = w_seg / w_seg.sum()
        if pv_prev_end is not None:
            start_val = pv_prev_end
        p0 = segment.iloc[0].replace(0, np.nan).fillna(method="ffill").fillna(method="bfill")
        shares = (start_val * w_seg) / p0
        pv = (segment * shares).sum(axis=1)
        pv_parts.append(pv)
        shares_row = pd.DataFrame([shares], index=[segment.index[0]])
        shares_parts.append(shares_row)
        pv_prev_end = float(pv.iloc[-1])

    pv_all = pd.concat(pv_parts, axis=0).sort_index()
    shares_df = pd.concat(shares_parts, axis=0).sort_index().reindex(columns=prices.columns).fillna(0.0)
    return pv_all, prices, shares_df

def backtrack_before_anchor(prices: pd.DataFrame, anchor_date: pd.Timestamp, weights: pd.Series, base_value: float=1000.0):
    """Construye una serie hacia atrás (t < anchor) usando los pesos del ancla."""
    pre = prices.loc[prices.index < anchor_date]
    if pre.empty:
        return pd.Series(dtype=float)
    cols = sorted(set(prices.columns) | set(weights.index))
    P = prices.reindex(columns=cols).ffill()
    w = weights.reindex(cols).fillna(0.0)
    if w.sum() == 0:
        return pd.Series(dtype=float)
    w = w / w.sum()
    after_anchor = P.loc[P.index >= anchor_date]
    if after_anchor.empty:
        return pd.Series(dtype=float)
    P_anchor = after_anchor.iloc[0]
    num_anchor = float((P_anchor * w).sum())
    pre_section = P.loc[P.index < anchor_date]
    num_pre = (pre_section * w).sum(axis=1)
    bt = base_value * (num_pre / num_anchor)
    return bt

# ---------- UI ----------
st.title("Índice Portafolio (base 1000 en 2025-08-01) — Backtrack + QQQ")

# Presets estilo Apple
preset = st.sidebar.selectbox("Periodo rápido", options=["Personalizado","5D","1M","3M","6M","YTD","1Y","Desde ancla"], index=6)
date_end = st.sidebar.date_input("Fin", value=pd.Timestamp.today().date())
date_start_custom = st.sidebar.date_input("Inicio (si Personalizado)", value=pd.Timestamp("2025-06-01").date())

def compute_window(preset_key: str, end_date: pd.Timestamp, custom_start: pd.Timestamp):
    end_ts = pd.Timestamp(end_date)
    if preset_key == "Personalizado":
        start_ts = pd.Timestamp(custom_start)
    elif preset_key == "5D":
        start_ts = end_ts - pd.Timedelta(days=7)
    elif preset_key == "1M":
        start_ts = end_ts - pd.Timedelta(days=31)
    elif preset_key == "3M":
        start_ts = end_ts - pd.Timedelta(days=93)
    elif preset_key == "6M":
        start_ts = end_ts - pd.Timedelta(days=186)
    elif preset_key == "1Y":
        start_ts = end_ts - pd.Timedelta(days=366)
    elif preset_key == "YTD":
        start_ts = pd.Timestamp(f"{end_ts.year}-01-01")
    elif preset_key == "Desde ancla":
        start_ts = ANCHOR_DATE
    else:
        start_ts = ANCHOR_DATE
    return start_ts, end_ts

win_start, win_end = compute_window(preset, date_end, date_start_custom)

st.sidebar.button("Force refresh now", on_click=lambda: st.experimental_rerun())
st.sidebar.caption("Prices are fetched fresh on each page refresh.")


# Carga CSVs de recomposición
with st.sidebar.expander("Recomposiciones (múltiples)", expanded=False):
    st.caption("Sube CSV long (date,ticker,weight) o wide. Se guardan si no hay conflicto de fecha exacta.")
    files = st.file_uploader("CSV(s)", type=["csv"], accept_multiple_files=True)
    if st.button("Borrar todas (conserva ancla)"):
        st.session_state.events = {ANCHOR_DATE: initial_weights}
        save_events(st.session_state.events)
        st.success("Recomposiciones eliminadas.")
    if files:
        def parse_df(file):
            try:
                return pd.read_csv(file)
            except Exception:
                file.seek(0)
                return pd.read_csv(file, sep=";")
        for f in files:
            df = parse_df(f)
            df.columns = [str(c).strip() for c in df.columns]
            date_col = next((c for c in df.columns if c.lower() in ("date","fecha")), df.columns[0])
            s = df[date_col].astype(str)
            d1 = pd.to_datetime(s, errors="coerce", dayfirst=False)
            d2 = pd.to_datetime(s, errors="coerce", dayfirst=True)
            parsed = d2 if s.str.contains(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b").any() else (d1 if d1.isna().sum() <= d2.isna().sum() else d2)
            parsed = parsed.dt.tz_localize(None).dt.normalize().map(next_business_day)
            df[date_col] = parsed
            lower = [c.lower() for c in df.columns]
            is_long = ("ticker" in lower or "activo" in lower) and ("weight" in lower or "peso" in lower)
            evs = []
            if is_long:
                tcol = next(c for c in df.columns if c.lower() in ("ticker","activo"))
                wcol = next(c for c in df.columns if c.lower() in ("weight","peso"))
                df[wcol] = df[wcol].astype(str).str.replace(",", ".")
                df[wcol] = pd.to_numeric(df[wcol], errors="coerce")
                for d, grp in df.groupby(date_col):
                    w = grp.set_index(tcol)[wcol].dropna()
                    if w.empty: continue
                    sers = (w / w.sum())
                    sers.index = [str(i).strip().upper() for i in sers.index]
                    evs.append((pd.Timestamp(d), sers))
            else:
                tickers = [c for c in df.columns if c != date_col]
                for c in tickers:
                    df[c] = df[c].astype(str).str.replace(",", ".")
                    df[c] = pd.to_numeric(df[c], errors="coerce")
                for _, row in df.iterrows():
                    d = row[date_col]
                    w = row[tickers].dropna()
                    if w.empty: continue
                    sers = (w / w.sum())
                    sers.index = [str(i).strip().upper() for i in sers.index]
                    evs.append((pd.Timestamp(d), sers))
            inserted = 0
            for d,sers in evs:
                if d in st.session_state.events and d != ANCHOR_DATE:
                    st.error(f"Conflicto: ya existe una recomposición para {d.date()}.")
                else:
                    st.session_state.events[d] = sers
                    inserted += 1
            if inserted>0:
                save_events(st.session_state.events)
            st.success(f"Procesado {getattr(f,'name','archivo')}: {inserted} evento(s)")

# --- Datos ---
events_ordered = sorted(st.session_state.events.items(), key=lambda x: x[0])
all_tickers = sorted(list({t for _, s in events_ordered for t in s.index}))

# Descarga suficiente historia para backtrack (al menos 1 año antes del ancla)
dl_start = min(win_start, ANCHOR_DATE - pd.Timedelta(days=400))
dl_end = pd.Timestamp(win_end) + pd.Timedelta(days=1)
prices = fetch_prices(all_tickers, start=dl_start, end=dl_end)

# Simulación (forward desde eventos) + backtrack (antes de ancla)
pv_forward, P_full, shares_df = simulate_with_recompositions(prices, events_ordered, start_value=1000.0)
pv_back = backtrack_before_anchor(P_full, ANCHOR_DATE, events_ordered[0][1], base_value=1000.0)
pv_all = pd.concat([pv_back, pv_forward], axis=0).sort_index()

# --- QQQ comparator ---
qqq = fetch_prices(["QQQ"], start=dl_start, end=dl_end)
if not qqq.empty:
    if (qqq.index >= ANCHOR_DATE).any():
        q_base = float(qqq.loc[qqq.index >= ANCHOR_DATE].iloc[0,0])
        qqq_idx = (qqq.iloc[:,0] * (1000.0 / q_base)).rename("QQQ (base 1000)")
    else:
        qqq_idx = pd.Series(dtype=float, name="QQQ (base 1000)")
else:
    qqq_idx = pd.Series(dtype=float, name="QQQ (base 1000)")

# --- Métricas diarias arriba (Índice vs QQQ) ---
with st.container():
    c1, c2, c3 = st.columns([2,1,1])
    c1.subheader("Resumen diario")
    idx_change, qqq_change = np.nan, np.nan
    if not pv_all.empty:
        # Rebase a 1000 en el primer punto >= ancla
        if (pv_all.index >= ANCHOR_DATE).any():
            base_val = float(pv_all.loc[pv_all.index >= ANCHOR_DATE].iloc[0])
            pv1000 = pv_all * (1000.0 / base_val)
        else:
            pv1000 = pv_all
        if pv1000.size >= 2:
            idx_change = float(pv1000.iloc[-1]/pv1000.iloc[-2]-1)
    if not qqq_idx.empty and qqq_idx.size >= 2:
        qqq_change = float(qqq_idx.iloc[-1]/qqq_idx.iloc[-2]-1)
    c2.metric("Índice — cambio hoy", f"{(0 if np.isnan(idx_change) else idx_change)*100:,.2f}%")
    c3.metric("QQQ — cambio hoy", f"{(0 if np.isnan(qqq_change) else qqq_change)*100:,.2f}%")

# --- Gráfico ---
st.subheader("Curva del índice (con backtrack) vs QQQ")
if pv_all.empty:
    st.warning("No hay datos para construir el índice.")
else:
    if (pv_all.index >= ANCHOR_DATE).any():
        base_val = float(pv_all.loc[pv_all.index >= ANCHOR_DATE].iloc[0])
        pv1000 = pv_all * (1000.0 / base_val)
    else:
        pv1000 = pv_all
    view = pv1000.loc[(pv1000.index >= win_start) & (pv1000.index <= win_end)]
    df_plot = view.rename("Índice (base 1000)").to_frame()
    if not qqq_idx.empty:
        view_q = qqq_idx.loc[(qqq_idx.index >= win_start) & (qqq_idx.index <= win_end)]
        df_plot = df_plot.join(view_q, how="outer")
    fig = px.line(df_plot, title=f"Evolución {preset} — Índice vs QQQ", labels={"value":"Índice","index":"Fecha","variable":"Serie"})
    st.plotly_chart(fig, width="stretch")
    if not view.empty:
        ret_idx = view.iloc[-1]/view.iloc[0]-1 if view.iloc[0]!=0 else np.nan
        st.metric("Retorno índice en ventana", f"{ret_idx*100:,.2f}%")
    if not qqq_idx.empty:
        vq = qqq_idx.loc[(qqq_idx.index >= win_start) & (qqq_idx.index <= win_end)]
        if not vq.empty:
            ret_q = vq.iloc[-1]/vq.iloc[0]-1 if vq.iloc[0]!=0 else np.nan
            st.metric("Retorno QQQ en ventana", f"{ret_q*100:,.2f}%")

# --- Breakdown ---
st.subheader("Breakdown (peso actual, ret 1d, ret ventana)")
if isinstance(shares_df, pd.DataFrame) and not shares_df.empty and not P_full.empty:
    last_date = shares_df.index.max()
    sh = shares_df.loc[last_date].fillna(0.0)
    last_px = P_full.reindex(columns=sh.index).ffill().iloc[-1].fillna(0.0)
    values = sh * last_px
    total = float(values.sum()) if values.sum()!=0 else 1.0
    w_last = (values / total)

    ret1d_df = P_full.pct_change().iloc[[-1]].T.squeeze()
    ret1d = ret1d_df.reindex(w_last.index).fillna(0.0)

    P_win = P_full.loc[(P_full.index >= win_start) & (P_full.index <= win_end)]
    def period_return(series: pd.Series) -> float:
        s = series.dropna()
        if s.empty: return np.nan
        first = s.iloc[0]; lastv = s.iloc[-1]
        return float(lastv/first - 1.0) if first!=0 else np.nan
    rets_period = P_win.apply(period_return, axis=0).reindex(w_last.index)

    dfb = pd.DataFrame({
        "Ticker": w_last.index,
        "Peso": w_last.values,
        "Ret_1d": ret1d.values,
        "Ret_ventana": rets_period.values
    }).sort_values("Peso", ascending=False)
    st.dataframe(dfb.style.format({"Peso":"{:.2%}","Ret_1d":"{:.2%}","Ret_ventana":"{:.2%}"}), width="stretch")
else:
    st.info("No hay datos suficientes para el breakdown.")

# --- Calendario ---
st.subheader("Calendario de recomposiciones")
if events_ordered:
    dfcal = pd.DataFrame([{"Fecha": d.date(), "N° tickers": len(s)} for d,s in events_ordered]).sort_values("Fecha")
    st.dataframe(dfcal, width="stretch")
