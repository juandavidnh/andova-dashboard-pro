
import os, time, warnings, requests, pytz
from typing import Optional, Dict, Tuple, List
from datetime import date, datetime
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

warnings.filterwarnings("ignore")
st.set_page_config(page_title="ANDOVA — Portfolio Dashboard", layout="wide")

# --------------------------
# Configuración general
# --------------------------
ANCHOR_DATE = pd.Timestamp("2025-08-01")
ANCHOR_VALUE = 1000.0
US_EASTERN = pytz.timezone("US/Eastern")

# --------------------------
# Utilidades de tiempo
# --------------------------
def to_index_tz(ts, index: pd.DatetimeIndex) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    idx_tz = getattr(index, "tz", None)
    if idx_tz is None:
        return t.tz_localize(None) if t.tzinfo is not None else t
    return t.tz_localize(idx_tz) if t.tzinfo is None else t.tz_convert(idx_tz)

# --------------------------
# Yahoo helpers
# --------------------------
@st.cache_data(show_spinner=False, ttl=20)
def yahoo_market_state(symbol: str = "QQQ") -> str:
    """'REGULAR' | 'PRE' | 'POST' | 'CLOSED' desde Yahoo Quote."""
    try:
        url = f"https://query1.finance.yahoo.com/v7/finance/quote?symbols={symbol}"
        r = requests.get(url, headers={"User-Agent":"Mozilla/5.0"}, timeout=6)
        if r.status_code == 200:
            res = r.json().get("quoteResponse", {}).get("result", [])
            if res:
                return (res[0].get("marketState") or "CLOSED").upper()
    except Exception:
        pass
    return "CLOSED"

@st.cache_data(ttl=10, show_spinner=False)
def yahoo_intraday_last(symbol: str):
    """Último precio 1m (incluye pre/post)."""
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range=1d&interval=1m&includePrePost=true"
    try:
        r = requests.get(url, headers={"User-Agent":"Mozilla/5.0"}, timeout=7)
        if r.status_code != 200: return None, None, None
        j = r.json(); res = (j.get("chart",{}) or {}).get("result", [])
        if not res: return None, None, None
        res0 = res[0]; ts = res0.get("timestamp") or []
        idx = pd.to_datetime(ts, unit="s", utc=True) if ts else None
        q0 = (res0.get("indicators", {}) or {}).get("quote", [{}])[0]; closes = q0.get("close") or []
        px, tstamp = None, None
        for i in range(len(closes)-1, -1, -1):
            if closes[i] is not None:
                px = float(closes[i]); tstamp = idx[i] if idx is not None and i < len(idx) else None; break
        pc = (res0.get("meta", {}) or {}).get("previousClose"); pc = float(pc) if pc is not None else None
        return px, pc, tstamp
    except Exception:
        return None, None, None

@st.cache_data(ttl=5, show_spinner=False)
def yahoo_batch_quotes(symbols: List[str]) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, Optional[pd.Timestamp]]]:
    """Precios y previousClose por batch; fallback a 1m para los que falten."""
    prices, prev_closes, ts_map = {}, {}, {}
    if not symbols: return prices, prev_closes, ts_map
    chunk = 40; missing=set()
    for i in range(0, len(symbols), chunk):
        syms = symbols[i:i+chunk]
        url = "https://query1.finance.yahoo.com/v7/finance/quote?symbols=" + ",".join(syms)
        try:
            r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=6)
            if r.status_code == 200:
                data = r.json().get("quoteResponse", {}).get("result", [])
                got=set()
                for q in data:
                    s=str(q.get("symbol")); got.add(s)
                    px=q.get("regularMarketPrice") or q.get("postMarketPrice")
                    pc=q.get("regularMarketPreviousClose")
                    t=q.get("regularMarketTime") or q.get("postMarketTime")
                    ts=pd.to_datetime(t, unit="s", utc=True) if t else None
                    if px is not None: prices[s]=float(px)
                    if pc is not None: prev_closes[s]=float(pc)
                    ts_map[s]=ts
                for s in syms:
                    if s not in got: missing.add(s)
            else:
                missing.update(syms)
        except Exception:
            missing.update(syms)
    for s in sorted(missing):
        px, pc, ts = yahoo_intraday_last(s)
        if px is not None: prices[s]=px
        if pc is not None and s not in prev_closes: prev_closes[s]=pc
        if ts is not None: ts_map[s]=ts
    return prices, prev_closes, ts_map

# --------------------------
# Descarga de precios (Yahoo REST v8, sin yfinance)
# --------------------------
@st.cache_data(show_spinner=True, ttl=600)
def download_prices(tickers: List[str], start, end) -> pd.DataFrame:
    def yahoo_chart_daily_series(ticker: str, start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> pd.DataFrame | None:
        p1 = int(pd.Timestamp(start_dt).tz_localize("UTC").timestamp())
        p2 = int(pd.Timestamp(end_dt).tz_localize("UTC").timestamp())
        url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
               f"?period1={p1}&period2={p2}&interval=1d&events=split,div&includeAdjClose=true")
        for _ in range(3):
            try:
                r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=8)
                if r.status_code != 200: time.sleep(0.4); continue
                j = r.json()
                result = (j.get("chart", {}) or {}).get("result", [])
                if not result: time.sleep(0.3); continue
                res0 = result[0]; ts = res0.get("timestamp")
                if not ts: time.sleep(0.3); continue
                idx = pd.to_datetime(ts, unit="s", utc=True).tz_convert("US/Eastern").normalize()
                ind = res0.get("indicators", {}) or {}
                adj = ((ind.get("adjclose") or [{}])[0]).get("adjclose")
                close = ((ind.get("quote") or [{}])[0]).get("close")
                arr = adj if (adj and any(v is not None for v in adj)) else close
                if not arr: time.sleep(0.3); continue
                s = pd.Series(arr, index=idx, name=ticker, dtype="float64").dropna()
                if s.empty: time.sleep(0.2); continue
                return s.to_frame()
            except Exception:
                time.sleep(0.3)
        return None

    tickers = [t for t in dict.fromkeys(tickers) if isinstance(t, str)]
    frames, failed = [], []
    for t in tickers:
        df = yahoo_chart_daily_series(t, pd.Timestamp(start), pd.Timestamp(end))
        if df is not None and not df.empty: frames.append(df)
        else: failed.append(t)
    if not frames: return pd.DataFrame()
    prices = pd.concat(frames, axis=1).sort_index()
    prices = prices.loc[:, ~prices.columns.duplicated()].dropna(axis=1, how="all")
    if failed: st.sidebar.warning("Sin datos tras reintentos (Yahoo REST): " + ", ".join(sorted(failed)))
    return prices

# --------------------------
# Pesos y sectores
# --------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def load_weights(path: str = "weights.csv") -> pd.Series:
    df = pd.read_csv(path)
    s = pd.Series(df["weight_pct"].values, index=df["ticker"].values).astype(float)
    s = s[s > 0]
    return s / s.sum()

# --------------------------
# Rebalanceo (15 y último hábil, al CIERRE del propio día)
# --------------------------
def first_trading_on_or_after(index: pd.DatetimeIndex, target: pd.Timestamp):
    i = index.searchsorted(target)
    return index[i] if i < len(index) else None

def rebalance_dates_semimonthly(prices: pd.DataFrame) -> set:
    idx = prices.index
    if len(idx) == 0: return set()
    rdates = set(prices.resample("ME").last().index)  # month-end
    months = pd.period_range(idx[0], idx[-1], freq="M")
    for p in months:
        fifteenth = pd.Timestamp(p.year, p.month, 15)
        fifteenth = to_index_tz(fifteenth, idx).normalize()
        cand = first_trading_on_or_after(idx, fifteenth)
        if cand is not None and cand.month == p.month:
            rdates.add(cand)
    return rdates

def simulate_with_holdings_from_start(prices: pd.DataFrame, weights: pd.Series, start_value: float = 1.0):
    tickers_avail = [t for t in weights.index if t in prices.columns]
    if not tickers_avail: raise ValueError("No hay tickers disponibles.")
    P = prices[tickers_avail].copy()

    # Alineamos inicio: primer día donde todos los tickers tienen dato (ffill)
    fvs = [P[c].first_valid_index() for c in P.columns if P[c].first_valid_index() is not None]
    if not fvs: raise ValueError("No hay datos válidos.")
    start_dt = max(fvs); P = P.loc[P.index >= start_dt].ffill()

    cols_ok = [c for c in P.columns if P[c].notna().all()]
    if not cols_ok: raise ValueError("No hay suficientes datos tras el inicio para todos los tickers.")
    if set(cols_ok) != set(P.columns):
        dropped_cols = sorted(set(P.columns) - set(cols_ok))
        st.sidebar.warning("Excluidos por cobertura incompleta: " + ", ".join(dropped_cols))
        P = P[cols_ok]

    w = (weights.loc[P.columns] / weights.loc[P.columns].sum()).copy()
    dates = P.index
    shares = (start_value * w / P.iloc[0]).values
    rdates = rebalance_dates_semimonthly(P)

    pv_list, shares_rows = [], []
    for d in dates:
        v = float(np.dot(shares, P.loc[d].values))      # PV al cierre de d con shares vigentes
        if d in rdates:                                  # rebalanceo AL CIERRE de d
            shares = (v * w / P.loc[d]).values          # nuevas shares desde ya (cierre de d)
        pv_list.append(v)
        shares_rows.append(pd.Series(shares, index=P.columns, name=d))  # shares post-rebalanceo

    pv_raw = pd.Series(pv_list, index=dates, name="pv_raw")
    shares_df = pd.DataFrame(shares_rows)
    return pv_raw, shares_df, P

def normalize_to_anchor(pv_raw: pd.Series, anchor_date: pd.Timestamp, anchor_value: float):
    if len(pv_raw) == 0: return pv_raw, 1.0, None
    idx = pv_raw.index; anchor_target = to_index_tz(anchor_date, idx).normalize()
    pos = idx.searchsorted(anchor_target); anchor = idx[pos] if pos < len(idx) else idx[-1]
    base_val = float(pv_raw.loc[anchor]); scale = anchor_value / base_val if (base_val and not np.isnan(base_val)) else 1.0
    return (pv_raw * scale).rename("pv"), scale, anchor

# --------------------------
# Estado de mercado (feriados fijos a CLOSED)
# --------------------------
def market_state(P_aligned: pd.DataFrame) -> str:
    ms = yahoo_market_state("QQQ")
    if ms in ("REGULAR", "POST"):
        return "open"
    return "preopen"

def prev_trading_close(P_aligned: pd.DataFrame):
    if P_aligned.empty: return None, None, False
    idx=P_aligned.index; last_idx=idx[-1]; today_et=datetime.now(US_EASTERN).date()
    if last_idx.date()==today_et and len(idx)>=2:
        yday_close=P_aligned.iloc[-2].copy(); today_close=P_aligned.iloc[-1].copy(); have_today=True
    else:
        yday_close=P_aligned.iloc[-1].copy(); today_close=None; have_today=False
    return yday_close, today_close, have_today

def latest_price_vectors(P_aligned: pd.DataFrame):
    state = market_state(P_aligned)
    yday_close, today_close, have_today = prev_trading_close(P_aligned)
    used: Dict[str, str] = {}; ts_map: Dict[str, Optional[pd.Timestamp]] = {}
    symbols = list(P_aligned.columns)
    if state=="open":
        prices_y, pcs_y, ts_y = yahoo_batch_quotes(symbols)
        lp_used = yday_close.copy()
        for t,v in prices_y.items():
            if pd.notna(v): lp_used[t]=float(v); used[t]="Yahoo Quote/1m"; ts_map[t]=ts_y.get(t)
        for t,pc in pcs_y.items():
            if pd.isna(yday_close.get(t)): yday_close[t]=pc
        pp_used = yday_close
    else:
        lp_used=yday_close; pp_used=yday_close
        ts_common = pd.Timestamp.combine(pd.Timestamp(yday_close.name), pd.Timestamp('16:00').time())
        for t in symbols: used[t]="DailyClose"; ts_map[t]=ts_common
    return lp_used, pp_used, used, state, ts_map

# --------------------------
# UI
# --------------------------
st.sidebar.title("🔧 Configuración")
weights = load_weights()
try: sectors = pd.read_csv("sectors.csv")
except Exception: sectors = pd.DataFrame(columns=["ticker","sector"])

tickers_all = list(weights.index)
selected = st.sidebar.multiselect("Filtra tickers", tickers_all, default=tickers_all)
default_start = date(2018,1,1)
start = st.sidebar.date_input("Fecha inicio descarga", default_start)
end = st.sidebar.date_input("Fecha fin", date.today())
st.sidebar.markdown("---")
st.sidebar.subheader("⚡ Live")
live_on = st.sidebar.checkbox("Activar live", value=True, key="live_on")
refresh_sec = st.sidebar.number_input("Auto-refresh (segundos)", min_value=5, max_value=120, value=20, step=5)

weights_sel = (weights.loc[selected] if selected else weights).copy()
weights_sel = weights_sel / weights_sel.sum()
st.sidebar.dataframe(pd.DataFrame({"weight": weights_sel}).style.format({"weight":"{:.2%}"}), width="stretch")

prices = download_prices(list(weights_sel.index), start, end)
if prices is None or prices.empty:
    st.error("No hay datos para el rango seleccionado."); st.stop()

# Forzamos ancla: quitamos tickers que no tienen historia hasta ANCHOR_DATE
anchor_target = to_index_tz(ANCHOR_DATE, prices.index).normalize()
eligible_cols = []; excluded_after_anchor = []
for c in prices.columns:
    fvi = prices[c].first_valid_index()
    if fvi is not None and fvi <= anchor_target: eligible_cols.append(c)
    else: excluded_after_anchor.append(c)
if excluded_after_anchor:
    st.sidebar.warning("Excluidos por no tener datos hasta el ancla: " + ", ".join(sorted(excluded_after_anchor)))
    prices = prices[eligible_cols] if eligible_cols else prices

weights_sel = weights_sel[[t for t in weights_sel.index if t in prices.columns]]
weights_sel = weights_sel / weights_sel.sum()
valid_cols = [c for c in prices.columns if prices[c].notna().sum() >= 2]
removed_sparse = sorted(set(prices.columns) - set(valid_cols))
if removed_sparse: st.sidebar.warning("Tickers sin suficientes datos: " + ", ".join(removed_sparse))
prices = prices[valid_cols] if valid_cols else prices
present = [t for t in weights_sel.index if t in prices.columns]
if not present: st.error("Ningún ticker con datos."); st.stop()
weights_sel = (weights_sel.loc[present] / weights_sel.loc[present].sum())

st.title("📈 Índice del Portafolio — estilo Apple Stocks")

try:
    pv_raw, shares_df, P_aligned = simulate_with_holdings_from_start(prices, weights_sel, start_value=1.0)
    pv, scale_k, anchor_dt = normalize_to_anchor(pv_raw, ANCHOR_DATE, ANCHOR_VALUE)

    lp_used, pp_used, used_live, mkt_state, ts_map = latest_price_vectors(P_aligned)

    # Shares vigentes DURANTE el día t = shares post-rebalanceo del día anterior
    shares_during = shares_df.shift(1)

    last_idx = P_aligned.index[-1]
    last_shares_for_today = shares_during.loc[last_idx] if last_idx in shares_during.index else shares_df.iloc[0]
    last_val = float(np.dot(last_shares_for_today.values, lp_used.values)) * scale_k if len(lp_used) else np.nan
    delta_str = "n/a"
    if pp_used is not None and (pp_used != 0).all():
        prev_val = float(np.dot(last_shares_for_today.values, pp_used.values)) * scale_k
        if prev_val: delta_str = f"{((last_val/prev_val)-1.0)*100.0:.2f}%"

    ms = yahoo_market_state("QQQ")
    weekday = datetime.now(US_EASTERN).weekday()
    live_badge = ""
    if live_on and mkt_state == "open":
        live_badge = " (LIVE)"
    elif ms == "CLOSED" and weekday < 5:
        live_badge = " (FERIADO)"
    else:
        live_badge = " (Cierre HOY)" if mkt_state != "open" else live_badge

    col1, col2, col3 = st.columns([1.6,1,1])
    with col1:
        st.metric(f"Índice normalizado a 1.000 en 2025-08-01 — {'snapshot' if mkt_state=='open' else 'último cierre'}{live_badge}",
                  f"{last_val:,.2f}", delta=delta_str)
        if anchor_dt is not None:
            st.caption(f"Anclado a 1,000 el {anchor_dt.date().isoformat()} (o primer hábil ≥ 2025-08-01).")
        if used_live: st.caption("Precios recientes utilizados para: " + ", ".join(sorted(used_live.keys())))
        st.caption(f"Estado de mercado: {mkt_state} • marketState={ms} • Hora ET: {datetime.now(US_EASTERN).strftime('%Y-%m-%d %H:%M:%S')}")

    @st.cache_data(ttl=20, show_spinner=False)
    def yahoo_quote_rest(symbol: str):
        url = f"https://query1.finance.yahoo.com/v7/finance/quote?symbols={symbol}"
        try:
            r = requests.get(url, headers={"User-Agent":"Mozilla/5.0"}, timeout=6)
            if r.status_code == 200:
                data = r.json(); q = data.get("quoteResponse", {}).get("result", [])
                if q:
                    q=q[0]; px=q.get("regularMarketPrice") or q.get("postMarketPrice")
                    pc=q.get("regularMarketPreviousClose")
                    ts=q.get("regularMarketTime") or q.get("postMarketTime")
                    ts=pd.to_datetime(ts, unit="s", utc=True) if ts else None
                    if px and pc: return float(px), float(pc), ts, "Yahoo Quote"
        except Exception:
            pass
        return None, None, None, None

    def qqq_day_return():
        px, pc, ts, src = yahoo_quote_rest("QQQ")
        if px is not None and pc not in (None, 0): return (px/pc-1.0)*100.0, src, ts
        px2, pc2, ts2 = yahoo_intraday_last("QQQ")
        if px2 is not None and pc2 not in (None, 0): return (px2/pc2-1.0)*100.0, "Yahoo Chart 1m", ts2
        return None, "N/A", None

    with col2:
        qqq_ret, q_src, q_ts = qqq_day_return()
        if qqq_ret is not None:
            st.metric("QQQ (día)", f"{qqq_ret:.2f}%"); st.caption(f"Fuente: {q_src}" + (f" • {q_ts}" if q_ts is not None else ""))
        else: st.metric("QQQ (día)", "n/a")

    # ----- Gráfico índice -------
    ranges = ["1D","1W","1M","3M","6M","YTD","1Y","2Y","5Y","MAX"]
    sel = st.radio("Rango", options=ranges, index=ranges.index("6M"), horizontal=True)
    pv_disp = pv.copy(); end_idx = pv_disp.index[-1]
    if sel=="1D": start_idx = pv_disp.index[-2] if len(pv_disp)>=2 else pv_disp.index[0]
    elif sel=="1W": start_idx = end_idx - pd.Timedelta(days=7)
    elif sel=="1M": start_idx = end_idx - pd.DateOffset(months=1)
    elif sel=="3M": start_idx = end_idx - pd.DateOffset(months=3)
    elif sel=="6M": start_idx = end_idx - pd.DateOffset(months=6)
    elif sel=="YTD": start_idx = pd.Timestamp(year=end_idx.year, month=1, day=1, tz=end_idx.tz)
    elif sel=="1Y": start_idx = end_idx - pd.DateOffset(years=1)
    elif sel=="2Y": start_idx = end_idx - pd.DateOffset(years=2)
    elif sel=="5Y": start_idx = end_idx - pd.DateOffset(years=5)
    else: start_idx = pv_disp.index[0]
    pv_disp = pv_disp.loc[pv_disp.index>=start_idx]
    if live_on and mkt_state=="open" and len(pv_disp)>0:
        now_idx = to_index_tz(datetime.now(US_EASTERN), pv_disp.index)
        pv_disp = pd.concat([pv_disp, pd.Series([last_val], index=[now_idx])])
    if len(pv_disp)>=2:
        range_delta_abs = pv_disp.iloc[-1]-pv_disp.iloc[0]; range_delta_pct=(pv_disp.iloc[-1]/pv_disp.iloc[0]-1.0)*100.0
    else: range_delta_abs, range_delta_pct = None, None
    color="green" if (range_delta_abs is None or range_delta_abs>=0) else "red"
    chart_df = pv_disp.rename("Índice").to_frame(); fig = px.area(chart_df, x=chart_df.index, y="Índice")
    fig.update_traces(line_color=color, fillcolor=color, opacity=0.25); fig.update_layout(margin=dict(l=0,r=0,t=10,b=0), showlegend=False, xaxis_title="", yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

    tab1, tab2, tab3, tab4 = st.tabs(["Métricas","Exposición","Datos","Breakdown"])

    # ----- Métricas -----
    with tab1:
        st.subheader("Métricas (serie normalizada)")
        def perf_stats(series: pd.Series) -> dict:
            s=series.dropna(); r=s.pct_change().dropna()
            if len(r)==0: return {"CAGR":np.nan,"Vol anual":np.nan,"Max DD":np.nan,"Sharpe~":np.nan}
            years=(s.index[-1]-s.index[0]).days/365.25
            cagr=(s.iloc[-1]/s.iloc[0])**(1/years)-1 if years>0 else np.nan
            vol=r.std()*np.sqrt(252); dd=(s/s.cummax()-1.0).min(); sharpe=cagr/vol if (vol and vol>0) else np.nan
            return {"CAGR":cagr,"Vol anual":vol,"Max DD":dd,"Sharpe~":sharpe}
        stats=perf_stats(pv)
        fmt={"CAGR": f"{stats['CAGR']:.2%}" if pd.notna(stats['CAGR']) else "n/a",
             "Vol anual": f"{stats['Vol anual']:.2%}" if pd.notna(stats['Vol anual']) else "n/a",
             "Max DD": f"{stats['Max DD']:.2%}" if pd.notna(stats['Max DD']) else "n/a",
             "Sharpe~": f"{stats['Sharpe~']:.2f}" if pd.notna(stats['Sharpe~']) else "n/a"}
        st.dataframe(pd.DataFrame(fmt, index=["Índice"]).T, width="stretch")

    # ----- Exposición por sector -----
    with tab2:
        st.subheader("Exposición por sector (pesos objetivo)")
        st.caption("Puedes editar 'sectors.csv' (ticker,sector).")
        def _clean_secs_df(df):
            df=df.copy()
            if "ticker" in df.columns: df["ticker"]=df["ticker"].astype(str).str.strip().str.upper()
            if "sector" in df.columns: df["sector"]=df["sector"].fillna("").astype(str).str.strip()
            return df.drop_duplicates(subset=["ticker"], keep="first")
        try:
            sec = pd.read_csv("sectors.csv")
        except Exception:
            sec = pd.DataFrame(columns=["ticker","sector"])
        sec = _clean_secs_df(sec)
        expos = pd.DataFrame({"ticker":[t.strip().upper() for t in weights_sel.index], "weight":weights_sel.values})
        out = expos.merge(sec, on="ticker", how="left"); out["sector"]=out["sector"].fillna("").replace("","Unknown")
        expos2=out.groupby("sector", as_index=False)["weight"].sum().sort_values("weight", ascending=False)
        fig2=px.bar(expos2, x="sector", y="weight", labels={"weight":"Peso"}); st.plotly_chart(fig2, use_container_width=True)
        st.dataframe(expos2.style.format({"weight":"{:.2%}"}), width="stretch")

    # ----- Datos base -----
    with tab3:
        st.subheader("Series base (precios) — diario + snapshot de hoy si LIVE")
        P_show=P_aligned.ffill().copy()
        if live_on and mkt_state=="open":
            now_idx2 = to_index_tz(datetime.now(US_EASTERN), P_show.index); P_show.loc[now_idx2]=lp_used; P_show=P_show.sort_index()
        st.dataframe(P_show.tail(200), width="stretch")
        st.download_button("Descargar precios CSV", data=P_show.to_csv().encode("utf-8"), file_name="prices.csv", mime="text/csv")
        out_idx = pd.DataFrame({"pv_normalizada": pv})
        st.download_button("Descargar índice CSV", data=out_idx.to_csv().encode("utf-8"), file_name="index_values.csv", mime="text/csv")

    # ----- Breakdown -----
    with tab4:
        st.subheader("Breakdown de cambios por ticker (con fuente, timestamp y peso actual)")
        st.caption("Regla: feriados y fines de semana → cierre del día hábil anterior.")

        if len(P_aligned)>=2:
            d_last=P_aligned.index[-1]
            # Shares vigentes DURANTE el último día:
            shares_today = shares_during.loc[d_last] if d_last in shares_during.index else shares_df.iloc[0]
            px_last=lp_used.copy(); px_prev=pp_used.copy()
            dP=px_last-px_prev
            abs_contrib=(shares_today*dP)*scale_k; idx_move=abs_contrib.sum()
            pct_move=(px_last/px_prev-1.0).rename("ret_%")
            pos_vals=(shares_df.loc[d_last]*px_last)*scale_k  # peso actual al cierre/snapshot con shares POST-rebalanceo
            total_val=pos_vals.sum()
            peso_actual=(pos_vals/total_val)*100.0 if total_val!=0 else pd.Series(np.nan, index=pos_vals.index)

            fuente_series=pd.Series({t: used_live.get(t,'DailyClose') for t in px_last.index})
            ts_series=pd.Series({t: ts_map.get(t) for t in px_last.index})
            last_day_df=pd.DataFrame({
                "precio_prev":px_prev,"precio_ult":px_last,"ret_%":pct_move*100.0,
                "contrib_abs_puntos":abs_contrib,
                "contrib_%_del_mov_indice":(abs_contrib/idx_move*100.0) if idx_move!=0 else np.nan,
                "peso_actual":peso_actual,"fuente":fuente_series,"snapshot_ts":ts_series
            }).sort_values("contrib_abs_puntos", ascending=False)
            st.markdown("**Último día / snapshot**")
            st.dataframe(last_day_df.style.format({
                "precio_prev":"{:.2f}","precio_ult":"{:.2f}","ret_%":"{:.2f}%",
                "contrib_abs_puntos":"{:.2f}","contrib_%_del_mov_indice":"{:.2f}%","peso_actual":"{:.2f}%"
            }), width="stretch")
            fig_ld=px.bar(last_day_df.reset_index().rename(columns={"index":"ticker"}), x="ticker", y="contrib_abs_puntos", title="Contribución absoluta (puntos) — último día/snapshot")
            st.plotly_chart(fig_ld, use_container_width=True)
        else:
            st.warning("No hay suficientes datos para el breakdown del último día.")

        # Rango
        pv_r=pv_disp.copy()
        if len(pv_r)<2: st.warning("Rango demasiado corto.")
        else:
            idx_all=P_aligned.index; start_ts=pv_disp.index[0]; end_ts=pv_disp.index[-1]
            start_pos=idx_all.searchsorted(start_ts); end_pos=idx_all.searchsorted(end_ts, side="right")-1
            if end_pos<=start_pos: st.warning("Rango demasiado corto para breakdown.")
            else:
                contrib=pd.Series(0.0, index=P_aligned.columns)
                for i in range(start_pos+1, end_pos+1):
                    d_i=idx_all[i]; d_prev_i=idx_all[i-1]
                    dP_i=P_aligned.loc[d_i]-P_aligned.loc[d_prev_i]
                    shares_i=shares_during.loc[d_i]  # shares vigentes DURANTE el día i
                    contrib=contrib.add(shares_i*dP_i, fill_value=0.0)
                # Si LIVE, añadimos el tramo intradía desde el último cierre al snapshot
                if live_on and mkt_state=="open":
                    last_daily_close_prices=P_aligned.loc[P_aligned.index[-1]]
                    intraday_move=lp_used-last_daily_close_prices
                    shares_today2=shares_during.loc[P_aligned.index[-1]]
                    contrib=contrib.add(shares_today2*intraday_move, fill_value=0.0)
                contrib=contrib*scale_k; total_pts=contrib.sum()

                base_prices=P_aligned.loc[idx_all[start_pos]]
                end_prices=(lp_used if (live_on and mkt_state=="open") else P_aligned.loc[idx_all[end_pos]])
                ret_range_pct=(end_prices/base_prices-1.0)*100.0

                shares_end=shares_df.loc[idx_all[end_pos]]
                pos_vals_end=(shares_end*end_prices)*scale_k; total_end=pos_vals_end.sum()
                peso_actual_rango=(pos_vals_end/total_end)*100.0 if total_end!=0 else pd.Series(np.nan, index=pos_vals_end.index)

                range_df=pd.DataFrame({
                    "ret_%_rango":ret_range_pct,
                    "contrib_abs_puntos":contrib,
                    "contrib_%_del_mov_indice":(contrib/total_pts*100.0) if total_pts!=0 else np.nan,
                    "peso_actual_fin_rango":peso_actual_rango
                }).sort_values("contrib_abs_puntos", ascending=False)

                st.dataframe(range_df.style.format({
                    "ret_%_rango":"{:.2f}%","contrib_abs_puntos":"{:.2f}",
                    "contrib_%_del_mov_indice":"{:.2f}%","peso_actual_fin_rango":"{:.2f}%"
                }), width="stretch")
                fig_rg=px.bar(range_df.reset_index().rename(columns={"index":"ticker"}), x="ticker", y="contrib_abs_puntos", title="Contribución absoluta (puntos) — rango")
                st.plotly_chart(fig_rg, use_container_width=True)

    st.info("Feriados y fines de semana: estado 'preopen' (no LIVE) gracias a marketState de Yahoo. Rebalanceo al CIERRE del día (15 y último hábil).")

    if live_on and refresh_sec and mkt_state=="open":
        time.sleep(int(refresh_sec)); st.rerun()

except Exception as e:
    st.error(f"Error: {e}")
