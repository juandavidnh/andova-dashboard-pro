
import os
import time
from typing import Optional, Dict
from datetime import date, datetime

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf
import requests
import pytz

st.set_page_config(page_title="Índice Portafolio (estilo Apple Stocks)", layout="wide")

# --------- Parámetros ---------
ANCHOR_DATE = pd.Timestamp("2025-08-01")   # valor=1000 en esta fecha (o primer hábil ≥)
ANCHOR_VALUE = 1000.0
US_EASTERN = pytz.timezone("US/Eastern")

# --------- Utilidades ---------
@st.cache_data(show_spinner=False, ttl=3600)
def load_weights(path: str = "weights.csv") -> pd.Series:
    df = pd.read_csv(path)
    s = pd.Series(df["weight_pct"].values, index=df["ticker"].values).astype(float)
    s = s[s > 0]
    return s / s.sum()

@st.cache_data(show_spinner=True, ttl=600)
def download_prices(tickers, start, end):
    df = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        prices = df["Close"].copy()
    else:
        prices = df[["Close"]].copy()
        prices.columns = [tickers[0]]
    prices.columns = [c.split("=")[0] if isinstance(c, str) else c for c in prices.columns]
    prices = prices.sort_index()
    prices = prices.dropna(axis=1, how="all")
    return prices

def first_trading_on_or_after(index: pd.DatetimeIndex, target: pd.Timestamp):
    i = index.searchsorted(target)
    return index[i] if i < len(index) else None

def rebalance_dates_semimonthly(prices: pd.DataFrame) -> set:
    idx = prices.index
    if len(idx) == 0:
        return set()
    rdates = set(prices.resample("M").last().index)  # month-end
    months = pd.period_range(idx[0], idx[-1], freq="M")
    for p in months:
        fifteenth = pd.Timestamp(p.year, p.month, 15)
        cand = first_trading_on_or_after(idx, fifteenth)
        if cand is not None and cand.month == p.month:
            rdates.add(cand)
    return rdates

def simulate_with_holdings_from_start(prices: pd.DataFrame, weights: pd.Series, start_value: float = 1.0):
    """Simula desde el primer día disponible del rango, con rebalanceos 15/fin de mes."""
    tickers_avail = [t for t in weights.index if t in prices.columns]
    P = prices[tickers_avail].copy().ffill().dropna(how="any")
    if len(P) == 0:
        raise ValueError("No hay suficientes datos tras el inicio para todos los tickers.")
    w = (weights.loc[P.columns] / weights.loc[P.columns].sum()).copy()
    dates = P.index
    shares = (start_value * w / P.iloc[0]).values
    rdates = rebalance_dates_semimonthly(P)

    pv_list, shares_rows = [], []
    for d in dates:
        shares_rows.append(pd.Series(shares, index=P.columns, name=d))
        v = float(np.dot(shares, P.loc[d].values))
        pv_list.append(v)
        if d in rdates:
            shares = (v * w / P.loc[d]).values

    pv_raw = pd.Series(pv_list, index=dates, name="pv_raw")
    shares_df = pd.DataFrame(shares_rows)
    return pv_raw, shares_df, P

def normalize_to_anchor(pv_raw: pd.Series, anchor_date: pd.Timestamp, anchor_value: float):
    if len(pv_raw) == 0:
        return pv_raw, 1.0, None
    idx = pv_raw.index
    anchor = first_trading_on_or_after(idx, anchor_date)
    if anchor is None:
        anchor = idx[-1]
    base_val = float(pv_raw.loc[anchor])
    scale = anchor_value / base_val if (base_val and not np.isnan(base_val)) else 1.0
    return (pv_raw * scale).rename("pv"), scale, anchor

# ---- Live quotes providers ----
@st.cache_data(ttl=5, show_spinner=False)
def yahoo_intraday_last_all(tickers) -> tuple[dict, Optional[pd.Timestamp]]:
    """Último precio intradía (posible retraso ~15m) + timestamp común."""
    out = {}
    ts = None
    try:
        data = yf.download(tickers, period="1d", interval="1m", auto_adjust=True, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            if "Close" in data.columns.levels[0]:
                close_df = data["Close"].dropna(how="all")
                if not close_df.empty:
                    last_row = close_df.iloc[-1]
                    ts = last_row.name if hasattr(last_row, 'name') else None
                    last = last_row.dropna()
                    out = {str(k): float(v) for k, v in last.to_dict().items()}
        else:
            last_series = data["Close"].dropna()
            if len(last_series) > 0:
                ts = last_series.index[-1]
                out[str(tickers[0])] = float(last_series.iloc[-1])
    except Exception:
        pass
    return out, ts

@st.cache_data(ttl=5, show_spinner=False)
def finnhub_last_quote(symbol, token: Optional[str]):
    try:
        if not token:
            return None, None, None
        url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={token}"
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            j = r.json()
            if "c" in j and j["c"] not in (None, 0):
                ts = j.get("t")
                ts = pd.to_datetime(ts, unit="s", utc=True) if ts else None
                return float(j["c"]), float(j.get("pc") or 0.0), ts
    except Exception:
        pass
    return None, None, None

def market_state(p_aligned: pd.DataFrame) -> str:
    """Devuelve:
    - 'postclose' SOLO si >=16:05 ET **y** ya existe barra diaria de HOY.
    - 'open' durante sesión regular **o** en after-hours (~16:00–20:00 ET) **o** si no hay barra diaria de hoy pero ya pasó el cierre (para usar intradía).
    - 'preopen' resto (antes de 9:30 o fines de semana/feriados).
    """
    now_et = datetime.now(US_EASTERN)
    weekday = now_et.weekday()  # 0=lunes ... 6=domingo
    has_today = len(p_aligned.index) > 0 and p_aligned.index[-1].date() == now_et.date()

    open_t = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    close_t = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
    postclose_gate = now_et.replace(hour=16, minute=5, second=0, microsecond=0)
    ah_end = now_et.replace(hour=20, minute=0, second=0, microsecond=0)

    if has_today and now_et >= postclose_gate:
        return "postclose"
    # mercado abierto o after-hours, o bien no hay barra diaria de hoy y ya pasó el cierre
    if weekday < 5 and (open_t <= now_et < ah_end or (now_et >= close_t and not has_today)):
        return "open"
    return "preopen"""

def latest_price_vectors(P_aligned: pd.DataFrame, provider: str, token: Optional[str]):
    """Regla + salidas extendidas:
       - lp_used: último precio (cierre de hoy si postclose; live si open; cierre de ayer si preopen)
       - pp_used: SIEMPRE cierre de ayer
       - used: fuente por ticker
       - state: estado de mercado
       - ts_map: timestamp por ticker (o común)
    """
    state = market_state(P_aligned)
    last_idx = P_aligned.index[-1]
    today_et = datetime.now(US_EASTERN).date()
    if last_idx.date() == today_et and len(P_aligned.index) >= 2:
        yday_close = P_aligned.loc[P_aligned.index[-2]].copy()
        today_close = P_aligned.loc[P_aligned.index[-1]].copy()
        have_today = True
    else:
        yday_close = P_aligned.loc[P_aligned.index[-1]].copy()
        today_close = None
        have_today = False

    used: Dict[str, str] = {}
    ts_map: Dict[str, Optional[pd.Timestamp]] = {}

    if state == "postclose" and have_today:
        lp_used = today_close
        pp_used = yday_close
        ts_common = pd.Timestamp.combine(pd.Timestamp(today_et), pd.Timestamp('16:00').time()).tz_localize('US/Eastern')
        for t in P_aligned.columns:
            used[t] = "DailyClose"
            ts_map[t] = ts_common
    elif state == "open":
        lp_used = yday_close.copy()
        use_finnhub = provider.startswith("Finnhub") and bool(token)
        if use_finnhub:
            for t in P_aligned.columns:
                px, pc, ts = finnhub_last_quote(t, token)
                if px is not None:
                    lp_used[t] = px
                    used[t] = "Finnhub"
                    ts_map[t] = ts
                if pd.isna(yday_close.get(t)) and pc is not None:
                    yday_close[t] = pc
        else:
            quotes, ts = yahoo_intraday_last_all(list(P_aligned.columns))
            for t, v in quotes.items():
                lp_used[t] = float(v)
                used[t] = "Yahoo intraday"
                ts_map[t] = ts
        pp_used = yday_close
    else:  # preopen
        lp_used = yday_close
        pp_used = yday_close
        ts_common = pd.Timestamp.combine(pd.Timestamp(yday_close.name), pd.Timestamp('16:00').time())
        for t in P_aligned.columns:
            used[t] = "DailyClose"
            ts_map[t] = ts_common

    return lp_used, pp_used, used, state, ts_map

def sector_exposures(weights: pd.Series, sector_df: pd.DataFrame) -> pd.DataFrame:
    if sector_df is None or sector_df.empty:
        return pd.DataFrame({"sector":["Unknown"], "weight":[1.0]})
    df = pd.DataFrame({"ticker": weights.index, "weight": weights.values})
    out = df.merge(sector_df, on="ticker", how="left")
    out["sector"] = out["sector"].fillna("Unknown").replace("", "Unknown")
    return out.groupby("sector", as_index=False)["weight"].sum().sort_values("weight", ascending=False)

def download_bytes(df: pd.DataFrame):
    return df.to_csv(index=True).encode("utf-8")

def apply_timerange(series: pd.Series, mode: str):
    if series.empty:
        return series
    end = series.index[-1]
    if mode == "1D":
        start = series.index[-2] if len(series) >= 2 else series.index[0]
    elif mode == "1W":
        start = end - pd.Timedelta(days=7)
    elif mode == "1M":
        start = end - pd.DateOffset(months=1)
    elif mode == "3M":
        start = end - pd.DateOffset(months=3)
    elif mode == "6M":
        start = end - pd.DateOffset(months=6)
    elif mode == "YTD":
        start = pd.Timestamp(year=end.year, month=1, day=1)
    elif mode == "1Y":
        start = end - pd.DateOffset(years=1)
    elif mode == "2Y":
        start = end - pd.DateOffset(years=2)
    elif mode == "5Y":
        start = end - pd.DateOffset(years=5)
    else:  # MAX
        start = series.index[0]
    return series.loc[series.index >= start]

# --------- Sidebar ---------
st.sidebar.title("🔧 Configuración")

weights = load_weights()
try:
    sectors = pd.read_csv("sectors.csv")
except Exception:
    sectors = pd.DataFrame(columns=["ticker","sector"])

tickers_all = list(weights.index)
selected = st.sidebar.multiselect("Filtra tickers (renormaliza pesos)", tickers_all, default=tickers_all)

default_start = date(2018,1,1)
start = st.sidebar.date_input("Fecha inicio descarga (para ver <1000, usa antes del 2025-08-01)", default_start)
end = st.sidebar.date_input("Fecha fin", date.today())

show_dd = st.sidebar.checkbox("Mostrar drawdown", value=False)

# Live options
st.sidebar.markdown("---")
st.sidebar.subheader("⚡ Actualización en vivo (opcional)")

live_on = st.sidebar.checkbox("Activar live", value=st.session_state.get("live_on", True), key="live_on")
provider = st.sidebar.selectbox("Proveedor", ["Yahoo (posible retraso ~15m)", "Finnhub (tiempo real)"])
refresh_sec = st.sidebar.number_input("Auto-refresh (segundos)", min_value=5, max_value=120, value=15, step=5)

# Finnhub key
finnhub_key = None
try:
    finnhub_key = st.secrets.get("FINNHUB_API_KEY", None)
except Exception:
    finnhub_key = None
if not finnhub_key:
    finnhub_key = os.environ.get("FINNHUB_API_KEY")

if provider.startswith("Finnhub") and live_on and not finnhub_key:
    st.sidebar.warning("Configura FINNHUB_API_KEY en `st.secrets` o como variable de entorno para tiempo real.")

# --------- Data & Sim ---------
weights_sel = (weights.loc[selected] if selected else weights).copy()
weights_sel = weights_sel / weights_sel.sum()

st.sidebar.write("Pesos efectivos (normalizados):")
st.sidebar.dataframe(pd.DataFrame({"weight": weights_sel}).style.format({"weight": "{:.2%}"}), use_container_width=True)

prices = download_prices(list(weights_sel.index), start, end)
missing = [t for t in weights_sel.index if t not in prices.columns]
if missing:
    st.sidebar.warning("Tickers omitidos por falta de datos: " + ", ".join(missing))
    keep = [t for t in weights_sel.index if t in prices.columns]
    if keep:
        weights_sel = (weights_sel.loc[keep] / weights_sel.loc[keep].sum())

st.title("📈 Índice del Portafolio — estilo Apple Stocks")

try:
    pv_raw, shares_df, P_aligned = simulate_with_holdings_from_start(prices, weights_sel, start_value=1.0)
    pv, scale_k, anchor_dt = normalize_to_anchor(pv_raw, ANCHOR_DATE, ANCHOR_VALUE)

    # ---- Snapshot de precios para HOY siguiendo la regla ----
    lp_used, pp_used, used_live, mkt_state, ts_map = latest_price_vectors(P_aligned, provider if live_on else "Yahoo", finnhub_key if live_on else None)

    # Persistir último snapshot válido para evitar vacíos por rate limits
    if "last_live_lp" not in st.session_state:
        st.session_state["last_live_lp"] = None
        st.session_state["last_live_used"] = None
        st.session_state["last_live_ts"] = None
        st.session_state["last_live_date"] = None

    # Guardar snapshot si viene válido
    if live_on and used_live and len(lp_used.dropna()) > 0:
        st.session_state["last_live_lp"] = lp_used.copy()
        st.session_state["last_live_used"] = used_live.copy()
        st.session_state["last_live_ts"] = ts_map.copy()
        st.session_state["last_live_date"] = datetime.now(US_EASTERN).date()

    # Reusar snapshot SOLO si es del mismo día calendario ET y estamos en open (incluye after-hours)
    elif live_on and st.session_state.get("last_live_lp") is not None and (mkt_state == "open")         and st.session_state.get("last_live_date") == datetime.now(US_EASTERN).date():
        lp_used = st.session_state["last_live_lp"].copy()
        used_live = st.session_state["last_live_used"].copy()
        ts_map = st.session_state["last_live_ts"].copy() if st.session_state["last_live_ts"] else {}

    # ---- Top metric (último valor & cambio del día) ----
    last_shares = shares_df.loc[P_aligned.index[-1]]
    last_val = float(np.dot(last_shares.values, lp_used.values)) * scale_k if len(lp_used) else np.nan

    delta_str = "n/a"
    live_badge = ""
    if pp_used is not None and (pp_used != 0).all():
        prev_val = float(np.dot(last_shares.values, pp_used.values)) * scale_k
        if prev_val:
            delta_str = f"{((last_val / prev_val) - 1.0) * 100.0:.2f}%"
    if live_on and mkt_state == "open":
        live_badge = " (LIVE)"
    elif mkt_state == "postclose":
        live_badge = " (Cierre HOY)"

    col1, col2, col3 = st.columns([1.5, 1, 1])
    with col1:
        st.metric(f"Índice normalizado a 1.000 en 2025-08-01 — último {'snapshot' if mkt_state=='open' else 'cierre'}{live_badge}", f"{last_val:,.2f}", delta=delta_str)
        if anchor_dt is not None:
            st.caption(f"Anclado a {ANCHOR_VALUE:,.0f} el {anchor_dt.date().isoformat()} (o primer hábil ≥ 2025-08-01).")
        if used_live:
            st.caption("Precios recientes utilizados para: " + ", ".join(sorted(used_live.keys())))
        st.caption(f"Estado de mercado: {mkt_state} • Hora ET: {datetime.now(US_EASTERN).strftime('%Y-%m-%d %H:%M:%S')}")

    # ---- Selector de rango (estilo Apple) ----
    ranges = ["1D", "1W", "1M", "3M", "6M", "YTD", "1Y", "2Y", "5Y", "MAX"]
    sel = st.radio("Rango", options=ranges, index=ranges.index("6M"), horizontal=True)
    pv_disp = apply_timerange(pv, sel)

    # Si abierto, añadimos snapshot como último punto virtual (sin alterar histórico)
    if live_on and mkt_state == "open" and len(pv_disp) > 0:
        pv_disp = pd.concat([pv_disp, pd.Series([last_val], index=[pv_disp.index[-1] + pd.Timedelta(seconds=1)])])

    # Variación en el rango
    range_delta_abs, range_delta_pct = None, None
    if len(pv_disp) >= 2:
        range_delta_abs = pv_disp.iloc[-1] - pv_disp.iloc[0]
        range_delta_pct = (pv_disp.iloc[-1] / pv_disp.iloc[0] - 1.0) * 100.0

    # Gráfico principal (área verde/roja)
    color = "green" if (range_delta_abs is None or range_delta_abs >= 0) else "red"
    chart_df = pv_disp.rename("Índice").to_frame()
    fig = px.area(chart_df, x=chart_df.index, y="Índice")
    fig.update_traces(line_color=color, fillcolor=color, opacity=0.25)
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), showlegend=False, xaxis_title="", yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

    if range_delta_abs is not None:
        sign = "▲" if range_delta_abs >= 0 else "▼"
        st.caption(f"{sign} {range_delta_abs:,.2f} puntos ({range_delta_pct:.2f}%) en {sel}{' • LIVE' if (live_on and mkt_state=='open') else ''}")

    # ---- Tabs (Métricas / Exposición / Datos / Breakdown) ----
    tab1, tab2, tab3, tab4 = st.tabs(["Métricas", "Exposición", "Datos", "Breakdown"])

    with tab1:
        st.subheader("Métricas (serie normalizada)")
        def perf_stats(series: pd.Series) -> dict:
            s = series.dropna()
            r = s.pct_change().dropna()
            if len(r) == 0:
                return {"CAGR": np.nan, "Vol anual": np.nan, "Max DD": np.nan, "Sharpe~": np.nan}
            years = (s.index[-1] - s.index[0]).days / 365.25
            cagr = (s.iloc[-1] / s.iloc[0]) ** (1/years) - 1 if years > 0 else np.nan
            vol = r.std() * np.sqrt(252)
            dd = (s / s.cummax() - 1.0).min()
            sharpe = cagr / vol if (vol and vol > 0) else np.nan
            return {"CAGR": cagr, "Vol anual": vol, "Max DD": dd, "Sharpe~": sharpe}
        stats = perf_stats(pv)
        fmt = {
            "CAGR": f"{stats['CAGR']:.2%}" if pd.notna(stats['CAGR']) else "n/a",
            "Vol anual": f"{stats['Vol anual']:.2%}" if pd.notna(stats['Vol anual']) else "n/a",
            "Max DD": f"{stats['Max DD']:.2%}" if pd.notna(stats['Max DD']) else "n/a",
            "Sharpe~": f"{stats['Sharpe~']:.2f}" if pd.notna(stats['Sharpe~']) else "n/a"
        }
        st.dataframe(pd.DataFrame(fmt, index=["Índice"]).T, use_container_width=True)

    with tab2:
        st.subheader("Exposición por sector (pesos objetivo)")
        expos = sector_exposures(weights_sel, sectors)
        fig2 = px.bar(expos, x="sector", y="weight", labels={"weight": "Peso"})
        st.plotly_chart(fig2, use_container_width=True)
        st.dataframe(expos.style.format({"weight": "{:.2%}"}), use_container_width=True)

    with tab3:
        st.subheader("Series base (diarias, Yahoo)")
        st.dataframe(P_aligned.ffill().tail(200), use_container_width=True)
        st.download_button("Descargar precios CSV", data=P_aligned.ffill().to_csv().encode("utf-8"), file_name="prices.csv", mime="text/csv")
        out = pd.DataFrame({"pv_normalizada": pv})
        st.download_button("Descargar índice CSV", data=out.to_csv().encode("utf-8"), file_name="index_values.csv", mime="text/csv")

    with tab4:
        st.subheader("Breakdown de cambios por ticker (con fuente y timestamp)")
        st.caption("Regla: cierre de hoy si ya cerró; si abierto → precio LIVE; si no abrió → cierre de ayer. `precio_prev` SIEMPRE = cierre de ayer.")

        if len(P_aligned) >= 2:
            d_last = P_aligned.index[-1]
            d_prev = P_aligned.index[-2]
            px_last = lp_used.copy()                   # último precio según regla
            px_prev = pp_used.copy()                   # siempre cierre de ayer

            shares_today = shares_df.loc[d_last]       # shares del último día histórico

            dP = px_last - px_prev
            abs_contrib = (shares_today * dP) * scale_k
            idx_move = abs_contrib.sum()
            pct_move = (px_last / px_prev - 1.0).rename("ret_%")

            # Fuente y timestamp por ticker
            fuente_series = pd.Series({t: used_live.get(t, 'DailyClose') for t in px_last.index})
            ts_series = pd.Series({t: ts_map.get(t) for t in px_last.index})

            last_day_df = pd.DataFrame({
                "precio_prev": px_prev,
                "precio_ult": px_last,
                "ret_%": pct_move * 100.0,
                "contrib_abs_puntos": abs_contrib,
                "contrib_%_del_mov_indice": (abs_contrib / idx_move * 100.0) if idx_move != 0 else np.nan,
                "fuente": fuente_series,
                "snapshot_ts": ts_series
            }).sort_values("contrib_abs_puntos", ascending=False)

            st.markdown("**Último día / snapshot**")
            st.dataframe(last_day_df.style.format({
                "precio_prev": "{:.2f}", "precio_ult": "{:.2f}", "ret_%": "{:.2f}%",
                "contrib_abs_puntos": "{:.2f}", "contrib_%_del_mov_indice": "{:.2f}%"
            }), use_container_width=True)
            fig_ld = px.bar(last_day_df.reset_index().rename(columns={"index": "ticker"}),
                            x="ticker", y="contrib_abs_puntos", title="Contribución absoluta (puntos) — último día/snapshot")
            st.plotly_chart(fig_ld, use_container_width=True)
        else:
            st.warning("No hay suficientes datos para el breakdown del último día.")

        # Rango (según selector superior)
        pv_r = apply_timerange(pv, sel)
        if len(pv_r) < 2:
            st.warning("Rango demasiado corto.")
        else:
            start_ts = pv_r.index[0]
            end_ts = pv_r.index[-1]
            idx = P_aligned.index
            start_pos = idx.searchsorted(start_ts)
            end_pos = idx.searchsorted(end_ts, side="right") - 1
            if end_pos <= start_pos:
                st.warning("Rango demasiado corto para breakdown.")
            else:
                contrib = pd.Series(0.0, index=P_aligned.columns)
                for i in range(start_pos, end_pos + 1):
                    d_i = idx[i]
                    d_prev_i = idx[i - 1]
                    dP_i = P_aligned.loc[d_i] - P_aligned.loc[d_prev_i]
                    shares_i = shares_df.loc[d_i]
                    contrib = contrib.add(shares_i * dP_i, fill_value=0.0)

                # Ajuste intradía si abierto: añade tramo (LIVE - cierre más reciente)
                if live_on and mkt_state == "open":
                    last_daily_close_prices = P_aligned.loc[P_aligned.index[-1]]
                    intraday_move = lp_used - last_daily_close_prices
                    shares_today2 = shares_df.loc[P_aligned.index[-1]]
                    contrib = contrib.add(shares_today2 * intraday_move, fill_value=0.0)

                contrib = contrib * scale_k
                total_pts = contrib.sum()
                base_prices = P_aligned.loc[idx[start_pos - 1]]
                end_prices = P_aligned.loc[idx[end_pos]]
                ret_range_pct = (end_prices / base_prices - 1.0) * 100.0

                range_df = pd.DataFrame({
                    "ret_%_rango": ret_range_pct,
                    "contrib_abs_puntos": contrib,
                    "contrib_%_del_mov_indice": (contrib / total_pts * 100.0) if total_pts != 0 else np.nan
                }).sort_values("contrib_abs_puntos", ascending=False)

                st.dataframe(range_df.style.format({
                    "ret_%_rango": "{:.2f}%",
                    "contrib_abs_puntos": "{:.2f}",
                    "contrib_%_del_mov_indice": "{:.2f}%"
                }), use_container_width=True)
                fig_rg = px.bar(range_df.reset_index().rename(columns={"index": "ticker"}),
                                x="ticker", y="contrib_abs_puntos", title="Contribución absoluta (puntos) — rango")
                st.plotly_chart(fig_rg, use_container_width=True)

    st.info("Regla aplicada: cierre de hoy si ya cerró; si abierto → LIVE; si no abrió → cierre de ayer. `precio_prev` SIEMPRE = cierre de ayer.")

    # ---- Auto-refresh (no recarga el navegador) ----
    if live_on and refresh_sec and mkt_state == "open":
        time.sleep(int(refresh_sec))
        st.rerun()

except Exception as e:
    st.error(f"Error: {e}")
