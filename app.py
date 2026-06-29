"""
app.py -- Order-Book Imbalance Backtester (dark cool-blue UI).
Reads from Supabase, runs the hysteresis backtest live.
Credentials in .streamlit/secrets.toml (never in this file).
Run:  streamlit run app.py
"""

import os
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Imbalance Backtester", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@500;600&display=swap');

/* base: deep navy-blue */
.stApp {
    background: radial-gradient(1200px 600px at 72% -10%, #14233f 0%, #0b1424 45%, #060b16 100%);
}
header[data-testid="stHeader"] { background: transparent; }
* { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 2.2rem; max-width: 1180px; }

/* sidebar */
section[data-testid="stSidebar"] {
    background: rgba(11,20,36,0.85); border-right: 1px solid rgba(96,165,250,0.12);
    backdrop-filter: blur(8px);
}
.sb-head {
    display:flex; align-items:center; gap:8px; font-family:'IBM Plex Mono',monospace;
    font-size:0.72rem; font-weight:600; letter-spacing:0.18em; text-transform:uppercase;
    color:#7eb6ff; margin:0.4rem 0 1.5rem 0;
}
.sb-head::before { content:""; width:4px; height:15px; border-radius:2px;
    background:#3b82f6; box-shadow:0 0 8px #3b82f6; }

section[data-testid="stSidebar"] .stSlider label {
    color:#c4d2e8 !important; font-weight:600 !important; font-size:0.85rem !important;
}
section[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] [data-testid="stSliderTrack"] > div {
    background: linear-gradient(90deg,#2563eb,#60a5fa) !important;
}
section[data-testid="stSidebar"] .stSlider [role="slider"] {
    background:#60a5fa !important; border:3px solid #0b1424 !important;
    box-shadow:0 0 0 2px #60a5fa, 0 0 12px 2px rgba(96,165,250,0.7) !important;
}
section[data-testid="stSidebar"] .stSlider [data-testid="stThumbValue"] {
    color:#93c5fd !important; font-weight:700 !important; font-family:'IBM Plex Mono',monospace !important;
}
section[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] > div > div {
    background:#1e2d48 !important;
}
.sb-meta {
    margin-top:2rem; padding:0.95rem 1.05rem; background:rgba(96,165,250,0.04);
    border:1px solid rgba(96,165,250,0.15); border-radius:12px;
    font-family:'IBM Plex Mono',monospace; font-size:0.72rem; color:#8197b5; line-height:1.9;
}
.sb-meta b { color:#93c5fd; font-weight:600; }

/* hero */
.brand { display:flex; align-items:center; gap:10px; margin-bottom:1.6rem; }
.brand-dot { width:28px; height:28px; border-radius:50%;
    background:radial-gradient(circle at 35% 35%,#93c5fd,#2563eb);
    box-shadow:0 0 14px rgba(96,165,250,0.7); }
.brand-name { font-weight:700; font-size:1.3rem; color:#eaf2ff; letter-spacing:-0.01em; }
.hero-tag { font-family:'IBM Plex Mono',monospace; font-size:0.7rem; letter-spacing:0.22em;
    text-transform:uppercase; color:#5b7ba8; margin-bottom:0.5rem; }
.hero-title { font-weight:800; font-size:2.7rem; color:#f2f7ff; letter-spacing:-0.02em;
    line-height:1.07; margin:0 0 0.6rem 0; }
.hero-title .accent { color:#60a5fa; text-shadow:0 0 24px rgba(96,165,250,0.45); }
.hero-sub { color:#8fa3c2; font-size:1.0rem; max-width:640px; line-height:1.55; }

/* metric cards */
.card-row { display:flex; gap:14px; margin:2rem 0 0.6rem 0; flex-wrap:wrap; }
.card { flex:1; min-width:158px; background:rgba(20,35,63,0.55);
    border:1px solid rgba(96,165,250,0.14); border-radius:14px; padding:1.15rem 1.3rem;
    box-shadow:0 4px 24px rgba(0,0,0,0.3); }
.card-label { font-family:'IBM Plex Mono',monospace; font-size:0.66rem; font-weight:600;
    letter-spacing:0.13em; text-transform:uppercase; color:#6885ad; margin-bottom:0.55rem; }
.card-value { font-family:'IBM Plex Mono',monospace; font-size:1.85rem; font-weight:600;
    color:#eaf2ff; line-height:1; }
.pos { color:#34d399 !important; text-shadow:0 0 16px rgba(52,211,153,0.4); }
.neg { color:#f87171 !important; text-shadow:0 0 16px rgba(248,113,113,0.35); }

.section-label { font-family:'IBM Plex Mono',monospace; font-size:0.7rem; font-weight:600;
    letter-spacing:0.18em; text-transform:uppercase; color:#6885ad; margin:2.2rem 0 0.8rem 0; }

/* verdict */
.verdict { margin-top:1.5rem; padding:1.1rem 1.5rem; border-radius:14px; font-size:0.96rem;
    font-weight:500; line-height:1.55; }
.verdict-pos { background:rgba(52,211,153,0.08); border:1px solid rgba(52,211,153,0.35);
    color:#6ee7b7; }
.verdict-neg { background:rgba(248,113,113,0.07); border:1px solid rgba(248,113,113,0.3);
    color:#fca5a5; }
</style>
""", unsafe_allow_html=True)


def get_credentials():
    try:
        url = st.secrets["SUPABASE_URL"]; key = st.secrets["SUPABASE_KEY"]
        if url and key: return url, key
    except Exception: pass
    try:
        from dotenv import load_dotenv
        from pathlib import Path
        for c in [Path(__file__).parent / ".env", Path(".env")]:
            if c.exists(): load_dotenv(c); break
    except Exception: pass
    return os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_KEY")


@st.cache_data(ttl=600)
def load_data_from_supabase():
    from supabase import create_client
    url, key = get_credentials()
    if not url or not key: return None
    sb = create_client(url, key)
    rows, start, page = [], 0, 1000
    while True:
        q = (sb.table("orderbook").select("*").order("timestamp")
               .range(start, start + page - 1).execute())
        if not q.data: break
        rows.extend(q.data); start += page
        if len(rows) >= 50000: break
    df = pd.DataFrame(rows)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601")
    return df


def compute_signals(df, enter, exit):
    df = df.copy()
    df["mid"] = (df["bid_price"] + df["ask_price"]) / 2
    df["imbalance"] = (df["bid_qty"] - df["ask_qty"]) / (df["bid_qty"] + df["ask_qty"])
    imb = df["imbalance"].values; pos = np.zeros(len(df)); cur = 0.0
    for i in range(len(df)):
        x = imb[i]
        if cur == 0.0:
            if x > enter: cur = 1.0
            elif x < -enter: cur = -1.0
        elif cur == 1.0 and x < exit: cur = 0.0
        elif cur == -1.0 and x > -exit: cur = 0.0
        pos[i] = cur
    df["position"] = pos
    return df


def run_backtest(df, cost_bps):
    df = df.copy()
    df["ret"] = df["mid"].shift(-1) / df["mid"] - 1
    df["gross_pnl"] = df["position"] * df["ret"]
    df["trade"] = df["position"].diff().abs().fillna(0) > 0
    df["cost"] = df["trade"] * (cost_bps / 1e4)
    df["net_pnl"] = df["gross_pnl"] - df["cost"]
    df["equity_gross"] = df["gross_pnl"].fillna(0).cumsum()
    df["equity_net"] = df["net_pnl"].fillna(0).cumsum()
    return df


def sharpe(s):
    return float(s.mean() / s.std() * np.sqrt(len(s))) if s.std() > 0 else 0.0


with st.sidebar:
    st.markdown('<div class="sb-head">Strategy parameters</div>', unsafe_allow_html=True)
    enter = st.slider("Entry threshold", 0.3, 0.9, 0.6, 0.05)
    exit = st.slider("Exit threshold", 0.0, 0.3, 0.1, 0.05)
    cost_bps = st.slider("Cost per trade (bps)", 0.0, 2.0, 1.0, 0.1)
    st.markdown('<div class="sb-meta"><b>Data</b> · Supabase (Postgres)<br>'
                '<b>Signal</b> · order-book imbalance<br>'
                '<b>Rule</b> · hysteresis entry / exit</div>', unsafe_allow_html=True)

st.markdown('<div class="brand"><div class="brand-dot"></div>'
            '<div class="brand-name">Imbalance</div></div>', unsafe_allow_html=True)
st.markdown('<div class="hero-tag">NSE order-book · live backtest</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">Order-book imbalance,<br>'
            '<span class="accent">backtested live.</span></div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">A short-horizon NSE signal built from buy/sell pressure. '
            'Slide the trading cost and watch a winning edge turn into a loss — the reason '
            'execution speed is worth paying for.</div>', unsafe_allow_html=True)

df = load_data_from_supabase()
if df is None or df.empty:
    st.markdown('<div class="verdict verdict-neg">No data loaded. Check credentials in '
                '.streamlit/secrets.toml and that the orderbook table has rows.</div>',
                unsafe_allow_html=True)
    st.stop()

bt = run_backtest(compute_signals(df, enter, exit), cost_bps)
net = bt["net_pnl"].dropna(); net_ret = net.sum(); net_sh = sharpe(net); n_tr = int(bt["trade"].sum())
cls = "pos" if net_ret > 0 else "neg"

st.markdown(f"""
<div class="card-row">
  <div class="card"><div class="card-label">Rows</div><div class="card-value">{len(bt):,}</div></div>
  <div class="card"><div class="card-label">Trades</div><div class="card-value">{n_tr:,}</div></div>
  <div class="card"><div class="card-label">Net return</div>
    <div class="card-value {cls}">{net_ret:+.4f}</div></div>
  <div class="card"><div class="card-label">Net Sharpe</div>
    <div class="card-value {cls}">{net_sh:+.1f}</div></div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-label">Equity curve · gross vs net of costs</div>',
            unsafe_allow_html=True)

import altair as alt
net_color = "#34d399" if net_ret > 0 else "#f87171"
plot = pd.DataFrame({
    "t": np.arange(len(bt)),
    "gross (before costs)": bt["equity_gross"].values,
    "net (after costs)": bt["equity_net"].values,
}).melt("t", var_name="series", value_name="equity")

chart = (alt.Chart(plot).mark_line(strokeWidth=2.5)
         .encode(
            x=alt.X("t:Q", title=None,
                    axis=alt.Axis(grid=False, labelColor="#6885ad",
                                  domainColor="#22344f", tickColor="#22344f")),
            y=alt.Y("equity:Q", title=None,
                    axis=alt.Axis(grid=True, gridColor="#13233c",
                                  labelColor="#6885ad", domainColor="#22344f", tickColor="#22344f")),
            color=alt.Color("series:N",
                    scale=alt.Scale(domain=["gross (before costs)", "net (after costs)"],
                                    range=["#60a5fa", net_color]),
                    legend=alt.Legend(title=None, labelColor="#8fa3c2",
                                      orient="top-left", direction="vertical",
                                      symbolType="stroke", symbolStrokeWidth=3,
                                      labelFontSize=12, rowPadding=6, offset=8)))
         .properties(height=380, background="transparent")
         .configure_view(strokeWidth=0))
st.altair_chart(chart, use_container_width=True)

if net_ret > 0:
    st.markdown(f'<div class="verdict verdict-pos">Profitable at {cost_bps:.1f}bp — the signal\'s '
                f'edge survives execution at this cost level.</div>', unsafe_allow_html=True)
else:
    st.markdown(f'<div class="verdict verdict-neg">Loss at {cost_bps:.1f}bp — costs exceed the edge. '
                f'Drop below ~0.5bp to flip it. The alpha is real, but thinner than the cost of '
                f'trading on it — which is exactly why HFT firms invest in low-latency execution.'
                f'</div>', unsafe_allow_html=True)