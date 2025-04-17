import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from scipy.stats import norm
from datetime import datetime, time
import plotly.express as px

# ---- Black-Scholes Functions ----
def black_scholes_price(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def calculate_greeks(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    delta = norm.cdf(d1) if option_type == 'call' else -norm.cdf(-d1)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100
    theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) -
             r * K * np.exp(-r * T) * norm.cdf(d2 if option_type == 'call' else -d2)) / 365
    rho = (K * T * np.exp(-r * T) * norm.cdf(d2 if option_type == 'call' else -d2)) / 100
    return {'Delta': delta, 'Gamma': gamma, 'Vega': vega, 'Theta': theta, 'Rho': rho}

# ---- App Layout ----
st.set_page_config("📈 Black-Scholes Option Pricing", layout="wide")
st.title("🧮 Black-Scholes Option Pricing Model")

# ---- Sidebar Inputs ----
st.sidebar.header("🔧 Input Parameters")
ticker = st.sidebar.text_input("Stock Ticker", "AAPL").upper()
option_type = st.sidebar.radio("Option Type", ["call", "put"])
vol_mode = st.sidebar.radio("Volatility", ["Implied", "Historical"])
strike = st.sidebar.number_input("Strike Price", value=180.0, step=1.0)
r = st.sidebar.slider("Risk-Free Rate (%)", min_value=0.0, max_value=10.0, value=5.0, step=0.25) / 100
show_chain = st.sidebar.checkbox("Show Full Option Chain")

# ---- Fetch Data ----
try:
    stock = yf.Ticker(ticker)
    S = stock.history(period='1d')['Close'].iloc[-1]
    expiries = stock.options
    default_expiry = expiries[0] if expiries else None
    expiry = st.sidebar.selectbox("Expiration Date", options=expiries, index=0)

    expiry_dt = datetime.combine(datetime.strptime(expiry, "%Y-%m-%d"), time(16))
    now = datetime.now()
    T = max((expiry_dt - now).total_seconds() / (365 * 24 * 60 * 60), 0.0001)

    # Options Chain
    chain = stock.option_chain(expiry)
    df = chain.calls if option_type == 'call' else chain.puts

    # Volatility
    row = df[df['strike'] == strike]
    if vol_mode == "Implied" and not row.empty and not np.isnan(row['impliedVolatility'].values[0]):
        sigma = row['impliedVolatility'].values[0]
    else:
        hist = stock.history(period='1y')
        log_returns = np.log(hist['Close'] / hist['Close'].shift(1)).dropna()
        sigma = np.std(log_returns) * np.sqrt(252)
        vol_mode += " (Fallback)"

    # Pricing + Greeks
    price = black_scholes_price(S, strike, T, r, sigma, option_type)
    greeks = calculate_greeks(S, strike, T, r, sigma, option_type)

    # ---- Display Results ----
    col1, col2 = st.columns(2)
    with col1:
        st.metric("📌 Stock Price", f"${S:.2f}")
        st.metric("🎯 Strike", f"${strike:.2f}")
        st.metric("📅 Time to Expiry", f"{T:.4f} years")
    with col2:
        st.metric("💰 Option Price", f"${price:.2f}")
        st.metric("📉 Volatility", f"{sigma:.2%} ({vol_mode})")
        st.metric("🏦 Risk-Free Rate", f"{r:.2%}")

    st.subheader("🧮 Greeks")
    st.table(pd.DataFrame(greeks, index=["Value"]).T.style.background_gradient(cmap="Blues"))

    # ---- Option Chain Display ----
    if show_chain:
        st.subheader(f"🔍 Option Chain: {option_type.title()}s @ {expiry}")
        filtered_df = df[["strike", "lastPrice", "bid", "ask", "impliedVolatility"]].copy()
        filtered_df["Moneyness"] = np.where(filtered_df["strike"] < S, "ITM", "OTM")
        st.dataframe(filtered_df)

        # Volatility Smile Chart
        fig = px.line(filtered_df, x="strike", y="impliedVolatility", color="Moneyness",
                      title="Volatility Smile", markers=True)
        st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"⚠️ Error: {e}")
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Could not fetch option chain: {e}")
