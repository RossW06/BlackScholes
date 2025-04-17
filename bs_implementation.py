import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from scipy.stats import norm
from datetime import datetime, time
import plotly.express as px

# --- Black-Scholes Functions ---
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

# --- Streamlit GUI ---
st.set_page_config(page_title="Black-Scholes Option Pricing", layout="wide")
st.title("📈 Black-Scholes Option Pricing Dashboard")

col1, col2 = st.columns([1, 2])
with col1:
    ticker = st.text_input("Enter Stock Ticker", value="AAPL")
    option_type = st.radio("Option Type", ["call", "put"])
    expiry_date = st.date_input("Expiration Date")
    strike = st.number_input("Strike Price", min_value=1.0, value=180.0, step=1.0)
    vol_mode = st.selectbox("Volatility Source", ["Implied Volatility", "Historical Volatility"])

    calc_button = st.button("📊 Calculate")

with col2:
    show_chain = st.checkbox("🔍 Show Full Option Chain")

# --- Backend Logic ---
if calc_button:
    try:
        stock = yf.Ticker(ticker)
        S = stock.history(period='1d')['Close'].iloc[-1]
        expiry_str = expiry_date.strftime("%Y-%m-%d")
        expiry_dt = datetime.combine(expiry_date, time(16, 0))
        now = datetime.now()
        T = (expiry_dt - now).total_seconds() / (365 * 24 * 60 * 60)
        r = 0.05  # Risk-free rate

        chain = stock.option_chain(expiry_str)
        df = chain.calls if option_type == 'call' else chain.puts

        if vol_mode == "Implied Volatility":
            row = df[df['strike'] == strike]
            if not row.empty and not np.isnan(row['impliedVolatility'].values[0]):
                sigma = row['impliedVolatility'].values[0]
            else:
                st.warning("Implied volatility not found. Falling back to historical volatility.")
                vol_mode = "Historical Volatility"

        if vol_mode == "Historical Volatility":
            hist = stock.history(period='1y')
            log_returns = np.log(hist['Close'] / hist['Close'].shift(1)).dropna()
            sigma = np.std(log_returns) * np.sqrt(252)

        price = black_scholes_price(S, strike, T, r, sigma, option_type)
        greeks = calculate_greeks(S, strike, T, r, sigma, option_type)

        st.success("✅ Calculation Complete")
        st.subheader("Option Price")
        st.metric("Black-Scholes Price", f"${price:.2f}")
        st.text(f"Stock Price: ${S:.2f} | Volatility ({vol_mode}): {sigma:.2%} | Time to Expiry: {T:.4f} yrs")

        st.subheader("Greeks")
        for g, val in greeks.items():
            st.write(f"**{g}**: {val:.4f}")

    except Exception as e:
        st.error(f"Error: {e}")

# --- Option Chain Viewer ---
if show_chain:
    try:
        chain = stock.option_chain(expiry_str)
        options_df = chain.calls if option_type == 'call' else chain.puts
        st.subheader(f"{option_type.capitalize()} Option Chain for {expiry_str}")
        st.dataframe(options_df[['strike', 'lastPrice', 'bid', 'ask', 'impliedVolatility']])
        
        # Plotting IV smile
        fig = px.line(options_df, x='strike', y='impliedVolatility',
                      title="Implied Volatility Smile", markers=True)
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Could not fetch option chain: {e}")
