import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.stats import norm
from datetime import datetime

# Page config MUST come first
st.set_page_config(page_title="Black-Scholes Option Pricing Dashboard", layout="wide")

st.title("📈 Black-Scholes Option Pricing Model")

# Black-Scholes option pricing
@st.cache_data
def black_scholes_price(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# Greeks calculation
@st.cache_data
def calculate_greeks(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    delta = norm.cdf(d1) if option_type == 'call' else -norm.cdf(-d1)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100
    theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) -
             r * K * np.exp(-r * T) * norm.cdf(d2 if option_type == 'call' else -d2)) / 365
    rho = (K * T * np.exp(-r * T) * norm.cdf(d2 if option_type == 'call' else -d2)) / 100

    return {
        'Delta': delta,
        'Gamma': gamma,
        'Vega': vega,
        'Theta': theta,
        'Rho': rho
    }

# Sidebar inputs
st.sidebar.header("Input Parameters")
ticker = st.sidebar.text_input("Stock Ticker", value="AAPL")
expiry = st.sidebar.date_input("Expiration Date", value=datetime(2025, 6, 21))
strike_price = st.sidebar.number_input("Strike Price", min_value=1.0, value=180.0)
risk_free_rate = st.sidebar.slider("Risk-Free Rate (annual %)", min_value=0.0, max_value=10.0, value=2.5) / 100
volatility = st.sidebar.slider("Implied Volatility (annual %)", min_value=1.0, max_value=200.0, value=30.0) / 100
option_type = st.sidebar.selectbox("Option Type", ["call", "put"])
show_graph = st.sidebar.checkbox("Show Stock Price Chart", value=True)
show_chain = st.sidebar.checkbox("Show Option Chain", value=False)

# Fetch current stock data
stock = yf.Ticker(ticker)
stock_data = stock.history(period="1mo")
current_price = stock.history(period="1d")['Close'][0]
st.write(f"### Current {ticker} Price: ${current_price:.2f}")

# Time to expiry in years
time_to_expiry = (expiry - datetime.today().date()).days / 365

# Pricing and Greeks
option_price = black_scholes_price(current_price, strike_price, time_to_expiry, risk_free_rate, volatility, option_type)
greeks = calculate_greeks(current_price, strike_price, time_to_expiry, risk_free_rate, volatility, option_type)

st.subheader("Option Price and Greeks")
st.markdown(f"**Black-Scholes {option_type.capitalize()} Price**: ${option_price:.2f}")

col1, col2, col3 = st.columns(3)
col1.metric("Delta", f"{greeks['Delta']:.3f}")
col2.metric("Gamma", f"{greeks['Gamma']:.3f}")
col3.metric("Vega", f"{greeks['Vega']:.3f}")
col1.metric("Theta", f"{greeks['Theta']:.3f}")
col2.metric("Rho", f"{greeks['Rho']:.3f}")

st.markdown("""
- **Delta**: Change in option price per $1 move in the stock.
- **Gamma**: Sensitivity of Delta to stock price changes.
- **Vega**: Change in option price with 1% change in volatility.
- **Theta**: Daily time decay of the option.
- **Rho**: Change in option price with 1% change in interest rate.
""")

# Greeks Plot (Matplotlib)
greek_values = {g: [] for g in ['Delta', 'Gamma', 'Vega', 'Theta', 'Rho']}
strike_range = np.arange(current_price - 20, current_price + 20, 2)
for K in strike_range:
    g = calculate_greeks(current_price, K, time_to_expiry, risk_free_rate, volatility, option_type)
    for key in greek_values:
        greek_values[key].append(g[key])

fig, ax = plt.subplots(figsize=(10, 6))
colors = {
    'Delta': 'orange',
    'Gamma': 'cyan',
    'Vega': 'magenta',
    'Theta': 'green',
    'Rho': 'red'
}
for greek in greek_values:
    ax.plot(strike_range, greek_values[greek], label=greek, color=colors[greek], linewidth=2)

ax.set_title('Greeks vs Strike Price')
ax.set_xlabel('Strike Price')
ax.set_ylabel('Greek Value')
ax.grid(True)
ax.legend()
st.pyplot(fig)

# Stock price graph
if show_graph and stock_data is not None:
    try:
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.plot(stock_data.index, stock_data['Close'], color='skyblue', linewidth=2)
        ax2.set_title(f"{ticker} Stock Price (Last 30 Days)")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Price (USD)")
        ax2.grid(True)
        fig2.autofmt_xdate()
        st.pyplot(fig2)

        st.markdown("""
        **Stock Price Over Time**: 
        This shows recent stock price movement. Black-Scholes uses current price and assumes log-normal behavior, so recent trends help contextualize the model.
        """)

    except Exception as e:
        st.error(f"❌ Error: {e}")

# Option chain (optional)
if show_chain:
    try:
        expiry_str = expiry.strftime("%Y-%m-%d")
        chain = stock.option_chain(expiry_str)
        options_df = chain.calls if option_type == 'call' else chain.puts
        st.subheader(f"Option Chain for {ticker} ({expiry_str})")
        st.dataframe(options_df[['strike', 'lastPrice', 'bid', 'ask', 'impliedVolatility']])
        st.markdown("""
        **Option Chain Explanation**:
        - **Strike**: Exercise price.
        - **Last Price**: Most recent option price.
        - **Bid/Ask**: Market buy/sell prices.
        - **Implied Volatility**: Market's forecast of future volatility.
        """)
    except Exception as e:
        st.error(f"❌ Error: {e}")
