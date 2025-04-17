import streamlit as st
import yfinance as yf
import numpy as np
from scipy.stats import norm
from datetime import datetime, time
import plotly.graph_objects as go
import plotly.express as px

# Ensure this is the very first Streamlit function
st.set_page_config(page_title="Black-Scholes Option Pricing Dashboard", layout="wide")

# Black-Scholes Formula for Option Pricing
def black_scholes_price(S, K, T, r, sigma, option_type='call'):
    """
    The Black-Scholes formula is used to calculate the theoretical price of an option.
    S: Stock price
    K: Strike price
    T: Time to expiration in years
    r: Risk-free interest rate
    sigma: Volatility of the stock
    option_type: 'call' or 'put' for call or put options
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# Greeks calculation
def calculate_greeks(S, K, T, r, sigma, option_type='call'):
    """
    Calculate the option Greeks (Delta, Gamma, Vega, Theta, Rho) for pricing sensitivity.
    These values help understand how sensitive the option's price is to different factors.
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    delta = norm.cdf(d1) if option_type == 'call' else -norm.cdf(-d1)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))  # Sensitivity to changes in the stock price
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100  # Sensitivity to changes in volatility
    theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) -
             r * K * np.exp(-r * T) * norm.cdf(d2 if option_type == 'call' else -d2)) / 365  # Sensitivity to time decay
    rho = (K * T * np.exp(-r * T) * norm.cdf(d2 if option_type == 'call' else -d2)) / 100  # Sensitivity to interest rate changes
    
    return {'Delta': delta, 'Gamma': gamma, 'Vega': vega, 'Theta': theta, 'Rho': rho}

# Streamlit app layout
st.title("📊 **Black-Scholes Option Pricing**")

# Sidebar input
col1, col2 = st.columns([1, 2])
with col1:
    ticker = st.text_input("Stock Ticker", value="AAPL", help="Enter the stock ticker symbol (e.g., AAPL for Apple).")
    option_type = st.radio("Option Type", ["call", "put"], help="Select the option type (Call or Put).")
    expiry_date = st.date_input("Expiration Date", help="Pick the expiration date for the option.")
    strike = st.number_input("Strike Price", min_value=1.0, value=180.0, help="Enter the strike price of the option.")
    volatility_type = st.selectbox("Volatility Source", ["Implied Volatility", "Historical Volatility"],
                                  help="Choose between implied or historical volatility to calculate option price.")

with col2:
    show_chain = st.checkbox("Show Option Chain", help="Check to view the full option chain.")

# Button to trigger calculation
calc_button = st.button("📈 **Calculate Option**", help="Click to calculate the option price and Greeks")

# If button is clicked, perform calculations
if calc_button:
    try:
        # Get the stock price
        stock = yf.Ticker(ticker)
        stock_price = stock.history(period='1d')['Close'].iloc[-1]
        
        # Calculate time to expiration
        expiry_str = expiry_date.strftime("%Y-%m-%d")
        expiry_datetime = datetime.combine(expiry_date, time(16, 0))
        time_to_expiry = (expiry_datetime - datetime.now()).total_seconds() / (365 * 24 * 60 * 60)
        
        # Risk-free rate assumption (this is a general constant)
        risk_free_rate = 0.05
        
        # Option chain data for selected expiration
        chain = stock.option_chain(expiry_str)
        options_data = chain.calls if option_type == 'call' else chain.puts
        
        # Calculate volatility based on selection
        if volatility_type == "Implied Volatility":
            selected_option = options_data[options_data['strike'] == strike]
            if not selected_option.empty:
                volatility = selected_option['impliedVolatility'].values[0]
            else:
                volatility_type = "Historical Volatility"
        
        if volatility_type == "Historical Volatility":
            historical_data = stock.history(period="1y")
            log_returns = np.log(historical_data['Close'] / historical_data['Close'].shift(1)).dropna()
            volatility = np.std(log_returns) * np.sqrt(252)  # Volatility based on annualized standard deviation
        
        # Option price using Black-Scholes formula
        option_price = black_scholes_price(stock_price, strike, time_to_expiry, risk_free_rate, volatility, option_type)
        
        # Calculate the Greeks
        greeks = calculate_greeks(stock_price, strike, time_to_expiry, risk_free_rate, volatility, option_type)
        
        # Display results
        st.success("✅ Calculation Complete")
        st.subheader("Option Price")
        st.metric("Black-Scholes Price", f"${option_price:.2f}")
        st.markdown("This is the theoretical price of the option based on the Black-Scholes model. It accounts for factors such as the stock price, strike price, time to expiration, risk-free rate, and volatility.")
        
        st.subheader("Greeks")
        for greek, value in greeks.items():
            st.write(f"**{greek}:** {value:.4f}")
        st.markdown("""
            - **Delta**: Sensitivity of the option price to changes in the stock price.
            - **Gamma**: Sensitivity of Delta to changes in the stock price.
            - **Vega**: Sensitivity of the option price to changes in volatility.
            - **Theta**: Sensitivity of the option price to the passage of time (time decay).
            - **Rho**: Sensitivity of the option price to changes in the risk-free interest rate.
        """)

        # Plot Greeks vs Strike Price
        strike_prices = np.arange(stock_price - 20, stock_price + 20, 5)
        greek_values = {greek: [] for greek in greeks}
        for K in strike_prices:
            price = black_scholes_price(stock_price, K, time_to_expiry, risk_free_rate, volatility, option_type)
            greek_results = calculate_greeks(stock_price, K, time_to_expiry, risk_free_rate, volatility, option_type)
            for greek, value in greek_results.items():
                greek_values[greek].append(value)
        
        greek_fig = go.Figure()
        for greek, values in greek_values.items():
            greek_fig.add_trace(go.Scatter(x=strike_prices, y=values, mode='lines', name=greek, line=dict(width=3)))
        
        greek_fig.update_layout(
            title="Greeks vs Strike Price", 
            xaxis_title="Strike Price", 
            yaxis_title="Greek Value", 
            template="plotly_dark",
            xaxis=dict(showgrid=True),
            yaxis=dict(showgrid=True)
        )
        st.plotly_chart(greek_fig, use_container_width=True)
        st.markdown("""
            The graph above shows how the Greeks change with different strike prices. 
            This is useful to understand how the option price reacts to changes in strike price.
        """)

        # Plot Stock Price over Time
        stock_data = stock.history(period="1mo")
        stock_fig = px.line(stock_data, x=stock_data.index, y="Close", title=f"{ticker} Stock Price (Last 30 Days)")
        stock_fig.update_layout(
            xaxis_title="Date", 
            yaxis_title="Price (USD)", 
            template="plotly_dark",
            xaxis=dict(showgrid=True),
            yaxis=dict(showgrid=True)
        )
        st.plotly_chart(stock_fig, use_container_width=True)
        st.markdown("""
            The graph above shows the stock price movement over the last 30 days. 
            This is important because the option price is closely tied to the price movement of the underlying stock.
        """)

    except Exception as e:
        st.error(f"❌ Error: {e}")

# Option Chain Viewer
if show_chain:
    try:
        chain = stock.option_chain(expiry_str)
        options_df = chain.calls if option_type == 'call' else chain.puts
        st.subheader(f"Option Chain for {ticker} ({expiry_str})")
        st.dataframe(options_df[['strike', 'lastPrice', 'bid', 'ask', 'impliedVolatility']])
        st.markdown("""
            The option chain provides details on various options with different strike prices and expiration dates. 
            It shows the current prices, bid-ask spreads, and implied volatility.
        """)
    except Exception as e:
