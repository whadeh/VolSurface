import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm
from scipy.interpolate import griddata
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import io

st.set_page_config(page_title="Volatility Surface Explorer", layout="wide")

# --- Black-Scholes Model ---
def bs_price(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if option_type == 'call':
        return S * norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    else:
        return K*np.exp(-r*T)*norm.cdf(-d2) - S * norm.cdf(-d1)

def implied_vol(price, S, K, T, r, option_type='call'):
    if price < 0.01 or T == 0:
        return np.nan
    try:
        return brentq(lambda sigma: bs_price(S, K, T, r, sigma, option_type) - price, 0.001, 5.0)
    except:
        return np.nan

# --- Get Options Data ---
@st.cache_data(show_spinner=False)
def get_iv_surface(ticker, cutoff_date, option_type='both', num_expirations=20):
    tk = yf.Ticker(ticker)
    spot = tk.history(period='1d')['Close'][0]
    r = 0.015
    surface_data = []

    expirations = tk.options
    if cutoff_date:
        expirations = [d for d in expirations if d >= cutoff_date]
    if not expirations:
        return pd.DataFrame(), spot

    for exp in expirations[:num_expirations]:  # Limit for performance
        opt_date = pd.to_datetime(exp)
        T = (opt_date - pd.Timestamp.today()).days / 365.0
        if T <= 0: continue

        oc = tk.option_chain(exp)

        if option_type in ['call', 'both']:
            for _, row in oc.calls.iterrows():
                K = row['strike']
                P = row['lastPrice']
                iv = implied_vol(P, spot, K, T, r, 'call')
                if not np.isnan(iv):
                    surface_data.append(['call', exp, K, T, iv])

        if option_type in ['put', 'both']:
            for _, row in oc.puts.iterrows():
                K = row['strike']
                P = row['lastPrice']
                iv = implied_vol(P, spot, K, T, r, 'put')
                if not np.isnan(iv):
                    surface_data.append(['put', exp, K, T, iv])

    df = pd.DataFrame(surface_data, columns=['Type', 'Expiration', 'Strike', 'Time', 'IV'])
    return df, spot

# --- Streamlit UI ---
st.title("📊 Volatility Surface Explorer")

col1, col2 = st.columns(2)
with col1:
    ticker = st.text_input("Stock Ticker", value="AAPL").upper()

with col2:
    cutoff = st.date_input("Expiration Cutoff (optional)", value=None)

option_choice = st.radio("Option Type", ['call', 'put', 'both'], horizontal=True)

num_expirations = st.slider("Number of Expirations to Include", min_value=1, max_value=60, value=20)

if st.button("Generate Volatility Surface"):
    with st.spinner("Fetching and computing implied vols..."):
        df, spot = get_iv_surface(ticker, str(cutoff) if cutoff else None, option_choice)

    if df.empty:
        st.warning("No valid option data found.")
    else:
        st.success(f"Fetched {len(df)} options. Spot price: ${spot:.2f}")

        # Export CSV
        csv = df.to_csv(index=False).encode()
        st.download_button("Download IV Data CSV", csv, f"{ticker}_iv_surface.csv", "text/csv")

        # --- 3D Surface (Interpolated) ---
        call_df = df[df['Type'] == 'call']
        if not call_df.empty:
            st.subheader("📈 Call Option Implied Volatility Surface")
            x = call_df['Strike']
            y = call_df['Time']
            z = call_df['IV']
            xi = np.linspace(min(x), max(x), 40)
            yi = np.linspace(min(y), max(y), 40)
            xi, yi = np.meshgrid(xi, yi)
            zi = griddata((x, y), z, (xi, yi), method='cubic')

            fig = go.Figure(data=[go.Surface(z=zi, x=xi, y=yi, colorscale='Viridis')])
            fig.update_layout(
                scene=dict(
                    xaxis_title="Strike",
                    yaxis_title="Time to Expiry (Years)",
                    zaxis_title="IV"
                ),
                margin=dict(l=0, r=0, t=40, b=0)
            )
            st.plotly_chart(fig, use_container_width=True)

        # --- IV Smile ---
        st.subheader("📉 IV Smiles by Expiration")
        fig_smile, ax = plt.subplots(figsize=(10, 5))
        for exp in df['Expiration'].unique():
            for opt_type in ['call', 'put']:
                subset = df[(df['Expiration'] == exp) & (df['Type'] == opt_type)]
                if len(subset) > 2:
                    ax.plot(subset['Strike'], subset['IV'], label=f"{exp} ({opt_type})")
        ax.set_title(f"{ticker} IV Smiles")
        ax.set_xlabel("Strike")
        ax.set_ylabel("Implied Volatility")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig_smile)
