import streamlit as st
import numpy as np
import pandas as pd
import random
from datetime import datetime
import pickle
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

# Access the variables
groq_key = os.getenv('GROQ_API_KEY')
# Groq client
client = Groq(api_key=groq_key)

# Load the model
with open('xgb_model.pkl', 'rb') as file:
    xgb_model = pickle.load(file)

columns = [
    'Year', 'Month', 'Day',
    'XAU BGNL', 'BDIY', 'DXY', 'JPY', 'GBP', 'Cl1', 'VIX',
    'USGG3M', 'EONIA', 'GTITL30YR', 'GTITL10YR', 'GTITL2YR',
    'GTJPY30YR', 'GTJPY10YR', 'GTJPY2YR', 'LMBITR', 'LUACTRUU',
    'LF98TRUU', 'LG30TRUU', 'LP01TREU', 'EMUSTRUU', 'MXUS', 'MXEU'
]

# List of assets to choose from
assets = [
    # Stocks (Large-Cap and Blue-Chip)
    "Apple Inc. (AAPL)",
    "Microsoft Corp. (MSFT)",
    "Alphabet Inc. (GOOGL)",
    "Amazon.com Inc. (AMZN)",
    "Tesla Inc. (TSLA)",
    "Berkshire Hathaway Inc. (BRK.B)",
    "Meta Platforms Inc. (META)",
    "Johnson & Johnson (JNJ)",
    "JPMorgan Chase & Co. (JPM)",
    "Procter & Gamble Co. (PG)",

    # ETFs (Equity and Sector-Focused)
    "SPDR S&P 500 ETF (SPY)",
    "Invesco QQQ Trust (QQQ)",
    "Vanguard FTSE All-World ex-US ETF (VEU)",
    "iShares MSCI Emerging Markets ETF (EEM)",
    "Vanguard Dividend Appreciation ETF (VIG)",
    "iShares U.S. Technology ETF (IYW)",
    "iShares Russell 2000 ETF (IWM)",
    "ARK Innovation ETF (ARKK)",

    # Bonds (Government and Corporate Bonds ETFs)
    "iShares 20+ Year Treasury Bond ETF (TLT)",
    "iShares iBoxx $ Investment Grade Corporate Bond ETF (LQD)",
    "Vanguard Total Bond Market ETF (BND)",
    "iShares U.S. Treasury Bond ETF (GOVT)",
    "SPDR Bloomberg Barclays High Yield Bond ETF (JNK)",

    # Commodities (ETFs for Precious Metals, Oil, etc.)
    "SPDR Gold Shares (GLD)",
    "United States Oil Fund (USO)",
    "Invesco DB Agriculture Fund (DBA)",
    "iShares Silver Trust (SLV)",

    # REITs (Real Estate Investment Trusts)
    "Vanguard Real Estate ETF (VNQ)",
    "iShares U.S. Real Estate ETF (IYR)",

    # Cash Equivalents and Short-Term Instruments
    "SPDR Bloomberg Barclays 1-3 Month T-Bill ETF (BIL)",
    "iShares Short Treasury Bond ETF (SHV)",

    # International Markets (ETFs)
    "iShares MSCI Japan ETF (EWJ)",
    "iShares MSCI Brazil ETF (EWZ)",
    "iShares MSCI China ETF (MCHI)",
    "iShares MSCI Europe ETF (IEUR)",

    # Currency ETFs
    "Invesco DB U.S. Dollar Index Bullish Fund (UUP)",
    "WisdomTree Chinese Yuan Strategy Fund (CYB)",

    # Volatility ETF
    "ProShares VIX Short-Term Futures ETF (VIXY)"
]


def generate_portfolio(portfolio_value, num_assets, assets):
    # Randomly select assets
    num_chosen_assets = num_assets
    chosen_assets = random.sample(assets, num_chosen_assets)

    # Generate random weights for the chosen assets
    weights = np.random.random(len(chosen_assets))
    weights /= weights.sum()  # Normalize weights to sum to 1

    # Random allocation for cash (5% to 20%)
    cash_weight = random.uniform(0.05, 0.20)
    remaining_weight = 1 - cash_weight  # Remaining weight for assets
    weights *= remaining_weight  # Scale asset weights accordingly

    # Calculate allocation in dollar amounts
    cash_allocation = cash_weight * portfolio_value
    allocations = [weight * portfolio_value for weight in weights]

    # Create a portfolio dataframe
    portfolio = pd.DataFrame({
        'Asset': chosen_assets + ['Cash'],
        'Weight': list(weights) + [cash_weight],
        'Allocation ($)': allocations + [cash_allocation]
    })

    return portfolio

def generate_market_data(columns):
    # Initialize an empty DataFrame for today's data
    today_df = pd.DataFrame()

    # Add Year, Month, and Day columns with today's date
    today = datetime.today()
    today_df['Year'] = [today.year]
    today_df['Month'] = [today.month]
    today_df['Day'] = [today.day]

    # Add each column individually with random values
    for col in columns:
        if col == 'XAU BGNL':
            today_df[col] = np.random.randint(4900, 5150, 1)
        elif col == 'BDIY':
            today_df[col] = np.random.randint(800, 1300, 1)
        elif col == 'DXY':
            today_df[col] = np.random.uniform(95, 115, 1)
        elif col == 'JPY':
            today_df[col] = np.random.uniform(140, 160, 1)
        elif col == 'GBP':
            today_df[col] = np.random.uniform(1.2, 1.4, 1)
        elif col == 'Cl1':
            today_df[col] = np.random.uniform(60, 85, 1)
        elif col == 'VIX':
            today_df[col] = np.random.uniform(15, 22, 1)
        elif col == 'USGG3M':
            today_df[col] = np.random.uniform(4, 5, 1)
        elif col == 'EONIA':
            today_df[col] = np.random.uniform(-0.5, 0.5, 1)
        elif col == 'GTITL30YR':
            today_df[col] = np.random.uniform(1.5, 3.5, 1)
        elif col == 'GTITL10YR':
            today_df[col] = np.random.uniform(0.4, 3, 1)
        elif col == 'GTITL2YR':
            today_df[col] = np.random.uniform(0.5, 2.5, 1)
        elif col == 'GTJPY30YR':
            today_df[col] = np.random.uniform(0, 1.5, 1)
        elif col == 'GTJPY10YR':
            today_df[col] = np.random.uniform(0, 1, 1)
        elif col == 'GTJPY2YR':
            today_df[col] = np.random.uniform(-0.1, 0.5, 1)
        elif col == 'LMBITR':
            today_df[col] = np.random.uniform(3000, 3500, 1)
        elif col == 'LUACTRUU':
            today_df[col] = np.random.uniform(3000, 5000, 1)
        elif col == 'LF98TRUU':
            today_df[col] = np.random.uniform(2500, 3000, 1)
        elif col == 'LG30TRUU':
            today_df[col] = np.random.uniform(1500, 1800, 1)
        elif col == 'LP01TREU':
            today_df[col] = np.random.uniform(400, 600, 1)
        elif col == 'EMUSTRUU':
            today_df[col] = np.random.uniform(1100, 1300, 1)
        elif col == 'MXUS':
            today_df[col] = np.random.uniform(4000, 5000, 1)
        elif col == 'MXEU':
            today_df[col] = np.random.uniform(300, 400, 1)

    return today_df

def adjust_portfolio(portfolio, P_crash, min_allocation=0.05, max_shift=0.2):
    """
    Adjusts portfolio allocations based on the model's prediction score (P_crash).

    Args:
        portfolio (pd.DataFrame): Current portfolio with 'Asset', 'Weight', and 'Allocation ($)' columns.
        P_crash (float): Model's predicted probability of a market crash (0 to 1).
        min_allocation (float): Minimum allocation (as a fraction) for each asset.
        max_shift (float): Maximum percentage shift (as a fraction) for any asset.

    Returns:
        pd.DataFrame: Updated portfolio with adjusted weights and allocations.
    """
    # Define risk levels for assets
    # High Risk Assets
    high_risk_assets = [
        # Stocks (Large-Cap and Blue-Chip)
        "Apple Inc. (AAPL)",
        "Microsoft Corp. (MSFT)",
        "Alphabet Inc. (GOOGL)",
        "Amazon.com Inc. (AMZN)",
        "Tesla Inc. (TSLA)",
        "Berkshire Hathaway Inc. (BRK.B)",
        "Meta Platforms Inc. (META)",
        "Johnson & Johnson (JNJ)",
        "JPMorgan Chase & Co. (JPM)",
        "Procter & Gamble Co. (PG)",

        # Equity and Sector-Focused ETFs
        "SPDR S&P 500 ETF (SPY)",
        "Invesco QQQ Trust (QQQ)",
        "Vanguard FTSE All-World ex-US ETF (VEU)",
        "iShares MSCI Emerging Markets ETF (EEM)",
        "Vanguard Dividend Appreciation ETF (VIG)",
        "iShares U.S. Technology ETF (IYW)",
        "iShares Russell 2000 ETF (IWM)",
        "ARK Innovation ETF (ARKK)",

        # REITs (Real Estate Investment Trusts)
        "Vanguard Real Estate ETF (VNQ)",
        "iShares U.S. Real Estate ETF (IYR)",

        # Volatility ETF
        "ProShares VIX Short-Term Futures ETF (VIXY)"
    ]

    # Moderate Risk Assets
    moderate_risk_assets = [
        # Corporate and High-Yield Bonds ETFs
        "iShares iBoxx $ Investment Grade Corporate Bond ETF (LQD)",
        "SPDR Bloomberg Barclays High Yield Bond ETF (JNK)",

        # Commodities ETFs
        "SPDR Gold Shares (GLD)",
        "United States Oil Fund (USO)",
        "Invesco DB Agriculture Fund (DBA)",
        "iShares Silver Trust (SLV)",

        # International Markets ETFs
        "iShares MSCI Japan ETF (EWJ)",
        "iShares MSCI Brazil ETF (EWZ)",
        "iShares MSCI China ETF (MCHI)",
        "iShares MSCI Europe ETF (IEUR)"
    ]

    # Low Risk Assets
    low_risk_assets = [
        # Government Bonds and Treasuries ETFs
        "iShares 20+ Year Treasury Bond ETF (TLT)",
        "Vanguard Total Bond Market ETF (BND)",
        "iShares U.S. Treasury Bond ETF (GOVT)",
        "SPDR Bloomberg Barclays 1-3 Month T-Bill ETF (BIL)",
        "iShares Short Treasury Bond ETF (SHV)"
    ]

    # Get the total portfolio value
    total_value = portfolio['Allocation ($)'].sum()

    # Adjust weights dynamically based on P_crash
    def adjust_weight(asset, weight):
        if asset in high_risk_assets:
            return max(weight * (1 - P_crash), min_allocation)
        elif asset in moderate_risk_assets:
            return weight * (1 - 0.5 * P_crash)
        elif asset in low_risk_assets:
            return min(weight + P_crash * max_shift, 1 - min_allocation * len(low_risk_assets))
        return weight

    # Apply adjustments
    portfolio['Adjusted Weight'] = portfolio.apply(
        lambda row: adjust_weight(row['Asset'], row['Weight']), axis=1
    )

    # Normalize weights to sum to 1
    portfolio['Adjusted Weight'] /= portfolio['Adjusted Weight'].sum()

    # Update allocations
    portfolio['Adjusted Allocation ($)'] = portfolio['Adjusted Weight'] * total_value

    return portfolio

def predict_probability(today_df):
    prediction = xgb_model.predict_proba(today_df)    
    
    # Get the probability of a crash (class 1)
    P_crash = prediction[0][0]

    return P_crash


def get_ai_explanation(updated_portfolio, today_df, P_crash, client):
    # Convert portfolio and DataFrame to strings
    portfolio_str = updated_portfolio.to_string()
    today_df_str = today_df.to_string()
    P_crash_str = str(P_crash)

    # Create the chat completion request
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a financial assistant with expertise in portfolio management, asset allocation, "
                    "and investment strategy. Your task is to explain portfolio adjustments in simple terms, "
                    "focusing on the reasoning behind those adjustments. While you do not need to dive into complex financial "
                    "jargon, ensure that the explanation is clear and provides enough detail for someone with little financial "
                    "background to understand. The explanation should consider the impact of predicted market conditions, "
                    "specifically the likelihood of a market crash (P_crash), and how the portfolio's asset allocation was "
                    "adjusted based on the indices of the assets, which are provided in today's data."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Explain the portfolio adjustments and the investment strategy behind the changes made. "
                    f"Here is the updated portfolio and the relevant indices for today: {today_df_str}. "
                    f"The model predicts a market crash with a probability of {P_crash_str}. "
                    "Please explain the reasoning behind the adjustments made in the portfolio based on this information."
                ),
            }
        ],
        model="llama-3.3-70b-versatile",
    )

    # Extract the AI's explanation
    ai_explanation = chat_completion.choices[0].message.content
    
    return ai_explanation



##############################################################################################################################################

# Initialize session state variables if they don't exist
session_keys = [
    'show_market_data', 
    'show_market_crash_probability', 
    'show_portfolio_before', 
    'show_portfolio_after', 
    'show_ai_explanation'
]

for key in session_keys:
    if key not in st.session_state:
        st.session_state[key] = False

# Main page content
st.title("Market Anomaly Detection")
st.header("Select Data to Display")

# Portfolio value and number of assets inputs
portfolio_value = st.number_input("Enter Portfolio Value ($)", min_value=0, value=100000, step=1000)
num_assets = st.number_input("Enter Number of Assets (5-20)", min_value=5, max_value=20, value=10, step=1)


# First button ("Today's Market Data") is always visible
if not st.session_state.show_market_data:
    show_market_data = st.button("Today's Market Data", help="Show the current market data.")
    if show_market_data:
        st.session_state.show_market_data = True
else:
    show_market_data = True
# The next button is revealed once the first button is pressed
if st.session_state.show_market_data and not st.session_state.show_market_crash_probability:
    show_market_crash_probability = st.button("Market Crash Probability", help="Show the probability of a market crash.")
    if show_market_crash_probability:
        st.session_state.show_market_crash_probability = True
else:
    show_market_crash_probability = False

# The next button is revealed once the second button is pressed
if st.session_state.show_market_crash_probability and not st.session_state.show_portfolio_before:
    show_portfolio_before = st.button("Portfolio Before Adjustment", help="Show the portfolio before any adjustments.")
    if show_portfolio_before:
        st.session_state.show_portfolio_before = True
else:
    show_portfolio_before = False

# The next button is revealed once the third button is pressed
if st.session_state.show_portfolio_before and not st.session_state.show_portfolio_after:
    show_portfolio_after = st.button("Portfolio After Adjustment", help="Show the portfolio after adjustments.")
    if show_portfolio_after:
        st.session_state.show_portfolio_after = True
else:
    show_portfolio_after = False

# The next button is revealed once the fourth button is pressed
if st.session_state.show_portfolio_after and not st.session_state.show_ai_explanation:
    show_ai_explanation = st.button("AI Explanation", help="Show the AI's explanation of adjustments.")
    if show_ai_explanation:
        st.session_state.show_ai_explanation = True
else:
    show_ai_explanation = False

# Display content in expandable sections to keep the UI clean
if st.session_state.show_market_data:
    today = datetime.today()
    today_df = generate_market_data(columns)
    with st.expander(f"Today's Market Data ({today.strftime('%Y-%m-%d')})"):
        st.write(today_df)

if st.session_state.show_market_crash_probability:
    with st.expander("Market Crash Probability"):
        P_crash = predict_probability(today_df)
        st.write(f"### Market Crash Probability: {P_crash * 100:.2f}%")

if st.session_state.show_portfolio_before:
    with st.expander("Portfolio Before Adjustment"):
        portfolio = generate_portfolio(portfolio_value, num_assets, assets)
        st.write(portfolio)

if st.session_state.show_portfolio_after:
    with st.expander("Portfolio After Adjustment"):
        updated_portfolio = adjust_portfolio(portfolio, P_crash)
        st.write(updated_portfolio)

if st.session_state.show_ai_explanation:
    with st.expander("AI Explanation of Adjustments"):
        ai_explanation = get_ai_explanation(updated_portfolio, today_df, P_crash, client)
        st.write(ai_explanation)