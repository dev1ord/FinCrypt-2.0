import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.svm import SVR
from datetime import datetime, timedelta
import requests

# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(
    page_title="FinCrypt 2.0: Bitcoin Price Predictor & Crypto News",
    page_icon="🪙",
    layout="wide",
)

# =====================================
# SIDEBAR
# =====================================
st.sidebar.title("Control Panel")

# ----- Prediction settings -----
st.sidebar.subheader("Prediction Settings")
prediction_days = st.sidebar.slider(
    "Number of days to predict",
    min_value=1,
    max_value=30,
    value=7,
    step=1,
)
uploaded_file = st.sidebar.file_uploader(
    "Upload custom Bitcoin price CSV (Date, Price)",
    type=["csv"],
    help="If not provided, the bundled `bitcoin.csv` file will be used.",
)

# ----- Crypto news settings -----
st.sidebar.markdown("---")
st.sidebar.subheader("Crypto News Settings")

news_api_key = st.sidebar.text_input(
    "NewsData.io API key",
    type="password",
    help="Create a free account on NewsData.io and paste your API key here.",
)

# Coin dropdown with “logos” (emojis/icons)
coin_options = {
    "🟧 Bitcoin (BTC)": "btc",
    "🟪 Ethereum (ETH)": "eth",
    "🟦 Solana (SOL)": "sol",
    "💠 Ripple (XRP)": "xrp",
    "🟨 Binance Coin (BNB)": "bnb",
    "🐕 Dogecoin (DOGE)": "doge",
    "🟥 Cardano (ADA)": "ada",
    "⚪ Litecoin (LTC)": "ltc",
    "🎯 Polkadot (DOT)": "dot",
    "🧊 Avalanche (AVAX)": "avax",
}

selected_coin_label = st.sidebar.selectbox(
    "Select Coin",
    list(coin_options.keys()),
)
news_coin = coin_options[selected_coin_label]  # API code

# Language dropdown with full names (including Hindi)
language_options = {
    "English": "en",
    "Spanish": "es",
    "German": "de",
    "French": "fr",
    "Italian": "it",
    "Portuguese": "pt",
    "Hindi": "hi",
}

selected_language_label = st.sidebar.selectbox(
    "Select Language",
    list(language_options.keys()),
)
news_language = language_options[selected_language_label]

# Dark / light mode toggle for news cards
news_dark_mode = st.sidebar.checkbox("Dark mode for news cards", value=False)

# =====================================
# DATA / MODEL HELPERS
# =====================================
@st.cache_data(show_spinner=False)
def load_price_data(file) -> pd.DataFrame:
    if file is not None:
        df = pd.read_csv(file)
    else:
        df = pd.read_csv("bitcoin.csv")

    expected_cols = {"Date", "Price"}
    if not expected_cols.issubset(df.columns):
        raise ValueError("CSV must contain at least 'Date' and 'Price' columns.")

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def prepare_features(df: pd.DataFrame):
    df = df.copy()
    df["Date_ordinal"] = df["Date"].map(datetime.toordinal)
    X = df[["Date_ordinal"]].values
    y = df["Price"].values
    return X, y


@st.cache_resource(show_spinner=False)
def train_model(X, y):
    tscv = TimeSeriesSplit(n_splits=5)

    param_grid = {
        "C": [1, 10, 100],
        "gamma": [0.001, 0.01, 0.1],
        "kernel": ["rbf"],
    }

    svr = SVR()
    grid_search = GridSearchCV(
        svr,
        param_grid,
        cv=tscv,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
    )
    grid_search.fit(X, y)

    best_model = grid_search.best_estimator_
    best_score = grid_search.best_score_
    rmse = float(np.sqrt(-best_score))

    return best_model, rmse, grid_search.best_params_


def make_forecast(model, df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    last_date = df["Date"].max()
    future_dates = [last_date + timedelta(days=i) for i in range(1, horizon + 1)]
    future_ordinals = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)
    preds = model.predict(future_ordinals)

    forecast_df = pd.DataFrame(
        {
            "Date": future_dates,
            "Predicted Price": preds,
        }
    )
    return forecast_df


def create_plot(df: pd.DataFrame, forecast_df: pd.DataFrame, horizon: int):
    fig, ax = plt.subplots(figsize=(11, 5))

    ax.plot(df["Date"], df["Price"], label="Historical Price")
    ax.plot(
        forecast_df["Date"],
        forecast_df["Predicted Price"],
        linestyle="--",
        marker="o",
        label=f"{horizon}-Day Forecast",
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.set_title("Bitcoin Price History and Forecast")
    ax.legend()
    fig.autofmt_xdate()

    return fig


# =====================================
# CRYPTO NEWS HELPERS
# =====================================
@st.cache_data(show_spinner=False)
def fetch_crypto_news(api_key: str, coin: str, language: str = "en", items: int = 20):
    """
    Fetch latest crypto news from NewsData.io Crypto News API.
    Example: https://newsdata.io/api/1/crypto?apikey=KEY&coin=btc
    """
    base_url = "https://newsdata.io/api/1/crypto"
    params = {
        "apikey": api_key,
        "coin": coin.lower(),
        "language": language,
    }

    try:
        resp = requests.get(base_url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if data.get("status") != "success":
            return [], f"API returned non-success status: {data.get('status', 'unknown')}"

        results = data.get("results") or data.get("articles") or []
        news_items = []
        for item in results[:items]:
            news_items.append(
                {
                    "title": item.get("title") or "Untitled",
                    "description": item.get("description") or "",
                    "link": item.get("link") or item.get("url") or "",
                    "pubDate": item.get("pubDate") or item.get("published_at") or "",
                    "source": (
                        item.get("source_name")
                        or item.get("source_id")
                        or (item.get("source") or {}).get("name", "")
                    ),
                }
            )

        return news_items, None

    except requests.exceptions.RequestException as e:
        return [], f"Network or API error: {e}"
    except Exception as e:
        return [], f"Unexpected error while parsing news: {e}"


def render_news_card(article, dark: bool):
    title = article["title"]
    desc = article["description"]
    link = article["link"]
    pub = article["pubDate"]
    source = article["source"]

    if dark:
        bg = "#111827"
        border = "#374151"
        title_color = "#e5e7eb"
        text_color = "#d1d5db"
        meta_color = "#9ca3af"
        link_color = "#60a5fa"
    else:
        bg = "#fafafa"
        border = "#e0e0e0"
        title_color = "#111827"
        text_color = "#374151"
        meta_color = "#6b7280"
        link_color = "#2563eb"

    meta_parts = []
    if source:
        meta_parts.append(f"<span>Source: <strong>{source}</strong></span>")
    if pub:
        meta_parts.append(f"<span>Published: <strong>{pub}</strong></span>")
    meta_html = " • ".join(meta_parts) if meta_parts else ""

    html = f"""
    <div style="
        border-radius: 0.9rem;
        border: 1px solid {border};
        background: {bg};
        padding: 1rem 1.2rem;
        margin-bottom: 0.9rem;
    ">
        <div style="font-weight: 600; font-size: 1.05rem; color: {title_color}; margin-bottom: 0.25rem;">
            {title}
        </div>
        <div style="font-size: 0.83rem; color: {meta_color}; margin-bottom: 0.4rem;">
            {meta_html}
        </div>
        <div style="font-size: 0.95rem; color: {text_color}; margin-bottom: 0.6rem;">
            {desc}
        </div>
        <div>
            <a href="{link}" target="_blank" style="
                font-size: 0.9rem;
                text-decoration: none;
                color: {link_color};
                font-weight: 600;
            ">
                🔗 Read full article
            </a>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_crypto_news_tab(dark_mode: bool):
    st.title("📰 FinCrypt for Crypto News")

    # Info header bar
    st.markdown(
        f"""
        <div style="padding: 0.75rem 1rem; border-radius: 0.75rem; 
                    border: 1px solid #e0e0e0;">
            <strong>Source:</strong> NewsData.io Crypto News API<br>
            <strong>Coin:</strong> {selected_coin_label} &nbsp;|&nbsp; 
            <strong>Language:</strong> {selected_language_label}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("")

    if not news_api_key:
        st.info("Enter your NewsData.io API key in the sidebar to load the latest crypto news.")
        return

    # Search bar for news
    search_query = st.text_input(
        "Search in headlines and descriptions",
        value="",
        placeholder="e.g. ETF, regulation, halving, DeFi...",
    )

    with st.spinner("Loading latest crypto headlines..."):
        news_items, error = fetch_crypto_news(
            api_key=news_api_key,
            coin=news_coin,
            language=news_language,
            items=20,
        )

    if error:
        st.error(error)
        return

    if not news_items:
        st.info("No news articles found for the selected coin / filters.")
        return

    # Filter by search query (client-side)
    if search_query.strip():
        q = search_query.strip().lower()
        filtered_items = []
        for article in news_items:
            t = (article.get("title") or "").lower()
            d = (article.get("description") or "").lower()
            if q in t or q in d:
                filtered_items.append(article)
        news_items = filtered_items

    if not news_items:
        st.warning("No articles match your search query.")
        return

    # Two-column layout for cards
    col_left, col_right = st.columns(2)

    for idx, article in enumerate(news_items):
        with (col_left if idx % 2 == 0 else col_right):
            render_news_card(article, dark_mode)


# =====================================
# MAIN TABS
# =====================================
tab_forecast, tab_news = st.tabs(["Bitcoin Forecast", "📰 Crypto News"])

with tab_forecast:
    st.title("FinCrypt 2.0: Bitcoin Price Prediction")

    st.markdown(
        """
This tool uses a Support Vector Regression (SVR) model to predict the price of Bitcoin.

1. Optional: upload your own CSV with `Date` and `Price` columns.
2. Choose a prediction horizon from the sidebar.
3. Click **Run Prediction** to see the forecast and chart.
"""
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Prediction Horizon", f"{prediction_days} days")
    col2.markdown("")
    col3.markdown("")

    run_clicked = st.button("Run Prediction", use_container_width=True)

    if run_clicked:
        try:
            df_prices = load_price_data(uploaded_file)

            st.subheader("Raw Data Preview")
            st.dataframe(df_prices.tail(10), use_container_width=True)

            X, y = prepare_features(df_prices)

            with st.spinner("Training SVR model with time-series cross-validation..."):
                model, rmse, best_params = train_model(X, y)

            c1, c2 = st.columns(2)
            with c1:
                st.metric("Cross-validated RMSE", f"{rmse:,.2f} USD")
            with c2:
                st.write("Best parameters:")
                st.json(best_params)

            forecast_df = make_forecast(model, df_prices, prediction_days)

            st.subheader(f"{prediction_days}-Day Price Forecast")
            st.dataframe(
                forecast_df.style.format({"Predicted Price": "${:,.2f}"}),
                use_container_width=True,
            )

            st.subheader("Historical vs Forecast Chart")
            fig = create_plot(df_prices, forecast_df, prediction_days)
            st.pyplot(fig)

        except FileNotFoundError:
            st.error(
                "Could not find `bitcoin.csv`. Ensure it is in the same folder as `app.py` "
                "or upload a CSV file via the sidebar."
            )
        except ValueError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"An unexpected error occurred while running the prediction: {e}")

with tab_news:
    render_crypto_news_tab(news_dark_mode)
