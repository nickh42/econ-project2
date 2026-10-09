from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from finance_dashboard.ai_assistant import ask_groq_assistant, contains_prediction_request, is_question_in_scope
from finance_dashboard.dashboard_controller import DashboardController
from finance_dashboard.favourite_list_of_stocks import FavouriteListOfStocks
from finance_dashboard.mongo_client import MongoDB
from finance_dashboard.services.auth_service import AuthenticationService
from finance_dashboard.stock_comparator import StockComparator
from finance_dashboard.marketaux_client import MarketauxClient
from finance_dashboard.trading import TradeEngine


st.set_page_config(page_title="Finance Dashboard", layout="wide")

# Incrementing counter to ensure per-render unique element keys
_render_card_counter = 0

controller = DashboardController()
favorites_store = FavouriteListOfStocks(Path("favorites.json"))

DEFAULT_HOLDINGS: dict[str, dict[str, float]] = {}


def get_auth_service() -> AuthenticationService | None:
    mongo_uri = st.secrets.get("mongodb_uri", "").strip()
    if not mongo_uri:
        return None
    mongo_db = MongoDB(mongo_uri)
    return AuthenticationService(mongo_db.user_repository)


def get_assistant_settings() -> tuple[str, str]:
    api_key = (
        st.secrets.get("groq_api_key")
        or st.secrets.get("GROQ_API_KEY")
        or ""
    ).strip()
    model = "openai/gpt-oss-120b"
    return api_key, model


def render_ai_assistant(page_name: str, current_ticker: str | None, observed_data: dict, *, context_label: str = "") -> None:
    with st.expander("Ask AI", expanded=False):
        st.caption(
            "Read-only guidance for stocks, SEC data, market news, and portfolio questions. "
            "No predictions, trade execution, or actions are allowed."
        )
        key_name = f"{page_name}_{current_ticker or 'portfolio'}_{context_label or 'panel'}"
        question = st.text_area(
            "Ask a question",
            key=f"ai_question_{key_name}",
            value=st.session_state.get(f"ai_question_{key_name}", ""),
            placeholder="Example: What does the current cash flow statement show?",
            height=120,
        )
        st.session_state[f"ai_question_{key_name}"] = question

        if st.button("Ask AI", key=f"ai_submit_{key_name}", use_container_width=True):
            if not question.strip():
                st.warning("Please enter a question.")
            elif not is_question_in_scope(question):
                st.info("I can only help with stock, SEC, news, and portfolio questions. Please ask about the current ticker, filings, market news, or your portfolio.")
            elif contains_prediction_request(question):
                st.info("I can explain the current information on this page, but I cannot make predictions or price forecasts.")
            else:
                api_key, model = get_assistant_settings()
                if not api_key:
                    st.info("Add groq_api_key to .streamlit/secrets.toml to enable Ask AI.")
                else:
                    try:
                        answer = ask_groq_assistant(
                            page_name=page_name,
                            current_ticker=current_ticker,
                            user_query=question,
                            observed_data=observed_data,
                            api_key=api_key,
                            model=model,
                        )
                        st.markdown(answer)
                    except ValueError as exc:
                        st.error(str(exc))


def render_metrics(summary: dict, ticker: str) -> None:
    metrics = summary.get(ticker, {})
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Average Return", f"{metrics.get('avg_return', 0.0):.4f}")
    col2.metric("Volatility", f"{metrics.get('volatility', 0.0):.4f}")
    col3.metric("Min Return", f"{metrics.get('min_return', 0.0):.4f}")
    col4.metric("Max Return", f"{metrics.get('max_return', 0.0):.4f}")


def get_default_range() -> tuple[pd.Timestamp, pd.Timestamp]:
    end = pd.Timestamp.today().normalize()
    start = end - pd.DateOffset(years=1)
    return start, end


def render_signed_line_chart(series: pd.Series, *, title: str, zero_reference: bool = True, color_by_sign: bool = False) -> None:
    values = series.dropna()
    if values.empty:
        st.info(f"No {title.lower()} data available.")
        return

    fig = go.Figure()
    x_values = values.index
    y_values = values.to_numpy(dtype=float)

    if zero_reference:
        fig.add_hline(
            y=0,
            line_dash="dot",
            line_color="gray",
            line_width=1.5,
            annotation_text="0",
            annotation_position="top left",
        )

    if color_by_sign:
        fig.add_hrect(
            y0=0,
            y1=max(y_values.max(), 0.0),
            fillcolor="rgba(76, 175, 80, 0.12)",
            line_width=0,
            layer="below",
        )
        fig.add_hrect(
            y0=min(y_values.min(), 0.0),
            y1=0,
            fillcolor="rgba(244, 67, 54, 0.12)",
            line_width=0,
            layer="below",
        )
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="lines",
                line=dict(color="#000000", width=3),
                connectgaps=True,
                showlegend=False,
                hovertemplate="%{x}<br>%{y:.4f}<extra></extra>",
            )
        )
        for idx, value in enumerate(y_values):
            if value < 0:
                fig.add_trace(
                    go.Scatter(
                        x=[x_values[idx]],
                        y=[value],
                        mode="lines",
                        line=dict(color="#ef5350", width=3),
                        showlegend=False,
                        hovertemplate="%{x}<br>%{y:.4f}<extra></extra>",
                    )
                )
    else:
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="lines",
                line=dict(color="#1565c0", width=3),
                connectgaps=True,
                name=title,
                hovertemplate="%{x}<br>%{y:.4f}<extra></extra>",
            )
        )

    fig.update_xaxes(
        range=[x_values.min(), x_values.max()],
        fixedrange=True,
        rangeslider=dict(visible=False),
    )
    fig.update_yaxes(range=[min(y_values.min(), 0.0), max(y_values.max(), 0.0)], fixedrange=True)
    fig.update_layout(
        template="plotly_white",
        title=title,
        margin=dict(l=20, r=20, t=40, b=20),
        height=320,
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "scrollZoom": False, "doubleClick": "reset"})


def build_sec_data_table(sec_data: dict) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for table_name, frame in sec_data.items():
        if not isinstance(frame, pd.DataFrame):
            continue
        table = frame.copy().reset_index()
        if table.empty:
            continue
        table.columns = [str(column) for column in table.columns]
        table.insert(0, "section", table_name)
        frames.append(table)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def reset_trade_state() -> None:
    for key in [
        "trade_ticker",
        "trade_side",
        "trade_step",
        "trade_shares",
        "trade_order_ready",
    ]:
        st.session_state.pop(key, None)


def render_trade_choice_dialog(ticker: str) -> None:
    @st.dialog(f"Trade {ticker}")
    def _trade_dialog() -> None:
        st.write("Choose the order type.")
        buy_col, sell_col = st.columns(2)
        if buy_col.button("Buy", type="primary", use_container_width=True):
            st.session_state["trade_ticker"] = ticker
            st.session_state["trade_side"] = "buy"
            st.session_state["trade_step"] = "order"
            st.session_state["trade_shares"] = 1.0
            st.rerun()
        if sell_col.button("Sell", type="secondary", use_container_width=True):
            st.session_state["trade_ticker"] = ticker
            st.session_state["trade_side"] = "sell"
            st.session_state["trade_step"] = "order"
            st.session_state["trade_shares"] = 1.0
            st.rerun()

    _trade_dialog()


def render_trade_order_page() -> None:
    ticker = st.session_state.get("trade_ticker")
    side = st.session_state.get("trade_side", "buy")
    if not ticker:
        return

    default_start, default_end = get_default_range()
    snapshot = controller.get_stock_snapshot(
        ticker,
        str(default_start.date()),
        str(default_end.date()),
    )
    current_price = float(snapshot["current_price"])
    user = st.session_state.user
    available_cash = float(getattr(user, "cash_balance", 0.0))
    owned_shares = float(st.session_state.portfolio_holdings.get(ticker, {}).get("shares", 0.0))

    st.title(f"{side.title()} {ticker}")
    st.caption(f"Current price: ${current_price:.2f}")

    if st.button("Back to detail page", key=f"trade_back_{ticker}"):
        st.session_state["detail_ticker"] = ticker
        reset_trade_state()
        st.rerun()
    if st.button("Clear trade", key=f"clear_trade_{ticker}"):
        reset_trade_state()
        st.rerun()

    shares = st.number_input(
        "Number of Shares",
        min_value=0.0,
        value=float(st.session_state.get("trade_shares", 1.0)),
        step=1.0,
        key=f"trade_shares_input_{ticker}",
    )
    st.session_state["trade_shares"] = float(shares)

    if side == "buy":
        estimated_cost = shares * current_price
        st.metric("Current Price", f"${current_price:.2f}")
        st.metric("Available Cash", f"${available_cash:.2f}")
        st.metric("Number of Shares", f"{shares:.0f}")
        st.metric("Estimated Cost", f"${estimated_cost:.2f}")
        if estimated_cost > available_cash:
            st.error("Insufficient funds for this buy order.")
        review_disabled = shares <= 0 or estimated_cost > available_cash
        review_label = "Review Buy Order"
    else:
        estimated_value = shares * current_price
        st.metric("Current Price", f"${current_price:.2f}")
        st.metric("Shares Owned", f"{owned_shares:.2f}")
        st.metric("Number of Shares", f"{shares:.0f}")
        st.metric("Estimated Proceeds", f"${estimated_value:.2f}")
        if shares > owned_shares:
            st.error(f"You only own {owned_shares:.2f} shares of {ticker}.")
        review_disabled = shares <= 0 or shares > owned_shares
        review_label = "Review Sell Order"

    if st.button(review_label, key=f"review_{side}_{ticker}", disabled=review_disabled, type="primary"):
        st.session_state["trade_step"] = "review"
        st.rerun()


def render_trade_review_page() -> None:
    ticker = st.session_state.get("trade_ticker")
    side = st.session_state.get("trade_side", "buy")
    if not ticker:
        return

    default_start, default_end = get_default_range()
    snapshot = controller.get_stock_snapshot(
        ticker,
        str(default_start.date()),
        str(default_end.date()),
    )
    current_price = float(snapshot["current_price"])
    shares = float(st.session_state.get("trade_shares", 0.0))
    order_total = shares * current_price
    user = st.session_state.user

    st.title(f"Review {side.title()} Order")
    st.subheader(f"{side.title()} {ticker}")
    st.write(f"Ticker: {ticker}")
    st.write(f"Side: {side.title()}")
    st.write(f"Current Price: ${current_price:.2f}")
    st.write(f"Number of Shares: {shares:.2f}")
    if side == "buy":
        st.write(f"Estimated Cost: ${order_total:.2f}")
        st.write(f"Available Cash After Order: ${float(user.cash_balance) - order_total:.2f}")
    else:
        st.write(f"Estimated Proceeds: ${order_total:.2f}")
        st.write(f"Cash After Sale: ${float(user.cash_balance) + order_total:.2f}")

    if st.button("Submit Order", key=f"submit_{side}_{ticker}", type="primary"):
        trade_engine = TradeEngine(st.session_state.get("processed_trade_keys", set()))
        try:
            updated_cash, updated_holdings = trade_engine.execute_order(
                side=side,
                ticker=ticker,
                shares=shares,
                price=current_price,
                cash_balance=float(user.cash_balance),
                holdings=st.session_state.portfolio_holdings,
            )
            user.cash_balance = updated_cash
            st.session_state.user = user
            st.session_state.portfolio_holdings = updated_holdings
            st.session_state["processed_trade_keys"] = trade_engine.processed_trade_keys
            st.success(f"{side.title()} order submitted successfully for {ticker}.")
            reset_trade_state()
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))

    if st.button("Back to order form", key=f"back_to_order_{ticker}"):
        st.session_state["trade_step"] = "order"
        st.rerun()


st.markdown(
    """
    <style>
    div.stButton > button[kind="primary"] {
        background-color: #2e7d32;
        color: white;
        border: 1px solid #2e7d32;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #1b5e20;
        border-color: #1b5e20;
    }
    div[data-testid="stMetric"] {
        min-height: 110px;
    }
    div[data-testid="stMetricLabel"] {
        white-space: normal;
        overflow-wrap: anywhere;
        font-size: 1.1rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.5rem;
        white-space: normal;
        overflow-wrap: anywhere;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_stock_card(
    ticker: str,
    snapshot: dict,
    *,
    show_add_to_watchlist: bool = True,
    show_detail_button: bool = True,
) -> None:
    global _render_card_counter
    _render_card_counter += 1
    unique_id = _render_card_counter

    is_positive = snapshot["day_return"] >= 0
    with st.container(border=True):
        left_col, right_col = st.columns([4, 2])
        with left_col:
            st.markdown(
                f"### {ticker} · ${snapshot['current_price']:.2f}"
                f" · <span style='color:{'green' if is_positive else 'red'}'>"
                f"{snapshot['day_return']:+.2%}</span>",
                unsafe_allow_html=True,
            )
        with right_col:
            st.metric("Day return", f"{snapshot['day_return']:+.2%}", delta=f"{snapshot['day_change']:+.2f}")

        st.caption(f"Last price: ${snapshot['current_price']:.2f} | Change: ${snapshot['day_change']:+.2f}")

        button_cols = st.columns(2 if show_add_to_watchlist and show_detail_button else 1)
        if show_add_to_watchlist:
            if button_cols[0].button("⭐ Add to watchlist", key=f"fav_{ticker}_{unique_id}", use_container_width=True):
                favorites_store.add(ticker)
                st.success(f"{ticker} added to watchlist.")

        if show_detail_button:
            detail_col = button_cols[1] if show_add_to_watchlist else button_cols[0]
            if detail_col.button(
                "Go to detail page →",
                key=f"detail_{ticker}_{unique_id}",
                type="primary",
                use_container_width=True,
            ):
                st.session_state["detail_ticker"] = ticker
                st.rerun()


if "user" not in st.session_state:
    st.session_state.user = None

if "processed_trade_keys" not in st.session_state:
    st.session_state.processed_trade_keys = set()

if "trade_ticker" in st.session_state and st.session_state["trade_ticker"]:
    if st.session_state.get("trade_side") is None:
        render_trade_choice_dialog(st.session_state["trade_ticker"])
        st.stop()
    if st.session_state.get("trade_step") == "review":
        render_trade_review_page()
    else:
        render_trade_order_page()
    st.stop()

if st.session_state.user is None:
    st.title("Trading Simulator")
    st.subheader("Welcome")
    auth_tab, register_tab = st.tabs(["Login", "Sign up"])

    with auth_tab:
        with st.form("login_form"):
            login_username = st.text_input("Username (valid email)", placeholder="name@example.com")
            login_password = st.text_input("Password", type="password")
            login_submitted = st.form_submit_button("Login")

        if login_submitted:
            service = get_auth_service()
            if service is None:
                st.error("Add your MongoDB URI to .streamlit/secrets.toml before logging in.")
            else:
                try:
                    user = service.login_user(login_username.strip(), login_password)
                    st.session_state.user = user
                    st.success(f"Welcome back, {user.username}!")
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))

    with register_tab:
        with st.form("registration_form"):
            username = st.text_input("Username (valid email)", placeholder="name@example.com", key="signup_username")
            password = st.text_input("Password", type="password", key="signup_password")
            confirm_password = st.text_input("Confirm password", type="password", key="signup_confirm_password")
            submitted = st.form_submit_button("Register")

        if submitted:
            service = get_auth_service()
            if service is None:
                st.error("Add your MongoDB URI to .streamlit/secrets.toml before registering.")
            else:
                try:
                    user = service.register_user(username.strip(), password, confirm_password)
                    st.session_state.user = user
                    st.success(f"Registration successful. Welcome, {user.username}!")
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))

    st.info("Use the MongoDB connection string stored in .streamlit/secrets.toml to persist user accounts.")
    st.stop()

if "portfolio_holdings" not in st.session_state:
    st.session_state.portfolio_holdings = {}

user = st.session_state.user
st.title("Finance Dashboard")
st.subheader(f"Profile: {user.username}")

if st.button("Log out"):
    st.session_state.user = None
    st.session_state.pop("detail_ticker", None)
    st.session_state.pop("search_snapshot", None)
    st.rerun()

default_start, default_end = get_default_range()

if "detail_ticker" in st.session_state and st.session_state["detail_ticker"]:
    detail_ticker = st.session_state["detail_ticker"]
    if st.button("Back to landing page", key="back_to_landing"):
        st.session_state.pop("detail_ticker", None)
        st.rerun()

    detail_start = st.date_input(
        "Detail start date",
        value=default_start.date(),
        min_value=pd.Timestamp("2000-01-01").date(),
        max_value=default_end.date(),
    )
    detail_end = st.date_input(
        "Detail end date",
        value=default_end.date(),
        min_value=detail_start,
        max_value=pd.Timestamp.today().date(),
    )

    try:
        detail_snapshot = controller.get_stock_detail(
            detail_ticker,
            str(detail_start),
            str(detail_end),
        )
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    # News and sentiment
    api_key = st.secrets.get("MARKETAUX_API_KEY", "").strip()
    if not api_key:
        st.info("Add MARKETAUX_API_KEY to .streamlit/secrets.toml to enable latest news and sentiment.")
    else:
        # Cache results per-ticker for 15 minutes to avoid slow repeated calls
        @st.cache_data(ttl=900)
        def _fetch_news(ticker: str) -> list:
            client = MarketauxClient(api_key)
            return client.fetch_news_for_ticker(ticker, limit=5)

        try:
            articles = _fetch_news(detail_ticker)
        except Exception:
            articles = []

        with st.expander("Latest news & sentiment", expanded=True):
            if not articles:
                st.info("No recent news articles were found for this ticker.")
            else:
                for art in articles:
                    title = art.get("title") or "Untitled"
                    description = art.get("description") or art.get("snippet") or ""
                    source = art.get("source", "")
                    url = art.get("url") or art.get("link") or ""
                    score = art.get("sentiment_score")
                    label = art.get("sentiment_label")

                    score_text = f"{score:.2f}" if isinstance(score, (int, float)) else "N/A"
                    label_text = label or "N/A"

                    st.markdown(f"**[{title}]({url})**")
                    if description:
                        st.caption(description)
                    st.text(f"Source: {source} — Sentiment: {score_text} ({label_text})")
                    st.write("---")

    stock_context = {
        "ticker": detail_ticker,
        "company_name": detail_snapshot["company_name"],
        "current_price": detail_snapshot["current_price"],
        "day_change": detail_snapshot["day_change"],
        "daily_percent_change": detail_snapshot["daily_percent_change"],
        "market_cap": detail_snapshot["market_cap"],
        "sector": detail_snapshot["sector"],
        "industry": detail_snapshot["industry"],
        "recent_filings": detail_snapshot["recent_filings"],
        "sec_financial_data": build_sec_data_table(detail_snapshot["sec_financial_data"]).to_dict(orient="records"),
        "news": articles if api_key else [],
    }
    render_ai_assistant(
        "stock_detail",
        detail_ticker,
        stock_context,
        context_label="stock",
    )

    st.title(f"{detail_snapshot['company_name']} ({detail_ticker})")
    st.caption(f"{detail_snapshot['sector']} / {detail_snapshot['industry']}")

    col_left, col_right = st.columns(2)
    with col_left:
        st.metric("Current Price", f"${detail_snapshot['current_price']:.2f}")
        st.metric("Market Cap", f"${detail_snapshot['market_cap']:,}")
    with col_right:
        st.metric("Daily % change", f"{detail_snapshot['daily_percent_change']:+.2%}")
        st.metric("Sector / industry", f"{detail_snapshot['sector']} / {detail_snapshot['industry']}")

    with st.expander("Historical price graph", expanded=True):
        historical = detail_snapshot["historical_prices"]["Close"].dropna()
        historical_fig = go.Figure()
        historical_fig.add_trace(
            go.Scatter(
                x=historical.index,
                y=historical.values,
                mode="lines",
                line=dict(color="#1f77b4", width=3),
                hovertemplate="%{x}<br>$%{y:,.2f}<extra></extra>",
            )
        )
        historical_fig.update_xaxes(
            range=[historical.index.min(), historical.index.max()],
            fixedrange=True,
            rangeslider=dict(visible=False),
        )
        historical_fig.update_yaxes(range=[historical.min(), historical.max()], fixedrange=True)
        historical_fig.update_layout(
            template="plotly_white",
            margin=dict(l=20, r=20, t=20, b=20),
            height=320,
            paper_bgcolor="white",
            plot_bgcolor="white",
        )
        st.plotly_chart(
            historical_fig,
            use_container_width=True,
            config={"displayModeBar": False, "scrollZoom": False, "doubleClick": "reset"},
        )

    with st.expander("Return graph"):
        render_signed_line_chart(
            detail_snapshot["daily_returns"],
            title="Daily Return",
            zero_reference=True,
            color_by_sign=True,
        )

    with st.expander("Volatility graph"):
        render_signed_line_chart(detail_snapshot["volatility_series"], title="Volatility", zero_reference=False, color_by_sign=False)

    with st.expander("SEC financial data"):
        sec_table = build_sec_data_table(detail_snapshot["sec_financial_data"])
        if sec_table.empty:
            st.info("No SEC financial data was returned for this stock.")
        else:
            st.dataframe(sec_table, use_container_width=True)

    with st.expander("Recent SEC filings"):
        filings = detail_snapshot["recent_filings"]
        if filings:
            st.dataframe(pd.DataFrame(filings), use_container_width=True)
        else:
            st.info("No recent SEC filings were returned for this stock.")

    trade_clicked = st.button(
        "Trade",
        key=f"trade_button_{detail_ticker}",
        type="primary",
        use_container_width=True,
    )
    if trade_clicked:
        st.session_state["trade_ticker"] = detail_ticker
        st.session_state.pop("trade_side", None)
        st.session_state.pop("trade_step", None)
        st.session_state["trade_shares"] = 1.0
        st.rerun()

    st.stop()

st.caption(f"Created: {user.created_at.isoformat() if user.created_at else 'N/A'}")

cash_col, holdings_value_col, total_value_col = st.columns(3)
portfolio_snapshot = controller.build_portfolio_snapshot(
    holdings=st.session_state.portfolio_holdings,
    prices={
        ticker: float(position.get("avg_cost", 0.0))
        for ticker, position in st.session_state.portfolio_holdings.items()
    },
    cash_balance=user.cash_balance,
)
cash_col.metric("Available cash", f"${portfolio_snapshot['cash']:.2f}")
holdings_value_col.metric("Invested value", f"${portfolio_snapshot['invested']:.2f}")
total_value_col.metric("Total portfolio value", f"${portfolio_snapshot['total_value']:.2f}")

search_query = st.text_input("Search stocks or tickers", placeholder="Try AAPL, MSFT, NVDA")
search_clicked = st.button("Search")

if search_clicked:
    if not search_query.strip():
        st.warning("Please enter a stock ticker or company symbol.")
    else:
        searched_ticker = search_query.strip().upper()
        try:
            snapshot = controller.get_stock_snapshot(
                searched_ticker,
                str((default_end - pd.DateOffset(days=30)).date()),
                str(default_end.date()),
            )
            st.session_state.search_snapshot = snapshot
        except ValueError as exc:
            st.error(str(exc))

if "search_snapshot" in st.session_state:
    snapshot = st.session_state["search_snapshot"]
    render_stock_card(snapshot["ticker"], snapshot, show_add_to_watchlist=True, show_detail_button=True)

st.subheader("Holdings")
favorite_list = favorites_store.list()
show_favorites_only = st.checkbox("Filter by favourites", value=False)
portfolio_rows = []
for ticker, position in st.session_state.portfolio_holdings.items():
    if show_favorites_only and ticker not in favorite_list:
        continue
    shares = float(position.get("shares", 0.0))
    avg_cost = float(position.get("avg_cost", 0.0))
    current_price = avg_cost
    portfolio_rows.append(
        {
            "Ticker": ticker,
            "Shares": shares,
            "Avg Cost": avg_cost,
            "Market Price": current_price,
            "Market Value": shares * current_price,
        }
    )

portfolio_context = {
    "portfolio_summary": {
        "cash": portfolio_snapshot["cash"],
        "invested": portfolio_snapshot["invested"],
        "total_value": portfolio_snapshot["total_value"],
    },
    "holdings": portfolio_rows,
    "favorite_tickers": favorite_list,
    "show_favorites_only": show_favorites_only,
}
render_ai_assistant(
    "portfolio",
    None,
    portfolio_context,
    context_label="portfolio",
)

if portfolio_rows:
    header_cols = st.columns((1.2, 1, 1.2, 1.2, 1.5))
    with header_cols[0]:
        st.caption("Ticker")
    with header_cols[1]:
        st.caption("Shares")
    with header_cols[2]:
        st.caption("Avg Cost")
    with header_cols[3]:
        st.caption("Market Price")
    with header_cols[4]:
        st.caption("Market Value")

    for row in portfolio_rows:
        ticker = row["Ticker"]
        cols = st.columns((1.2, 1, 1.2, 1.2, 1.5))
        with cols[0]:
            if st.button(ticker, key=f"holding_detail_{ticker}", use_container_width=True):
                st.session_state["detail_ticker"] = ticker
                st.rerun()
        with cols[1]:
            st.write(f"{row['Shares']:.2f}")
        with cols[2]:
            st.write(f"${row['Avg Cost']:.2f}")
        with cols[3]:
            st.write(f"${row['Market Price']:.2f}")
        with cols[4]:
            st.write(f"${row['Market Value']:.2f}")
else:
    st.info("No holdings to display yet.")

st.subheader("Cash vs invested")
figure = go.Figure(
    data=[
        go.Pie(
            labels=["Cash", "Invested"],
            values=[portfolio_snapshot["cash"], portfolio_snapshot["invested"]],
            hole=0.45,
            marker_colors=["#2E8B57", "#1F77B4"],
            textinfo="label+value",
            hovertemplate="%{label}: $%{value:,.2f}<extra></extra>",
        )
    ]
)
figure.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=350)
st.plotly_chart(figure, use_container_width=True)

if favorite_list:
    st.subheader("Favourites")
    for favorite in favorite_list:
        try:
            favorite_snapshot = controller.get_stock_snapshot(
                favorite,
                str((default_end - pd.DateOffset(days=30)).date()),
                str(default_end.date()),
            )
        except ValueError:
            continue
        render_stock_card(
            favorite_snapshot["ticker"],
            favorite_snapshot,
            show_add_to_watchlist=False,
            show_detail_button=True,
        )
else:
    st.info("No favourites added yet.")
