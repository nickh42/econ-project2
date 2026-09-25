from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from finance_dashboard.dashboard_controller import DashboardController
from finance_dashboard.favourite_list_of_stocks import FavouriteListOfStocks
from finance_dashboard.mongo_client import MongoDB
from finance_dashboard.services.auth_service import AuthenticationService
from finance_dashboard.stock_comparator import StockComparator


st.set_page_config(page_title="Finance Dashboard", layout="wide")

controller = DashboardController()
favorites_store = FavouriteListOfStocks(Path("favorites.json"))

DEFAULT_HOLDINGS: dict[str, dict[str, float]] = {}


def get_auth_service() -> AuthenticationService | None:
    mongo_uri = st.secrets.get("mongodb_uri", "").strip()
    if not mongo_uri:
        return None
    mongo_db = MongoDB(mongo_uri)
    return AuthenticationService(mongo_db.user_repository)


def render_metrics(summary: dict, ticker: str) -> None:
    metrics = summary.get(ticker, {})
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Average Return", f"{metrics.get('avg_return', 0.0):.4f}")
    col2.metric("Volatility", f"{metrics.get('volatility', 0.0):.4f}")
    col3.metric("Min Return", f"{metrics.get('min_return', 0.0):.4f}")
    col4.metric("Max Return", f"{metrics.get('max_return', 0.0):.4f}")


def get_default_range() -> tuple[pd.Timestamp, pd.Timestamp]:
    end = pd.Timestamp.today().normalize()
    start = end - pd.DateOffset(months=6)
    return start, end


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
            if button_cols[0].button("⭐ Add to watchlist", key=f"fav_{ticker}", use_container_width=True):
                favorites_store.add(ticker)
                st.success(f"{ticker} added to watchlist.")

        if show_detail_button:
            detail_col = button_cols[1] if show_add_to_watchlist else button_cols[0]
            if detail_col.button(
                "Go to detail page →",
                key=f"detail_{ticker}",
                type="primary",
                use_container_width=True,
            ):
                st.session_state["detail_ticker"] = ticker
                st.rerun()


if "user" not in st.session_state:
    st.session_state.user = None

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

    try:
        detail_dashboard = controller.build_dashboard(
            [detail_ticker],
            str(default_start.date()),
            str(default_end.date()),
        )
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    if detail_dashboard.errors:
        for ticker, msg in detail_dashboard.errors.items():
            st.warning(msg)
        st.stop()

    resampled_data = StockComparator.resample_data(detail_dashboard.data, "Daily")
    detail_dashboard.data = resampled_data
    detail_dashboard.summary = StockComparator.compute_summary(detail_dashboard.data)
    detail_dashboard.returns = {
        ticker: metrics["returns"] for ticker, metrics in detail_dashboard.summary.items()
    }

    st.title(f"{detail_ticker} Detail Page")
    detail_returns = detail_dashboard.returns.get(detail_ticker)
    if detail_returns is not None and not detail_returns.empty:
        st.line_chart(detail_returns)

    render_metrics(detail_dashboard.summary, detail_ticker)
    stock_summary = detail_dashboard.summary.get(detail_ticker, {})
    if stock_summary:
        st.json(stock_summary)
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

holding_table = pd.DataFrame(portfolio_rows)
if not holding_table.empty:
    st.dataframe(holding_table, use_container_width=True)
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
