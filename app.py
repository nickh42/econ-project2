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

DEFAULT_FAVORITES = ["AAPL", "MSFT", "GOOG"]


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


if "user" not in st.session_state:
    st.session_state.user = None

if st.session_state.user is None:
    st.title("Trading Simulator")
    st.subheader("Create your account")

    with st.form("registration_form"):
        username = st.text_input("Username (valid email)", placeholder="name@example.com")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm password", type="password")
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

user = st.session_state.user
st.title("Finance Dashboard")
st.subheader(f"Profile: {user.username}")
profile_col, balance_col = st.columns(2)
profile_col.write(f"Created: {user.created_at.isoformat() if user.created_at else 'N/A'}")
balance_col.metric("Cash Balance", f"${user.cash_balance:,.2f}")

if st.button("Log out"):
    st.session_state.user = None
    st.rerun()

with st.sidebar:
    st.header("Inputs")
    tickers_input = st.text_input("Tickers (comma separated)", value="AAPL, MSFT, GOOG")
    data_frequency = st.selectbox("Data frequency", ["Daily", "Weekly", "Monthly", "Yearly"], index=0)
    default_end = pd.Timestamp.today().normalize().to_pydatetime().date()
    default_start = default_end - pd.DateOffset(months=6)
    start_date, end_date = st.date_input(
        "Date range",
        value=(default_start, default_end),
    )

    if end_date < start_date:
        st.warning("End date must be after start date.")
    elif end_date == start_date:
        st.warning("Select a date range longer than one day for market data.")

    favorite_list = favorites_store.list() or DEFAULT_FAVORITES
    favorites = st.multiselect("Favourite stocks", options=favorite_list, default=favorite_list[:3])
    if st.button("Save selected favorites"):
        for symbol in favorites:
            favorites_store.add(symbol)
        st.success("Favorites saved.")

if st.button("Load Dashboard"):
    symbols = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    valid_symbols = [s for s in symbols if s]
    if not valid_symbols:
        st.error("Please enter at least one ticker.")
    else:
        effective_start = start_date
        effective_end = end_date
        if effective_end <= effective_start:
            effective_end = effective_start + timedelta(days=1)
            st.warning("The selected range was too short, so the app extended it by one day to fetch data.")
        try:
            dashboard = controller.build_dashboard(valid_symbols, str(effective_start), str(effective_end))
            if dashboard.errors:
                for ticker, msg in dashboard.errors.items():
                    st.warning(msg)
            if not dashboard.data:
                st.info("No dashboard data available for the selected range.")
            else:
                resampled_data = StockComparator.resample_data(dashboard.data, data_frequency)
                dashboard.data = resampled_data
                dashboard.summary = StockComparator.compute_summary(dashboard.data)
                dashboard.returns = {ticker: metrics["returns"] for ticker, metrics in dashboard.summary.items()}
                st.session_state["dashboard"] = dashboard
                st.session_state["selected_view"] = st.session_state.get("selected_view", "Returns")
                st.session_state["data_frequency"] = data_frequency
        except ValueError as exc:
            st.error(str(exc))

if "dashboard" in st.session_state:
    dashboard = st.session_state["dashboard"]
    view_names = ["Returns", "Volatility", "Correlation", "Max/Min/Avg Returns"]
    selected_view = st.radio(
        "Dashboard view",
        view_names,
        index=view_names.index(st.session_state.get("selected_view", "Returns")),
        horizontal=True,
    )
    st.session_state["selected_view"] = selected_view

    if selected_view == "Returns":
        for ticker in dashboard.tickers:
            if ticker in dashboard.returns:
                st.write(f"## {ticker}")
                st.line_chart(dashboard.returns[ticker])

    elif selected_view == "Volatility":
        volatility_data = StockComparator.volatility_over_time(dashboard.data)
        if not volatility_data:
            st.info("No volatility data available for the selected range.")
        for ticker in dashboard.tickers:
            if ticker in volatility_data and not volatility_data[ticker].empty:
                st.write(f"## {ticker}")
                st.line_chart(volatility_data[ticker])

    elif selected_view == "Correlation":
        series_map = {
            ticker: dashboard.returns.get(ticker, pd.Series(dtype=float))
            for ticker in dashboard.tickers
            if ticker in dashboard.returns
        }
        if series_map:
            matrix = pd.DataFrame(series_map).corr()
            heatmap = go.Figure(
                data=go.Heatmap(
                    z=matrix.to_numpy(),
                    x=matrix.columns,
                    y=matrix.index,
                    colorscale="RdYlBu_r",
                    zmid=0,
                    hoverongaps=False,
                )
            )
            heatmap.update_layout(title="Stock correlation heatmap", height=500)
            st.plotly_chart(heatmap, use_container_width=True)
        else:
            st.info("No correlation data available for the selected tickers.")

    elif selected_view == "Max/Min/Avg Returns":
        for ticker in dashboard.tickers:
            if ticker in dashboard.summary:
                st.write(f"## {ticker}")
                render_metrics(dashboard.summary, ticker)

    stock_tabs = st.tabs([ticker for ticker in dashboard.tickers if ticker])
    for tab, ticker in zip(stock_tabs, dashboard.tickers):
        with tab:
            st.write(f"### {ticker} summary")
            if ticker in dashboard.summary:
                st.json(dashboard.summary[ticker])
            else:
                st.info("No summary available for this ticker.")
else:
    st.info("Use the sidebar to enter tickers and load a dashboard.")
