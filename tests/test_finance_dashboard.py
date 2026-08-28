import pandas as pd

from finance_dashboard.dashboard import Dashboard
from finance_dashboard.dashboard_controller import DashboardController
from finance_dashboard.stock_comparator import StockComparator
from finance_dashboard.favourite_list_of_stocks import FavouriteListOfStocks


def test_dashboard_controller_handles_multiindex_yfinance_columns():
    controller = DashboardController()

    multiindex = pd.MultiIndex.from_product(
        [["Close", "Open", "Volume"], ["AAPL"]],
        names=["Price", "Ticker"],
    )
    sample = pd.DataFrame(
        [[100.0, 101.0, 2000], [101.0, 102.0, 2100]],
        columns=multiindex,
        index=pd.date_range("2024-01-01", periods=2, freq="D"),
    )

    controller.fetch_stock_data = lambda ticker, start_date, end_date: sample  # type: ignore[assignment]

    result = controller.build_dashboard(["AAPL"], "2024-01-01", "2024-02-01")
    assert result.tickers == ["AAPL"]
    assert "Close" in result.data["AAPL"].columns
    assert result.errors == {}


def test_stock_comparator_daily_returns_and_summary():
    prices = pd.DataFrame(
        {
            "Close": [100.0, 110.0, 121.0, 108.9],
        },
        index=pd.date_range("2024-01-01", periods=4, freq="D"),
    )

    returns = StockComparator.daily_returns(prices["Close"])
    assert returns.iloc[0] == 0.1
    assert returns.iloc[1] == 0.1
    assert returns.iloc[2] == -0.1

    summary = StockComparator.compute_summary({"AAPL": prices})
    assert "AAPL" in summary
    assert "returns" in summary["AAPL"]
    assert "volatility" in summary["AAPL"]


def test_stock_comparator_resamples_daily_to_monthly_data():
    prices = pd.DataFrame(
        {"Close": [100.0, 110.0, 121.0, 108.9, 120.0, 130.0]},
        index=pd.date_range("2024-01-01", periods=6, freq="D"),
    )

    monthly = StockComparator.resample_data({"AAPL": prices}, "monthly")
    assert list(monthly["AAPL"].index[:1])[0].month == 1
    assert "Close" in monthly["AAPL"].columns


def test_dashboard_dataclass_tracks_tickers():
    dashboard = Dashboard(
        tickers=["AAPL", "MSFT"],
        start_date="2024-01-01",
        end_date="2024-02-01",
        data={
            "AAPL": pd.DataFrame({"Close": [1.0, 2.0]}),
            "MSFT": pd.DataFrame({"Close": [3.0, 4.0]}),
        },
        summary={},
        returns={},
        errors={},
    )

    assert dashboard.tickers == ["AAPL", "MSFT"]
    assert len(dashboard.data) == 2


def test_stock_comparator_builds_volatility_series_and_correlation_matrix():
    price_data = {
        "AAPL": pd.DataFrame(
            {"Close": [100.0, 110.0, 121.0, 108.9, 120.0]},
            index=pd.date_range("2024-01-01", periods=5, freq="D"),
        ),
        "MSFT": pd.DataFrame(
            {"Close": [200.0, 210.0, 220.0, 215.0, 225.0]},
            index=pd.date_range("2024-01-01", periods=5, freq="D"),
        ),
    }

    vol = StockComparator.volatility_over_time(price_data)
    assert set(vol) == {"AAPL", "MSFT"}
    assert list(vol["AAPL"].index) == list(price_data["AAPL"].index)

    corr = StockComparator.correlation_table(
        {
            "AAPL": StockComparator.daily_returns(price_data["AAPL"]["Close"]),
            "MSFT": StockComparator.daily_returns(price_data["MSFT"]["Close"]),
        }
    )
    assert set(corr.columns) == {"AAPL", "MSFT"}
    assert corr.shape == (2, 2)


def test_favourite_list_round_trip(tmp_path):
    storage = tmp_path / "favorites.json"
    favorites = FavouriteListOfStocks(str(storage))

    favorites.add("AAPL")
    favorites.add("MSFT")
    favorites.remove("MSFT")

    assert favorites.list() == ["AAPL"]
