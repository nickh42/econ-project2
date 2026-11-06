import pandas as pd
import pytest

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


def test_dashboard_controller_builds_stock_snapshot():
    controller = DashboardController()
    sample = pd.DataFrame(
        {"Close": [100.0, 101.0, 99.0, 102.0]},
        index=pd.date_range("2024-01-01", periods=4, freq="D"),
    )
    controller.fetch_stock_data = lambda ticker, start_date, end_date: sample  # type: ignore[assignment]

    snapshot = controller.get_stock_snapshot("AAPL", "2024-01-01", "2024-02-01")

    assert snapshot["ticker"] == "AAPL"
    assert snapshot["current_price"] == pytest.approx(102.0)
    assert snapshot["day_return"] == pytest.approx((102.0 - 99.0) / 99.0)


def test_dashboard_controller_builds_portfolio_snapshot():
    controller = DashboardController()

    portfolio = controller.build_portfolio_snapshot(
        holdings={
            "AAPL": {"shares": 10, "avg_cost": 100.0},
            "MSFT": {"shares": 5, "avg_cost": 200.0},
        },
        prices={"AAPL": 120.0, "MSFT": 220.0},
        cash_balance=5000.0,
    )

    assert portfolio["cash"] == 5000.0
    assert portfolio["invested"] == pytest.approx(1200.0 + 1100.0)
    assert portfolio["total_value"] == pytest.approx(7300.0)


def test_dashboard_controller_builds_stock_detail_payload():
    controller = DashboardController()
    sample = pd.DataFrame(
        {"Close": [100.0, 101.0, 99.0, 102.0]},
        index=pd.date_range("2024-01-01", periods=4, freq="D"),
    )
    controller.fetch_stock_data = lambda ticker, start_date, end_date: sample  # type: ignore[assignment]
    controller.fetch_stock_profile = lambda ticker: {  # type: ignore[assignment]
        "company_name": "Apple Inc.",
        "market_cap": 3000000000000,
        "sector": "Technology",
        "industry": "Consumer Electronics",
    }
    controller.fetch_sec_financial_data = lambda ticker: {"Revenue": {"2024": 1000}}  # type: ignore[assignment]
    controller.fetch_recent_sec_filings = lambda ticker: [{"form": "10-K", "filing_date": "2024-01-30"}]  # type: ignore[assignment]
    controller.fetch_edgar_data = lambda ticker: {"source": "EDGAR", "latest_filings": [{"form": "10-K", "link": "https://www.sec.gov"}]}  # type: ignore[assignment]

    detail = controller.get_stock_detail("AAPL", "2024-01-01", "2024-02-01")

    assert detail["ticker"] == "AAPL"
    assert detail["company_name"] == "Apple Inc."
    assert detail["market_cap"] == 3000000000000
    assert detail["sector"] == "Technology"
    assert detail["historical_prices"]["Close"].iloc[-1] == pytest.approx(102.0)
    assert detail["daily_returns"].iloc[-1] == pytest.approx(0.03)
    assert detail["recent_filings"][0]["form"] == "10-K"
    assert detail["edgar_data"]["source"] == "EDGAR"
    assert detail["edgar_data"]["latest_filings"][0]["link"].startswith("https://")


def test_favourite_list_round_trip(tmp_path):
    storage = tmp_path / "favorites.json"
    favorites = FavouriteListOfStocks(str(storage))

    assert favorites.list() == []

    favorites.add("AAPL")
    favorites.add("MSFT")
    favorites.remove("MSFT")

    assert favorites.list() == ["AAPL"]


def test_trade_engine_executes_buy_and_updates_cash_and_holdings():
    from finance_dashboard.trading import TradeEngine

    engine = TradeEngine()
    cash_balance, holdings = engine.execute_order(
        side="buy",
        ticker="AAPL",
        shares=5,
        price=100.0,
        cash_balance=100000.0,
        holdings={},
    )

    assert cash_balance == pytest.approx(99500.0)
    assert holdings["AAPL"]["shares"] == pytest.approx(5.0)
    assert holdings["AAPL"]["avg_cost"] == pytest.approx(100.0)


def test_trade_engine_executes_sell_and_rejects_duplicate_orders():
    from finance_dashboard.trading import TradeEngine

    engine = TradeEngine()
    cash_balance, holdings = engine.execute_order(
        side="sell",
        ticker="AAPL",
        shares=3,
        price=120.0,
        cash_balance=1000.0,
        holdings={"AAPL": {"shares": 10.0, "avg_cost": 100.0}},
    )

    assert cash_balance == pytest.approx(1360.0)
    assert holdings["AAPL"]["shares"] == pytest.approx(7.0)

    with pytest.raises(ValueError, match="Duplicate trade rejected"):
        engine.execute_order(
            side="sell",
            ticker="AAPL",
            shares=3,
            price=120.0,
            cash_balance=1360.0,
            holdings={"AAPL": {"shares": 7.0, "avg_cost": 100.0}},
        )


def test_stock_direction_features_align_news_at_or_before_each_trading_close():
    from finance_dashboard.stock_direction_predictor import StockDirectionPredictor

    trading_days = pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-04"])
    prices = pd.DataFrame(
        {"Close": [100.0, 101.0, 102.0], "Volume": [1000, 1100, 1200]},
        index=trading_days,
    )
    articles = [
        {"published_at": "2024-01-02T20:30:00Z", "sentiment_score": 0.6},
        {"published_at": "2024-01-02T22:00:00Z", "sentiment_score": -0.5},
        {"published_at": "2024-01-04T22:00:00Z", "sentiment_score": 1.0},
    ]

    features, target, sentiment_available = StockDirectionPredictor.build_features(prices, articles)

    assert sentiment_available is True
    assert features.loc["2024-01-02", "Daily_sentiment_mean"] == pytest.approx(0.6)
    assert features.loc["2024-01-02", "News_article_count"] == 1
    assert features.loc["2024-01-03", "Daily_sentiment_mean"] == pytest.approx(-0.5)
    assert features.loc["2024-01-03", "News_article_count"] == 1
    assert features.loc["2024-01-04", "News_article_count"] == 0
    assert target.loc["2024-01-02"] == 1
    assert pd.isna(target.iloc[-1])


def test_stock_direction_prediction_uses_price_only_fallback_and_chronological_split():
    import numpy as np

    from finance_dashboard.stock_direction_predictor import StockDirectionPredictor

    rng = np.random.default_rng(42)
    returns = rng.normal(0.0005, 0.015, size=240)
    close = 100 * np.cumprod(1 + returns)
    prices = pd.DataFrame(
        {
            "Close": close,
            "Volume": rng.integers(100_000, 1_000_000, size=len(close)),
        },
        index=pd.bdate_range("2024-01-02", periods=len(close)),
    )

    result = StockDirectionPredictor().predict(prices)

    assert result["direction"] in {"UP", "DOWN"}
    assert 0.0 <= result["probability"] <= 1.0
    assert result["sentiment_available"] is False
    assert result["training_observations"] >= 60
    assert result["validation_start"] < result["feature_date"]


def test_marketaux_client_fetches_date_bounded_historical_sentiment(monkeypatch):
    from finance_dashboard.marketaux_client import MarketauxClient

    captured = {}

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "data": [
                    {
                        "published_at": "2024-01-02T15:00:00Z",
                        "entities": [
                            {"symbol": "AAPL", "sentiment_score": 0.0},
                        ],
                    }
                ]
            }

    def fake_get(url, *, params, timeout):
        captured.update(params)
        return Response()

    monkeypatch.setattr("finance_dashboard.marketaux_client.requests.get", fake_get)

    articles = MarketauxClient("test-key").fetch_historical_news_for_ticker(
        "AAPL",
        "2024-01-01",
        "2024-01-03",
    )

    assert captured["published_after"] == "2024-01-01"
    assert captured["published_before"] == "2024-01-03"
    assert captured["limit"] == 100
    assert articles[0]["sentiment_score"] == 0.0
