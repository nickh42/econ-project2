"""Finance dashboard package."""

from .dashboard import Dashboard
from .dashboard_controller import DashboardController
from .favourite_list_of_stocks import FavouriteListOfStocks
from .stock_comparator import StockComparator

__all__ = [
    "Dashboard",
    "DashboardController",
    "FavouriteListOfStocks",
    "StockComparator",
]
