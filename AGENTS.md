# AGENT WORKING NOTES

## Current status
- Project workspace was initialized on 2026-08-11.
- Need to scaffold a Streamlit-based finance dashboard using Python, pandas, yfinance, and modular service classes.
- Need to create a virtual environment, install dependencies, then validate with tests and a run smoke check.

## Plan
1. Create a project skeleton and failing tests for core finance logic.
2. Implement modular dashboard components and Streamlit UI.
3. Install dependencies in a local virtual environment and run verification checks.
4. Leave the repository in a runnable state with notes here for the next agent.

## Commands to keep handy
- python3 -m venv .venv
- source .venv/bin/activate
- pip install -r requirements.txt
- pytest -q
- streamlit run app.py

## Notes
- The app should support up to three tickers, date-range filters, favorite stock persistence, and dashboard navigation.
- Errors for missing tickers or failed data pulls should be surfaced to the user rather than crashing the app.
