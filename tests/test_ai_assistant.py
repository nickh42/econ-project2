from finance_dashboard.ai_assistant import LLMClient, build_ai_prompt, is_question_in_scope, portfolio_has_holdings


def test_build_ai_prompt_includes_stock_context_and_guardrails():
    prompt = build_ai_prompt(
        page_name="Stock detail",
        current_ticker="AAPL",
        user_query="What does the recent SEC filing say about liquidity?",
        observed_data={
            "company_name": "Apple Inc.",
            "current_price": 210.5,
            "sector": "Technology",
            "news": [{"title": "Apple launches new chip", "source": "Reuters"}],
            "sec_financial_data": {"financials": {"2024": [1000.0]}},
        },
    )

    assert "AAPL" in prompt
    assert "Apple Inc." in prompt
    assert "Technology" in prompt
    assert "read-only" in prompt.lower()
    assert "cannot make predictions" in prompt.lower()
    assert "liquidity" in prompt.lower()


def test_is_question_in_scope_rejects_unrelated_queries():
    assert is_question_in_scope("Can you write me a poem about the ocean?") is False
    assert is_question_in_scope("Explain the current portfolio and recent SEC filings for AAPL") is True
    assert is_question_in_scope("Can you tell me about Tax Rate For Calcs") is False
    assert is_question_in_scope(
        "Can you tell me about Tax Rate For Calcs",
        page_name="stock_detail",
        current_ticker="AAPL",
    ) is True


def test_build_ai_prompt_prioritizes_sec_facts_matching_question():
    facts = [
        {"label": f"Revenue item {idx}", "value": idx}
        for idx in range(10)
    ]
    facts.append(
        {
            "label": "Effective Income Tax Rate Continuing Operations",
            "value": 0.24,
            "period": "2024",
        }
    )

    prompt = build_ai_prompt(
        page_name="Stock detail",
        current_ticker="AAPL",
        user_query="Can you tell me about Tax Rate For Calcs?",
        observed_data={"sec_financial_data": facts},
    )

    assert "Effective Income Tax Rate Continuing Operations" in prompt
    assert "0.24" in prompt


def test_portfolio_has_holdings_handles_empty_portfolio():
    assert portfolio_has_holdings({"holdings": []}) is False
    assert portfolio_has_holdings({"holdings": [{"Ticker": "AAPL", "Shares": 2}]}) is True


def test_build_ai_prompt_summarizes_large_stock_context():
    observed_data = {
        "company_name": "Apple Inc.",
        "current_price": 210.5,
        "recent_filings": [{"form": f"10-Q-{idx}", "filing_date": "2024-01-01"} for idx in range(50)],
        "news": [{"title": f"News {idx}", "source": "Reuters"} for idx in range(50)],
        "sec_financial_data": {
            "financials": {"2024": [float(idx) for idx in range(200)]},
            "balance_sheet": {"2024": [float(idx) for idx in range(200)]},
        },
    }

    prompt = build_ai_prompt(
        page_name="Stock detail",
        current_ticker="AAPL",
        user_query="What does the current cash flow statement show?",
        observed_data=observed_data,
    )

    assert "observed data summary" in prompt.lower()
    assert "summary" in prompt.lower()
    assert "truncated" in prompt.lower()
    assert "additional fields omitted" in prompt.lower()


def test_llm_client_uses_supported_default_model_and_validates_api_key():
    from finance_dashboard.ai_assistant import DEFAULT_MODEL

    assert DEFAULT_MODEL == "openai/gpt-oss-120b"

    try:
        LLMClient(api_key="   ")
    except ValueError as exc:
        assert "missing" in str(exc).lower()
    else:
        raise AssertionError("Expected a ValueError for a blank Groq API key")
