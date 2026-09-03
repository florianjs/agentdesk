from agentdesk.config import Settings


def test_settings_load_without_required_env() -> None:
    """No required field: the app boots in CI without secrets."""
    s = Settings()
    assert s.model_smart and s.model_judge
    assert s.triagely_url.endswith("/v1") and s.docpilot_url.endswith("/v1")


def test_env_overrides_default(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("DOCPILOT_URL", "https://docpilot.example.com/v1")
    assert Settings().docpilot_url == "https://docpilot.example.com/v1"
