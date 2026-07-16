from datetime import datetime, timezone

from litellm.models.team import LiteLLM_TeamTable
from litellm.proxy.spend_tracking.team_token_quota import (
    TeamTokenQuotaConfig,
    _get_config,
    _get_period,
    estimate_request_max_tokens,
    get_actual_tokens,
)


def test_team_token_quota_config_reads_team_metadata():
    team = LiteLLM_TeamTable(
        team_id="team-1",
        metadata={
            "token_quota": {
                "limit": 1_000_000,
                "duration": "monthly",
                "warning_thresholds": [0.8, 0.9],
                "max_output_tokens": 2048,
            }
        },
    )

    config = _get_config(team)

    assert config == TeamTokenQuotaConfig(1_000_000, "monthly", (0.8, 0.9), 2048)


def test_team_token_quota_uses_monthly_utc_period():
    period = _get_period(datetime(2026, 7, 16, 12, tzinfo=timezone.utc), "monthly")

    assert period.start == datetime(2026, 7, 1, tzinfo=timezone.utc)
    assert period.end == datetime(2026, 8, 1, tzinfo=timezone.utc)


def test_team_token_quota_estimates_input_and_requested_output():
    config = TeamTokenQuotaConfig(1_000_000, "monthly", (0.8, 0.9), 4096)

    estimated = estimate_request_max_tokens(
        request_body={
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 10,
        },
        route="/chat/completions",
        config=config,
    )

    assert estimated is not None
    assert estimated >= 10


def test_team_token_quota_reads_provider_usage():
    response = {"usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20}}

    assert get_actual_tokens(response=response, kwargs={}) == 20
