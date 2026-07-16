"""Team-scoped periodic token quota admission and reconciliation."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import litellm
from fastapi import HTTPException

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import CallInfo, LiteLLM_TeamTable, Litellm_EntityType, UserAPIKeyAuth
from litellm.proxy.auth.route_checks import RouteChecks
from litellm.proxy.utils import PrismaClient, ProxyLogging

_DEFAULT_MAX_OUTPUT_TOKENS = 4096
_COUNTER_PREFIX = "team_token_quota"
_LOCAL_LOCK = asyncio.Lock()
_LOCAL_ALERT_KEYS: set[str] = set()

# 2026-07-16: Team Token policy is read from existing Team metadata so the
# feature works with the current generated Prisma client and API surface.


@dataclass(frozen=True)
class TeamTokenQuotaConfig:
    limit: int
    duration: str
    warning_thresholds: tuple[float, ...]
    max_output_tokens: int
    max_input_tokens: Optional[int] = None


@dataclass(frozen=True)
class TeamTokenQuotaPeriod:
    start: datetime
    end: datetime


def _get_config(team_object: Optional[LiteLLM_TeamTable]) -> Optional[TeamTokenQuotaConfig]:
    if team_object is None or not isinstance(team_object.metadata, dict):
        return None
    raw = team_object.metadata.get("token_quota")
    if not isinstance(raw, dict):
        return None
    try:
        limit = int(raw.get("limit", 0))
        duration = str(raw.get("duration", "monthly")).lower()
        max_output_tokens = int(raw.get("max_output_tokens", _DEFAULT_MAX_OUTPUT_TOKENS))
        max_input_tokens = raw.get("max_input_tokens")
        max_input_tokens = int(max_input_tokens) if max_input_tokens is not None else None
        thresholds = tuple(sorted({float(value) for value in raw.get("warning_thresholds", (0.8, 0.9))}))
    except (TypeError, ValueError):
        verbose_proxy_logger.warning("Invalid metadata.token_quota for team=%s", team_object.team_id)
        return None
    if (
        limit <= 0
        or max_output_tokens <= 0
        or (max_input_tokens is not None and max_input_tokens <= 0)
        or duration not in {"daily", "weekly", "monthly"}
    ):
        return None
    if any(value <= 0 or value >= 1 for value in thresholds):
        return None
    return TeamTokenQuotaConfig(
        limit=limit,
        duration=duration,
        warning_thresholds=thresholds,
        max_output_tokens=max_output_tokens,
        max_input_tokens=max_input_tokens,
    )


def _get_period(now: datetime, duration: str) -> TeamTokenQuotaPeriod:
    now = now.astimezone(timezone.utc)
    if duration == "daily":
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1)
    elif duration == "weekly":
        start = (now - timedelta(days=now.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=7)
    else:
        start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        end = (start.replace(day=28) + timedelta(days=4)).replace(day=1)
    return TeamTokenQuotaPeriod(start=start, end=end)


def _request_model(request_body: dict[str, Any]) -> Optional[str]:
    model = request_body.get("model")
    return model if isinstance(model, str) and model else None


def _estimate_input_tokens(request_body: dict[str, Any], model: str, fallback: Optional[int]) -> Optional[int]:
    try:
        if "messages" in request_body:
            return int(litellm.token_counter(model=model, messages=request_body.get("messages") or []))
        for key in ("prompt", "input", "query", "documents"):
            if key in request_body:
                value = request_body.get(key)
                return int(litellm.token_counter(model=model, text=str(value or "")))
    except Exception:
        verbose_proxy_logger.debug("Unable to estimate Team Token quota input", exc_info=True)
    # 2026-07-16: Private model tokenizers may be unavailable; an administrator
    # supplied max_input_tokens keeps admission conservative instead of bypassing quota.
    return fallback


def estimate_request_max_tokens(
    request_body: dict[str, Any], route: str, config: TeamTokenQuotaConfig
) -> Optional[int]:
    """Estimate a finite upper bound without requiring model pricing metadata."""
    model = _request_model(request_body)
    if model is None:
        return None
    input_tokens = _estimate_input_tokens(request_body=request_body, model=model, fallback=config.max_input_tokens)
    if input_tokens is None:
        return None
    output_tokens = 0 if any(value in route for value in ("embedding", "moderation")) else None
    if output_tokens is None:
        output_tokens = next(
            (
                int(request_body[key])
                for key in ("max_completion_tokens", "max_tokens", "max_output_tokens")
                if request_body.get(key) is not None
            ),
            config.max_output_tokens,
        )
        output_tokens = min(output_tokens, config.max_output_tokens)
    return input_tokens + max(output_tokens, 0)


def get_actual_tokens(response: Any, kwargs: dict[str, Any]) -> Optional[int]:
    for candidate in (response, kwargs.get("complete_streaming_response"), kwargs.get("combined_usage_object")):
        usage = getattr(candidate, "usage", None)
        if usage is None and isinstance(candidate, dict):
            usage = candidate.get("usage")
        if usage is None:
            continue
        if isinstance(usage, dict):
            total = usage.get("total_tokens")
            prompt = usage.get("prompt_tokens", 0)
            completion = usage.get("completion_tokens", 0)
        else:
            total = getattr(usage, "total_tokens", None)
            prompt = getattr(usage, "prompt_tokens", 0)
            completion = getattr(usage, "completion_tokens", 0)
        if total is not None:
            return max(int(total), 0)
        if prompt is not None or completion is not None:
            return max(int(prompt or 0) + int(completion or 0), 0)
    return None


def _counter_key(team_id: str, period: TeamTokenQuotaPeriod) -> str:
    return f"{_COUNTER_PREFIX}:team:{team_id}:{period.start.isoformat()}"


async def _seed_counter(
    counter_key: str,
    team_id: str,
    period: TeamTokenQuotaPeriod,
    prisma_client: Optional[PrismaClient],
    ttl: int,
) -> None:
    from litellm.proxy.proxy_server import spend_counter_cache

    if await spend_counter_cache.async_get_cache(key=counter_key) is not None:
        return
    if prisma_client is None:
        raise HTTPException(status_code=503, detail={"error": "Team Token quota requires a connected database"})
    try:
        rows = await prisma_client.db.query_raw(
            'SELECT COALESCE(SUM("total_tokens"), 0) AS total_tokens '
            'FROM "LiteLLM_SpendLogs" WHERE "team_id" = $1 '
            'AND "startTime" >= $2::timestamp AND "startTime" < $3::timestamp',
            team_id,
            period.start.replace(tzinfo=None),
            period.end.replace(tzinfo=None),
        )
        total = int(rows[0].get("total_tokens") or 0) if rows else 0
    except Exception as exc:
        verbose_proxy_logger.warning("Unable to seed Team Token quota from spend logs", exc_info=True)
        raise HTTPException(status_code=503, detail={"error": "Unable to load Team Token quota state"}) from exc
    # 2026-07-16: SETNX prevents a second worker's cold-start DB seed from
    # overwriting a reservation that another worker already admitted.
    await spend_counter_cache.async_set_cache(key=counter_key, value=total, ttl=max(ttl, 60), nx=True)


async def _atomic_reserve(counter_key: str, amount: int, limit: int, ttl: int) -> tuple[bool, int]:
    from litellm.proxy.proxy_server import spend_counter_cache

    redis_cache = spend_counter_cache.redis_cache
    if redis_cache is not None:
        # 2026-07-16: Lua keeps the limit check and increment atomic across
        # proxy workers; a read-then-increment sequence would oversubscribe a Team.
        script = """
local current = tonumber(redis.call('GET', KEYS[1]) or '0')
local requested = tonumber(ARGV[1])
local limit = tonumber(ARGV[2])
local projected = current + requested
if projected > limit then return {0, current} end
redis.call('INCRBYFLOAT', KEYS[1], requested)
redis.call('EXPIRE', KEYS[1], tonumber(ARGV[3]))
return {1, projected}
"""
        result = await redis_cache.async_register_script(script)(
            keys=[counter_key],
            args=[str(amount), str(limit), str(max(ttl, 60))],
        )
        return bool(int(result[0])), int(float(result[1]))
    async with _LOCAL_LOCK:
        current = float(await spend_counter_cache.async_get_cache(key=counter_key) or 0)
        projected = current + amount
        if projected > limit:
            return False, int(current)
        await spend_counter_cache.async_set_cache(key=counter_key, value=projected, ttl=max(ttl, 60))
        return True, int(projected)


async def _adjust_counter(counter_key: str, adjustment: int) -> None:
    from litellm.proxy.proxy_server import _increment_spend_counter_cache

    if adjustment != 0:
        await _increment_spend_counter_cache(counter_key=counter_key, increment=float(adjustment))


async def _send_warning(
    team_object: LiteLLM_TeamTable,
    valid_token: UserAPIKeyAuth,
    config: TeamTokenQuotaConfig,
    period: TeamTokenQuotaPeriod,
    projected: int,
    threshold: float,
    proxy_logging_obj: ProxyLogging,
) -> None:
    from litellm.proxy.proxy_server import spend_counter_cache

    alert_key = f"{_counter_prefix(team_object.team_id, period)}:warning:{threshold}"
    if spend_counter_cache.redis_cache is not None:
        claimed = await spend_counter_cache.async_set_cache(
            key=alert_key,
            value=True,
            ttl=max(int((period.end - datetime.now(timezone.utc)).total_seconds()), 60),
            nx=True,
        )
        if claimed not in (True, "OK", b"OK"):
            return
    else:
        async with _LOCAL_LOCK:
            if alert_key in _LOCAL_ALERT_KEYS:
                return
            _LOCAL_ALERT_KEYS.add(alert_key)
    verbose_proxy_logger.warning(
        "Team Token quota warning: team=%s projected_tokens=%s limit=%s threshold=%s",
        team_object.team_id,
        projected,
        config.limit,
        threshold,
    )
    await proxy_logging_obj.budget_alerts(
        type="token_budget",
        user_info=CallInfo(
            spend=float(projected),
            max_budget=float(config.limit),
            token=valid_token.token,
            user_id=valid_token.user_id,
            team_id=team_object.team_id,
            team_alias=valid_token.team_alias,
            organization_id=valid_token.org_id,
            event_group=Litellm_EntityType.TEAM,
        ),
    )


def _counter_prefix(team_id: str, period: TeamTokenQuotaPeriod) -> str:
    return _counter_key(team_id, period)


async def reserve_team_token_quota(
    request_body: dict[str, Any],
    route: str,
    valid_token: UserAPIKeyAuth,
    team_object: Optional[LiteLLM_TeamTable],
    prisma_client: Optional[PrismaClient],
    proxy_logging_obj: ProxyLogging,
) -> Optional[dict[str, Any]]:
    config = _get_config(team_object)
    if (
        config is None
        or valid_token.team_id is None
        or team_object is None
        or not RouteChecks.is_llm_api_route(route=route)
    ):
        return None
    requested = estimate_request_max_tokens(request_body=request_body, route=route, config=config)
    if requested is None:
        raise HTTPException(status_code=503, detail={"error": "Unable to estimate Team Token quota for this request"})
    period = _get_period(datetime.now(timezone.utc), config.duration)
    ttl = int((period.end - datetime.now(timezone.utc)).total_seconds()) + 86400
    key = _counter_key(valid_token.team_id, period)
    async with _LOCAL_LOCK:
        await _seed_counter(
            counter_key=key, team_id=valid_token.team_id, period=period, prisma_client=prisma_client, ttl=ttl
        )
    allowed, projected = await _atomic_reserve(counter_key=key, amount=requested, limit=config.limit, ttl=ttl)
    if not allowed:
        raise HTTPException(
            status_code=429, detail={"error": "Team Token quota exceeded", "team_id": valid_token.team_id}
        )
    ratio = projected / config.limit
    for threshold in config.warning_thresholds:
        if ratio >= threshold:
            await _send_warning(team_object, valid_token, config, period, projected, threshold, proxy_logging_obj)
    return {
        "counter_key": key,
        "reserved_tokens": requested,
        "period_start": period.start.isoformat(),
        "finalized": False,
    }


async def get_team_token_quota_status(
    team_object: Optional[LiteLLM_TeamTable],
    prisma_client: Optional[PrismaClient],
) -> Optional[dict[str, Any]]:
    """Return the current Team Token quota state for Dashboard consumers."""
    config = _get_config(team_object)
    if config is None or team_object is None:
        return None
    period = _get_period(datetime.now(timezone.utc), config.duration)
    ttl = int((period.end - datetime.now(timezone.utc)).total_seconds()) + 86400
    key = _counter_key(team_object.team_id, period)
    async with _LOCAL_LOCK:
        await _seed_counter(
            counter_key=key,
            team_id=team_object.team_id,
            period=period,
            prisma_client=prisma_client,
            ttl=ttl,
        )
    from litellm.proxy.proxy_server import spend_counter_cache

    current_value = await spend_counter_cache.async_get_cache(key=key)
    projected_tokens = max(int(float(current_value or 0)), 0)
    usage_ratio = projected_tokens / config.limit if config.limit > 0 else 0.0
    return {
        "enabled": True,
        "limit": config.limit,
        "projected_tokens": projected_tokens,
        "usage_ratio": usage_ratio,
        "warning_thresholds": list(config.warning_thresholds),
        "duration": config.duration,
        "period_start": period.start.isoformat(),
        "period_end": period.end.isoformat(),
    }


async def reconcile_team_token_quota(reservation: Optional[dict[str, Any]], actual_tokens: Optional[int]) -> None:
    if not reservation or reservation.get("finalized") is True:
        return
    reserved = int(reservation.get("reserved_tokens") or 0)
    actual = reserved if actual_tokens is None else max(actual_tokens, 0)
    await _adjust_counter(counter_key=str(reservation["counter_key"]), adjustment=actual - reserved)
    reservation["actual_tokens"] = actual
    reservation["finalized"] = True


async def release_team_token_quota(reservation: Optional[dict[str, Any]]) -> None:
    await reconcile_team_token_quota(reservation=reservation, actual_tokens=0)
