"""Metrics endpoint and middleware for observability."""

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

from shared.utils.metrics import get_metrics

router = APIRouter(tags=["observability"])


@router.get(
    "/metrics",
    response_class=PlainTextResponse,
    responses={
        200: {
            "description": "Prometheus-formatted metrics",
            "content": {"text/plain": {"example": "# HELP pmmds_requests_total ..."}},
        },
    },
)
async def metrics_prometheus() -> PlainTextResponse:
    """Prometheus metrics endpoint.

    Returns all application metrics in Prometheus exposition format.
    Compatible with Prometheus scraping.

    Returns:
        Prometheus-formatted metrics text.
    """
    metrics = get_metrics()
    return PlainTextResponse(
        content=metrics.to_prometheus(),
        media_type="text/plain; charset=utf-8",
    )


@router.get(
    "/metrics/json",
    responses={
        200: {
            "description": "JSON-formatted metrics",
        },
    },
)
async def metrics_json() -> dict:
    """JSON metrics endpoint.

    Returns all application metrics in JSON format for easier
    programmatic consumption and debugging.

    Returns:
        Dictionary with all metrics.
    """
    metrics = get_metrics()
    return metrics.to_dict()
