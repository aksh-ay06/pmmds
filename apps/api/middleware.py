"""FastAPI middleware for metrics collection and request logging."""

import time
from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from shared.utils import get_logger
from shared.utils.metrics import get_metrics

logger = get_logger(__name__)


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware to collect request metrics.

    Tracks:
    - Request count by method, endpoint, and status
    - Request latency by method and endpoint
    - Errors by endpoint and type
    """

    def __init__(self, app: ASGIApp, exclude_paths: list[str] | None = None) -> None:
        """Initialize middleware.

        Args:
            app: ASGI application.
            exclude_paths: Paths to exclude from metrics (e.g., /healthz, /metrics).
        """
        super().__init__(app)
        self.exclude_paths = exclude_paths or ["/healthz", "/ready", "/metrics"]
        self.metrics = get_metrics()

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Process request and collect metrics.

        Args:
            request: Incoming request.
            call_next: Next middleware/handler.

        Returns:
            Response from the handler.
        """
        # Skip metrics collection for excluded paths
        path = request.url.path
        if any(path.startswith(excluded) for excluded in self.exclude_paths):
            return await call_next(request)

        # Normalize path for metrics (replace IDs with placeholders)
        normalized_path = self._normalize_path(path)
        method = request.method

        start_time = time.perf_counter()
        status_code = 500
        error_type = None

        try:
            response = await call_next(request)
            status_code = response.status_code
            return response

        except Exception as e:
            error_type = type(e).__name__
            raise

        finally:
            # Calculate latency
            latency = time.perf_counter() - start_time

            # Record request metrics
            self.metrics.record_request(
                method=method,
                endpoint=normalized_path,
                status=status_code,
                latency_seconds=latency,
            )

            # Record errors
            if status_code >= 400:
                self.metrics.record_error(
                    endpoint=normalized_path,
                    error_type=error_type or f"http_{status_code}",
                )

            # Structured request logging
            log_data = {
                "method": method,
                "path": path,
                "status": status_code,
                "latency_ms": round(latency * 1000, 2),
                "client_ip": self._get_client_ip(request),
            }

            if status_code >= 500:
                logger.error("request_error", **log_data)
            elif status_code >= 400:
                logger.warning("request_client_error", **log_data)
            else:
                logger.debug("request_completed", **log_data)

    def _normalize_path(self, path: str) -> str:
        """Normalize path for metrics aggregation.

        Replaces dynamic segments (UUIDs, IDs) with placeholders.

        Args:
            path: Original request path.

        Returns:
            Normalized path.
        """
        import re

        # Replace UUIDs
        path = re.sub(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
            "{id}",
            path,
            flags=re.IGNORECASE,
        )
        # Replace numeric IDs
        path = re.sub(r"/\d+(/|$)", "/{id}\\1", path)

        return path

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request.

        Handles X-Forwarded-For header for proxied requests.

        Args:
            request: Incoming request.

        Returns:
            Client IP address.
        """
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for detailed request/response logging.

    Logs request details with structured JSON format.
    """

    def __init__(
        self,
        app: ASGIApp,
        exclude_paths: list[str] | None = None,
        log_request_body: bool = False,
        log_response_body: bool = False,
    ) -> None:
        """Initialize middleware.

        Args:
            app: ASGI application.
            exclude_paths: Paths to exclude from logging.
            log_request_body: Whether to log request body (careful with PII).
            log_response_body: Whether to log response body.
        """
        super().__init__(app)
        self.exclude_paths = exclude_paths or ["/healthz", "/ready", "/metrics"]
        self.log_request_body = log_request_body
        self.log_response_body = log_response_body

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Process request with detailed logging.

        Args:
            request: Incoming request.
            call_next: Next middleware/handler.

        Returns:
            Response from the handler.
        """
        path = request.url.path
        if any(path.startswith(excluded) for excluded in self.exclude_paths):
            return await call_next(request)

        # Generate or extract request ID
        request_id = request.headers.get("x-request-id", "")
        if not request_id:
            import uuid

            request_id = str(uuid.uuid4())

        # Add request ID to structlog context
        import structlog

        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)

        # Log request start
        logger.info(
            "request_started",
            method=request.method,
            path=path,
            query=str(request.query_params),
            user_agent=request.headers.get("user-agent", ""),
        )

        response = await call_next(request)

        # Add request ID to response headers
        response.headers["x-request-id"] = request_id

        return response
