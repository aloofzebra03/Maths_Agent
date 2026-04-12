"""
Shared exception definitions for API tracker and agent logic.

These typed exceptions allow different layers (tracker, graph, API server)
to react differently to different exhaustion conditions without string parsing.
"""

from typing import Optional


class APITrackerError(RuntimeError):
    """Base class for all API tracker related errors."""
    pass


class MinuteLimitExhaustedError(APITrackerError):
    """
    Raised when ALL API-model combinations have exhausted
    their per-minute rate limits.

    This is a HARD backpressure condition.
    """
    def __init__(self, retry_after_seconds: int = 60):
        self.retry_after_seconds = retry_after_seconds
        super().__init__(
            f"All API-model combinations exhausted per-minute limits. "
            f"Retry after {retry_after_seconds} seconds."
        )


class DayLimitExhaustedError(APITrackerError):
    """
    Raised when ALL API-model combinations have exhausted
    their daily rate limits.

    This is a SOFT exhaustion condition (quota fully consumed).
    """
    def __init__(self,message: str | None = None):
        self.message = message or "All API-model combinations exhausted daily limits."
        super().__init__(self.message)
