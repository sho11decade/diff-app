import time
from collections import defaultdict, deque

from fastapi import HTTPException, status

from app.core.config import API_KEY, MAX_REQUESTS_PER_MINUTE

_request_log: dict[str, deque[float]] = defaultdict(deque)


def check_api_key(x_api_key: str | None) -> str:
    if x_api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="X-API-Key header is required.",
        )

    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key.",
        )

    return x_api_key


def check_rate_limit(client_id: str) -> None:
    now = time.time()
    window_start = now - 60

    queue = _request_log[client_id]
    while queue and queue[0] < window_start:
        queue.popleft()

    if len(queue) >= MAX_REQUESTS_PER_MINUTE:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please retry later.",
        )

    queue.append(now)
