from datetime import datetime, timedelta
from typing import Any, Dict, Optional
from uuid import uuid4
from zoneinfo import ZoneInfo

from backend.app.services import (
    memory_service,
)


SHANGHAI_TIMEZONE = ZoneInfo("Asia/Shanghai")


def build_agent_context(
    user_id: str,
    workspace_id: int,
    top_k: int,
    *,
    now: Optional[datetime] = None,
    request_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Build one immutable request context shared by every graph node."""

    request_now = now or datetime.now(SHANGHAI_TIMEZONE)
    if request_now.tzinfo is None:
        request_now = request_now.replace(tzinfo=SHANGHAI_TIMEZONE)
    else:
        request_now = request_now.astimezone(SHANGHAI_TIMEZONE)

    memory_context = (
        memory_service
        .load_memory_context(
            workspace_id=workspace_id,
            user_id=user_id,
        )
    )

    memory_prompt = (
        memory_service
        .build_memory_prompt(
            memory_context
        )
    )

    return {
        "request_id": request_id or uuid4().hex,
        "user_id": user_id,
        "workspace_id": workspace_id,
        "top_k": top_k,
        "memory_prompt": memory_prompt,
        "current_datetime": request_now.strftime("%Y-%m-%d %H:%M"),
        "current_date": request_now.strftime("%Y-%m-%d"),
        "yesterday_date": (request_now - timedelta(days=1)).strftime("%Y-%m-%d"),
        "tomorrow_date": (request_now + timedelta(days=1)).strftime("%Y-%m-%d"),
        "timezone": "Asia/Shanghai",
    }
