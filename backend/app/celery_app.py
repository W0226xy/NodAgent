import logging

from celery import Celery

from backend.app.core.config import (
    CELERY_BROKER_URL,
    CELERY_RESULT_BACKEND,
)
from backend.app.services.mcp_service import (
    get_github_mcp_diagnostic,
)


logger = logging.getLogger(
    __name__
)


github_mcp_problem = get_github_mcp_diagnostic()

if github_mcp_problem:
    logger.warning(
        "GitHub MCP unavailable in worker startup: %s. The current "
        "document worker does not call GitHub MCP, and future external "
        "tasks will degrade gracefully.",
        github_mcp_problem,
    )
else:
    logger.info(
        "GitHub MCP configuration is present in worker startup."
    )


celery_app = Celery(
    "nodagent",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=[
        (
            "backend.app.tasks."
            "document_tasks"
        ),
    ],
)


celery_app.conf.update(
    task_serializer="json",

    result_serializer="json",

    accept_content=[
        "json",
    ],

    task_track_started=True,

    broker_connection_retry_on_startup=True,

    result_expires=3600,

    enable_utc=True,

    worker_prefetch_multiplier=1,

    task_routes={
        (
            "backend.app.tasks."
            "document_tasks."
            "process_document"
        ): {
            "queue": "documents",
        },
    },
)
