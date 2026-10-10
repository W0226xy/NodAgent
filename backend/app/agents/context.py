from typing_extensions import TypedDict


class AgentContext(TypedDict):
    request_id: str

    user_id: str

    workspace_id: int

    top_k: int

    memory_prompt: str

    current_datetime: str

    current_date: str

    yesterday_date: str

    tomorrow_date: str

    timezone: str
