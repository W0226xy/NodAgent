import logging
from contextlib import AsyncExitStack
from typing import (
    List,
    Literal,
)

from langchain.agents import (
    create_agent,
)
from langchain.mcp import (
    MCPAdapter,
)
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
)

from backend.app.services.llm_service import (
    get_chat_model,
)
from backend.app.services.mcp_service import (
    create_external_info_client,
    create_github_mcp_client,
)


logger = logging.getLogger(
    __name__
)


MCPSelection = Literal[
    "external_info",
    "github",
    "both",
]


EXTERNAL_INFO_KEYWORDS = (
    "天气",
    "气温",
    "温度",
    "天气预报",
    "下雨",
    "风速",
    "现在几点",
    "现在时间",
    "当前时间",
    "时区",
    "weather",
    "temperature",
    "forecast",
    "time",
    "timezone",
)


GITHUB_KEYWORDS = (
    "github",
    "仓库",
    "代码库",
    "repository",
    "repo",
    "readme",
    "issue",
    "pull request",
    "pr",
)


EXTERNAL_AGENT_SYSTEM_PROMPT = """
你是 NodAgent 的 External Tool Agent。

你的职责是处理需要访问外部系统
或实时外部数据的问题。

当前连接两个 MCP Server。

1. External Info MCP

负责：

- 当前天气
- 当前时间


2. GitHub MCP

负责：

- Repository 信息
- Repository 搜索
- Repository 文件
- README
- Issue
- Pull Request


==================================================
工具调用规则
==================================================

涉及实时信息或外部系统数据时，
必须优先调用 MCP Tool。

不要根据模型训练知识猜测：

- 当前天气
- 当前时间
- GitHub Repository 当前状态
- 当前 Issue
- 当前 Pull Request
- Repository 当前文件内容


==================================================
GitHub 权限
==================================================

当前 GitHub MCP 使用只读模式。

不得执行或声称已经执行：

- 创建 Issue
- 修改 Issue
- 删除 Issue
- 创建 Pull Request
- Merge Pull Request
- Push 代码
- 修改 Repository


==================================================
职责边界
==================================================

你不负责：

- Workspace 知识库 RAG
- Workspace 文档管理
- Long-term Memory 管理
- Workspace 文档删除

这些任务由其他 Specialist Agent 负责。


==================================================
回答规则
==================================================

必须根据 MCP Tool 返回的真实数据回答。

调用 Tool 后，
将结果整理成自然语言。

不要向用户暴露：

- MCP JSON-RPC
- MCPAdapter
- stdio
- HTTP Transport
- Docker 内部实现
- ToolMessage 内部格式

如果 Tool 调用失败，
明确说明无法获取对应外部数据，
不要编造。
""".strip()


def select_mcp_servers(
    task: str,
) -> MCPSelection:
    normalized_task = task.lower()

    needs_external_info = any(
        keyword in normalized_task
        for keyword in EXTERNAL_INFO_KEYWORDS
    )

    needs_github = any(
        keyword in normalized_task
        for keyword in GITHUB_KEYWORDS
    )

    if needs_external_info and needs_github:
        return "both"

    if needs_github:
        return "github"

    # The main graph sends real-time weather/time
    # requests to this agent. Default to the local
    # server so a GitHub outage cannot affect them.
    return "external_info"


def get_unavailable_message(
    selection: MCPSelection,
) -> str:
    if selection == "github":
        return (
            "暂时无法获取 GitHub 信息，"
            "请稍后重试。"
        )

    if selection == "both":
        return (
            "暂时无法完成外部信息查询，"
            "请稍后重试。"
        )

    return (
        "暂时无法获取实时天气或时间信息，"
        "请稍后重试。"
    )


def get_selection_prompt(
    selection: MCPSelection,
) -> str:
    if selection == "github":
        return (
            EXTERNAL_AGENT_SYSTEM_PROMPT
            + "\n\n本次请求仅加载了 GitHub 工具。"
        )

    if selection == "both":
        return EXTERNAL_AGENT_SYSTEM_PROMPT

    return (
        EXTERNAL_AGENT_SYSTEM_PROMPT
        + "\n\n本次请求仅加载了天气和时间工具。"
    )


def get_latest_ai_content(
    messages: List,
) -> str:
    for message in reversed(
        messages
    ):
        if isinstance(
            message,
            AIMessage,
        ):
            content = message.content

            if isinstance(
                content,
                str,
            ):
                return content.strip()

    return ""


async def run_external_agent(
    task: str,
) -> str:
    selection = select_mcp_servers(
        task
    )

    try:
        async with AsyncExitStack() as stack:
            tools = []

            if selection in (
                "external_info",
                "both",
            ):
                external_info_adapter = await (
                    stack.enter_async_context(
                        MCPAdapter(
                            create_external_info_client()
                        )
                    )
                )

                tools.extend(
                    await external_info_adapter.list_tools(
                        cache_mode="refresh"
                    )
                )

            if selection in (
                "github",
                "both",
            ):
                github_adapter = await (
                    stack.enter_async_context(
                        MCPAdapter(
                            create_github_mcp_client()
                        )
                    )
                )

                tools.extend(
                    await github_adapter.list_tools(
                        cache_mode="refresh"
                    )
                )

            if not tools:
                logger.warning(
                    "No MCP tools available for %s task",
                    selection,
                )
                return get_unavailable_message(
                    selection
                )

            agent = create_agent(
                model=get_chat_model(),
                tools=tools,
                system_prompt=get_selection_prompt(
                    selection
                ),
                name="external_agent",
            )

            result = await agent.ainvoke(
                {
                    "messages": [
                        HumanMessage(
                            content=task
                        )
                    ]
                }
            )

            answer = get_latest_ai_content(
                result.get(
                    "messages",
                    [],
                )
            )

            if not answer:
                logger.warning(
                    "External agent returned no answer "
                    "for %s task",
                    selection,
                )
                return get_unavailable_message(
                    selection
                )

            return answer

    except Exception as exc:
        logger.warning(
            "External agent failed for %s task (%s)",
            selection,
            type(exc).__name__,
        )
        return get_unavailable_message(
            selection
        )
