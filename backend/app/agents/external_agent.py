import logging
import json
import re
import time
from ast import literal_eval
import zlib
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    List,
    Literal,
)

from langchain.agents import (
    create_agent,
)
from langchain.mcp import (
    MCPAdapter,
)
from langchain_core.callbacks import (
    BaseCallbackHandler,
)
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
)

from backend.app.services.llm_service import (
    get_chat_model,
    get_finish_reason,
    is_length_limited,
)
from backend.app.services.citation_service import (
    get_safe_web_url,
)
from backend.app.core.config import (
    LLM_MODEL,
    TAVILY_API_KEY,
    TAVILY_MCP_URL,
)
from backend.app.services.mcp_service import (
    create_external_info_client,
    create_github_mcp_client,
    create_web_search_client,
)


logger = logging.getLogger(
    __name__
)


MCPSelection = Literal[
    "external_info",
    "github",
    "both",
    "web_search",
    "external_info_and_web_search",
    "github_and_web_search",
    "all",
]


@dataclass
class ExternalAgentResult:
    answer: str#Agent 的最终文本回答
    retrieved_web_sources: List[Dict[str, Any]]#从 Tavily 工具结果提取出的网页来源
    incomplete: bool = False#模型是否因输出长度限制而截断
    web_search_status: str = "not_requested"#网络搜索状态
    web_search_message: str = ""#搜索失败或无结果时给用户的说明
    web_result_count: int = 0#Tavily 返回的结果数
    web_url_count: int = 0#有安全可用 URL 的结果数


MAX_WEB_RESULTS = 5
MAX_WEB_SNIPPET_CHARS = 1200


def classify_mcp_error(error: object) -> str:
    detail = str(error).lower()#统一转成小写文本
    #限流或配额问题
    if "429" in detail or "rate limit" in detail or "quota" in detail:
        return "rate_limited"
    #网络超时或连接问题
    if "timeout" in detail or "timed out" in detail:
        return "timeout"
    #鉴权或凭证问题
    if "401" in detail or "403" in detail or "auth" in detail or "credential" in detail:
        return "authentication"
    #MCP 工具参数或协议结构不匹配
    if "schema" in detail or "validation" in detail or "invalid argument" in detail:
        return "schema"
    #网络连接问题
    if any(marker in detail for marker in ("connection", "connect", "network", "dns", "refused")):
        return "network"
    return "unknown"

#把 MCP 工具调用参数写入日志，但自动删除 API Key、Token、请求头等敏感信息
def safe_tool_input(input_str: object) -> str:
    """Log search parameters while never serializing credentials or headers."""

    try:
        payload = json.loads(str(input_str))#按 JSON 格式解析
    except (TypeError, ValueError):
        try:
            payload = literal_eval(str(input_str))#JSON 解析失败时，尝试解析 Python 字典格式
        except (TypeError, ValueError, SyntaxError):
            return "<unparseable>"
    if not isinstance(payload, dict):#参数不是字典时也不展示
        return "<non-object>"
    safe = {#字典推导式过滤敏感字段
        key: value
        for key, value in payload.items()
        if key.lower() not in {"api_key", "token", "authorization", "headers"}
    }
    return json.dumps(safe, ensure_ascii=False)[:500]#转回 JSON，并限制日志长度

#无论 Tavily MCP 的结果被包装成什么样，最后都统一提取成“网页结果字典列表
def get_web_results_from_payload(output: object) -> List[Dict[str, Any]]:
    """Normalize both Tavily MCP list output and JSON/ToolMessage wrappers."""

    def decode(payload: object) -> List[Dict[str, Any]]:
        if isinstance(payload, str):
            try:
                return decode(json.loads(payload))
            except (TypeError, ValueError):
                return []
        if isinstance(payload, dict):
            if "results" in payload:
                return decode(payload.get("results"))
            # Streamable MCP wraps textual tool results as content blocks.
            if payload.get("type") == "text" and isinstance(payload.get("text"), str):
                return decode(payload["text"])
            return [payload] if "url" in payload else []
        if not isinstance(payload, list):
            return []
        direct_results = [item for item in payload if isinstance(item, dict) and "url" in item]
        if direct_results:
            return direct_results
        nested_results: List[Dict[str, Any]] = []
        for item in payload:
            nested_results.extend(decode(item))
        return nested_results

    content = getattr(output, "content", output)
    try:
        return decode(content)
    except (TypeError, ValueError):
        return []


def count_web_results_from_payload(output: object) -> tuple[int, int]:
    results = get_web_results_from_payload(output)
    return len(results), sum(
        1 for item in results
        if isinstance(item, dict) and get_safe_web_url(item.get("url"))
    )


class MCPToolLoggingHandler(
    BaseCallbackHandler,
):
    """Emit request-correlated MCP tool events without sensitive payloads."""

    def __init__(
        self,
        request_id: str,
        github_tool_names: set[str],
        tavily_tool_names: set[str],
    ):
        self.request_id = request_id
        self.github_tool_names = github_tool_names
        self.tavily_tool_names = tavily_tool_names
        self.tool_runs: Dict[str, tuple[str, float]] = {}

    def on_tool_start(
        self,
        serialized,
        input_str,
        *,
        run_id,
        **kwargs,
    ) -> None:
        del kwargs
        tool_name = (
            serialized.get("name", "unknown")
            if isinstance(serialized, dict)
            else "unknown"
        )

        if tool_name in self.github_tool_names | self.tavily_tool_names:
            self.tool_runs[str(run_id)] = (tool_name, time.monotonic())
            logger.warning(
                "llm_tool_call_requested request_id=%s tool=%s params=%s",
                self.request_id,
                tool_name,
                safe_tool_input(input_str),
            )
        if tool_name in self.tavily_tool_names:
            logger.warning(
                "tavily_search_started request_id=%s tool=%s params=%s",
                self.request_id,
                tool_name,
                safe_tool_input(input_str),
            )

    def on_tool_end(
        self,
        output,
        *,
        run_id,
        **kwargs,
    ) -> None:
        del kwargs
        tool_run = self.tool_runs.pop(str(run_id), None)
        if tool_run is None:
            return
        tool_name, started = tool_run
        elapsed_ms = int((time.monotonic() - started) * 1000)
        if tool_name in self.tavily_tool_names:
            result_count, url_count = count_web_results_from_payload(output)
            logger.warning(
                "tavily_search_succeeded request_id=%s tool=%s duration_ms=%d result_count=%d url_count=%d",
                self.request_id, tool_name, elapsed_ms, result_count, url_count,
            )
        else:
            logger.warning(
                "github_mcp_tool_succeeded request_id=%s tool=%s duration_ms=%d",
                self.request_id, tool_name, elapsed_ms,
            )

    def on_tool_error(
        self,
        error,
        *,
        run_id,
        **kwargs,
    ) -> None:
        del kwargs
        tool_run = self.tool_runs.pop(str(run_id), None)
        if tool_run is None:
            return
        tool_name, started = tool_run
        elapsed_ms = int((time.monotonic() - started) * 1000)
        error_category = classify_mcp_error(error)
        event = "tavily_search_failed" if tool_name in self.tavily_tool_names else "github_mcp_tool_failed"
        logger.warning(
            "%s request_id=%s tool=%s duration_ms=%d error_category=%s error_type=%s",
            event, self.request_id, tool_name, elapsed_ms, error_category, type(error).__name__,
        )


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

本次只会连接完成任务所需的 MCP 工具。

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

3. Web Search MCP

负责公开互联网的最新资讯、可核验资料和网页链接。

搜索和总结限制：

- 单次搜索最多保留 5 条结果；优先使用最相关的 3 至 5 条。
- 每条搜索摘要只保留回答所需的重点，避免逐字复述长网页内容。
- 默认总结 3 至 5 条重点新闻；每条说明事件、意义和来源。


==================================================
工具调用规则
==================================================

涉及实时信息或外部系统数据时，
必须优先调用 MCP Tool。

网页搜索结果是候选来源。使用某个网页形成结论时，必须在对应事实后
附上工具结果给出的原始 URL，优先写成 [网页标题](https://example.com)。
不得用 [1] 网站名一类的纯编号来源列表代替链接；不要编造、截断 URL，
也不要引用未使用的候选。时效性结论应说明“截至检索时”。
来源相互矛盾或信息不足时，应明确说明不确定性。

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

#根据任务内容和是否需要联网搜索，决定本次 External Agent 要连接哪些 MCP 服务
def select_mcp_servers(
    task: str,
    need_web_search: bool = False,
) -> MCPSelection:
    normalized_task = task.lower()#统一转小写

    #判断是否需要天气/时间工具
    needs_external_info = any(
        keyword in normalized_task
        for keyword in EXTERNAL_INFO_KEYWORDS
    )
    #判断是否需要 GitHub 工具
    needs_github = any(
        keyword in normalized_task
        for keyword in GITHUB_KEYWORDS
    )

    #根据判断结果选择 MCP 服务
    if needs_external_info and needs_github and need_web_search:
        return "all"
    if needs_external_info and needs_github:
        return "both"
    if needs_external_info and need_web_search:
        return "external_info_and_web_search"
    if needs_github and need_web_search:
        return "github_and_web_search"
    if needs_github:
        return "github"
    if need_web_search:
        return "web_search"
    return "external_info"


def get_unavailable_message(
    selection: MCPSelection,
) -> str:
    if "web_search" in selection or selection == "all":
        return "暂时无法联网检索，请稍后重试。"
    if selection == "github":
        return (
            "暂时无法获取 GitHub 信息，"
            "请稍后重试。"
        )

    return (
        "暂时无法获取实时天气或时间信息，"
        "请稍后重试。"
    )


def get_selection_prompt(
    selection: MCPSelection,
    github_unavailable: bool = False,
    runtime_context: Dict[str, Any] | None = None,
) -> str:
    if selection == "github":
        prompt = (
            EXTERNAL_AGENT_SYSTEM_PROMPT
            + "\n\n本次请求仅加载了 GitHub 工具。"
        )
    else:
        prompt = (
            EXTERNAL_AGENT_SYSTEM_PROMPT
            + f"\n\n本次仅加载：{selection}。"
        )

    if github_unavailable:
        prompt += (
            "\n\nGitHub MCP 本次不可用。不要声称已读取 GitHub；"
            "若仍有其他可用工具，可仅基于其结果回答，并明确 GitHub 数据未能获取。"
        )

    if runtime_context:
        prompt += (
            "\n\n当前运行时北京时间（所有相对日期必须以此为准）：\n"
            f"当前时间：{runtime_context.get('current_datetime')}（{runtime_context.get('timezone')}，北京时间）\n"
            f"今天：{runtime_context.get('current_date')}\n"
            f"昨天：{runtime_context.get('yesterday_date')}\n"
            f"明天：{runtime_context.get('tomorrow_date')}\n"
            "涉及今天、昨天、明天、近期或最新的新闻，必须先调用 tavily_search；"
            "不得自行把上述日期判断为未来，也不得使用训练知识代替搜索。"
            "若没有可靠网页结果，明确说明“未检索到足够可靠的结果”。"
        )

    return prompt


def safe_exception_detail(
    exc: Exception,
) -> str:
    """Keep diagnostic logs useful without allowing credentials into logs."""

    detail = str(exc)
    detail = re.sub(
        r"(?i)bearer\s+[^\s,]+",
        "Bearer <redacted>",
        detail,
    )
    return detail[:500] or type(exc).__name__


async def discover_mcp_tools(
    stack: AsyncExitStack,
    server_name: str,
    client_factory,
    request_id: str = "unknown",
) -> List[Any]:
    """Connect and discover one MCP server, returning an empty list on failure."""

    try:
        adapter = await stack.enter_async_context(
            MCPAdapter(
                client_factory()
            )
        )
        tools = await adapter.list_tools(
            cache_mode="refresh"
        )
    except Exception as exc:
        logger.warning(
            "mcp_tools_discovered request_id=%s server=%s success=false error_category=%s error_type=%s error=%s",
            request_id,
            server_name,
            classify_mcp_error(exc),
            type(exc).__name__,
            safe_exception_detail(exc),
        )
        return []

    logger.warning(
        "mcp_tools_discovered request_id=%s server=%s success=true count=%d tools=%s",
        request_id,
        server_name,
        len(tools),
        ",".join(
            str(getattr(tool, "name", "unknown"))
            for tool in tools
        ),
    )

    return tools


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


def get_web_result_observability(
    messages: List[Any],
) -> tuple[int, List[int]]:
    """Return result counts and lengths only; never log tool payload content."""

    snippet_lengths: List[int] = []
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        for result in get_web_results_from_payload(message)[:MAX_WEB_RESULTS]:
            snippet = result.get("content")
            if isinstance(snippet, str):
                snippet_lengths.append(len(snippet))
    return len(snippet_lengths), snippet_lengths

#用来从 Agent 执行结束后的 messages 中，统计 Tavily 搜索是否真的被调用、返回了几条结果、有多少可用 URL，以及是否出现错误
def get_tavily_observability(messages: List[Any]) -> tuple[int, int, int, str]:
    """Return calls, results, URLs and safe failure category from final messages."""

    calls = 0#LLM 发起 tavily_search 的次数
    results = 0#Tavily 返回的网页结果总数
    urls = 0#Tavily 返回的网页结果中，有安全可用 URL 的数量
    failure_category = ""#Tavily 搜索的失败类别
    #遍历所有消息，统计 Tavily 搜索调用次数、结果数、URL 数量，并分类错误
    for message in messages:
        if isinstance(message, AIMessage):
            calls += sum(
                1 for call in getattr(message, "tool_calls", [])
                if isinstance(call, dict) and call.get("name") == "tavily_search"
            )
        #如果消息不是 ToolMessage，则跳过
        if not isinstance(message, ToolMessage):
            continue
        #如果消息的工具名称不是 tavily_search，则跳过
        tool_name = getattr(message, "name", "")
        if tool_name and tool_name != "tavily_search":
            continue
        #统计网页结果数和安全 URL 数
        count, url_count = count_web_results_from_payload(message)
        results += count
        urls += url_count
        #工具报错时归类失败原因
        if getattr(message, "status", "success") == "error":
            failure_category = classify_mcp_error(message.content)
    return calls, results, urls, failure_category


def compact_web_snippet(snippet: str) -> str:
    """Bound source-card snippets at a sentence boundary, not URLs/citations."""

    if len(snippet) <= MAX_WEB_SNIPPET_CHARS:
        return snippet

    boundary = max(
        snippet.rfind(marker, 0, MAX_WEB_SNIPPET_CHARS)
        for marker in ("。", "！", "？", ". ", "! ", "? ")
    )
    if boundary < MAX_WEB_SNIPPET_CHARS // 2:
        boundary = snippet.rfind(" ", 0, MAX_WEB_SNIPPET_CHARS)
    if boundary < MAX_WEB_SNIPPET_CHARS // 2:
        boundary = MAX_WEB_SNIPPET_CHARS
    return snippet[:boundary].rstrip() + "…"


async def run_external_agent(
    task: str,#Router 整理后的独立任务
    need_web_search: bool = False,#是否必须使用 Tavily 做公开网页搜索
    return_result: bool = False,#是否返回 ExternalAgentResult 对象而非最终文本回答
    runtime_context: Dict[str, Any] | None = None,#运行时上下文，包含当前时间、时区等信息
) -> str | ExternalAgentResult:
    #获取请求 ID，选择 MCP 组合
    request_id = (runtime_context or {}).get("request_id", "unknown")
    #根据任务内容和是否需要网络搜索，选择要使用的 MCP 工具组合
    selection = select_mcp_servers(
        task,
        need_web_search=need_web_search,
    )

    #定义一个内部函数，用于返回不可用的结果
    def unavailable_result(web_search_message: str = ""):
        result = ExternalAgentResult(
            answer=get_unavailable_message(selection),
            retrieved_web_sources=[],
            incomplete=False,
            web_search_status="failed" if need_web_search else "not_requested",
            web_search_message=web_search_message,
        )
        return result if return_result else result.answer

    #检查 Tavily 配置是否缺失，如果缺失则返回不可用结果
    if (
        selection in (#只有本次任务真的需要加载网页搜索 MCP 时，才检查
            "web_search",
            "external_info_and_web_search",
            "github_and_web_search",
            "all",
        )
        and (
            not TAVILY_MCP_URL
            or not TAVILY_API_KEY
        )
    ):
        logger.warning(
            "external_agent_started request_id=%s selection=%s need_web_search=true tavily_configured=false",
            request_id, selection,
        )
        return unavailable_result("联网搜索服务暂时不可用：配置缺失。")

    try:
        #使用 AsyncExitStack 管理异步上下文，发现并连接所需的 MCP 工具
        async with AsyncExitStack() as stack:
            #初始化工具列表
            tools = []#最终传给 Agent 的全部工具。
            github_tools: List[Any] = []#GitHub MCP 工具列表
            tavily_tools: List[Any] = []#Tavily MCP 工具列表
            github_unavailable = False#GitHub MCP 是否不可用
            logger.warning(
                "external_agent_started request_id=%s selection=%s need_web_search=%s",
                request_id, selection, need_web_search,
            )
            #根据选择结果，按需发现工具
            if selection in ("external_info", "external_info_and_web_search", "both", "all"):
                tools.extend(
                    #发现 External info MCP 工具(天气、时间等实时信息)
                    await discover_mcp_tools(
                        stack,
                        "External info",
                        create_external_info_client,
                        request_id,
                    )
                )
            #根据选择结果，按需发现 GitHub MCP 工具
            if selection in ("github", "github_and_web_search", "both", "all"):
                github_tools = await discover_mcp_tools(
                    stack,
                    "GitHub",
                    create_github_mcp_client,
                    request_id,
                )
                github_unavailable = not github_tools
                tools.extend(github_tools)
            #根据选择结果，按需发现 Tavily MCP 工具
            if selection in ("web_search", "external_info_and_web_search", "github_and_web_search", "all"):
                tavily_tools = await discover_mcp_tools(
                    stack, "Web search", create_web_search_client, request_id,
                )
                tools.extend(tavily_tools)

            #检查是否需要网络搜索但未发现可用的 Tavily 工具
            if need_web_search and not tavily_tools:
                return unavailable_result("联网搜索服务暂时不可用：未发现可用的搜索工具。")
            #检查是否没有任何可用的 MCP 工具
            if not tools:
                logger.warning(
                    "No MCP tools available for %s task",
                    selection,
                )
                return unavailable_result()
            #创建带工具的 External Agent
            agent = create_agent(
                model=get_chat_model(),
                tools=tools,
                system_prompt=get_selection_prompt(
                    selection,
                    github_unavailable=github_unavailable,
                    runtime_context=runtime_context,
                ),
                name="external_agent",
            )
            #调用 Agent 处理任务，并传入日志回调以记录工具调用情况
            result = await agent.ainvoke(
                {
                    "messages": [
                        HumanMessage(
                            content=task
                        )
                    ]
                },
                config={
                    "callbacks": [
                        MCPToolLoggingHandler(
                            request_id,
                            {
                                str(getattr(tool, "name", ""))
                                for tool in github_tools
                            },
                            {
                                str(getattr(tool, "name", ""))
                                for tool in tavily_tools
                            },
                        )
                    ]
                },
            )
            #从 Agent 的结果中提取最终的文本回答
            answer = get_latest_ai_content(
                result.get(
                    "messages",
                    [],
                )
            )
            #如果没有得到任何回答，则记录警告并返回不可用结果
            if not answer:
                logger.warning(
                    "External agent returned no answer "
                    "for %s task",
                    selection,
                )
                return unavailable_result()

            #获取最终的 AI 消息
            final_message = next(
                (
                    message
                    for message in reversed(result.get("messages", []))
                    if isinstance(message, AIMessage)
                ),
                None,
            )
            finish_reason = get_finish_reason(final_message)
            #获取 Tavily MCP 的可观测性数据
            tavily_calls, tavily_results, tavily_urls, tavily_failure = get_tavily_observability(
                result.get("messages", [])
            )
            #获取网络结果的可观测性数据
            result_count, snippet_lengths = get_web_result_observability(
                result.get("messages", [])
            )
            #记录外部代理完成的日志，包括请求 ID、模型、输入消息数量、字符数、网络结果数量、摘要长度、输出字符数、完成原因以及 Tavily 调用情况
            logger.warning(
                "external_agent_completed request_id=%s model=%s input_messages=%d "
                "input_chars=%d web_results=%d web_snippet_lengths=%s "
                "web_context_chars=%d output_chars=%d finish_reason=%s tavily_calls=%d tavily_urls=%d tavily_failure=%s",
                request_id,
                LLM_MODEL,
                1,
                len(task),
                result_count,
                snippet_lengths,
                sum(snippet_lengths),
                len(answer),
                finish_reason or "unknown",
                tavily_calls,
                tavily_urls,
                tavily_failure or "none",
            )
            #从结果中提取网页来源
            web_sources = extract_web_sources(result.get("messages", []))
            #判断“联网搜索是否真的成功”
            web_search_status = "not_requested"
            web_search_message = ""
            if need_web_search:
                if tavily_failure:
                    web_search_status = "failed"
                    web_search_message = "联网搜索暂时不可用：搜索请求失败。"
                elif not tavily_calls:
                    web_search_status = "failed"
                    web_search_message = "联网搜索暂时不可用：未完成搜索调用。"
                elif not tavily_urls:
                    web_search_status = "no_results"
                    web_search_message = "未检索到足够可靠的结果。"
                else:
                    web_search_status = "succeeded"
            #根据是否需要网络搜索以及搜索状态，构建 ExternalAgentResult 对象并返回
            external_result = ExternalAgentResult(
                answer=(web_search_message if need_web_search and web_search_status != "succeeded" else answer),
                retrieved_web_sources=web_sources,
                incomplete=is_length_limited(final_message),
                web_search_status=web_search_status,
                web_search_message=web_search_message,
                web_result_count=tavily_results,
                web_url_count=tavily_urls,
            )
            return external_result if return_result else answer

    except Exception as exc:
        logger.warning(
            "external_agent_completed request_id=%s selection=%s success=false error_category=%s error_type=%s error=%s",
            request_id,
            selection,
            classify_mcp_error(exc),
            type(exc).__name__,
            safe_exception_detail(exc),
        )
        return unavailable_result("联网搜索服务暂时不可用：请求失败。")

#从 External Agent 的所有工具返回消息中，提取真正的网页搜索结果，并转换成系统统一的“来源对象”格式
def extract_web_sources(messages: List[Any]) -> List[Dict[str, Any]]:
    """Extract web candidates from MCP ToolMessages without exposing them yet."""
    #初始化网页来源列表和已见 URL 集合
    sources: List[Dict[str, Any]] = []
    seen_urls = set()
    #只处理工具返回消息（message中可能包含其他类型的消息）
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        #统一解析网页结果
        for result in get_web_results_from_payload(message):
            url = get_safe_web_url(#原始网页链接
                result.get("url")
            )
            title = result.get("title")#网页标题
            snippet = result.get("content")#网页摘要内容
            source_number = result.get("source_number")#来源编号
            #过滤掉无效的 URL 或重复的 URL
            if not isinstance(url, str):
                continue
            #过滤掉已经处理过的 URL，避免重复来源
            if url in seen_urls:
                continue
            seen_urls.add(url)
            #如果来源编号不是整数，则使用当前来源列表长度 + 1 作为编号
            if not isinstance(source_number, int):
                source_number = len(sources) + 1
            #对标题和摘要进行规范化处理，确保它们是字符串类型
            normalized_title = (
                title.strip()
                if isinstance(title, str)
                else ""
            )
            normalized_snippet = (
                snippet
                if isinstance(snippet, str)
                else ""
            )
            #将网页来源信息添加到 sources 列表中，包含编号、引用格式、类型、内容等字段
            sources.append({
                "source_number": source_number,
                "citation": f"[{source_number}]",
                "source_type": "web",
                "chunk_id": zlib.crc32(url.encode("utf-8")),
                "document_id": 0,
                "filename": normalized_title or url,
                "chunk_index": 0,
                "page_number": None,
                "content": compact_web_snippet(
                    normalized_snippet
                ),
                "metadata_json": {},
                "distance": None,
                "similarity": None,
                "keyword_score": None,
                "vector_rank": None,
                "keyword_rank": None,
                "rrf_score": None,
                "url": url,
                "title": normalized_title or None,
                "published_date": result.get("published_date"),
            })

    return sources
