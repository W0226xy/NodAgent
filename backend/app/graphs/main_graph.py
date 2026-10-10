import asyncio
import logging
import re
from functools import lru_cache
from typing import (
    Annotated,
    Any,
    Dict,
    List,
    Literal,
    Optional,
)

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import (
    END,
    START,
    StateGraph,
    add_messages,
)
from langgraph.runtime import Runtime
from langgraph.types import (
    interrupt,
)
from pydantic import (
    BaseModel,
    Field,
)
from typing_extensions import TypedDict

from backend.app.agents.context import (
    AgentContext,
)
from backend.app.agents.document_agent import (
    get_document_agent,
)
from backend.app.agents.knowledge_agent import (
    get_knowledge_agent,#
)
from backend.app.agents.memory_agent import (
    get_memory_agent,
)
from backend.app.agents.external_agent import (
    run_external_agent,
)
from backend.app.services import (
    citation_service,
    document_action_service,
)
from backend.app.services.document_action_service import (
    DocumentActionError,
)
from backend.app.services.llm_service import (
    get_chat_model,
    get_finish_reason,
    is_length_limited,
)
from backend.app.core.config import (
    LLM_MODEL,
)


logger = logging.getLogger(__name__)


#这份代码是 NodAgent 的主 LangGraph 工作流。
#它决定一条用户消息要交给哪个专业 Agent 处理，并最终由 Chat Agent 整理为面向用户的回答。
#用户消息
  # → Router 判断意图
  # → 专业节点处理
  #    ├─ 知识库问答
  #    ├─ 文档管理查询
  #    ├─ 长期记忆管理
  #    ├─ 外部 MCP / 联网搜索
  #    └─ 删除文档（需人工确认）
  # → Chat Agent 生成最终回答
  # → 保存 checkpoint

EXTERNAL_REQUEST_MARKERS = (
    "天气",
    "气温",
    "温度",
    "几点",
    "时间",
    "github",
)

WEB_SEARCH_MARKERS = (
    "搜索", "查一下", "联网", "找资料", "给链接", "核实", "验证",
    "最新", "最近", "新闻", "实时价格", "汇率", "股价", "比赛结果",
    "航班", "verify", "search", "latest", "news",
)

KNOWLEDGE_CONTEXT_MARKERS = (
    "知识库",
    "文档",
    "上传文件",
    "workspace",
    "项目资料",
)

KNOWLEDGE_FACT_MARKERS = (
    "在哪里",
    "位于",
    "地址",
    "容纳",
    "容量",
    "多少人",
    "located",
    "where is",
    "how many",
)

# 防止“文档事实问题”误走外部服务。文档/知识库本身只是上下文，
# 还必须同时出现事实型问法；否则“删除这个文档”等操作会被误判。
def should_route_fact_question_to_knowledge(
    question: str,#用户问题
) -> bool:
    """Keep stable facts out of the realtime/external path.

    The LLM router occasionally treated any "where" question as a request
    for a geolocation service.  Uploaded-document questions use the same
    wording, so protect that class of fact questions before graph routing.
    """

    normalized = question.lower()#问题转小写

    if any(#先检查问题里是否出现了明确属于外部实时服务的关键词
        marker in normalized
        for marker in EXTERNAL_REQUEST_MARKERS
    ):
        return False

    has_knowledge_context = any(
        marker in normalized
        for marker in KNOWLEDGE_CONTEXT_MARKERS
    )
    has_fact_question = any(#检查是否包含知识库事实关键词
        marker in normalized
        for marker in KNOWLEDGE_FACT_MARKERS
    )
    return has_knowledge_context and has_fact_question

#用户明确说“搜索、新闻、最新”时，强制联网
def should_request_web_search(question: str) -> bool:
    normalized = question.lower()
    if any(marker in normalized for marker in EXTERNAL_REQUEST_MARKERS):
        return False
    return any(marker in normalized for marker in WEB_SEARCH_MARKERS)

#
def shift_citations(text: str, offset: int) -> str:
    return re.sub(
        r"\[(\d+)\]",
        lambda match: f"[{int(match.group(1)) + offset}]",
        text,
    ) if offset > 0 else text


def shift_web_sources(sources: List[Dict[str, Any]], offset: int) -> List[Dict[str, Any]]:
    shifted_sources = []
    for source in sources:
        shifted = dict(source)
        source_number = int(shifted["source_number"]) + offset
        shifted["source_number"] = source_number
        shifted["citation"] = f"[{source_number}]"
        shifted_sources.append(shifted)
    return shifted_sources


RouteName = Literal[
    "direct",
    "knowledge",
    "document",
    "memory",
    "external",
    "delete_document",
]


def finalize_router_route(
    route: RouteName,#Router LLM 初步判断的路由
    question: str,#用户问题
    need_web_search: bool,#Router LLM 初步判断是否需要联网搜索
) -> tuple[RouteName, bool, Optional[str]]:
    """Apply deterministic guards without overriding explicit operations.

    The model owns semantic classification. These guards only correct document
    fact questions sent to direct/external and explicit web requests sent to
    direct. Document, memory, and delete routes remain higher-priority actions.
    """
    #判断用户是否明确要求联网搜索
    explicit_web_search = should_request_web_search(question)
    #判断是否是知识库事实问题
    is_knowledge_fact_question = should_route_fact_question_to_knowledge(question)
    override_reason = None
    #如果路由是 direct 或 external，并且是知识库事实问题，则强制改为 knowledge 路由
    if route in ("direct", "external") and is_knowledge_fact_question:
        route = "knowledge"
        override_reason = "knowledge_fact_guard"
    #如果路由是 direct，并且用户明确要求联网搜索，则强制改为 external 路由
    elif route in ("direct", "external") and need_web_search:
        route = "external"
        override_reason = "explicit_web_search"

    # Explicit operation routes do not execute public searches. Knowledge
    # questions can request web supplementation, but a plain Workspace fact
    # question must not trigger an unnecessary public search.
    if route in ("document", "memory", "delete_document"):
        need_web_search = False
    if route == "knowledge" and not explicit_web_search:
        need_web_search = False

    return route, need_web_search, override_reason

#整条工作流共享的状态
class MainGraphState(
    TypedDict,
    total=False,
):
    #完整对话消息，使用 add_messages 自动追加
    messages: Annotated[
        List[AnyMessage],
        add_messages,
    ]
    #Router 最终决定走哪条路径
    route: RouteName
    #Router 传给专业节点的任务描述，必须是完整、独立可理解的任务
    specialist_task: str
    #专业 Agent 的返回结果
    specialist_answer: str
    #结果来自哪个专业 Agent
    specialist_agent: str
    #是否需要进行网页搜索
    need_web_search: bool
    #知识库的回答
    knowledge_answer: str

    sources: List[
        Dict[str, Any]
    ]
    retrieved_sources: List[
        Dict[str, Any]
    ]

    cited_sources: List[
        Dict[str, Any]
    ]

    documents: List[
        Dict[str, Any]
    ]

    action_document_id: Optional[
        int
    ]

    action_document_filename: Optional[
        str
    ]

    action_ready: bool

    action_approved: bool

    pending_action: Dict[
        str,
        Any,
    ]

    answer_incomplete: bool

    external_status: Dict[str, Any]

#Router：先让模型判断，再用规则兜底
#Router 不能随意输出一段自然语言，而必须按规定返回结构化结果
#比如：
# {
#   "route": "delete_document",
#   "task": "删除 report.pdf。",
#   "need_web_search": false,
#   "document_id": null,
#   "document_filename": "report.pdf"
# }
class RouteDecision(BaseModel):
    route: RouteName = Field(
        description=(
            "Destination for the "
            "current request."
        )
    )

    task: str = Field(
        description=(
            "Standalone task with "
            "references resolved from "
            "conversation history."
        )
    )

    need_web_search: bool = Field(
        default=False,
        description="Whether public web search is needed for this request.",
    )

    web_search_reason: Optional[str] = Field(
        default=None,
        description="Brief safe reason for web-search routing.",
    )

    document_id: Optional[int] = Field(
        default=None,
        description=(
            "Document ID when the "
            "request targets a specific "
            "document. Do not invent "
            "an ID."
        ),
    )

    document_filename: Optional[str] = Field(
        default=None,
        description=(
            "Exact filename when a delete "
            "request identifies a document "
            "by name rather than ID. "
            "Do not invent a filename."
        ),
    )


class DeleteApproval(BaseModel):
    decision: Literal[
        "approve",
        "reject",
    ]


ROUTER_SYSTEM_PROMPT = """
你是 NodAgent 的 Router。

你不负责回答问题。

你只负责根据当前问题和完整聊天历史，
决定应该进入哪个处理节点。

只能选择：

direct
knowledge
document
memory
external
delete_document

==================================================
direct
==================================================

普通聊天、通用知识问题，
以及已经能够直接从长期记忆中回答的问题。

例如：

Python 字典是什么？
什么是二分查找？
FastAPI Depends 是什么？

我之前对代码修改有什么要求？
当前 Workspace 主要是什么项目？

如果当前用户或 Workspace 的长期记忆
已经足以回答问题，应选择 direct，
不要为了重复确认而进入 knowledge。

==================================================
knowledge
==================================================

只有当回答确实需要读取 Workspace
中的文档正文、Chunk 或知识库证据时，
才选择 knowledge。

例如：

这篇论文主要讲了什么？
论文有什么创新？
总结知识库里的文档。
根据我的简历介绍项目。

如果长期记忆已经包含足够的信息，
不要进入 knowledge。

==================================================
document
==================================================

查询文档管理信息。

例如：

我有哪些文档？
哪些文档处理完成？
7 号文档状态是什么？
某个文档有多少 Chunk？

==================================================
memory
==================================================

用户明确要求管理长期记忆时选择 memory。

包括：

记住以后代码修改给完整文件。
以后回答尽量一步一步讲。
把这个偏好保存下来。
我有哪些长期记忆？
你现在记得我什么？
把代码修改偏好改成只给关键修改。
忘掉我的代码修改偏好。
删除这个长期记忆。

只是在询问已经存在的某条长期信息，
并且当前长期记忆已经足够回答时，
可以选择 direct。

只要用户是在保存、更新、列出或删除长期记忆，
必须选择 memory。

==================================================
external
==================================================

需要访问外部系统或实时外部信息时选择 external。

例如：

东京现在天气怎么样？
上海今天气温多少？
纽约现在几点？
东京当前时间是多少？

查看 GitHub 某个 Repository。
读取 GitHub Repository 的 README。
查看 GitHub Issue。
查看 GitHub Pull Request。
读取 GitHub Repository 当前文件内容。

这类问题依赖实时数据或外部系统，
不能只依赖模型训练知识回答，
应交给 External Agent 调用 MCP Tool。

如果问题只是通用知识，
不需要实时外部数据，
选择 direct。

联网搜索判断：用户明确要求搜索、联网、链接、核实，或问题需要最新
新闻、价格、汇率、比赛、航班、近期产品/API/政策版本时，选择 external
并将 need_web_search 设为 true。稳定概念、代码原理、翻译、写作和头脑风暴
选择 direct。天气、时间和 GitHub 请求使用专用 MCP，need_web_search 为 false。
涉及我的文档、知识库、上传文件、项目资料或 Workspace 文档事实时优先
knowledge；除非用户明确要求网上核实或补充外部资料，否则不得绕过知识库。
两者都需要时选择 knowledge 并设置 need_web_search 为 true。

==================================================
delete_document
==================================================

用户明确要求删除某个文档。

例如：

删除 7 号文档。
把刚才那个文件删掉。
删除 document_id=12。
删除 CET6_202209_322110221202516_1.pdf。

如果用户在删除对话中补充文件名、
回答候选编号，或说“确认删除”，
仍应根据聊天历史进入 delete_document。
聊天里的“确认”不能代替人工确认步骤。

删除属于有副作用的高风险操作，
必须进入 delete_document，
不能进入普通 document。

==================================================

如果用户说：

它
这个
这个文件
刚才那个文档
这篇论文

必须结合聊天历史确定具体对象。

如果能够从历史明确得到 document_id，
请写入 document_id。

如果只知道文档的完整文件名，
请把不带引号和路径的名称写入 document_filename，
document_id 保持 null。
文件名是否存在由后续节点查询，
不要因为不知道 ID 就拒绝删除。

如果无法可靠确定 document_id：

document_id 必须为 null。

绝对不要猜 document_id。
也不要猜 document_filename。

task 必须是一个完整、独立可理解的任务。

例如：

用户：
“把它删掉”

历史中明确知道它是 document_id=7。

则：

route = delete_document
document_id = 7
task = 删除 document_id=7 的文档。

用户：
“删除 report.pdf”

则：

route = delete_document
document_id = null
document_filename = report.pdf
task = 删除 report.pdf。
""".strip()

#时间问题注入运行时日期
def get_runtime_time_prompt(context: AgentContext) -> str:
    return (
        "当前运行时北京时间（所有相对日期必须以此为准）：\n"
        f"当前时间：{context.get('current_datetime')}（{context.get('timezone')}，北京时间）\n"
        f"今天：{context.get('current_date')}\n"
        f"昨天：{context.get('yesterday_date')}\n"
        f"明天：{context.get('tomorrow_date')}\n"
        "“今天、昨天、明天、近期、最新”等相对时间必须以以上信息为准。"
        "新闻问题必须路由到 external 并设置 need_web_search=true，不能因模型自行判断日期未来而拒绝搜索。"
    )


CHAT_SYSTEM_PROMPT = """
你是 NodAgent 最终负责与用户交流的 Chat Agent。

你的职责：

1. 根据用户问题、聊天历史以及专业节点结果，
   生成最终面向用户的回答。

2. 如果问题涉及 Workspace 文档正文、
   论文、简历或知识库证据，
   必须以专业节点结果为准。

   如果问题只是询问长期记忆中已经明确保存的
   用户偏好或 Workspace 高层业务信息，
   可以直接依据长期记忆回答，
   不要求再次查询知识库。

3. 本地知识库的 [1]、[2] 等引用编号必须原样保留，不能新增、删除或改写。
   网页资料必须保留已有的 Markdown 原始链接；不要把网页来源写成
   “### 参考来源\n[1] 网站名”这类不能点击的编号列表。

4. 如果操作节点告诉你操作成功，
   明确说明操作已经完成。

5. 如果操作被用户取消，
   明确告诉用户操作已经取消。

6. 如果操作无法执行，
   只根据专业节点给出的原因进行说明。
   不要猜测文件名、路径或文档是否存在。
   不要声称已经检索过没有实际查询的记录。

7. 系统支持删除文档，但执行前需要用户确认。
   不要因为 Document Agent 只有查询工具，
   就告诉用户整个系统无法删除文档。

8. 不要告诉用户内部存在 Router、
   StateGraph、节点、ToolRuntime 等实现细节。

9. direct 路由没有 specialist_answer 时，
   直接根据聊天历史回答用户。

10. 对联网资讯默认提炼 3 至 5 条重点；每条说明事件、意义和来源。
    每条用 1 至 2 句，总正文控制在约 1,200 个中文字符内。
    优先保证结论完整收束，不要因罗列过多细节而中断在半句。
""".strip()


INCOMPLETE_ANSWER_SUFFIX = (
    "\n\n回答因输出长度限制未完成，请回复“继续”。"
)

# 获取路由模型
#会返回一个 ChatOpenAI 对象，带有结构化输出 RouteDecision 的功能
#类似：
#RouteDecision(
#     route="external",
#     task="搜索并总结今天苏州地区与人工智能相关的新闻，提供原始网页链接。",
#     need_web_search=True,
#     document_id=None,
#     document_filename=None
# )
@lru_cache(maxsize=1)
def get_router_model():
    return (
        get_chat_model()
        .with_structured_output(
            RouteDecision,
            method="function_calling",
        )
    )


def get_message_text(
    message: Any,
) -> str:
    content = getattr(
        message,
        "content",
        "",
    )

    if isinstance(
        content,
        str,
    ):
        return content

    if isinstance(
        content,
        list,
    ):
        text_parts: List[str] = []

        for block in content:
            if isinstance(
                block,
                str,
            ):
                text_parts.append(
                    block
                )

            elif isinstance(
                block,
                dict,
            ):
                text = block.get(
                    "text"
                )

                if text:
                    text_parts.append(
                        str(text)
                    )

        return "".join(
            text_parts
        )

    return ""


def get_latest_user_message(
    messages: List[AnyMessage],
) -> str:
    for message in reversed(
        messages
    ):
        if isinstance(
            message,
            HumanMessage,
        ):
            return get_message_text(
                message
            )

    return ""


def get_latest_ai_answer(
    messages: List[Any],
) -> str:
    for message in reversed(
        messages
    ):
        if not isinstance(
            message,
            AIMessage,
        ):
            continue

        text = get_message_text(
            message
        )

        if text:
            return text

    return ""


def extract_sources(
    messages: List[Any],
) -> List[
    Dict[str, Any]
]:
    sources_by_chunk: Dict[
        int,
        Dict[str, Any],
    ] = {}

    for message in messages:
        if not isinstance(
            message,
            ToolMessage,
        ):
            continue

        artifact = (
            message.artifact
        )

        if not isinstance(
            artifact,
            dict,
        ):
            continue

        raw_sources = artifact.get(
            "sources",
            [],
        )

        if not isinstance(
            raw_sources,
            list,
        ):
            continue

        for raw_source in raw_sources:
            if not isinstance(
                raw_source,
                dict,
            ):
                continue

            chunk_id = raw_source.get(
                "chunk_id"
            )

            if chunk_id is None:
                continue

            sources_by_chunk[
                int(chunk_id)
            ] = raw_source

    return list(
        sources_by_chunk.values()
    )


def extract_documents(
    messages: List[Any],
) -> List[
    Dict[str, Any]
]:
    documents_by_id: Dict[
        int,
        Dict[str, Any],
    ] = {}

    for message in messages:
        if not isinstance(
            message,
            ToolMessage,
        ):
            continue

        artifact = (
            message.artifact
        )

        if not isinstance(
            artifact,
            dict,
        ):
            continue

        raw_documents = artifact.get(
            "documents",
            [],
        )

        if not isinstance(
            raw_documents,
            list,
        ):
            continue

        for raw_document in (
            raw_documents
        ):
            if not isinstance(
                raw_document,
                dict,
            ):
                continue

            document_id = (
                raw_document.get(
                    "id"
                )
            )

            if document_id is None:
                continue

            documents_by_id[
                int(document_id)
            ] = raw_document

    return list(
        documents_by_id.values()
    )

#主工作流的第一个节点。它的任务不是回答用户，而是：
# 1. 读取聊天记录、长期记忆和当前时间；
# 2. 调用 LLM Router，得到结构化的路由决定；
# 3. 用代码规则修正模型可能做错的判断；
# 4. 将路由结果和本轮初始状态写回 MainGraphState
async def router_node(
    state: MainGraphState,#当前工作流状态，state预期收到一个MainGraphState 类型的对象
    runtime: Runtime[#运行时上下文
        AgentContext
    ],
):
    #从共享状态中取出完整聊天记录
    messages = state.get(
        "messages",#若 messages 存在，返回它
        [],#若 messages 不存在，返回空列表
    )
    #构造 Router 的基础提示词（定义六种路由什么情况选择+当时的北京时间）
    router_system_prompt = (
        ROUTER_SYSTEM_PROMPT + "\n\n" + get_runtime_time_prompt(runtime.context)
    )
    #从运行时上下文中取出长期记忆提示词（如果存在）
    memory_prompt = (
        runtime.context.get(
            "memory_prompt",
            "",
        )
    )

    if memory_prompt:
        router_system_prompt += (
            "\n\n"
            "==================================================\n"
            "当前可用长期记忆\n"
            "==================================================\n"
            f"{memory_prompt}\n\n"
            "这些长期记忆已经可以作为 Router "
            "判断路由时的上下文。"
            "如果长期记忆本身足以回答用户问题，"
            "请选择 direct。"
            "如果用户要保存、更新、列出或删除长期记忆，"
            "请选择 memory。"
            "只有确实需要读取 Workspace 文档正文时，"
            "才选择 knowledge。"
        )

    #调用 Router 模型，得到路由决策
    #decision 是 RouteDecision 对象，包含 route、task、need_web_search、document_id、document_filename 等属性
    decision = await (#await 表示等待异步 LLM 调用返回
        get_router_model().ainvoke(
            [
                SystemMessage(
                    content=(
                        router_system_prompt
                        #包括SystemMessage：路由规则 + 当前时间 + 长期记忆
                        #HumanMessage / AIMessage：完整聊天历史
                    )
                ),
                *messages,#messages 是 Python 的列表解包，相当于把 messages 列表中的每个消息对象逐个放进新列表。
            ]
        )
    )
    #从历史消息中倒序找到最新一条 HumanMessage 的文本
    latest_question = get_latest_user_message(messages)

    model_route = decision.route
    route = model_route
    task = decision.task
    need_web_search = (#LLM判断或者关键词命中，只要一个true就是需要联网搜索
        decision.need_web_search
        or should_request_web_search(latest_question)
    )

    route, need_web_search, override_reason = finalize_router_route(
        route,
        latest_question,
        need_web_search,
    )
    #路由是 direct 或 external，并且是知识库事实问题，则强制改为 knowledge 路由
    if override_reason == "knowledge_fact_guard":
        # The raw question is a safer standalone task than a task generated
        # for the incorrect direct/external route.
        route = "knowledge"
        task = latest_question

    logger.warning(
        "router_decision request_id=%s model_route=%s final_route=%s "
        "need_web_search=%s override_reason=%s web_search_reason=%r "
        "specialist_task=%r",
        runtime.context.get("request_id", "unknown"),
        model_route,
        route,
        need_web_search,
        override_reason,
        decision.web_search_reason,
        task[:500],
    )

    return {
        "route": route,

        "specialist_task": (
            task
        ),

        "action_document_id": (
            decision.document_id
        ),

        "action_document_filename": (
            decision.document_filename
        ),

        "specialist_answer": "",

        "specialist_agent": "",

        "need_web_search": need_web_search,

        "knowledge_answer": "",

        "sources": [],

        "retrieved_sources": [],

        "cited_sources": [],

        "answer_incomplete": False,

        "documents": [],

        "action_ready": False,

        "action_approved": False,

        "pending_action": {},
        "external_status": {},
    }

#LangGraph 的条件路由函数：从当前工作流状态中读出 Router 已经决定好的路由名称，交给图框架选择下一个节点。
def route_after_router(
    state: MainGraphState,
) -> RouteName:
    return state["route"]#直接取出 "route" 对应的值并返回。
    # 比如，如果 state["route"] 是 "knowledge"，那么函数就返回 "knowledge"。


def route_after_knowledge(
    state: MainGraphState,
) -> Literal["external", "chat"]:
    return "external" if state.get("need_web_search", False) else "chat"

#将 specialist_task 交给 Knowledge Agent
#1.取出知识 Agent 最后的 AI 回复；
#2.从工具消息的 artifact["sources"] 中提取检索来源；
#3.检查最终回答是否真的引用了这些来源。
async def knowledge_node(#知识库问答节点
    state: MainGraphState,
    runtime: Runtime[
        AgentContext
    ],
):
    task = state.get(
        "specialist_task",#这里的具体任务是 Router 节点生成的、完整可理解的任务描述
        "",
    )

    #如果 specialist_task 为空，则从聊天记录中取出最新一条用户消息作为任务
    if not task:
        task = (
            get_latest_user_message(
                state.get(
                    "messages",
                    [],
                )
            )
        )
    #调用知识库专业 Agent，传入任务和运行时上下文，得到结果
    result = await (
        get_knowledge_agent()
        .ainvoke(
            {
                "messages": [
                    HumanMessage(
                        content=task
                    )
                ]
            },
            context=runtime.context,
        )
    )
    #从结果中提取消息
    messages = result.get(
        "messages",
        [],
    )
    #从消息中取出最新的 AI 回复作为最终回答
    answer = (
        get_latest_ai_answer(
            messages
        )
    )
    #从消息中提取检索来源
    retrieved_sources = extract_sources(
        messages
    )
    #检查最终回答是否真的引用了这些来源
    cited_sources = (
        citation_service.filter_cited_sources(
            answer,
            retrieved_sources,
        )
        if answer
        else []
    )

    if not cited_sources:
        logger.warning(
            "Knowledge agent returned no usable citations; refusing "
            "to present an unsupported document answer."
        )
        answer = (
            "当前知识库未返回可引用的内容，"
            "暂时无法基于文档回答。"
        )
    elif not answer:
        answer = (
            "知识库专业 Agent "
            "没有生成有效回答。"
        )

    return {#将结果写回主图 State
        "specialist_agent": (
            "knowledge"
        ),
        "specialist_answer": answer,
        "knowledge_answer": answer,
        # The retrieved set remains available to the final Chat Agent, while
        # sources/cited_sources are safe to expose outside the graph.
        "retrieved_sources": retrieved_sources,
        "cited_sources": cited_sources,
        "sources": cited_sources,
        "documents": [],
    }

#文档管理查询
#调用 Document Agent，适合处理：
# - 有哪些文件；
# - 某文档状态；
# - 文档处理是否完成；
# - 某文档被切成多少个 Chunk。
async def document_node(
    state: MainGraphState,
    runtime: Runtime[
        AgentContext
    ],
):
    #specialist_task 通常由前面的 router_node 生成
    task = state.get(
        "specialist_task",
        "",
    )
    #如果 specialist_task 为空，则从聊天记录中取出最新一条用户消息作为任务
    if not task:
        task = (
            get_latest_user_message(
                state.get(
                    "messages",
                    [],
                )
            )
        )
    #调用文档专业 Agent，传入任务和运行时上下文，得到结果
    result = await (
        get_document_agent()#这里会额外提供2个工具：list_workspace_documents，get_document_details。方便文档查询
        .ainvoke(
            {
                "messages": [
                    HumanMessage(
                        content=task
                    )
                ]
            },
            context=runtime.context,
        )
    )
    #从结果中提取消息
    messages = result.get(
        "messages",
        [],
    )
    #从消息中取出最新的 AI 回复作为最终回答
    answer = (
        get_latest_ai_answer(
            messages
        )
    )
    #从消息中提取文档信息
    documents = extract_documents(
        messages
    )

    if not answer:
        answer = (
            "文档专业 Agent "
            "没有生成有效回答。"
        )

    return {
        "specialist_agent": (
            "document"
        ),
        "specialist_answer": answer,
        "sources": [],
        "documents": documents,
    }

#长期记忆管理
#调用 Memory Agent，用于处理：
# - “记住我希望分步骤讲解”
# - “以后只给关键修改”
# - “你记得我什么？”
# - “忘掉某条偏好”
async def memory_node(
    state: MainGraphState,
    runtime: Runtime[
        AgentContext
    ],
):
    #specialist_task 通常由前面的 router_node 生成
    task = state.get(
        "specialist_task",
        "",
    )
    #如果 specialist_task 为空，则从聊天记录中取出最新一条用户消息作为任务
    if not task:
        task = (
            get_latest_user_message(
                state.get(
                    "messages",
                    [],
                )
            )
        )
    #调用长期记忆专业 Agent，传入任务和运行时上下文，得到结果
    result = await (
        get_memory_agent()#额外提供save_memory、list_memories、delete_memory等工具，方便长期记忆管理
        .ainvoke(
            {
                "messages": [
                    HumanMessage(
                        content=task
                    )
                ]
            },
            context=runtime.context,
        )
    )

    messages = result.get(
        "messages",
        [],
    )

    answer = (
        get_latest_ai_answer(
            messages
        )
    )

    if not answer:
        answer = (
            "长期记忆 Agent "
            "没有生成有效回答。"
        )

    return {
        "specialist_agent": (
            "memory"
        ),
        "specialist_answer": answer,
        "sources": [],
        "documents": [],
    }

#外部 MCP 与联网搜索
#负责：
# - 天气、时间；
# - GitHub 仓库、Issue、PR、文件；
# - Tavily 等联网搜索；
# - 最新新闻、价格、API 版本等。
#如果一个问题既需要知识库内容，又需要网络补充，会先执行
#Knowledge Agent → External Agent → Chat Agent
async def external_node(
    state: MainGraphState,
    runtime: Runtime[AgentContext],
):
    #获取待执行任务
    task = state.get(
        "specialist_task",
        "",
    )
    #如果为空，则回退到用户最新消息：
    if not task:
        task = (
            get_latest_user_message(
                state.get(
                    "messages",
                    [],
                )
            )
        )
    #调用 External Agent，主要负责调用外部 MCP 工具，比如Github Mcp、Tavily MCP
    result = await (
        run_external_agent(
            task,
            need_web_search=state.get("need_web_search", False),
            return_result=True,
            runtime_context=runtime.context,
        )
    )

    #情况1：只需要外部信息，不需要查知识库
    #情况2：本地知识库 + 网络补充
    #读取前面知识库节点留下的内容（比如用户问：总结我的项目，并联网补充相关技术的最新趋势）
    knowledge_answer = state.get("knowledge_answer", "")
    retrieved_sources = state.get("retrieved_sources", [])
    web_sources = result.retrieved_web_sources

    #如果知识库引用和联网搜索的引用出现序号冲突，则优先展示本地文档，联网的引用序号往后移
    if knowledge_answer and web_sources:
        max_document_number = max(
            (int(source["source_number"]) for source in retrieved_sources),
            default=0,
        )
        web_sources = shift_web_sources(web_sources, max_document_number)
        answer = (
            "知识库内容：\n" f"{knowledge_answer}\n\n"
            "网络补充内容（截至检索时）：\n"
            f"{shift_citations(result.answer, max_document_number)}"
        )
    elif knowledge_answer:
        answer = f"知识库内容：\n{knowledge_answer}\n\n网络补充内容：\n{result.answer}"
    else:
        answer = result.answer

    return {
        "specialist_agent": (
            "knowledge_and_web" if knowledge_answer else "external"
        ),
        "specialist_answer": answer,
        "retrieved_sources": retrieved_sources + web_sources,
        "cited_sources": [],
        "sources": [],
        "documents": [],
        "answer_incomplete": result.incomplete,
        "external_status": {
            "state": result.web_search_status,
            "message": result.web_search_message,
            "result_count": result.web_result_count,
            "url_count": result.web_url_count,
        },
    }

#文档删除准备
#prepare_delete → confirm_delete → execute_delete → chat
async def prepare_delete_node(
    state: MainGraphState,
    runtime: Runtime[
        AgentContext
    ],
):
    document_id = state.get(
        "action_document_id"
    )

    document_filename = state.get(
        "action_document_filename"
    )

    workspace_id = (
        runtime.context[
            "workspace_id"
        ]
    )

    if document_id is not None:
        try:
            document = await asyncio.to_thread(
                (
                    document_action_service
                    .get_document_snapshot
                ),
                workspace_id,
                document_id,
            )

        except DocumentActionError as exc:
            return {
                "action_ready": False,
                "specialist_agent": (
                    "document_action"
                ),
                "specialist_answer": (
                    f"无法准备删除文档："
                    f"{exc}"
                ),
            }

    elif document_filename:
        matches = await asyncio.to_thread(
            (
                document_action_service
                .find_document_snapshots_by_filename
            ),
            workspace_id,
            document_filename,
        )

        if not matches:
            return {
                "action_ready": False,
                "specialist_agent": (
                    "document_action"
                ),
                "specialist_answer": (
                    "当前 Workspace 没有找到"
                    f"文件名为 {document_filename} "
                    "的文档。"
                ),
            }

        if len(matches) > 1:
            document_ids = ", ".join(
                str(document["id"])
                for document in matches
            )

            return {
                "action_ready": False,
                "specialist_agent": (
                    "document_action"
                ),
                "specialist_answer": (
                    f"文件名 {document_filename} "
                    "对应多个文档，ID 分别为 "
                    f"{document_ids}。"
                    "请指定要删除的文档 ID。"
                ),
            }

        document = matches[0]
        document_id = document["id"]

    else:
        return {
            "action_ready": False,
            "specialist_agent": (
                "document_action"
            ),
            "specialist_answer": (
                "无法确定需要删除的文档。"
                "请提供文档 ID 或完整文件名。"
            ),
        }

    if (
        document_filename
        and document["filename"]
        != (
            document_action_service
            .normalize_document_filename(
                document_filename
            )
        )
    ):
        return {
            "action_ready": False,
            "specialist_agent": (
                "document_action"
            ),
            "specialist_answer": (
                "文档 ID 与文件名不一致。"
                "请核对后重新发起删除。"
            ),
        }

    processing_status = (
        document[
            "processing_status"
        ]
    )

    if processing_status in {
        "queued",
        "processing",
    }:
        return {
            "action_ready": False,
            "specialist_agent": (
                "document_action"
            ),
            "specialist_answer": (
                f"文档 "
                f"{document_id} "
                f"当前状态为 "
                f"{processing_status}，"
                "正在处理的文档暂时"
                "不能删除。"
            ),
        }

    return {
        "action_ready": True,

        "specialist_agent": (
            "document_action"
        ),

        "pending_action": {
            "action": (
                "delete_document"
            ),
            "document_id": (
                document["id"]
            ),
            "filename": (
                document["filename"]
            ),
            "processing_status": (
                processing_status
            ),
        },
    }


def route_after_prepare_delete(
    state: MainGraphState,
) -> Literal[
    "confirm_delete",
    "chat",
]:
    if state.get(
        "action_ready",
        False,
    ):
        return "confirm_delete"

    return "chat"


def confirm_delete_node(
    state: MainGraphState,
):
    pending_action = state.get(
        "pending_action",
        {},
    )

    document_id = (
        pending_action.get(
            "document_id"
        )
    )

    filename = (
        pending_action.get(
            "filename"
        )
    )

    approval = interrupt(
        {
            "type": (
                "approval_required"
            ),

            "action": (
                "delete_document"
            ),

            "document_id": (
                document_id
            ),

            "filename": (
                filename
            ),

            "message": (
                f"确认删除文档 "
                f"{document_id} "
                f"({filename})？"
                "删除后文档记录、"
                "向量 Chunk 和本地文件"
                "都会被移除。"
            ),
        },
        response_schema=(
            DeleteApproval
        ),
    )

    if isinstance(
        approval,
        DeleteApproval,
    ):
        decision = (
            approval.decision
        )

    elif isinstance(
        approval,
        dict,
    ):
        decision = (
            approval.get(
                "decision"
            )
        )

    else:
        decision = None

    approved = (
        decision == "approve"
    )

    if approved:
        return {
            "action_approved": True,
            "specialist_answer": "",
        }

    return {
        "action_approved": False,

        "specialist_agent": (
            "document_action"
        ),

        "specialist_answer": (
            f"已取消删除文档 "
            f"{document_id} "
            f"({filename})。"
        ),
    }


def route_after_delete_confirmation(
    state: MainGraphState,
) -> Literal[
    "execute_delete",
    "chat",
]:
    if state.get(
        "action_approved",
        False,
    ):
        return "execute_delete"

    return "chat"


async def execute_delete_node(
    state: MainGraphState,
    runtime: Runtime[
        AgentContext
    ],
):
    pending_action = state.get(
        "pending_action",
        {},
    )

    document_id = (
        pending_action.get(
            "document_id"
        )
    )

    if document_id is None:
        return {
            "specialist_agent": (
                "document_action"
            ),
            "specialist_answer": (
                "删除失败：缺少 "
                "document_id。"
            ),
        }

    workspace_id = (
        runtime.context[
            "workspace_id"
        ]
    )

    try:
        result = await asyncio.to_thread(
            (
                document_action_service
                .delete_document
            ),
            workspace_id,
            int(document_id),
        )

    except DocumentActionError as exc:
        return {
            "specialist_agent": (
                "document_action"
            ),

            "specialist_answer": (
                f"删除文档失败："
                f"{exc}"
            ),

            "pending_action": {},
        }

    return {
        "specialist_agent": (
            "document_action"
        ),

        "specialist_answer": (
            f"文档 "
            f"{result['document_id']} "
            f"({result['filename']}) "
            "已删除。"
        ),

        "pending_action": {},

        "action_ready": False,

        "action_approved": False,

        "documents": [],

        "sources": [],
    }

#所有专业节点处理完，不会直接把原始结果扔给用户，而是进入 chat_node
#根据原始聊天记录、专业节点结果、相关引用结果和长期记忆，生成最终用户回答
async def chat_node(
    state: MainGraphState,
    runtime: Runtime[
        AgentContext
    ],
):
    specialist_answer = (
        state.get(
            "specialist_answer",
            "",
        )
    )

    specialist_agent = (
        state.get(
            "specialist_agent",
            "",
        )
    )

    system_prompt = (
        CHAT_SYSTEM_PROMPT + "\n\n" + get_runtime_time_prompt(runtime.context)
    )

    memory_prompt = (
        runtime.context.get(
            "memory_prompt",
            "",
        )
    )

    if memory_prompt:
        system_prompt += (
            "\n\n"
            "以下是当前用户和 Workspace "
            "的长期记忆。"
            "这些信息可以作为回答时的长期上下文。"
            "如果长期记忆与用户当前消息冲突，"
            "以用户当前消息为准。"
            "不要向用户暴露记忆系统的内部实现。"
            "\n\n"
            f"{memory_prompt}"
        )

    if specialist_answer:
        system_prompt += (
            "\n\n"
            "本轮已有专业处理结果："
            "\n"
            f"专业类型："
            f"{specialist_agent}"
            "\n\n"
            "结果："
            "\n"
            "--------------------"
            "\n"
            f"{specialist_answer}"
            "\n"
            "--------------------"
            "\n"
            "请基于这个结果生成"
            "最终用户回答。"
        )

    model_messages = [
        SystemMessage(
            content=system_prompt
        ),
        *state.get("messages", []),
    ]
    input_chars = sum(
        len(get_message_text(message))
        for message in model_messages
    )
    answer_incomplete = state.get("answer_incomplete", False)

    if answer_incomplete:
        response = AIMessage(
            content=(
                specialist_answer.rstrip()
                + INCOMPLETE_ANSWER_SUFFIX
            )
        )
        logger.warning(
            "Final Chat Agent skipped regeneration because the external "
            "answer ended due to output length"
        )
    else:
        response = await get_chat_model().ainvoke(model_messages)
        finish_reason = get_finish_reason(response)
        logger.warning(
            "final_answer_completed request_id=%s model=%s input_messages=%d "
            "input_chars=%d specialist_chars=%d output_chars=%d "
            "finish_reason=%s",
            runtime.context.get("request_id", "unknown"), LLM_MODEL,
            len(model_messages),
            input_chars,
            len(specialist_answer),
            len(get_message_text(response)),
            finish_reason or "unknown",
        )
        if is_length_limited(response):
            answer_incomplete = True
            response = AIMessage(
                content=(
                    get_message_text(response).rstrip()
                    + INCOMPLETE_ANSWER_SUFFIX
                ),
                response_metadata={
                    "finish_reason": finish_reason,
                },
            )
            logger.warning(
                "Final Chat Agent output hit its length limit; "
                "marking the SSE response incomplete"
            )

    cited_sources = state.get(
        "cited_sources",
        [],
    )
    retrieved_sources = state.get(
        "retrieved_sources",
        [],
    )

    if retrieved_sources:
        answer = get_message_text(response)
        formatted_answer = (
            citation_service.render_web_citations(
                answer,
                retrieved_sources,
            )
        )
        cited_sources = (
            citation_service.filter_cited_sources(
                formatted_answer,
                retrieved_sources,
            )
        )
        document_sources = [
            source
            for source in retrieved_sources
            if source.get("source_type") == "document"
        ]
        web_sources = [
            source
            for source in retrieved_sources
            if source.get("source_type") == "web"
        ]
        cited_document_sources = [
            source
            for source in cited_sources
            if source.get("source_type") == "document"
        ]

        if (
            document_sources
            and state.get("knowledge_answer")
            and not cited_document_sources
        ):
            logger.warning(
                "Final Chat Agent answer omitted usable document citations."
            )
            response = AIMessage(
                content=(
                    "当前检索结果未返回可引用的内容，"
                    "暂时无法基于文档回答。"
                )
            )
            cited_sources = []
        elif web_sources and not any(
            source.get("source_type") == "web"
            for source in cited_sources
        ):
            logger.warning(
                "Final Chat Agent answer omitted usable web URLs."
            )
            response = AIMessage(
                content=(
                    "联网检索未返回可追溯的原始网页链接，"
                    "暂时无法基于该检索结果回答。"
                )
            )
            cited_sources = []
        elif formatted_answer != answer:
            response = AIMessage(
                content=formatted_answer,
                response_metadata=getattr(
                    response,
                    "response_metadata",
                    {},
                ),
            )

    return {
        "messages": [
            response
        ],
        "cited_sources": cited_sources,
        "sources": cited_sources,
        "answer_incomplete": answer_incomplete,
    }

#编排graph
def build_main_graph(
    checkpointer,
):
    graph = StateGraph(
        MainGraphState,
        context_schema=AgentContext,
    )

    graph.add_node(
        "router",
        router_node,
    )

    graph.add_node(
        "knowledge",
        knowledge_node,
    )

    graph.add_node(
        "document",
        document_node,
    )

    graph.add_node(
        "memory",
        memory_node,
    )

    graph.add_node(
        "external",
        external_node,
    )

    graph.add_node(
        "prepare_delete",
        prepare_delete_node,
    )

    graph.add_node(
        "confirm_delete",
        confirm_delete_node,
    )

    graph.add_node(
        "execute_delete",
        execute_delete_node,
    )

    graph.add_node(
        "chat",
        chat_node,
    )

    graph.add_edge(
        START,
        "router",
    )

    graph.add_conditional_edges(
        "router",
        route_after_router,
        {
            "direct": "chat",

            "knowledge": (
                "knowledge"
            ),

            "document": (
                "document"
            ),

            "memory": (
                "memory"
            ),

            "external": (
                "external"
            ),

            "delete_document": (
                "prepare_delete"
            ),
        },
    )

    graph.add_conditional_edges(
        "knowledge",
        route_after_knowledge,
        {"external": "external", "chat": "chat"},
    )

    graph.add_edge(
        "document",
        "chat",
    )

    graph.add_edge(
        "memory",
        "chat",
    )

    graph.add_edge(
        "external",
        "chat",
    )

    graph.add_conditional_edges(
        "prepare_delete",
        route_after_prepare_delete,
        {
            "confirm_delete": (
                "confirm_delete"
            ),

            "chat": "chat",
        },
    )

    graph.add_conditional_edges(
        "confirm_delete",
        route_after_delete_confirmation,
        {
            "execute_delete": (
                "execute_delete"
            ),

            "chat": "chat",
        },
    )

    graph.add_edge(
        "execute_delete",
        "chat",
    )

    graph.add_edge(
        "chat",
        END,
    )

    return graph.compile(
        checkpointer=checkpointer,
        name="nodagent_main_graph",
    )
