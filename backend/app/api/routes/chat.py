from collections.abc import (
    AsyncIterator,
)
from dataclasses import dataclass
from typing import Union

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
    status,
)
from fastapi.sse import (
    EventSourceResponse,
    ServerSentEvent,
)

from backend.app.db.database import (
    SessionLocal,
)
from backend.app.schemas.chat import (
    ChatApiResponse,
    ChatRequest,
    ChatResumeRequest,
)
from backend.app.services import (
    agent_service,
    chat_history_service,
)
from backend.app.services.agent_service import (
    NoPendingInterruptError,
    PendingInterruptError,
)

# 前端发送消息
#   → 校验工作空间与会话线程
#   → 获取已初始化的 Agent Graph
#   → agent_service 执行 / 恢复 Agent
#   → 一次性 JSON 返回，或 SSE 流式推送事件

#把前端的聊天请求交给 Agent 图执行，并支持两种交互方式
#普通请求：等 Agent 全部执行完，再一次性返回结果；
#流式请求：通过 SSE 将生成过程实时推给前端。
router = APIRouter(
    prefix=(#完整接口前缀为/api/workspaces/{workspace_id}/chat
        "/workspaces/"
        "{workspace_id}/chat"
    ),
    tags=[
        "chat",
    ],
)


@dataclass(
    frozen=True
)
class ChatContext:#聊天请求的统一上下文
    data: Union[#已经经过 Pydantic 校验的请求数据
        ChatRequest,#普通聊天使用 ChatRequest
        ChatResumeRequest,#恢复聊天使用 ChatResumeRequest
    ]

    chat_thread_id: int#数据库中会话线程的真实整数主键。

#校验工作空间和聊天线程
def validate_chat_thread(
    workspace_id: int,
    user_id: str,
    thread_id: str,
) -> int:
    with SessionLocal() as db:
        workspace = (#工作空间是否存在
            chat_history_service
            .get_workspace(
                db=db,
                workspace_id=(
                    workspace_id
                ),
            )
        )

        if workspace is None:
            raise HTTPException(
                status_code=(
                    status
                    .HTTP_404_NOT_FOUND
                ),
                detail=(
                    "Workspace not found"
                ),
            )

        thread = (#聊天线程是否存在
            chat_history_service
            .get_thread(
                db=db,
                workspace_id=(
                    workspace_id
                ),
                user_id=user_id,
                thread_id=thread_id,
            )
        )

        if thread is None:
            raise HTTPException(
                status_code=(
                    status
                    .HTTP_404_NOT_FOUND
                ),
                detail=(
                    "Chat thread "
                    "not found"
                ),
            )

        return thread.id

#根据请求类型获取聊天上下文
# 从 URL 读取 workspace_id；
# 从请求体读取 JSON；
# 用 ChatRequest 校验 JSON；
# 执行 get_chat_context()；
# 将返回结果注入到接口参数 chat_context
def get_chat_context(
    workspace_id: int,
    data: ChatRequest,
) -> ChatContext:
    chat_thread_id = (
        validate_chat_thread(
            workspace_id=workspace_id,
            user_id=data.user_id,
            thread_id=data.thread_id,
        )
    )

    return ChatContext(
        data=data,
        chat_thread_id=(
            chat_thread_id
        ),
    )

#恢复聊天时，获取上下文
def get_resume_context(
    workspace_id: int,
    data: ChatResumeRequest,
) -> ChatContext:
    chat_thread_id = (
        validate_chat_thread(
            workspace_id=workspace_id,
            user_id=data.user_id,
            thread_id=data.thread_id,
        )
    )

    return ChatContext(
        data=data,
        chat_thread_id=(
            chat_thread_id
        ),
    )


@router.post(
    "",#完整接口为 /api/workspaces/{workspace_id}/chat
    response_model=ChatApiResponse,
)
async def chat(#普通聊天请求接口。只有当 Agent 完整执行结束，才返回最终 JSON
    workspace_id: int,
    request: Request,
    chat_context: ChatContext = Depends(
        get_chat_context
    ),
):
    data = chat_context.data#已经经过 Pydantic 校验的请求数据

    agent = (#从 FastAPI 应用的共享状态中取出启动时初始化好的 Agent 图
        request.app.state.agent_graph
    )

    try:
        return await agent_service.chat(
            agent=agent,
            chat_thread_id=(
                chat_context
                .chat_thread_id
            ),
            workspace_id=(
                workspace_id
            ),
            user_id=data.user_id,
            thread_id=(
                data.thread_id
            ),
            message=data.message,
            top_k=data.top_k,
        )

    except PendingInterruptError as exc:
        raise HTTPException(
            status_code=(
                status.HTTP_409_CONFLICT
            ),
            detail=(
                "This thread is waiting "
                "for human approval. "
                "Use /chat/resume."
            ),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=(
                status
                .HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail=(
                "Failed to execute "
                "agent graph"
            ),
        ) from exc

#恢复被人工中断的聊天
#它服务于 Human-in-the-loop（人在回路）场景，比如有删除操作需要人工审核
@router.post(
    "/resume",
    response_model=ChatApiResponse,
)
async def resume_chat(
    workspace_id: int,
    request: Request,
    chat_context: ChatContext = Depends(
        get_resume_context
    ),
):
    data = chat_context.data

    agent = (
        request.app.state.agent_graph
    )

    try:
        return await (
            agent_service.resume_chat(
                agent=agent,
                chat_thread_id=(
                    chat_context
                    .chat_thread_id
                ),
                workspace_id=(
                    workspace_id
                ),
                user_id=(
                    data.user_id
                ),
                thread_id=(
                    data.thread_id
                ),
                decision=(#人工审批结果（approve 表示通过，reject 表示拒绝）
                    data.decision
                ),
                top_k=data.top_k,
            )
        )

    except (
        NoPendingInterruptError
    ) as exc:
        raise HTTPException(
            status_code=(
                status.HTTP_409_CONFLICT
            ),
            detail=(
                "This thread has no "
                "pending approval."
            ),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=(
                status
                .HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail=(
                "Failed to resume "
                "agent graph"
            ),
        ) from exc

#流式聊天接口
#这个接口和普通 chat() 的业务功能基本一致，区别是它不会等到 Agent 执行结束后才统一返回，而是把过程中产生的事件逐步发送给前端。
@router.post(
    "/stream",
    response_class=(
        EventSourceResponse
    ),
)
async def stream_chat(
    workspace_id: int,
    request: Request,
    chat_context: ChatContext = Depends(
        get_chat_context
    ),
) -> AsyncIterator[
    ServerSentEvent
]:
    data = chat_context.data

    agent = (
        request.app.state.agent_graph
    )
    #agent_service.stream_chat() 是异步生成器，会不断产生事件
    async for stream_event in (
        agent_service.stream_chat(
            agent=agent,
            chat_thread_id=(
                chat_context
                .chat_thread_id
            ),
            workspace_id=(
                workspace_id
            ),
            user_id=data.user_id,
            thread_id=(
                data.thread_id
            ),
            message=data.message,
            top_k=data.top_k,
        )
    ):
        yield ServerSentEvent(#把每个事件包装成 ServerSentEvent 对象，发送给前端
            event=(
                stream_event[
                    "event"
                ]
            ),
            data=(
                stream_event[
                    "data"
                ]
            ),
        )
# 传输过程中格式大致类似
# event: token
# data: 正在分析文档……
#
# event: token
# data: 已找到三条相关内容。
#
# event: done
# data: {"message_id": 23}



@router.post(
    "/stream/resume",
    response_class=(
        EventSourceResponse
    ),
)
async def stream_resume_chat(
    workspace_id: int,
    request: Request,
    chat_context: ChatContext = Depends(
        get_resume_context
    ),
) -> AsyncIterator[
    ServerSentEvent
]:
    data = chat_context.data

    agent = (
        request.app.state.agent_graph
    )

    async for stream_event in (
        agent_service
        .stream_resume_chat(
            agent=agent,
            chat_thread_id=(
                chat_context
                .chat_thread_id
            ),
            workspace_id=(
                workspace_id
            ),
            user_id=data.user_id,
            thread_id=(
                data.thread_id
            ),
            decision=(
                data.decision
            ),
            top_k=data.top_k,
        )
    ):
        yield ServerSentEvent(
            event=(
                stream_event[
                    "event"
                ]
            ),
            data=(
                stream_event[
                    "data"
                ]
            ),
        )