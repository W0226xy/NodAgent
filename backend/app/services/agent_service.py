import logging
from typing import (
    Any,
    AsyncIterator,
    Dict,
    List,
    Union,
)

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
)
from langgraph.graph.state import (
    CompiledStateGraph,
)
from langgraph.types import (
    Command,
)

from backend.app.db.database import (
    SessionLocal,
)
from backend.app.models.chat_thread import (
    ChatThread,
)
from backend.app.schemas.chat import (
    ChatInterruptResponse,
    ChatResponse,
    ChatSource,
)
from backend.app.services import (
    agent_context_service,
    chat_history_service,
    citation_service,
)


logger = logging.getLogger(
    __name__
)


class PendingInterruptError(
    RuntimeError
):
    pass


class NoPendingInterruptError(
    RuntimeError
):
    pass


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


def get_current_turn_messages(
    messages: List[Any],
) -> List[Any]:
    for index in range(
        len(messages) - 1,
        -1,
        -1,
    ):
        if isinstance(
            messages[index],
            HumanMessage,
        ):
            return messages[
                index + 1:
            ]

    return messages


def get_latest_ai_answer(
    messages: List[Any],
) -> str:
    for message in reversed(
        messages
    ):
        if isinstance(
            message,
            AIMessage,
        ):
            text = get_message_text(
                message
            )

            if text:
                return text

    return ""


def parse_sources(
    raw_sources: Any,
) -> List[ChatSource]:
    if not isinstance(
        raw_sources,
        list,
    ):
        return []

    sources_by_chunk: Dict[
        int,
        ChatSource,
    ] = {}

    for raw_source in raw_sources:
        if not isinstance(
            raw_source,
            dict,
        ):
            continue

        try:
            source = ChatSource(
                **raw_source
            )

        except Exception:
            continue

        sources_by_chunk[
            source.chunk_id
        ] = source

    return list(
        sources_by_chunk.values()
    )


def filter_answer_sources(
    answer: str,
    sources: List[ChatSource],
) -> List[ChatSource]:
    """Filter graph-provided candidates to citations in the final answer."""

    filtered_payload = (
        citation_service.filter_cited_sources(
            answer,
            [source.model_dump() for source in sources],
        )
    )

    return parse_sources(filtered_payload)


def save_user_message(
    chat_thread_id: int,
    message: str,
) -> None:
    with SessionLocal() as db:
        thread = (
            db.query(ChatThread)
            .filter(
                ChatThread.id
                == chat_thread_id
            )
            .first()
        )

        if thread is None:
            raise ValueError(
                "Chat thread not found"
            )

        (
            chat_history_service
            .create_message(
                db=db,
                chat_thread_id=(
                    thread.id
                ),
                role="user",
                content=message,
            )
        )

        (
            chat_history_service
            .update_thread_title_from_message(
                thread=thread,
                message=message,
            )
        )

        chat_history_service.touch_thread(
            thread=thread
        )

        db.commit()


def save_assistant_message(
    chat_thread_id: int,
    answer: str,
    sources: List[
        ChatSource
    ],
) -> int:
    with SessionLocal() as db:
        thread = (
            db.query(ChatThread)
            .filter(
                ChatThread.id
                == chat_thread_id
            )
            .first()
        )

        if thread is None:
            raise ValueError(
                "Chat thread not found"
            )

        source_payload = [
            source.model_dump()
            for source in sources
        ]

        assistant_message = (
            chat_history_service
            .create_message(
                db=db,
                chat_thread_id=(
                    thread.id
                ),
                role="assistant",
                content=answer,
                sources=source_payload,
            )
        )

        chat_history_service.touch_thread(
            thread=thread
        )

        db.commit()

        db.refresh(
            assistant_message
        )

        return assistant_message.id


def build_thread_config(
    user_id: str,
    workspace_id: int,
    thread_id: str,
):
    internal_thread_id = (
        chat_history_service
        .build_internal_thread_id(
            user_id=user_id,
            workspace_id=workspace_id,
            thread_id=thread_id,
        )
    )

    return {
        "configurable": {
            "thread_id": (
                internal_thread_id
            )
        }
    }


def build_agent_context(
    user_id: str,
    workspace_id: int,
    top_k: int,
):
    return (
        agent_context_service
        .build_agent_context(
            user_id=user_id,
            workspace_id=workspace_id,
            top_k=top_k,
        )
    )


def build_interrupt_response(
    thread_id: str,
    snapshot,
) -> ChatInterruptResponse:
    pending_interrupt = (
        snapshot.interrupts[0]
    )

    value = (
        pending_interrupt.value
    )

    if not isinstance(
        value,
        dict,
    ):
        value = {
            "message": str(
                value
            )
        }

    return ChatInterruptResponse(
        thread_id=thread_id,
        interrupt_id=(
            pending_interrupt.id
        ),
        interrupt=value,
    )


async def ensure_no_pending_interrupt(
    agent: CompiledStateGraph,
    config,
) -> None:
    snapshot = await agent.aget_state(
        config
    )

    if snapshot.interrupts:
        raise PendingInterruptError(
            "This thread has a "
            "pending human approval"
        )


async def chat(
    agent: CompiledStateGraph,
    chat_thread_id: int,
    workspace_id: int,
    user_id: str,
    thread_id: str,
    message: str,
    top_k: int,
) -> Union[
    ChatResponse,
    ChatInterruptResponse,
]:
    config = build_thread_config(
        user_id=user_id,
        workspace_id=workspace_id,
        thread_id=thread_id,
    )

    context = build_agent_context(
        user_id=user_id,
        workspace_id=workspace_id,
        top_k=top_k,
    )

    await ensure_no_pending_interrupt(
        agent=agent,
        config=config,
    )

    save_user_message(
        chat_thread_id=(
            chat_thread_id
        ),
        message=message,
    )

    result = await agent.ainvoke(
        {
            "messages": [
                HumanMessage(
                    content=message
                )
            ]
        },
        config=config,
        context=context,
    )

    snapshot = await agent.aget_state(
        config
    )

    if snapshot.interrupts:
        return build_interrupt_response(
            thread_id=thread_id,
            snapshot=snapshot,
        )

    messages = result.get(
        "messages",
        [],
    )

    current_turn_messages = (
        get_current_turn_messages(
            messages
        )
    )

    answer = get_latest_ai_answer(
        current_turn_messages
    )

    if not answer:
        raise RuntimeError(
            "Main graph returned no "
            "assistant answer"
        )

    sources = parse_sources(
        result.get(
            "sources",
            [],
        )
    )
    sources = filter_answer_sources(answer, sources)

    save_assistant_message(
        chat_thread_id=(
            chat_thread_id
        ),
        answer=answer,
        sources=sources,
    )

    return ChatResponse(
        thread_id=thread_id,
        answer=answer,
        sources=sources,
    )


async def resume_chat(
    agent: CompiledStateGraph,
    chat_thread_id: int,
    workspace_id: int,
    user_id: str,
    thread_id: str,
    decision: str,
    top_k: int,
) -> Union[
    ChatResponse,
    ChatInterruptResponse,
]:
    config = build_thread_config(
        user_id=user_id,
        workspace_id=workspace_id,
        thread_id=thread_id,
    )

    context = build_agent_context(
        user_id=user_id,
        workspace_id=workspace_id,
        top_k=top_k,
    )

    snapshot = await agent.aget_state(
        config
    )

    if not snapshot.interrupts:
        raise NoPendingInterruptError(
            "This thread has no "
            "pending human approval"
        )

    result = await agent.ainvoke(
        Command(
            resume={
                "decision": decision
            }
        ),
        config=config,
        context=context,
    )

    snapshot = await agent.aget_state(
        config
    )

    if snapshot.interrupts:
        return build_interrupt_response(
            thread_id=thread_id,
            snapshot=snapshot,
        )

    messages = result.get(
        "messages",
        [],
    )

    current_turn_messages = (
        get_current_turn_messages(
            messages
        )
    )

    answer = get_latest_ai_answer(
        current_turn_messages
    )

    if not answer:
        raise RuntimeError(
            "Main graph returned no "
            "assistant answer"
        )

    sources = parse_sources(
        result.get(
            "sources",
            [],
        )
    )
    sources = filter_answer_sources(answer, sources)

    save_assistant_message(
        chat_thread_id=(
            chat_thread_id
        ),
        answer=answer,
        sources=sources,
    )

    return ChatResponse(
        thread_id=thread_id,
        answer=answer,
        sources=sources,
    )


async def _stream_graph_run(
    agent: CompiledStateGraph,
    graph_input,
    chat_thread_id: int,
    thread_id: str,
    config,
    context,
) -> AsyncIterator[
    Dict[str, Any]
]:
    answer_parts: List[str] = []

    final_answer = ""
    emitted_token_count = 0
    emitted_char_count = 0

    try:
        async for part in agent.astream(
            graph_input,
            config=config,
            context=context,
            stream_mode=[
                "updates",
                "messages",
            ],
            version="v2",
        ):
            part_type = part[
                "type"
            ]

            if part_type == "updates":
                update_data = part[
                    "data"
                ]

                external_update = update_data.get("external")
                if isinstance(external_update, dict):
                    external_status = external_update.get("external_status")
                    if isinstance(external_status, dict):
                        state = external_status.get("state")
                        if state == "succeeded":
                            message = f"已找到 {external_status.get('url_count', 0)} 条网页来源。"
                        elif state in {"failed", "no_results"}:
                            message = str(external_status.get("message") or "联网搜索暂时不可用。")
                        else:
                            message = ""
                        if message:
                            yield {"event": "status", "data": {"message": message, "state": state}}

                chat_update = (
                    update_data.get(
                        "chat"
                    )
                )

                if isinstance(
                    chat_update,
                    dict,
                ):
                    chat_messages = (
                        chat_update.get(
                            "messages",
                            [],
                        )
                    )

                    for chat_message in (
                        chat_messages
                    ):
                        if not isinstance(
                            chat_message,
                            AIMessage,
                        ):
                            continue

                        text = (
                            get_message_text(
                                chat_message
                            )
                        )

                        if text:
                            final_answer = text

            elif part_type == "messages":
                (
                    message_chunk,
                    metadata,
                ) = part[
                    "data"
                ]

                if (
                    metadata.get(
                        "langgraph_node"
                    )
                    != "chat"
                ):
                    continue

                if not isinstance(
                    message_chunk,
                    AIMessageChunk,
                ):
                    continue

                token = get_message_text(
                    message_chunk
                )

                if not token:
                    continue

                answer_parts.append(
                    token
                )
                emitted_token_count += 1
                emitted_char_count += len(token)

                yield {
                    "event": "token",
                    "data": {
                        "content": token
                    },
                }

        snapshot = await agent.aget_state(
            config
        )

        if snapshot.interrupts:
            interrupt_response = (
                build_interrupt_response(
                    thread_id=thread_id,
                    snapshot=snapshot,
                )
            )

            yield {
                "event": "interrupt",
                "data": (
                    interrupt_response
                    .model_dump()
                ),
            }

            return

        answer = (
            final_answer.strip()
            or "".join(
                answer_parts
            ).strip()
        )

        if not answer:
            snapshot_messages = (
                snapshot.values.get(
                    "messages",
                    [],
                )
            )

            answer = (
                get_latest_ai_answer(
                    snapshot_messages
                )
            )

        if not answer:
            raise RuntimeError(
                "Main graph returned no "
                "assistant answer"
            )

        snapshot_sources = (
            snapshot.values.get(
                "sources",
                [],
            )
        )

        sources = parse_sources(
            snapshot_sources
        )
        sources = filter_answer_sources(answer, sources)
        answer_incomplete = bool(
            snapshot.values.get(
                "answer_incomplete",
                False,
            )
        )

        message_id = (
            save_assistant_message(
                chat_thread_id=(
                    chat_thread_id
                ),
                answer=answer,
                sources=sources,
            )
        )

        # Chat-model tokens are streamed before the graph node has finished
        # normalizing web [n] markers into original Markdown URLs. Replace the
        # provisional text with the persisted final answer so stream and JSON
        # responses have identical citation rendering.
        yield {
            "event": "answer",
            "data": {
                "answer": answer,
            },
        }

        if sources:
            yield {
                "event": "sources",
                "data": {
                    "sources": [
                        source.model_dump()
                        for source in sources
                    ],
                },
            }

        logger.warning(
            "final_answer_completed request_id=%s thread_id=%s final_chars=%d emitted_chunks=%d "
            "emitted_chars=%d sources=%d incomplete=%s",
            context.get("request_id", "unknown"), thread_id,
            len(answer),
            emitted_token_count,
            emitted_char_count,
            len(sources),
            answer_incomplete,
        )
        yield {
            "event": "done",
            "data": {
                "thread_id": thread_id,
                "message_id": (
                    message_id
                ),
                "incomplete": answer_incomplete,
            },
        }
        logger.warning(
            "SSE done sent thread_id=%s incomplete=%s",
            thread_id,
            answer_incomplete,
        )

    except Exception as exc:
        logger.warning(
            "Main graph stream failed before done thread_id=%s (%s)",
            thread_id,
            type(exc).__name__,
        )

        yield {
            "event": "error",
            "data": {
                "message": (
                    "暂时无法完成本次请求，"
                    "请稍后重试。"
                )
            },
        }


async def stream_chat(
    agent: CompiledStateGraph,
    chat_thread_id: int,
    workspace_id: int,
    user_id: str,
    thread_id: str,
    message: str,
    top_k: int,
) -> AsyncIterator[
    Dict[str, Any]
]:
    config = build_thread_config(
        user_id=user_id,
        workspace_id=workspace_id,
        thread_id=thread_id,
    )

    context = build_agent_context(
        user_id=user_id,
        workspace_id=workspace_id,
        top_k=top_k,
    )

    snapshot = await agent.aget_state(
        config
    )

    if snapshot.interrupts:
        yield {
            "event": "error",
            "data": {
                "message": (
                    "Thread has a pending "
                    "human approval"
                )
            },
        }

        return

    save_user_message(
        chat_thread_id=(
            chat_thread_id
        ),
        message=message,
    )

    yield {
        "event": "start",
        "data": {
            "thread_id": thread_id,
        },
    }

    async for event in (
        _stream_graph_run(
            agent=agent,
            graph_input={
                "messages": [
                    HumanMessage(
                        content=message
                    )
                ]
            },
            chat_thread_id=(
                chat_thread_id
            ),
            thread_id=thread_id,
            config=config,
            context=context,
        )
    ):
        yield event


async def stream_resume_chat(
    agent: CompiledStateGraph,
    chat_thread_id: int,
    workspace_id: int,
    user_id: str,
    thread_id: str,
    decision: str,
    top_k: int,
) -> AsyncIterator[
    Dict[str, Any]
]:
    config = build_thread_config(
        user_id=user_id,
        workspace_id=workspace_id,
        thread_id=thread_id,
    )

    context = build_agent_context(
        user_id=user_id,
        workspace_id=workspace_id,
        top_k=top_k,
    )

    snapshot = await agent.aget_state(
        config
    )

    if not snapshot.interrupts:
        yield {
            "event": "error",
            "data": {
                "message": (
                    "Thread has no pending "
                    "human approval"
                )
            },
        }

        return

    yield {
        "event": "resume",
        "data": {
            "thread_id": thread_id,
            "decision": decision,
        },
    }

    async for event in (
        _stream_graph_run(
            agent=agent,
            graph_input=Command(
                resume={
                    "decision": decision
                }
            ),
            chat_thread_id=(
                chat_thread_id
            ),
            thread_id=thread_id,
            config=config,
            context=context,
        )
    ):
        yield event
