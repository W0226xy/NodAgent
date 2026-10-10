from typing import (
    Annotated,
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Union,
)

from pydantic import (
    BaseModel,
    Field,
)
#这段是聊天接口的 Pydantic 数据模型（Schema）。它统一规定了：
# 前端能传什么数据、格式是什么
#         ↓
# FastAPI / Pydantic 自动校验
#         ↓
# 后端返回什么数据、字段如何组织

class ChatSource(BaseModel):
    source_type: Literal["document", "web"] = "document"#来源类型，文档或网页

    source_number: int#来源编号

    citation: str#引用信息

    chunk_id: int#块 ID

    document_id: int#文档 ID

    filename: str#文件名

    chunk_index: int#块索引

    page_number: Optional[int] = None#原文页码

    content: str#实际被检索命中的文本内容

    metadata_json: Dict[#其他元数据，例如标题、位置、解析器信息等
        str,
        Any,
    ]

    distance: Optional[float] = None#向量距离

    similarity: Optional[float] = None#向量相似度
    keyword_score: Optional[float] = None#关键词匹配分数

    vector_rank: Optional[int] = None#向量检索排名

    keyword_rank: Optional[int] = None#关键词检索排名

    rrf_score: Optional[float] = None#RRF 融合后的最终排序得分。

    url: Optional[str] = None#来源 URL

    title: Optional[str] = None#标题

    published_date: Optional[str] = None#发布日期

#普通聊天请求
class ChatRequest(BaseModel):
    user_id: str#用户 ID

    thread_id: str#线程 ID，用于区分不同的聊天会话

    message: str#用户发送的消息内容

    top_k: int = Field(#RAG 检索时,返回最相关的 Chunk 数量
        default=5,#默认5条，最少1条，最多20条
        ge=1,
        le=20,
    )

#人工审批中断的聊天请求
class ChatResumeRequest(BaseModel):
    user_id: str#用户 ID

    thread_id: str#线程 ID，用于区分不同的聊天会话

    decision: Literal[#批准 Agent 继续执行, 或者拒绝 Agent 继续执行
        "approve",
        "reject",
    ]

    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
    )

#当 Agent 完整执行成功时，返回该结构。
class ChatResponse(BaseModel):
    status: Literal[#让前端能知道：这不是等待审批的中断结果，而是已完成的回答
        "completed"
    ] = "completed"

    thread_id: str#线程 ID，用于区分不同的聊天会话

    answer: str#Agent 的回答内容

    sources: List[#引用来源列表
        ChatSource
    ] = Field(
        default_factory=list
    )

#当 Agent 被人工审批中断时，返回该结构。
class ChatInterruptResponse(BaseModel):
    status: Literal[#当 Agent 处理到需要人工确认的节点时，不会返回最终答案，而是返回“中断信息”。
        "interrupted"
    ] = "interrupted"

    thread_id: str#线程 ID，用于区分不同的聊天会话

    interrupt_id: str#中断 ID

    interrupt: Dict[
        str,
        Any,
    ]


ChatApiResponse = Annotated[#联合响应类型，聊天接口的返回值有两种合法结构：ChatResponse 或 ChatInterruptResponse
    #FastAPI / Pydantic 会通过 status 字段自动区分应该使用哪一种模型
    Union[
        ChatResponse,
        ChatInterruptResponse,
    ],
    Field(
        discriminator="status"
    ),
]
