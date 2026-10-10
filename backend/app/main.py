from contextlib import (
    asynccontextmanager,
)

from fastapi import FastAPI#FastAPI 是 Web 后端框架，负责暴露 /api/chat、/api/documents 等接口
from langgraph.checkpoint.postgres.aio import (#管理应用生命周期
    AsyncPostgresSaver,
    #AsyncPostgresSaver 是 LangGraph 提供的 PostgreSQL 检查点保存器。
    #它不是普通聊天记录，而是专门保存 Graph 的运行状态。
)
from sqlalchemy import text

from backend.app.core.config import (
    CHECKPOINT_DATABASE_URL,#LangGraph Checkpointer 使用的数据库连接地址
)
from backend.app.db.database import (
    Base,#所有 SQLAlchemy 数据表模型的父类。
    engine,#SQLAlchemy 的数据库引擎，也就是 Python 连接 PostgreSQL 的统一入口
)
from backend.app.models import (
    ChatMessage,
    ChatThread,
    Document,
    DocumentChunk,
    Workspace,
)


with engine.begin() as connection:
    connection.execute(
        text("CREATE EXTENSION IF NOT EXISTS vector")
    )


Base.metadata.create_all(
    bind=engine,
)


from backend.app.api.router import (#导入 API 路由
    api_router,
)
from backend.app.graphs.main_graph import (
    build_main_graph,
)
from backend.app.services.mcp_service import (
    log_github_mcp_startup_diagnostics,
)


@asynccontextmanager
async def lifespan(#应用启动时构建 Agent 工作流
    app: FastAPI,
):
    await log_github_mcp_startup_diagnostics()

    #1.连接PostgreSQL数据库，创建检查点保存器 AsyncPostgresSaver
    #2.用 async with 确保 FastAPI 关闭时连接能正确释放
    async with (
        AsyncPostgresSaver
        .from_conn_string(
            CHECKPOINT_DATABASE_URL
        )
    ) as checkpointer:
        await checkpointer.setup()

        app.state.checkpointer = (
            checkpointer
        )
        #3.构建 LangGraph 主图，并将其存储在 FastAPI 的 app.state 中，以便在应用的生命周期内使用
        app.state.agent_graph = (
            build_main_graph(
                checkpointer=(
                    checkpointer
                )
            )
        )

        yield


app = FastAPI(
    title="NodAgent",
    description=(
        "AI Agent Knowledge "
        "Base System"
    ),
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/")
def root():
    return {
        "message": (
            "NodAgent is running"
        )
    }


app.include_router(
    api_router
)
