from fastapi import APIRouter

from backend.app.api.routes.chat import (
    router as chat_router,
)
from backend.app.api.routes.chat_history import (
    router as chat_history_router,
)
from backend.app.api.routes.documents import (
    router as documents_router,
)
from backend.app.api.routes.health import (
    router as health_router,
)
from backend.app.api.routes.memories import (
    router as memories_router,
)
from backend.app.api.routes.search import (
    router as search_router,
)
from backend.app.api.routes.workspaces import (
    router as workspaces_router,
)

#FastAPI 项目的“总路由注册中心”。
#它把分散在不同功能模块中的路由统一挂到 /api 前缀下，最终由主程序一次性注册。

api_router = APIRouter(
    prefix="/api",
)

api_router.include_router(
    health_router#健康检查
)

api_router.include_router(
    workspaces_router#工作空间管理（创建、查询、切换）
)

api_router.include_router(
    documents_router#文档上传、查询、删除等
)

api_router.include_router(
    search_router#检索、知识库搜索
)

api_router.include_router(
    memories_router#记忆管理
)

api_router.include_router(
    chat_router#聊天管理、流式回答、Agent调用
)

api_router.include_router(
    chat_history_router#聊天历史查询
)
