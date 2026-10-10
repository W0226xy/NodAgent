from typing import List

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from backend.app.db.database import get_db
from backend.app.schemas.workspace import (
    WorkspaceCreate,
    WorkspaceResponse,
)
from backend.app.services import workspace_service


router = APIRouter(
    prefix="/workspaces",#该文件中所有接口自动带上 /workspaces 前缀
    tags=["workspaces"],#用于 Swagger 接口文档分类。打开 /docs 时，这个接口会出现在 workspaces 分组下；它不影响实际访问地址。
)

#获取工作空间列表
@router.get(
    "",#表示定义一个 GET 请求，路径为空，因此它直接对应当前前缀GET /api/workspaces
    response_model=List[WorkspaceResponse],
)
def get_workspaces(
    db: Session = Depends(get_db),#FastAPI 会在执行函数前自动调用 get_db()，获取一个数据库会话，并把得到的会话对象赋给 db
):
    return workspace_service.get_workspaces(db)#随后路由把 db 交给业务层

#创建工作空间
@router.post(
    "",#表示定义一个 POST 请求，路径为空，因此它直接对应当前前缀POST /api/workspaces
    response_model=WorkspaceResponse,
    #FastAPI 会先依据 WorkspaceCreate 自动完成校验（JSON 格式是否正确、必填字段是否存在等）
    status_code=status.HTTP_201_CREATED,#表示创建成功后返回 201 状态码，而不是默认的 200
)
def create_workspace(
    workspace: WorkspaceCreate,
    db: Session = Depends(get_db),#FastAPI 会在执行函数前自动调用 get_db()，获取一个数据库会话，并把得到的会话对象赋给 db
):
    return workspace_service.create_workspace(#路由层将数据库会话和已验证的 workspace 对象交给 Service 层
        db=db,
        workspace=workspace,
    )

#前端发送 POST /api/workspaces
#           ↓
# FastAPI 接收 JSON 请求
#           ↓
# 校验数据是否符合 WorkspaceCreate
#           ↓
# 调用 workspace_service 创建数据库记录
#           ↓
# 返回 JSON 给前端