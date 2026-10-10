import shutil
from pathlib import Path
from uuid import uuid4

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Response,
    UploadFile,
    status,
)
from sqlalchemy.orm import Session

from backend.app.core.document_status import (
    DocumentProcessingStatus,
)
from backend.app.db.database import (
    get_db,
)
from backend.app.models.document import (
    Document,
)
from backend.app.models.document_chunk import (
    DocumentChunk,
)
from backend.app.models.workspace import (
    Workspace,
)
from backend.app.schemas.document import (
    DocumentChunkResponse,
    DocumentProcessingResponse,
    DocumentResponse,
)
from backend.app.services import (
    document_action_service,
)
from backend.app.services.document_action_service import (
    DocumentBusyError,
    DocumentNotFoundError,
)
from backend.app.services.task_dispatcher import (
    DocumentAlreadyProcessingError,
    TaskDispatchError,
    dispatch_document_processing,
)


router = APIRouter(
    prefix=(
        "/workspaces/"
        "{workspace_id}/documents"#workspace_id 是路径参数，表示文档属于哪个工作空间。
    ),#例如GET /api/workspaces/3/documents
    tags=[
        "documents",#用于 Swagger 接口文档分类。打开 /docs 时，这个接口会出现在 documents 分组下；它不影响实际访问地址。
    ],
)


UPLOAD_DIR = Path(#上传的原始文件保存在项目/data/uploads/下
    "data/uploads"
)


ALLOWED_SUFFIXES = {#允许的文件类型
    ".pdf",
    ".txt",
    ".md",
}


def get_workspace_or_404(#按id查询工作空间，如果不存在则抛出 404 异常
    db: Session,
    workspace_id: int,
) -> Workspace:
    workspace = (
        db.query(Workspace)
        .filter(
            Workspace.id
            == workspace_id
        )
        .first()
    )

    if workspace is None:
        raise HTTPException(
            status_code=(
                status.HTTP_404_NOT_FOUND
            ),
            detail="Workspace not found",
        )

    return workspace


def get_document_or_404(#按id查询文档，如果不存在则抛出 404 异常
    db: Session,
    workspace_id: int,
    document_id: int,
) -> Document:
    document = (
        db.query(Document)
        .filter(
            Document.id
            == document_id,
            Document.workspace_id#文档必须属于指定的工作空间
            == workspace_id,
        )
        .first()
    )

    if document is None:
        raise HTTPException(
            status_code=(
                status.HTTP_404_NOT_FOUND
            ),
            detail="Document not found",
        )

    return document

#获取工作空间下的文档列表
@router.get(
    "",#GET /api/workspaces/{workspace_id}/documents
    response_model=list[
        DocumentResponse
    ],
)
def list_documents(
    workspace_id: int,
    db: Session = Depends(
        get_db
    ),
):
    get_workspace_or_404(
        db=db,
        workspace_id=workspace_id,
    )

    return (
        db.query(Document)
        .filter(
            Document.workspace_id
            == workspace_id
        )
        .order_by(
            Document.id.desc()
        )
        .all()
    )

#上传文档接口
@router.post(
    "",#POST /api/workspaces/{workspace_id}/documents
    response_model=DocumentResponse,
    status_code=(
        status.HTTP_202_ACCEPTED#上传文档后，返回 202 Accepted，表示请求已接受，但处理尚未完成。客户端可以通过 /status 接口查询处理状态。
    ),
)
def upload_document(
    workspace_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(
        get_db
    ),
):
    get_workspace_or_404(#确认工作空间存在
        db=db,
        workspace_id=workspace_id,
    )

    original_filename = (#检查文件后缀
        file.filename
        or "document"
    )

    suffix = (
        Path(
            original_filename
        )
        .suffix
        .lower()
    )

    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=(
                status.HTTP_400_BAD_REQUEST
            ),
            detail=(
                "Unsupported file type. "
                "Supported types: "
                "pdf, txt, md"
            ),
        )

    UPLOAD_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    stored_filename = (#生成一个唯一的文件名，避免文件名冲突
        f"{uuid4().hex}"
        f"{suffix}"
    )

    file_path = (
        UPLOAD_DIR
        / stored_filename
    )

    try:
        with file_path.open(
            "wb"
        ) as buffer:
            shutil.copyfileobj(
                file.file,
                buffer,
            )

    except Exception as exc:
        if file_path.exists():
            file_path.unlink()

        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail="Failed to save file",
        ) from exc

    if (#检查文件是否为空
        not file_path.exists()
        or file_path.stat().st_size == 0
    ):
        if file_path.exists():
            file_path.unlink()

        raise HTTPException(
            status_code=(
                status.HTTP_400_BAD_REQUEST
            ),
            detail="Uploaded file is empty",
        )

    document = Document(#创建一个 Document 实例，保存到数据库
        workspace_id=workspace_id,
        filename=original_filename,
        file_path=str(
            file_path
        ),
        content_type=(
            file.content_type
        ),
        processing_status=(
            DocumentProcessingStatus
            .UPLOADED
            .value
        ),
    )

    try:
        db.add(document)
        db.commit()
        db.refresh(document)

    except Exception as exc:
        db.rollback()

        if file_path.exists():
            file_path.unlink()

        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail=(
                "Failed to create "
                "document"
            ),
        ) from exc

    try:
        dispatch_document_processing(#投递后台解析任务，通常会将任务提交给 Celery Worker 或其他任务队列
            db=db,
            document=document,
        )

    except TaskDispatchError as exc:
        raise HTTPException(
            status_code=(
                status.HTTP_503_SERVICE_UNAVAILABLE#任务队列或 Worker 不可用
            ),
            detail=(#文件和数据库记录已经创建成功，但处理任务没有成功进入队列
                "Document was uploaded, "
                "but processing could not "
                "be queued"
            ),
        ) from exc

    db.refresh(document)

    return document

#重新处理文档接口，对已有文档手动重新解析（比如上次解析失败）
@router.post(
    "/{document_id}/process",#POST /api/workspaces/{workspace_id}/documents/{document_id}/process
    response_model=(
        DocumentProcessingResponse
    ),
    status_code=(
        status.HTTP_202_ACCEPTED
    ),
)
def process_document_again(
    workspace_id: int,
    document_id: int,
    db: Session = Depends(
        get_db
    ),
):
    document = (
        get_document_or_404(
            db=db,
            workspace_id=workspace_id,
            document_id=document_id,
        )
    )

    try:
        task_id = (
            dispatch_document_processing(
                db=db,
                document=document,
            )
        )

    except (
        DocumentAlreadyProcessingError
    ) as exc:
        raise HTTPException(
            status_code=(
                status.HTTP_409_CONFLICT
            ),
            detail=str(exc),
        ) from exc

    except TaskDispatchError as exc:
        raise HTTPException(
            status_code=(
                status.HTTP_503_SERVICE_UNAVAILABLE
            ),
            detail=(
                "Failed to queue "
                "document processing"
            ),
        ) from exc

    db.refresh(document)

    return DocumentProcessingResponse(
        document_id=document.id,
        status=(
            document.processing_status
        ),
        task_id=task_id,
        error=(
            document.processing_error
        ),
    )

#查询文档处理状态接口，返回文档的当前处理状态、任务 ID 和错误信息（如果有）
@router.get(
    "/{document_id}/status",
    response_model=(
        DocumentProcessingResponse
    ),
)
def get_document_status(
    workspace_id: int,
    document_id: int,
    db: Session = Depends(
        get_db
    ),
):
    document = (
        get_document_or_404(
            db=db,
            workspace_id=workspace_id,
            document_id=document_id,
        )
    )

    return DocumentProcessingResponse(
        document_id=document.id,
        status=(
            document.processing_status
        ),
        task_id=(
            document.processing_task_id
        ),
        error=(
            document.processing_error
        ),
    )

#查询文档的分块内容接口，返回文档的所有分块信息，按 chunk_index 升序排列
@router.get(
    "/{document_id}/chunks",#GET /api/workspaces/{workspace_id}/documents/{document_id}/chunks
    response_model=list[
        DocumentChunkResponse
    ],
)
def list_document_chunks(
    workspace_id: int,
    document_id: int,
    db: Session = Depends(
        get_db
    ),
):
    document = (
        get_document_or_404(
            db=db,
            workspace_id=workspace_id,
            document_id=document_id,
        )
    )

    return (
        db.query(
            DocumentChunk
        )
        .filter(
            DocumentChunk.document_id
            == document.id
        )
        .order_by(
            DocumentChunk
            .chunk_index
            .asc()
        )
        .all()
    )

#删除文档接口，删除文档及其分块内容，同时删除原始文件。注意：如果文档正在处理或被其他操作占用，会返回 409 Conflict。
@router.delete(
    "/{document_id}",#DELETE /api/workspaces/{workspace_id}/documents/{document_id}
    status_code=(
        status.HTTP_204_NO_CONTENT
    ),
)
def delete_document(
    workspace_id: int,
    document_id: int,
):
    try:
        document_action_service.delete_document(
            workspace_id=workspace_id,
            document_id=document_id,
        )

    except DocumentNotFoundError as exc:
        raise HTTPException(
            status_code=(
                status.HTTP_404_NOT_FOUND
            ),
            detail=str(exc),
        ) from exc

    except DocumentBusyError as exc:
        raise HTTPException(
            status_code=(
                status.HTTP_409_CONFLICT
            ),
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail=(
                "Failed to delete "
                "document"
            ),
        ) from exc

    return Response(
        status_code=(
            status.HTTP_204_NO_CONTENT
        )
    )
