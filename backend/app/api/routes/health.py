from fastapi import APIRouter

# 健康检查路由
router = APIRouter(
    prefix="/health",#该文件中所有接口自动带上 /health 前缀
    tags=["health"],#用于 Swagger 接口文档分类。打开 /docs 时，这个接口会出现在 health 分组下；它不影响实际访问地址。
)

#主要用途是让前端、Docker 健康检查、负载均衡器或运维系统判断后端进程是否仍可访问。
#比如curl http://localhost:8000/api/health
#如果返回 {"status": "ok"}，说明FastAPI 服务已启动、路由已正确注册、HTTP 请求可以正常到达后端。
@router.get("")
def health():
    return {
        "status": "ok"
    }