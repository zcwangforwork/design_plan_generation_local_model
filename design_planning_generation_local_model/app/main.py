"""
QMS Document Generator - FastAPI Application
医疗器械质量体系文档生成工具
"""

from dotenv import load_dotenv
load_dotenv()

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.api.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期: 启动时初始化Agent"""
    # 启动时初始化Agent
    try:
        from app.services.agent_engine import init_agent
        await init_agent()
        print("[main] Agent initialized successfully")
    except Exception as e:
        print(f"[main] Agent initialization failed (non-fatal): {e}")
        print("[main] Agent endpoints will return errors until fixed")

    yield

    # 关闭时清理
    try:
        from app.services.agent_state import close_checkpointer
        await close_checkpointer()
        print("[main] Checkpointer closed")
    except Exception:
        pass


app = FastAPI(
    title="QMS Document Generator",
    description="医疗器械质量体系文档生成工具 - 基于AI自动生成符合法规的质量体系文档",
    version="2.0.0",
    lifespan=lifespan,
)

# Include API routes
app.include_router(router, prefix="/api")


@app.get("/")
async def root():
    """返回前端页面 (旧版表单模式)"""
    return FileResponse("app/static/index.html")


@app.get("/agent")
async def agent_page():
    """返回Agent对话页面 (新版聊天模式)"""
    return FileResponse("app/static/agent.html")


@app.get("/agent/review/{project_id}")
async def agent_review_page(project_id: str):
    """返回Agent文档审阅页面"""
    return FileResponse("app/static/review.html")


@app.get("/api/health")
async def health_check():
    """健康检查接口"""
    return {"status": "ok", "agent": "enabled"}


if __name__ == "__main__":
    import uvicorn
    import socket

    def _try_bind(host: str, port: int) -> bool:
        """检查端口是否可用"""
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind((host, port))
            s.close()
            return True
        except OSError:
            return False

    port = 8002
    if not _try_bind("0.0.0.0", port):
        print(f"[main] Port {port} is occupied, trying 8003...")
        port = 8003
        if not _try_bind("0.0.0.0", port):
            print(f"[main] Port {port} also occupied, trying 8004...")
            port = 8004

    print(f"[main] Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
