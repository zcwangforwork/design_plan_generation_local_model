"""
Agent State Management - 设计策划文档Agent状态管理

基于PRD Section 4.3的运行时状态快照设计。
使用LangGraph TypedDict + SqliteSaver实现持久化。
"""
import os
from pathlib import Path
from typing import TypedDict, Annotated, Optional
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


# ── State Schema (PRD Section 4.3 状态快照) ──

class AgentState(TypedDict, total=False):
    """设计策划文档Agent的全局状态

    与PRD Section 4.3的JSON状态快照一一对应。
    LangGraph自动通过SqliteSaver持久化此状态。
    """
    # ── 消息历史 ──
    messages: Annotated[list[BaseMessage], add_messages]

    # ── 步骤1: 产品画像 ──
    product_name: Optional[str]
    product_classification: Optional[str]
    product_intended_use: Optional[str]
    product_status: Optional[str]  # "confirmed" | "partial"

    # ── 步骤2: 标准适用性清单 ──
    confirmed_standards: list[str]
    candidate_standards: list[str]
    standards_status: Optional[str]  # "confirmed" | "partial" | "not_started"

    # ── 步骤3: 结构化内容采集 ──
    # 结构: {内容域: {status, items_confirmed, items_skipped}}
    content_sections: dict

    # ── 步骤4: 文档生成 ──
    generated_sections: dict   # {章节标题: content}
    document_status: Optional[str]

    # ── 步骤5: 追溯矩阵 ──
    traceability_status: Optional[str]

    # ── 步骤6: 审核+评审 ──
    review_status: Optional[str]

    # ── 待处理项 ──
    unresolved_items: list[str]

    # ── 文档类型 ──
    doc_type: Optional[str]  # 目标文档类型 (design_development_plan, risk_management_plan, etc.)

    # ── 自动模式 ──
    auto_mode: Optional[bool]  # True时跳过所有HITL确认

    # ── 多代理协作 ──
    outline: Optional[str]          # JSON字符串, 完整的文档框架
    outline_status: Optional[str]   # "not_started" | "draft" | "confirmed" | "revised"
    current_chapter: Optional[str]  # 当前正在生成/审核的章节名
    chapter_write_queue: list[str]  # 待生成章节列表 (章节名称)

    # ── 附件管理 ──
    # 结构: [{"file_id": str, "filename": str, "char_count": int, "preview": str, "full_text": str, "status": str}, ...]
    attachments: list[dict]


# ── 默认初始状态 ──

def create_initial_state() -> AgentState:
    """创建新项目的默认空状态"""
    return AgentState(
        messages=[],
        product_name=None,
        product_classification=None,
        product_intended_use=None,
        product_status="not_started",
        confirmed_standards=[],
        candidate_standards=[],
        standards_status="not_started",
        content_sections={},
        generated_sections={},
        document_status="not_started",
        traceability_status="not_started",
        review_status="not_started",
        unresolved_items=[],
        auto_mode=False,
        outline=None,
        outline_status="not_started",
        current_chapter=None,
        chapter_write_queue=[],
        attachments=[],
    )


# ── SqliteSaver 实例管理 ──

_checkpointer_instance: Optional[AsyncSqliteSaver] = None
_ctx = None  # from_conn_string() returns async context manager in 3.x
_db_path: Optional[str] = None


async def get_checkpointer(db_path: Optional[str] = None) -> AsyncSqliteSaver:
    """获取或创建SqliteSaver检查点实例（单例）

    Args:
        db_path: SQLite数据库路径，默认在项目根目录 project_store/
    """
    global _checkpointer_instance, _ctx, _db_path

    if db_path is None:
        project_root = Path(__file__).parent.parent.parent
        db_dir = project_root / "project_store"
        db_dir.mkdir(parents=True, exist_ok=True)
        db_path = str(db_dir / "agent_checkpoints.db")

    # 如果路径变了，关闭旧实例
    if _checkpointer_instance is not None and db_path != _db_path:
        if _ctx:
            try:
                await _ctx.__aexit__(None, None, None)
            except Exception:
                pass
            _ctx = None
        _checkpointer_instance = None

    if _checkpointer_instance is None:
        # from_conn_string() returns async context manager (not awaitable) in 3.x
        _ctx = AsyncSqliteSaver.from_conn_string(db_path)
        _checkpointer_instance = await _ctx.__aenter__()
        _db_path = db_path

    return _checkpointer_instance


async def close_checkpointer():
    """关闭检查点实例（应用关闭时调用）"""
    global _checkpointer_instance, _ctx
    if _ctx:
        try:
            await _ctx.__aexit__(None, None, None)
        except Exception:
            pass
        _ctx = None
    _checkpointer_instance = None


# ── 状态辅助函数 ──

def build_state_snapshot(state: AgentState) -> dict:
    """从State提取JSON快照，用于注入System Prompt

    对应PRD Section 4.3的JSON结构
    """
    return {
        "product": {
            "name": state.get("product_name"),
            "classification": state.get("product_classification"),
            "intended_use": state.get("product_intended_use"),
            "status": state.get("product_status", "not_started"),
        },
        "standards": {
            "confirmed": state.get("confirmed_standards", []),
            "candidate": state.get("candidate_standards", []),
            "status": state.get("standards_status", "not_started"),
        },
        "content_sections": state.get("content_sections", {}),
        "document_generation": {
            "status": state.get("document_status", "not_started"),
            "sections_generated": list(state.get("generated_sections", {}).keys()),
            "doc_type": state.get("doc_type", "design_development_plan"),
        },
        "traceability": {"status": state.get("traceability_status", "not_started")},
        "review": {"status": state.get("review_status", "not_started")},
        "unresolved_items": state.get("unresolved_items", []),
        "outline": {
            "status": state.get("outline_status", "not_started"),
            "chapters_count": _count_chapters(state.get("outline")),
        },
        "multi_agent": {
            "current_chapter": state.get("current_chapter"),
            "pending_chapters": state.get("chapter_write_queue", []),
        },
        "attachments": [
            {
                "file_id": a.get("file_id"),
                "filename": a.get("filename"),
                "char_count": a.get("char_count", 0),
                "preview": a.get("preview", ""),
                "status": a.get("status", "unknown"),
            }
            for a in state.get("attachments", [])
        ],
    }


def _count_chapters(outline_json: Optional[str]) -> int:
    """从框架JSON中提取章节数"""
    if not outline_json:
        return 0
    try:
        import json
        data = json.loads(outline_json)
        return len(data.get("chapters", []))
    except (json.JSONDecodeError, TypeError):
        return 0
