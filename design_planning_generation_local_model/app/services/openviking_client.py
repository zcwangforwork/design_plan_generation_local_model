"""
OpenVikingService - OpenViking 集成封装（Phase 1: capture-only）

设计原则（与项目"静默降级"一致）：
- OPENVIKING_ENABLED=false 时不初始化，所有方法返回空/无操作
- OpenViking 服务不可用时不影响现有 Agent 功能
- 所有 OpenViking 调用包在 try/except 中，异常仅记日志

Phase 1 范围（评审决策 #3 合规收窄）：
- ✅ capture：每轮 agent 对话后捕获消息到 OpenViking
- ✅ capture-only 工具：viking_store / viking_add_resource（LLM 可主动调用）
- ❌ recall：移到 Phase 3 + audit log（医疗器械 QMS 合规）

集成方式：手动集成（非 middleware）
- 原因：StateGraph.compile() 不支持 middleware 参数（LangGraph 1.2.5），
  OpenVikingContextMiddleware 是 langchain.agents.create_agent() 工厂的概念，
  不适用于项目自定义 StateGraph。
- 做法：在 agent_engine._agent_node 末尾直接调用 capture_messages()。
"""
import os
import logging
from typing import Optional, Any

logger = logging.getLogger(__name__)


class OpenVikingService:
    """OpenViking 集成服务封装（单例，Phase 1 capture-only）。"""

    def __init__(self):
        self.url = os.getenv("OPENVIKING_URL", "http://localhost:1933")
        self.api_key = os.getenv("OPENVIKING_API_KEY", "")
        self.account = os.getenv("OPENVIKING_ACCOUNT", "my-team")
        self.user = os.getenv("OPENVIKING_USER", "wangzichen")
        self.actor_peer_id = os.getenv("OPENVIKING_ACTOR_PEER_ID", "design-plan-agent")
        self.enabled = os.getenv("OPENVIKING_ENABLED", "false").lower() == "true"

        self._recorder = None
        self._capture_tools: list = []
        # 每个 session 已捕获的消息数（增量捕获，避免重复写入）
        self._session_captured_count: dict[str, int] = {}

    async def initialize(self) -> bool:
        """初始化 OpenViking 连接，返回是否成功（失败时静默降级）。"""
        if not self.enabled:
            logger.info("[OpenViking] Disabled by OPENVIKING_ENABLED=false")
            return False

        try:
            from langchain_openviking.recording import OpenVikingSessionRecorder
            from langchain_openviking.tools import create_openviking_tools

            # 初始化 recorder（capture 用）
            self._recorder = OpenVikingSessionRecorder(
                url=self.url,
                api_key=self.api_key,
                account=self.account,
                user=self.user,
                actor_peer_id=self.actor_peer_id,
                auto_initialize=True,
            )

            # 冒烟测试 + 获取 capture-only 工具（决策 #3 合规收窄）
            all_tools = create_openviking_tools(
                url=self.url,
                api_key=self.api_key,
                account=self.account,
                user=self.user,
                actor_peer_id=self.actor_peer_id,
                auto_initialize=True,
            )
            self._capture_tools = [
                t for t in all_tools if t.name in ("viking_store", "viking_add_resource")
            ]

            logger.info(
                "[OpenViking] Initialized: %d capture tools, recorder ready",
                len(self._capture_tools),
            )
            return True

        except Exception as e:
            logger.warning("[OpenViking] Initialization failed (non-fatal): %s", e)
            self._recorder = None
            self._capture_tools = []
            return False

    def is_available(self) -> bool:
        """OpenViking capture 是否可用。"""
        return self._recorder is not None

    def get_capture_tools(self) -> list:
        """返回 capture-only 工具列表（决策 #3 合规收窄）。

        返回 viking_store / viking_add_resource，不含 recall 类工具。
        """
        return list(self._capture_tools)

    async def capture_messages(self, messages: list, session_id: str) -> None:
        """增量捕获消息到 OpenViking（静默降级）。

        在 agent_engine._agent_node 末尾调用，捕获当前对话消息。
        使用 _session_captured_count 跟踪每个 session 的已捕获量，
        只写入增量消息，避免重复捕获。

        Args:
            messages: 当前 state 中的完整消息列表
            session_id: 会话 ID（LangGraph thread_id）
        """
        if not self._recorder or not session_id:
            return

        try:
            prev_count = self._session_captured_count.get(session_id, 0)
            total = len(messages)
            if total <= prev_count:
                # 无新增消息（如 HITL resume 未产生新回复）
                return

            new_messages = messages[prev_count:]
            if not new_messages:
                return

            await self._recorder.arecord(session_id, new_messages)
            self._session_captured_count[session_id] = total
            logger.debug(
                "[OpenViking] Captured %d messages for session %s (total: %d)",
                len(new_messages),
                session_id,
                total,
            )
        except Exception as e:
            # 静默降级：capture 失败不影响 Agent 运行
            logger.warning("[OpenViking] capture_messages failed (degraded): %s", e)

    async def close(self) -> None:
        """清理资源（应用关闭时调用）。"""
        if self._recorder is not None:
            try:
                await self._recorder.aclose()
            except Exception as e:
                logger.warning("[OpenViking] recorder close failed (non-fatal): %s", e)
            self._recorder = None
        self._capture_tools = []
        self._session_captured_count.clear()
        logger.info("[OpenViking] Service closed")


# ── 模块级单例 ──

_service: Optional[OpenVikingService] = None


async def init_openviking() -> bool:
    """初始化 OpenViking 服务（应用启动时调用）。

    Returns:
        True if OpenViking 可用，False if 降级（不可用但不影响现有功能）
    """
    global _service
    _service = OpenVikingService()
    ok = await _service.initialize()
    if ok:
        print("[openviking] Service initialized successfully")
    else:
        print("[openviking] Service not available (non-fatal, agent will work without it)")
    return ok


def get_openviking_service() -> Optional[OpenVikingService]:
    """获取 OpenViking 服务单例（未初始化时返回 None）。"""
    return _service


async def close_openviking() -> None:
    """关闭 OpenViking 服务（应用关闭时调用）。"""
    global _service
    if _service is not None:
        await _service.close()
        _service = None
