"""快速验证"精炼生成"开关功能（mock get_agent，无需启动服务）

覆盖:
1. 模式切换端点: concise=true / false 写入 state
2. 模式切换端点: 新线程(无 checkpoint) 用初始状态 seed
3. build_system_prompt: concise_mode=True 注入精炼指令块（仅文档内容生成）
4. build_system_prompt: concise_mode=False 输出与基线一致（回归护栏）
5. build_state_snapshot: 暴露 concise_mode 供前端初始化
6. create_initial_state: concise_mode 默认 False
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch


def _fake_state(values):
    """构造 aget_state 返回值"""
    class _State:
        def __init__(self, values):
            self.values = values
    return _State(values)


async def test_mode_endpoint_on():
    """开启精炼模式: 写入 concise_mode=True"""
    from app.api import routes as r

    fake_agent = MagicMock()
    fake_agent.aget_state = AsyncMock(return_value=_fake_state({"concise_mode": False}))
    fake_agent.aupdate_state = AsyncMock()
    with patch("app.services.agent_engine.get_agent", return_value=fake_agent):
        resp = await r.agent_set_generation_mode("proj_x", concise=True)

    assert resp["success"] is True and resp["concise_mode"] is True
    fake_agent.aupdate_state.assert_awaited_with(
        {"configurable": {"thread_id": "proj_x"}},
        {"concise_mode": True},
        as_node="after_tools",
    )
    print("场景1 OK: 开启精炼模式")


async def test_mode_endpoint_off():
    """关闭精炼模式: 写入 concise_mode=False"""
    from app.api import routes as r

    fake_agent = MagicMock()
    fake_agent.aget_state = AsyncMock(return_value=_fake_state({"concise_mode": True}))
    fake_agent.aupdate_state = AsyncMock()
    with patch("app.services.agent_engine.get_agent", return_value=fake_agent):
        resp = await r.agent_set_generation_mode("proj_x", concise=False)

    assert resp["success"] is True and resp["concise_mode"] is False
    fake_agent.aupdate_state.assert_awaited_with(
        {"configurable": {"thread_id": "proj_x"}},
        {"concise_mode": False},
        as_node="after_tools",
    )
    print("场景2 OK: 关闭精炼模式")


async def test_mode_endpoint_fresh_thread():
    """新线程无 checkpoint: 用 create_initial_state 合并模式 seed"""
    from app.api import routes as r

    fake_agent = MagicMock()
    fake_agent.aget_state = AsyncMock(return_value=_fake_state({}))  # 空 values → 无 checkpoint
    fake_agent.aupdate_state = AsyncMock()
    with patch("app.services.agent_engine.get_agent", return_value=fake_agent):
        resp = await r.agent_set_generation_mode("proj_new", concise=True)

    assert resp["success"] is True and resp["concise_mode"] is True
    call_args = fake_agent.aupdate_state.await_args
    state_vals = call_args.args[1]
    assert state_vals["concise_mode"] is True
    assert state_vals["messages"] == []           # 初始状态被 seed
    assert state_vals["attachments"] == []        # 初始状态被 seed
    assert call_args.kwargs["as_node"] == "after_tools"
    print("场景3 OK: 新线程 seed 初始状态")


async def test_prompt_concise_on():
    """concise_mode=True: 注入精炼指令块，且限定文档内容生成"""
    from app.services.agent_prompt import build_system_prompt

    prompt = build_system_prompt({
        "concise_mode": True,
        "product_name": "测试产品",
        "attachments": [],
    })
    assert "精炼生成模式（已开启）" in prompt, "应包含精炼模式指令块"
    assert "write_chapter" in prompt, "应限定文档生成工具作用域"
    assert "不改变聊天回复风格" in prompt, "应声明聊天回复不受影响"
    assert "# 回复风格" in prompt, "原有回复风格段应保留"
    print("场景4 OK: 开启时注入精炼指令块")


async def test_prompt_concise_off_baseline():
    """concise_mode=False/缺省: 输出与基线完全一致（回归护栏）"""
    from app.services.agent_prompt import build_system_prompt

    baseline = build_system_prompt({})
    off = build_system_prompt({"concise_mode": False})
    assert "精炼生成模式" not in baseline
    assert "精炼生成模式" not in off
    assert baseline == off, "关闭模式时提示词必须与基线一致"
    print("场景5 OK: 关闭时输出与基线一致")


async def test_snapshot_exposes_mode():
    """build_state_snapshot 暴露 concise_mode 供前端初始化"""
    from app.services.agent_state import build_state_snapshot, create_initial_state

    s_on = build_state_snapshot({"concise_mode": True})
    assert s_on["concise_mode"] is True

    s_off = build_state_snapshot({})
    assert s_off["concise_mode"] is False

    init = create_initial_state()
    assert init["concise_mode"] is False, "初始状态 concise_mode 应为 False"
    print("场景6 OK: 快照暴露 concise_mode + 初始状态默认 False")


async def main():
    await test_mode_endpoint_on()
    await test_mode_endpoint_off()
    await test_mode_endpoint_fresh_thread()
    await test_prompt_concise_on()
    await test_prompt_concise_off_baseline()
    await test_snapshot_exposes_mode()
    print("\nALL CONCISE-MODE TESTS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
