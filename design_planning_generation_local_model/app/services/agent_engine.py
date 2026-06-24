"""
Agent Engine — 设计策划文档Agent核心引擎

基于LangGraph StateGraph构建ReAct Agent循环:
- LLM自主决定调用工具或直接回复
- generate_section工具通过interrupt()实现HITL确认
- SqliteSaver自动检查点持久化
- astream_events() SSE流式输出
"""
import os
import re
import json
from typing import Optional, Literal
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import interrupt, Command
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

from app.services.agent_state import (
    AgentState,
    create_initial_state,
    get_checkpointer,
    build_state_snapshot,
)
from app.services.agent_prompt import build_system_prompt
from app.services.agent_tools import PHASE1_TOOLS, generate_section
from app.services.context_manager import maybe_compress_messages


# ── 全局Agent实例 (应用启动时初始化) ──

_agent_graph = None
_model = None


# ── LLM模型初始化 ──

def _get_model() -> ChatOpenAI:
    """获取或创建ChatOpenAI实例 (单例) - 使用本地Ollama"""
    global _model
    if _model is None:
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11435") + "/v1"
        model = os.getenv("OLLAMA_MODEL", "qwen3.5:122b")
        api_key = os.getenv("MINIMAX_API_KEY", "ollama")

        _model = ChatOpenAI(
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=0.3,
            max_tokens=4096,
        )
    return _model


def _get_model_with_tools():
    """获取绑定了工具的模型实例"""
    return _get_model().bind_tools(PHASE1_TOOLS)


# ── Graph节点 ──

async def _agent_node(state: AgentState) -> dict:
    """Agent节点: LLM决策 + 生成回复 + 决定工具调用

    这是StateGraph的核心节点。每轮对话执行:
    1. 压缩旧消息 (如果需要)
    2. 构建System Prompt (含当前状态快照)
    3. 调用LLM
    4. 返回回复 (文本或tool_calls)
    """
    messages = list(state.get("messages", []))

    # 上下文压缩检查
    messages, was_compressed = await maybe_compress_messages(messages)
    if was_compressed:
        print("[agent_engine] Context compressed — old messages summarized")

    # 同步文档上下文到工具层，确保 build_docx 无需 LLM 传入 markdown
    _sync_doc_context(state)

    # 同步附件上下文到工具层，确保 search_attachment 可访问附件内容
    _sync_attachment_context(state)

    # 构建并注入System Prompt
    system_prompt = build_system_prompt(state)
    full_messages = [SystemMessage(content=system_prompt)] + messages

    # 调用LLM
    model = _get_model_with_tools()
    response = await model.ainvoke(full_messages)

    # 如果发生了压缩，也更新状态中的messages
    if was_compressed:
        return {
            "messages": messages + [response],
        }

    return {"messages": [response]}


async def _pre_tool_node(state: AgentState) -> dict:
    """工具执行前节点: 对generate_section实现HITL暂停

    当LLM决定调用 generate_section 时，通过interrupt()暂停，
    等待用户在前端确认/修改/拒绝后再继续执行。
    auto_mode=True 时跳过所有HITL确认。
    """
    messages = state.get("messages", [])
    if not messages:
        return {}

    last_message = messages[-1]
    tool_calls = getattr(last_message, "tool_calls", None)

    if not tool_calls:
        return {}

    # 自动模式: 跳过所有HITL确认
    if state.get("auto_mode"):
        return {}

    # 检查是否有 generate_section 调用
    for tc in tool_calls:
        if tc.get("name") == "generate_section":
            section_name = tc.get("args", {}).get("section_name", "未知章节")
            # HITL暂停: 等待用户确认
            # Python 3.10 下 LangGraph get_config() 的 contextvar 可能跨 async
            # 任务传播失败。此时跳过 HITL 直接放行，避免阻塞文档生成流程。
            try:
                decision = interrupt({
                    "type": "generate_section_approval",
                    "tool_call_id": tc["id"],
                    "section_name": section_name,
                    "message": f"即将生成「{section_name}」章节。请确认、修改指令或跳过。",
                })
                print(f"[agent_engine] HITL decision for {section_name}: {decision}")
            except RuntimeError as e:
                if "get_config" in str(e):
                    print(f"[agent_engine] get_config unavailable, skipping HITL for {section_name}")
                else:
                    raise
            break

    return {}


# ── after_tools 节点: 将工具生成结果写入 state ──

async def _after_tools_node(state: AgentState) -> dict:
    """工具执行后节点: 将 generate_section / revise_section 的结果
    写入 generated_sections，使导出下载按钮可见。

    同时将 build_docx 的下载信息写入 state，供前端直接读取。

    重要: 遍历本轮 ALL 新增的 ToolMessage（而非仅 messages[-1]），
    修复多工具并行调用时前几个工具结果被丢弃导致章节内容丢失的 bug。
    """
    messages = state.get("messages", [])
    if not messages:
        return {}

    # 收集本轮新增的所有连续 ToolMessage（从末尾向前扫描）
    new_tool_messages = []
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            new_tool_messages.append(msg)
        else:
            break  # 遇到非 ToolMessage 即停止

    if not new_tool_messages:
        return {}

    # 构建 tool_call_id → (tool_name, tool_args) 的完整映射
    tool_call_map = {}
    for msg in messages:
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            for tc in tool_calls:
                tc_id = tc.get("id")
                if tc_id:
                    tool_call_map[tc_id] = (tc.get("name"), tc.get("args", {}))

    generated_sections = dict(state.get("generated_sections", {}))
    sections_updated = False
    doc_status = None
    outline_data = None
    outline_status = None

    for tool_msg in reversed(new_tool_messages):
        tool_call_id = getattr(tool_msg, "tool_call_id", None)
        if not tool_call_id:
            continue

        tool_name, tool_args = tool_call_map.get(tool_call_id, (None, {}))

        if tool_name in ("generate_section", "revise_section"):
            section_name = tool_args.get("section_name", "")
            if section_name:
                content = str(tool_msg.content)
                content = re.sub(r'^\[(?:章节|已修改):\s*[^\]]+\]\s*\n*', '', content)
                generated_sections[section_name] = content
                sections_updated = True
                print(f"[agent_engine] Updated generated_sections['{section_name}'] "
                      f"({len(content)} chars)")

        if tool_name == "build_docx":
            try:
                docx_result = json.loads(str(tool_msg.content))
                if docx_result.get("status") == "ok":
                    doc_status = "completed"
            except Exception:
                pass

        # ── 新增: design_outline 结果处理 ──
        if tool_name == "design_outline":
            outline = str(tool_msg.content)
            try:
                json.loads(outline)
                outline_data = outline
                outline_status = "draft"
                print(f"[agent_engine] outline stored "
                      f"({len(outline)} chars)")
            except json.JSONDecodeError:
                print("[agent_engine] design_outline returned invalid JSON")

        # ── 新增: write_chapter 结果处理 ──
        if tool_name == "write_chapter":
            chapter_name = tool_args.get("chapter_name", "")
            if chapter_name:
                # 优先从旁路 dict 读取完整内容（避免大段内容进入 LLM 对话历史）
                from app.services.agent_tools import get_pending_chapter_content
                pending = get_pending_chapter_content(chapter_name)
                content = pending.get("full_content", "")
                if not content:
                    # 回退：从 ToolMessage 读取（兼容旧格式）
                    content = str(tool_msg.content)
                generated_sections[chapter_name] = content
                sections_updated = True
                print(f"[agent_engine] Subagent wrote '{chapter_name}' "
                      f"({len(content)} chars)")

        # ── 新增: update_outline 结果处理 ──
        if tool_name == "update_outline":
            outline = str(tool_msg.content)
            try:
                json.loads(outline)
                outline_data = outline
                outline_status = "revised"
                print(f"[agent_engine] outline revised "
                      f"({len(outline)} chars)")
            except json.JSONDecodeError:
                pass

    result = {}
    if sections_updated:
        result["generated_sections"] = generated_sections
        _sync_doc_context(state, generated_sections)
    if doc_status:
        result["document_status"] = doc_status
    if outline_data:
        result["outline"] = outline_data
        result["outline_status"] = outline_status

    return result


def _sync_doc_context(state: AgentState,
                      generated_sections: dict = None) -> None:
    """将当前文档上下文同步到 agent_tools 的 contextvars，供 build_docx 自动读取"""
    from app.services.agent_tools import set_current_doc_context

    doc_type = state.get("doc_type", "design_development_plan")
    product_name = state.get("product_name", "贴敷式胰岛素泵")
    sections = generated_sections or state.get("generated_sections", {})

    # 组装完整 Markdown
    parts = []
    for name, content in sections.items():
        parts.append(f"# {name}\n\n{content}\n\n")
    full_md = "\n".join(parts)

    set_current_doc_context(doc_type, product_name, full_md)


def _sync_attachment_context(state: AgentState) -> None:
    """将当前附件上下文同步到 agent_tools 的 contextvars，供 search_attachment 使用"""
    from app.services.agent_tools import set_current_attachments

    attachments = state.get("attachments", [])
    set_current_attachments(attachments)


# ── Graph构建 ──

def _build_graph() -> StateGraph:
    """构建LangGraph StateGraph

    图结构:
        START → pre_tools → tools → after_tools → agent ←┐
                  ↑          ↓                           │
                  └──────────┼───────────────────────────┘
                             │
                             └────────────────────────────┘
                             (tools_condition循环)
    """
    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("agent", _agent_node)
    workflow.add_node("pre_tools", _pre_tool_node)
    workflow.add_node("tools", ToolNode(PHASE1_TOOLS))
    workflow.add_node("after_tools", _after_tools_node)

    # 添加边
    workflow.add_edge(START, "agent")

    # agent → 条件路由: 有tool_calls → pre_tools; 无 → END
    workflow.add_conditional_edges(
        "agent",
        tools_condition,
        {
            "tools": "pre_tools",
            END: END,
        },
    )

    # pre_tools → tools (如果未被interrupt)
    workflow.add_edge("pre_tools", "tools")

    # tools → after_tools → agent (after_tools 将工具结果写入 state)
    workflow.add_edge("tools", "after_tools")
    workflow.add_edge("after_tools", "agent")

    return workflow


# ── Agent 编译与初始化 ──

async def init_agent(db_path: Optional[str] = None):
    """初始化Agent (应用启动时调用一次)

    Args:
        db_path: SQLite检查点数据库路径
    """
    global _agent_graph

    checkpointer = await get_checkpointer(db_path)
    workflow = _build_graph()
    _agent_graph = workflow.compile(checkpointer=checkpointer)

    print(f"[agent_engine] Agent initialized with {len(PHASE1_TOOLS)} tools")
    print(f"[agent_engine] Checkpointer: {db_path}")
    return _agent_graph


def get_agent():
    """获取已编译的Agent图"""
    global _agent_graph
    if _agent_graph is None:
        raise RuntimeError("Agent not initialized. Call init_agent() first.")
    return _agent_graph


# ── Agent 调用接口 ──

async def invoke_agent(
    user_message: str,
    thread_id: str,
    initial_state: Optional[AgentState] = None,
) -> dict:
    """发送消息到Agent，获取完整回复 (非流式)

    Args:
        user_message: 用户消息文本
        thread_id: 会话/项目ID (用于检查点隔离)
        initial_state: 初始状态 (新项目时提供)

    Returns:
        更新后的状态dict
    """
    agent = get_agent()

    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 100,
    }

    if initial_state:
        state = dict(initial_state)
        state["messages"] = [HumanMessage(content=user_message)]
    else:
        state = {"messages": [HumanMessage(content=user_message)]}

    result = await agent.ainvoke(state, config=config)
    return result


async def stream_agent_events(
    user_message: str,
    thread_id: str,
    initial_state: Optional[AgentState] = None,
):
    """发送消息到Agent，返回SSE事件流

    事件类型:
    - on_chat_model_stream: LLM逐token输出 → 前端打字机效果
    - on_tool_start: 工具调用开始 → 前端展示"🔧检索中..."
    - on_tool_end: 工具调用结束 → 前端展示结果摘要
    - on_interrupt: HITL暂停 → 前端展示确认按钮
    - done: 流结束

    Yields:
        SSE格式的dict事件
    """
    agent = get_agent()

    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 100,
    }

    if initial_state:
        state = dict(initial_state)
        state["messages"] = [HumanMessage(content=user_message)]
    else:
        state = {"messages": [HumanMessage(content=user_message)]}

    async for event in agent.astream_events(state, config=config, version="v2"):
        event_type = event.get("event", "")
        event_name = event.get("name", "")

        if event_type == "on_chat_model_stream":
            # LangGraph 0.x: LLM token-by-token stream
            chunk = event["data"]["chunk"]
            if chunk.content:
                yield {
                    "type": "token",
                    "content": chunk.content,
                }

        elif event_type == "on_chain_stream" and event_name == "agent":
            # LangGraph 1.x: agent node output captured as chain stream
            # chunk is a dict: {"messages": [AIMessage(...)]}
            chunk_data = event["data"].get("chunk", {})
            messages = chunk_data.get("messages", [])
            for msg in messages:
                if hasattr(msg, "content") and msg.content:
                    yield {
                        "type": "token",
                        "content": msg.content,
                    }

        elif event_type == "on_tool_start":
            tool_name = event.get("name", "unknown")
            tool_input = event["data"].get("input", {})
            # 过滤敏感参数
            safe_input = {k: v for k, v in tool_input.items() if k not in ("api_key",)}
            yield {
                "type": "tool_start",
                "tool": tool_name,
                "input": safe_input,
            }
            # ── 子代理进度事件 ──
            if tool_name == "design_outline":
                yield {
                    "type": "subagent_start",
                    "agent": "outline_agent",
                    "message": "框架设计师正在设计文档结构..."
                }
            elif tool_name == "write_chapter":
                chapter = tool_input.get("chapter_name", "")
                yield {
                    "type": "subagent_start",
                    "agent": "chapter_agent",
                    "chapter": chapter,
                    "message": f"正在编写「{chapter}」..."
                }

        elif event_type == "on_tool_end":
            tool_name = event.get("name", "unknown")
            output = str(event["data"].get("output", ""))
            yield {
                "type": "tool_end",
                "tool": tool_name,
                "output_preview": output[:200],
            }
            # build_docx 完成时发射 file_ready 事件，前端弹出下载按钮
            if tool_name == "build_docx":
                import json as _json
                try:
                    result = _json.loads(output)
                    if result.get("status") == "ok" and result.get("download_id"):
                        yield {
                            "type": "file_ready",
                            "download_id": result["download_id"],
                            "filename": result["filename"],
                            "size_bytes": result.get("size_bytes", 0),
                        }
                except Exception:
                    pass
            # generate_section / revise_section 完成时发射 sections_ready 事件
            if tool_name in ("generate_section", "revise_section"):
                yield {
                    "type": "sections_ready",
                    "message": "文档章节已生成，可以下载",
                }
            # ── 子代理完成事件 ──
            if tool_name == "design_outline":
                yield {
                    "type": "subagent_complete",
                    "agent": "outline_agent",
                    "message": "文档框架已设计完成"
                }
            elif tool_name == "write_chapter":
                tool_input_end = event["data"].get("input", {})
                chapter = tool_input_end.get("chapter_name", "")
                yield {
                    "type": "subagent_complete",
                    "agent": "chapter_agent",
                    "chapter": chapter,
                    "message": f"「{chapter}」编写完成"
                }

        elif event_type == "on_interrupt":
            yield {
                "type": "waiting_approval",
                "message": "Agent等待你的确认...",
                "interrupt_data": event["data"],
            }

    yield {"type": "done"}


async def resume_agent(
    thread_id: str,
    decision: str,
):
    """HITL暂停后恢复Agent执行

    Args:
        thread_id: 会话/项目ID
        decision: 用户决定 ("approve" | "reject" | "edit:xxx")

    Yields:
        SSE格式的dict事件
    """
    agent = get_agent()
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 100,
    }

    if decision.startswith("edit:"):
        edited_content = decision[5:]
        resume_value = Command(resume={"action": "edit", "content": edited_content})
    elif decision == "reject":
        resume_value = Command(resume={"action": "reject"})
    else:
        resume_value = Command(resume={"action": "approve"})

    async for event in agent.astream_events(
        resume_value,
        config=config,
        version="v2",
    ):
        event_type = event.get("event", "")
        event_name = event.get("name", "")

        if event_type == "on_chat_model_stream":
            # LangGraph 0.x: LLM token-by-token stream
            chunk = event["data"]["chunk"]
            if chunk.content:
                yield {
                    "type": "token",
                    "content": chunk.content,
                }

        elif event_type == "on_chain_stream" and event_name == "agent":
            # LangGraph 1.x: agent node output captured as chain stream
            chunk_data = event["data"].get("chunk", {})
            messages = chunk_data.get("messages", [])
            for msg in messages:
                if hasattr(msg, "content") and msg.content:
                    yield {
                        "type": "token",
                        "content": msg.content,
                    }

        elif event_type == "on_tool_start":
            tool_name = event.get("name", "unknown")
            tool_input = event["data"].get("input", {})
            yield {
                "type": "tool_start",
                "tool": tool_name,
            }
            # ── 子代理进度事件 ──
            if tool_name == "design_outline":
                yield {
                    "type": "subagent_start",
                    "agent": "outline_agent",
                    "message": "框架设计师正在设计文档结构..."
                }
            elif tool_name == "write_chapter":
                chapter = tool_input.get("chapter_name", "")
                yield {
                    "type": "subagent_start",
                    "agent": "chapter_agent",
                    "chapter": chapter,
                    "message": f"正在编写「{chapter}」..."
                }

        elif event_type == "on_tool_end":
            tool_name = event.get("name", "unknown")
            output = str(event["data"].get("output", ""))
            yield {
                "type": "tool_end",
                "tool": tool_name,
                "output_preview": output[:200],
            }
            if tool_name == "build_docx":
                import json as _json
                try:
                    result = _json.loads(output)
                    if result.get("status") == "ok" and result.get("download_id"):
                        yield {
                            "type": "file_ready",
                            "download_id": result["download_id"],
                            "filename": result["filename"],
                            "size_bytes": result.get("size_bytes", 0),
                        }
                except Exception:
                    pass
            if tool_name in ("generate_section", "revise_section"):
                yield {
                    "type": "sections_ready",
                    "message": "文档章节已生成，可以下载",
                }
            # ── 子代理完成事件 ──
            if tool_name == "design_outline":
                yield {
                    "type": "subagent_complete",
                    "agent": "outline_agent",
                    "message": "文档框架已设计完成"
                }
            elif tool_name == "write_chapter":
                tool_input_end = event["data"].get("input", {})
                chapter = tool_input_end.get("chapter_name", "")
                yield {
                    "type": "subagent_complete",
                    "agent": "chapter_agent",
                    "chapter": chapter,
                    "message": f"「{chapter}」编写完成"
                }

        elif event_type == "on_interrupt":
            yield {
                "type": "waiting_approval",
                "message": "Agent等待你的确认...",
            }

    yield {"type": "done"}


async def get_agent_state(thread_id: str) -> dict:
    """获取指定会话的当前状态

    Args:
        thread_id: 会话/项目ID

    Returns:
        状态快照dict (供前端进度面板使用)
    """
    agent = get_agent()
    config = {"configurable": {"thread_id": thread_id}}

    try:
        state = await agent.aget_state(config)
        if state and state.values:
            return build_state_snapshot(state.values)
        return build_state_snapshot(create_initial_state())
    except Exception:
        return build_state_snapshot(create_initial_state())
