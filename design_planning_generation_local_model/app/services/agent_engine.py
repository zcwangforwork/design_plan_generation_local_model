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
import time
from typing import Optional, Literal
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import interrupt, Command
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage
from langchain_core.runnables import RunnableConfig

from app.services.agent_state import (
    AgentState,
    create_initial_state,
    get_checkpointer,
    build_state_snapshot,
)
from app.services.agent_prompt import build_system_prompt
from app.services.agent_tools import (
    PHASE1_TOOLS,
    generate_section,
    _user_asks_realtime,
    _looks_like_realtime_refusal,
    _user_asks_attachment_process,
)
from app.services.context_manager import maybe_compress_messages


# ── 全局Agent实例 (应用启动时初始化) ──

_agent_graph = None
_model = None

# 携带 doc_type 参数的文档生成相关工具：用于从工具调用参数中提取文档类型并持久化到 state
_DOC_TYPE_TOOLS = (
    "design_outline",
    "outline_from_attachment",
    "outline_from_template",
    "write_chapter",
    "generate_section",
    "revise_section",
    "revise_paragraph",
    "build_docx",
)


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

async def _agent_node(state: AgentState, config: RunnableConfig = None) -> dict:
    """Agent节点: LLM决策 + 生成回复 + 决定工具调用

    这是StateGraph的核心节点。每轮对话执行:
    1. 压缩旧消息 (如果需要)
    2. 自动检索 OpenViking 相关记忆 (Phase 3 recall)
    3. 构建System Prompt (含当前状态快照 + 记忆上下文)
    4. 调用LLM
    5. 返回回复 (文本或tool_calls)
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

    # 同步模板上下文到工具层，确保 outline_from_template 可访问模板内容
    _sync_template_context(state)

    # ── Phase 3 recall: 自动检索 OpenViking 相关记忆 ──
    # 用当前用户消息作为查询，从 OpenViking 跨会话检索相关历史记忆，
    # 注入 system prompt，让 Agent 无需主动调用工具也能获得历史上下文。
    memory_context = await _recall_memory_context(messages)

    # 构建并注入System Prompt（含记忆上下文）
    system_prompt = build_system_prompt(state, memory_context=memory_context)
    full_messages = [SystemMessage(content=system_prompt)] + messages

    # 调用LLM
    model = _get_model_with_tools()
    response = await model.ainvoke(full_messages)

    # OpenViking capture（静默降级，Phase 3 capture + recall）
    # 每轮 agent 对话后捕获消息到 OpenViking，为后续 recall 积累数据
    thread_id = ""
    if config:
        thread_id = config.get("configurable", {}).get("thread_id", "")
    await _capture_to_openviking(messages + [response], thread_id)

    # ── 代码级兜底：LLM 未调用工具但给出"无法获取实时信息"式拒绝，且用户问题涉及实时话题 ──
    # 此时强制插入 web_search 工具调用，让图路由到 ToolNode 执行真实网络搜索，
    # 搜索完成后自动回到本节点，LLM 依据真实结果重新回答。
    # 场景：本地 LLM 偶尔不遵循提示词强规则（如问天气时直接拒绝），此兜底保证触发真实搜索。
    if not getattr(response, "tool_calls", None):
        user_text = str(messages[-1].content) if messages else ""
        resp_text = str(getattr(response, "content", "")) or ""
        if _user_asks_realtime(user_text) and _looks_like_realtime_refusal(resp_text):
            query = user_text.strip()[:200] or resp_text[:80]
            forced = AIMessage(
                content="",  # 仅携带工具调用，触发真实网络搜索
                tool_calls=[{
                    "name": "web_search",
                    "args": {"query": query},
                    "id": f"call_backstop_{int(time.time() * 1000)}",
                    "type": "tool_call",
                }],
            )
            print(f"[agent_engine] 代码兜底: 强制触发 web_search (query={query[:40]})")
            return {"messages": [forced]}

        # ── 代码级兜底：用户要求处理上传附件（修改/补全/精简），但 LLM 未调用任何工具 ──
        # 本地 LLM 偶尔不遵循提示词，用户说"帮我补充/修改/精简这篇上传的文档"时，
        # 可能直接口头回答甚至伪造"已生成/已下载"，而不调用附件处理工具。
        # 此兜底检测到用户意图 + 存在附件时，强制插入对应工具调用，
        # 让工具真正生成可下载的修改版/补全版/精简版文档。
        # 仅当本轮最后一条消息是用户消息（HumanMessage）时触发；工具执行后的轮次
        # messages[-1] 是 ToolMessage，跳过以防空转/死循环。
        last_msg = messages[-1] if messages else None
        if getattr(last_msg, "type", None) == "human":
            action = _user_asks_attachment_process(user_text)
            if action and state.get("attachments"):
                tool_name = {
                    "modify": "modify_attachment",
                    "enrich": "enrich_attachment",
                    "summarize": "summarize_attachment",
                }[action]
                if action == "summarize":
                    args = {"file_id": ""}
                else:
                    args = {"instruction": user_text.strip(), "file_id": ""}
                forced = AIMessage(
                    content="",  # 仅携带工具调用，触发真实附件处理
                    tool_calls=[{
                        "name": tool_name,
                        "args": args,
                        "id": f"call_attach_backstop_{int(time.time() * 1000)}",
                        "type": "tool_call",
                    }],
                )
                print(f"[agent_engine] 代码兜底: 强制触发 {tool_name} (action={action})")
                return {"messages": [forced]}

    # 如果发生了压缩，也更新状态中的messages
    if was_compressed:
        return {
            "messages": messages + [response],
        }

    return {"messages": [response]}


async def _recall_memory_context(messages: list) -> str:
    """从 OpenViking 自动检索相关历史记忆（静默降级）。

    Phase 3 recall：每轮对话前，用最新用户消息作为查询，
    从 OpenViking 跨会话语义检索相关记忆，格式化为 system prompt 注入块。

    失败或不可用时返回空字符串，不影响 Agent 运行。
    """
    try:
        from app.services.openviking_client import get_openviking_service
        ov_service = get_openviking_service()
        if not ov_service or not ov_service.is_available():
            return ""

        # 取最后一条用户消息作为查询
        user_query = ""
        for msg in reversed(messages):
            if hasattr(msg, "type") and msg.type == "human":
                user_query = str(getattr(msg, "content", ""))[:500]
                break

        if not user_query or not user_query.strip():
            return ""

        memories = await ov_service.search_memories(query=user_query, limit=5)
        if not memories:
            return ""

        # 格式化记忆为 system prompt 注入块
        lines = [
            "## 历史相关记忆（来自 OpenViking 自动检索）",
            "以下是从过往会话中检索到的相关信息，可作为当前对话的参考：",
            "",
        ]
        for i, mem in enumerate(memories):
            abstract = (mem.get("abstract") or "").strip()
            overview = (mem.get("overview") or "").strip()
            uri = mem.get("uri", "")
            score = mem.get("score", 0)
            text = abstract or overview
            if not text:
                continue
            # 截断每条记忆，避免占用过多 context window
            text = text[:600]
            lines.append(f"### 记忆 {i + 1}（相关度: {score:.2f}）")
            lines.append(text)
            lines.append("")

        if len(lines) <= 3:
            return ""  # 没有有效内容

        context = "\n".join(lines)
        print(f"[agent_engine] OpenViking recall: {len(memories)} memories injected "
              f"({len(context)} chars)")
        return context

    except Exception:
        return ""  # 静默降级


async def _capture_to_openviking(messages: list, thread_id: str = "") -> None:
    """将对话消息捕获到 OpenViking（静默降级）。

    Phase 3 capture + recall：每轮 agent 对话后记录消息到 OpenViking，
    为后续 recall 积累数据。失败时不影响 Agent 运行。

    使用 LangGraph thread_id 作为 OpenViking session_id（Phase 0 验证：
    _resolve_session_id 直接读 thread_id，无需前缀映射）。

    thread_id 优先由 _agent_node 的 config 参数传入（可靠，不依赖 contextvar）；
    未传入时再回退 get_config() 兜底（覆盖非流式 ainvoke 等场景）。
    """
    try:
        from app.services.openviking_client import get_openviking_service
        ov_service = get_openviking_service()
        if not ov_service or not ov_service.is_available():
            return

        # 兜底：config 未传入 thread_id 时，尝试从 get_config() 读取。
        # Python 3.10 下 LangGraph get_config() 的 contextvar 可能跨 async
        # 任务传播失败，故仅作兜底，主路径已通过 config 参数传入。
        if not thread_id:
            try:
                from langgraph.config import get_config
                config = get_config()
                thread_id = config.get("configurable", {}).get("thread_id", "")
            except (RuntimeError, Exception):
                pass

        if not thread_id:
            return

        await ov_service.capture_messages(messages, thread_id)
    except Exception:
        pass  # 静默降级：capture 失败不影响 Agent 运行


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
    modifications = []  # attachment_modifications 新增项
    detected_doc_type = None  # 本轮工具调用中检测到的文档类型
    supplementary_prompts = None  # 本轮累积的新增补充提示词
    # ── 撤销快照：记录本次变更前状态，供「回退」恢复 ──
    sections_before = {}  # 被覆盖章节旧值（章节名 -> 旧内容或 None，None 表示原本不存在）
    attachments_before_len = len(state.get("attachment_modifications", []) or [])
    removed_file_ids = []  # 本次新增附件修改的 file_id（回退时前端用于清理下载按钮）
    undo_desc_parts = []  # 撤销描述片段

    for tool_msg in reversed(new_tool_messages):
        tool_call_id = getattr(tool_msg, "tool_call_id", None)
        if not tool_call_id:
            continue

        tool_name, tool_args = tool_call_map.get(tool_call_id, (None, {}))

        # ── 从工具参数中提取文档类型，持久化到 state，避免文档名称恒为"设计开发策划书" ──
        if tool_name in _DOC_TYPE_TOOLS and tool_args.get("doc_type"):
            detected_doc_type = tool_args["doc_type"]

        if tool_name in ("generate_section", "revise_section", "revise_paragraph"):
            section_name = tool_args.get("section_name", "")
            if section_name:
                content = str(tool_msg.content)
                content = re.sub(r'^\[(?:章节|已修改|已修改段落):\s*[^\]]+\]\s*\n*', '', content)
                if section_name not in sections_before:
                    sections_before[section_name] = generated_sections.get(section_name)
                generated_sections[section_name] = content
                sections_updated = True
                undo_desc_parts.append(
                    f"生成章节「{section_name}」" if tool_name == "generate_section"
                    else f"修改章节「{section_name}」")
                print(f"[agent_engine] Updated generated_sections['{section_name}'] "
                      f"({len(content)} chars)")

        if tool_name == "build_docx":
            try:
                docx_result = json.loads(str(tool_msg.content))
                if docx_result.get("status") == "ok":
                    doc_status = "completed"
            except Exception:
                pass

        # ── 新增: generate_supplementary_prompt 结果处理 ──
        # 将生成的补充提示词累积到 state，追加到后续文档生成提示词末尾（纯补充）
        if tool_name == "generate_supplementary_prompt":
            try:
                parsed = json.loads(str(tool_msg.content))
            except Exception:
                parsed = {}
            sp_text = (parsed.get("supplementary_prompt", "") if isinstance(parsed, dict) else "") or ""
            if sp_text and sp_text.strip():
                if supplementary_prompts:
                    supplementary_prompts = (supplementary_prompts + "\n\n" + sp_text.strip()).strip()
                else:
                    supplementary_prompts = sp_text.strip()
                print(f"[agent_engine] accumulate supplementary_prompts "
                      f"(+{len(sp_text.strip())} chars)")

        # ── 新增: design_outline / outline_from_attachment / outline_from_template 结果处理 ──
        if tool_name in ("design_outline", "outline_from_attachment", "outline_from_template"):
            outline = str(tool_msg.content)
            try:
                parsed = json.loads(outline)
                # 仅当返回有效框架（含 chapters 字段）时才视为新文档；
                # 避免把 error 返回（{"status":"error"}）误判为新框架而清空已有章节
                if parsed.get("status") == "error" or "chapters" not in parsed:
                    print(f"[agent_engine] {tool_name} returned error/empty, skip reset")
                else:
                    outline_data = outline
                    outline_status = "draft"
                    # 新文档框架已生成 → 清空上一份文档的已生成章节，
                    # 避免前几次生成的文档内容累积拼接到新文档中
                    generated_sections = {}
                    sections_updated = True
                    print(f"[agent_engine] outline stored via {tool_name}, "
                          f"reset generated_sections ({len(outline)} chars)")
            except json.JSONDecodeError:
                print(f"[agent_engine] {tool_name} returned invalid JSON")

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
                if chapter_name not in sections_before:
                    sections_before[chapter_name] = generated_sections.get(chapter_name)
                generated_sections[chapter_name] = content
                sections_updated = True
                undo_desc_parts.append(f"写入章节「{chapter_name}」")
                print(f"[agent_engine] Subagent wrote '{chapter_name}' "
                      f"({len(content)} chars)")

        # ── 新增: summarize_section 结果处理 ──
        # 与 write_chapter 类似：从 _pending_chapter_contents 旁路读取精简后完整内容，
        # 用精简后内容替换 generated_sections[section_name]
        if tool_name == "summarize_section":
            section_name = tool_args.get("section_name", "")
            if section_name:
                from app.services.agent_tools import get_pending_chapter_content
                pending = get_pending_chapter_content(section_name)
                content = pending.get("full_content", "")
                if content:
                    if section_name not in sections_before:
                        sections_before[section_name] = generated_sections.get(section_name)
                    generated_sections[section_name] = content
                    sections_updated = True
                    undo_desc_parts.append(f"精简章节「{section_name}」")
                    print(f"[agent_engine] Summarized '{section_name}' "
                          f"({len(content)} chars)")
                else:
                    print(f"[agent_engine] summarize_section '{section_name}': "
                          f"no pending content (旁路为空，可能工具失败)")

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

        # ── 新增: modify_attachment / enrich_attachment 结果处理 ──
        # 完整修改后（或补全后）文档从旁路 dict 读取（避免大段内容进入 LLM 对话历史），
        # 工具返回的 JSON 提供 file_id / filename / summary / download_id。
        if tool_name in ("modify_attachment", "enrich_attachment", "summarize_attachment"):
            try:
                parsed = json.loads(str(tool_msg.content))
            except Exception:
                parsed = {}
            if parsed.get("status") == "ok":
                from app.services.agent_tools import get_pending_modified_document
                from datetime import datetime as _dt
                file_id = tool_args.get("file_id", "") or parsed.get("file_id", "")
                pending = get_pending_modified_document(file_id) if file_id else None
                pending = pending or {}
                _filename = parsed.get("filename", "") or pending.get("filename", "")
                _summary = parsed.get("summary", "")
                _timestamp = _dt.now().isoformat(timespec="seconds")

                # 段落级手术：pending 携带编辑指令 ops + 原文件路径，下载时在原 docx 上执行
                ops = pending.get("ops")
                original_path = pending.get("original_path", "")
                if ops:
                    modifications.append({
                        "file_id": file_id,
                        "filename": _filename,
                        "kind": pending.get("kind", ""),
                        "ops": ops,
                        "original_path": original_path,
                        "summary": _summary,
                        "modified_chars": parsed.get("modified_chars", 0),
                        "timestamp": _timestamp,
                    })
                    if file_id:
                        removed_file_ids.append(file_id)
                    undo_desc_parts.append(
                        f"补充附件「{_filename or file_id}」" if tool_name == "enrich_attachment"
                        else f"修改附件「{_filename or file_id}」")
                    print(f"[agent_engine] Saved docx-edit attachment "
                          f"'{_filename}' ({len(ops)} ops)")
                else:
                    markdown = pending.get("markdown", "")
                    if not markdown:
                        markdown = parsed.get("modified_markdown", "")
                    if markdown:
                        modifications.append({
                            "file_id": file_id,
                            "filename": _filename,
                            "modified_markdown": markdown,
                            "summary": _summary,
                            "modified_chars": parsed.get("modified_chars", len(markdown)),
                            "timestamp": _timestamp,
                        })
                        if file_id:
                            removed_file_ids.append(file_id)
                        undo_desc_parts.append(
                            f"补充附件「{_filename or file_id}」" if tool_name == "enrich_attachment"
                            else f"修改附件「{_filename or file_id}」")
                        print(f"[agent_engine] Saved modified attachment "
                              f"'{parsed.get('filename', '?')}' ({len(markdown)} chars)")

    # 优先使用本轮工具调用中检测到的 doc_type，其次沿用已有 state 值
    effective_doc_type = detected_doc_type or state.get("doc_type") or "design_development_plan"

    result = {}
    if detected_doc_type:
        result["doc_type"] = detected_doc_type
    if sections_updated:
        result["generated_sections"] = generated_sections
        _sync_doc_context(state, generated_sections, doc_type=effective_doc_type)
    if doc_status:
        result["document_status"] = doc_status
    if outline_data:
        result["outline"] = outline_data
        result["outline_status"] = outline_status
    if modifications:
        existing_mods = list(state.get("attachment_modifications", []) or [])
        result["attachment_modifications"] = existing_mods + modifications
    if supplementary_prompts:
        existing_sp = (state.get("supplementary_prompts") or "").strip()
        merged_sp = (existing_sp + "\n\n" + supplementary_prompts).strip() if existing_sp else supplementary_prompts
        result["supplementary_prompts"] = merged_sp

    # ── 撤销快照：本轮若覆盖了章节或新增了附件修改，压入撤销栈（单步回退） ──
    if sections_before or modifications:
        undo_entry = {
            "sections_before": sections_before,
            "attachments_before_len": attachments_before_len,
            "removed_file_ids": removed_file_ids,
            "description": "、".join(undo_desc_parts) if undo_desc_parts else "修改文档",
        }
        existing_undo = list(state.get("undo_stack", []) or [])
        result["undo_stack"] = (existing_undo + [undo_entry])[-50:]  # 限制栈深 50

    return result


def _sync_doc_context(state: AgentState,
                      generated_sections: dict = None,
                      doc_type: str = None) -> None:
    """将当前文档上下文同步到 agent_tools 的 contextvars，供 build_docx 自动读取"""
    from app.services.agent_tools import set_current_doc_context

    if doc_type is None:
        doc_type = state.get("doc_type", "design_development_plan")
    product_name = state.get("product_name", "贴敷式胰岛素泵")
    product_classification = state.get("product_classification", "")
    product_intended_use = state.get("product_intended_use", "")
    confirmed_standards = state.get("confirmed_standards", [])
    sections = generated_sections or state.get("generated_sections", {})

    # 组装完整 Markdown
    parts = []
    for name, content in sections.items():
        parts.append(f"# {name}\n\n{content}\n\n")
    full_md = "\n".join(parts)

    supplementary_prompts = state.get("supplementary_prompts", "") or ""

    set_current_doc_context(
        doc_type,
        product_name,
        full_md,
        product_classification=product_classification,
        product_intended_use=product_intended_use,
        confirmed_standards=confirmed_standards,
        supplementary_prompts=supplementary_prompts,
    )


def _sync_attachment_context(state: AgentState) -> None:
    """将当前附件上下文同步到 agent_tools 的 contextvars，供 search_attachment 使用"""
    from app.services.agent_tools import set_current_attachments

    attachments = state.get("attachments", [])
    set_current_attachments(attachments)


def _sync_template_context(state: AgentState) -> None:
    """将当前模板上下文同步到 agent_tools 的 contextvars，供 outline_from_template 使用"""
    from app.services.agent_tools import set_current_templates

    templates = state.get("templates", [])
    set_current_templates(templates)


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

    # 初始化领域 SQLite 数据库（建表 + 一次性种子，幂等；agent 的 SQL 工具查询源）
    try:
        from app.services.sql_db import init_db
        print(f"[agent_engine] SQL domain db: {init_db()}")
    except Exception as e:
        print(f"[agent_engine] SQL domain db init failed (non-fatal): {e}")

    # 初始化 OpenViking（静默降级，评审决策 #3 Phase 1 capture-only）
    from app.services.openviking_client import init_openviking, get_openviking_service
    await init_openviking()

    # 追加 OpenViking capture 工具到 PHASE1_TOOLS（去重）
    # capture-only: viking_store / viking_add_resource（合规收窄，无 recall）
    ov_service = get_openviking_service()
    if ov_service and ov_service.is_available():
        capture_tools = ov_service.get_capture_tools()
        for t in capture_tools:
            if t not in PHASE1_TOOLS:
                PHASE1_TOOLS.append(t)
        print(f"[agent_engine] OpenViking tools (capture + recall) added: {len(capture_tools)}")
    else:
        print("[agent_engine] OpenViking not available (non-fatal)")

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


class _LeakFilter:
    """流式输出兜底过滤器：剥掉弱模型复读的工具结果 JSON / 进度摘要。

    本地模型（qwen 等）偶发把 ToolMessage 原文或上下文压缩摘要原样写进回复正文，
    且经常把多个工具结果 JSON 对象首尾拼接在一行（`{...}{...}{...}`）再复读。
    旧实现用 json.loads() 只能解析单个对象，拼接后解析失败导致整行漏放。
    这里按「完整行」过滤以下泄漏：
    - 工具结果 JSON（单个或多个拼接），按含 "status" 字段的对象识别并剥离
    - 以 [进度回顾] / （系统内部上下文摘要 起始的块，及随后的摘要正文（直到遇到真正的 Markdown 结构）
    - 与某条已见工具输出逐字相同的行
    """
    _PROGRESS_MARKERS = ("[进度回顾]", "（系统内部上下文摘要")

    def __init__(self):
        self._buf = ""
        self._tool_outputs = set()
        self._in_summary_block = False
        self._summary_lines = 0

    def add_tool_output(self, output: str) -> None:
        s = (output or "").strip()
        if len(s) >= 16:
            self._tool_outputs.add(s)

    def feed(self, chunk: str) -> list[str]:
        """喂入 chunk，返回可安全输出的完整行（含换行符）。"""
        self._buf += chunk
        if "\n" not in self._buf:
            return []
        parts = self._buf.split("\n")
        self._buf = parts[-1]
        out = []
        for line in parts[:-1]:
            kept = self._clean_line(line)
            if kept is not None:
                out.append(kept + "\n")
        return out

    def flush(self) -> list[str]:
        """流结束时冲刷剩余内容。"""
        if not self._buf:
            return []
        line = self._buf
        self._buf = ""
        kept = self._clean_line(line)
        return [kept] if kept is not None else []

    def _clean_line(self, line: str):
        raw = line.strip()
        if not raw:
            return line  # 空行保留

        # 1. 摘要块内：持续丢弃，直到遇到真正的 Markdown 结构（标题/加粗/表格/列表）
        if self._in_summary_block:
            if self._looks_like_real_answer(raw) or self._summary_lines > 40:
                self._in_summary_block = False
                self._summary_lines = 0
            else:
                self._summary_lines += 1
                return None

        # 2. 剥离工具结果 JSON（支持多个对象首尾拼接）
        s = self._strip_tool_json(raw)
        if s is None:
            return None

        # 3. 摘要起始标记（可能在行首，也可能紧跟在被剥离的 JSON 之后）
        marker_idx = self._find_summary_marker(s)
        if marker_idx is not None:
            self._in_summary_block = True
            self._summary_lines = 0
            before = s[:marker_idx].strip()
            return before if before else None

        # 4. 与已见工具输出逐字相同的行
        if s in self._tool_outputs:
            return None

        # 无修改则保留原文，否则返回剥离 JSON 后的剩余文本
        return line if s == raw else s

    def _find_summary_marker(self, s: str):
        """返回摘要标记在 s 中的起始下标；未找到返回 None。"""
        for m in self._PROGRESS_MARKERS:
            idx = s.find(m)
            if idx != -1:
                return idx
        return None

    def _looks_like_real_answer(self, s: str) -> bool:
        """判断一行是否像模型真正回答的开头（而非摘要正文的复读）。"""
        return (
            s.startswith(("#", "**", "|", "> ", "- ", "* ")) or
            (len(s) > 1 and s[0].isdigit() and s[1] in ".、)")
        )

    def _strip_tool_json(self, s: str):
        """剥离行内的工具结果 JSON 对象（含多个首尾拼接）。返回剩余文本或 None（整行为 JSON）。"""
        if '"status"' not in s or '{' not in s:
            return s
        decoder = json.JSONDecoder()
        out = []
        idx = 0
        stripped = False
        while True:
            brace = s.find('{', idx)
            if brace == -1:
                out.append(s[idx:])
                break
            out.append(s[idx:brace])
            try:
                obj, end = decoder.raw_decode(s, brace)
            except json.JSONDecodeError:
                # 非法 JSON，保留该 '{' 并继续扫描后续
                out.append(s[brace])
                idx = brace + 1
                continue
            if isinstance(obj, dict) and "status" in obj:
                stripped = True
            else:
                out.append(s[brace:end])
            idx = end
        if not stripped:
            return s
        remainder = "".join(out).strip()
        return remainder if remainder else None


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

    leak_filter = _LeakFilter()

    async for event in agent.astream_events(state, config=config, version="v2"):
        event_type = event.get("event", "")
        event_name = event.get("name", "")

        if event_type == "on_chat_model_stream":
            # LangGraph 0.x: LLM token-by-token stream
            chunk = event["data"]["chunk"]
            if chunk.content:
                for line in leak_filter.feed(chunk.content):
                    yield {
                        "type": "token",
                        "content": line,
                    }

        elif event_type == "on_chain_stream" and event_name == "agent":
            # LangGraph 1.x: agent node output captured as chain stream
            # chunk is a dict: {"messages": [AIMessage(...)]}
            chunk_data = event["data"].get("chunk", {})
            messages = chunk_data.get("messages", [])
            for msg in messages:
                if hasattr(msg, "content") and msg.content:
                    for line in leak_filter.feed(msg.content):
                        yield {
                            "type": "token",
                            "content": line,
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
            leak_filter.add_tool_output(output)
            yield {
                "type": "tool_end",
                "tool": tool_name,
                "output_preview": output[:200],
            }
            # search_kb / search_attachment 完成时，推送完整检索结果到前端
            if tool_name in ("search_kb", "search_attachment"):
                try:
                    import json as _json
                    parsed = _json.loads(output)
                    if parsed.get("status") == "ok" and parsed.get("results"):
                        yield {
                            "type": "rag_results",
                            "tool": tool_name,
                            "query": parsed.get("query", ""),
                            "count": parsed.get("count", 0),
                            "results": parsed["results"],
                        }
                except Exception:
                    pass
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
            # modify_attachment 完成时发射 modified_doc_ready 事件，前端弹出修改版下载
            if tool_name in ("modify_attachment", "enrich_attachment", "summarize_attachment"):
                import json as _json
                try:
                    result = _json.loads(output)
                    if result.get("status") == "ok":
                        yield {
                            "type": "modified_doc_ready",
                            "file_id": result.get("file_id", ""),
                            "filename": result.get("filename", ""),
                            "modified_chars": result.get("modified_chars", 0),
                            "summary": result.get("summary", ""),
                            "kind": result.get("kind", "modify"),
                        }
                except Exception:
                    pass
            # generate_section / revise_section / revise_paragraph 完成时发射 sections_ready 事件
            if tool_name in ("generate_section", "revise_section", "revise_paragraph"):
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

    for line in leak_filter.flush():
        yield {
            "type": "token",
            "content": line,
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

    leak_filter = _LeakFilter()

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
                for line in leak_filter.feed(chunk.content):
                    yield {
                        "type": "token",
                        "content": line,
                    }

        elif event_type == "on_chain_stream" and event_name == "agent":
            # LangGraph 1.x: agent node output captured as chain stream
            chunk_data = event["data"].get("chunk", {})
            messages = chunk_data.get("messages", [])
            for msg in messages:
                if hasattr(msg, "content") and msg.content:
                    for line in leak_filter.feed(msg.content):
                        yield {
                            "type": "token",
                            "content": line,
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
            leak_filter.add_tool_output(output)
            yield {
                "type": "tool_end",
                "tool": tool_name,
                "output_preview": output[:200],
            }
            # search_kb / search_attachment 完成时，推送完整检索结果到前端
            if tool_name in ("search_kb", "search_attachment"):
                try:
                    import json as _json
                    parsed = _json.loads(output)
                    if parsed.get("status") == "ok" and parsed.get("results"):
                        yield {
                            "type": "rag_results",
                            "tool": tool_name,
                            "query": parsed.get("query", ""),
                            "count": parsed.get("count", 0),
                            "results": parsed["results"],
                        }
                except Exception:
                    pass
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
            if tool_name in ("modify_attachment", "enrich_attachment", "summarize_attachment"):
                import json as _json
                try:
                    result = _json.loads(output)
                    if result.get("status") == "ok":
                        yield {
                            "type": "modified_doc_ready",
                            "file_id": result.get("file_id", ""),
                            "filename": result.get("filename", ""),
                            "modified_chars": result.get("modified_chars", 0),
                            "summary": result.get("summary", ""),
                            "kind": result.get("kind", "modify"),
                        }
                except Exception:
                    pass
            if tool_name in ("generate_section", "revise_section", "revise_paragraph"):
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

    for line in leak_filter.flush():
        yield {
            "type": "token",
            "content": line,
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
