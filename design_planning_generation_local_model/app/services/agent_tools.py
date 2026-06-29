"""
Agent Tools — 设计策划文档Agent工具集 (Phase 1: 4核心工具)

Tool 1: search_kb     — 检索贴敷式胰岛素泵知识库
Tool 2: generate_section — 基于策划内容生成指定章节
Tool 3: revise_section   — 根据用户指令修改章节内容
Tool 4: build_docx       — 构建Word文档并提供下载

所有工具封装为LangChain Tool，直接复用现有模块 (minimax.py, vector_store.py)
"""
import json
import os
import time
import asyncio
import contextvars
from langchain_core.tools import tool

# 跨工具共享: _after_tools_node 写入, build_docx 读取
# 使用 contextvars 确保 async task 隔离
_current_generated_markdown: contextvars.ContextVar[str] = contextvars.ContextVar(
    'generated_markdown', default=''
)
_current_product_name: contextvars.ContextVar[str] = contextvars.ContextVar(
    'product_name', default=''
)
_current_doc_type: contextvars.ContextVar[str] = contextvars.ContextVar(
    'doc_type', default='design_development_plan'
)


_current_attachments: contextvars.ContextVar[list[dict]] = contextvars.ContextVar(
    'attachments', default=[]
)

# write_chapter 完整内容旁路: 工具返回简短摘要，完整内容通过此字典
# 传递给 _after_tools_node，避免大段内容进入 LLM 对话历史。
# 使用模块级 dict 而非 contextvar，避免 LangGraph 异步节点切换时
# contextvar 丢失导致回退到 JSON 摘要被写入文档。
_pending_chapter_contents: dict = {}  # {chapter_name: full_content}


def set_current_doc_context(doc_type: str, product_name: str, markdown: str) -> None:
    """由 agent_engine 在每轮开始前调用，同步当前文档上下文"""
    _current_doc_type.set(doc_type)
    _current_product_name.set(product_name)
    _current_generated_markdown.set(markdown)


def set_current_attachments(attachments: list[dict]) -> None:
    """由 agent_engine 在每轮开始前调用，同步当前附件列表"""
    _current_attachments.set(attachments or [])


def get_pending_chapter_content(chapter_name: str) -> dict:
    """读取并删除 write_chapter 写入的指定章节完整内容"""
    return {"chapter_name": chapter_name, "full_content": _pending_chapter_contents.pop(chapter_name, "")}


# ── Tool 1: search_kb ──

@tool
async def search_kb(query: str, top_k: int = 5) -> str:
    """检索贴敷式胰岛素泵知识库 (标准、法规、技术文档、测试报告等)。

    当需要查找具体标准条款、技术参数限值、法规要求、同类产品数据时，必须先调用此工具。
    不要在未检索的情况下编造标准条款号和具体限值。

    Args:
        query: 搜索查询关键词。应为具体的标准号、参数名或技术问题。
        top_k: 返回结果数量，默认5条。

    Returns:
        JSON格式的检索结果列表，每项包含 content, source, score。
    """
    try:
        from app.services.rag.vector_store import VectorStore

        def _do_retrieve():
            store = VectorStore()
            return store.retrieve_hybrid(
                query=query,
                top_k=top_k,
                vector_weight=0.85,
            )

        results = await asyncio.wait_for(
            asyncio.to_thread(_do_retrieve),
            timeout=90.0,
        )

        if not results:
            return json.dumps({
                "status": "no_results",
                "message": f'未找到与"{query}"直接相关的知识库内容。请用已有知识回答，并告知用户此为基于经验的建议，建议用户自行查证最新标准。',
                "results": [],
            }, ensure_ascii=False)

        formatted = []
        for r in results:
            formatted.append({
                "content": r.get("text", ""),
                "source": r.get("source_file", "未知来源"),
                "score": round(r.get("similarity", 0), 3),
            })

        return json.dumps({
            "status": "ok",
            "query": query,
            "count": len(formatted),
            "results": formatted,
        }, ensure_ascii=False)

    except ImportError:
        return json.dumps({
            "status": "unavailable",
            "message": "知识库服务暂时不可用。请用已有知识回答，并告知用户当前无法检索知识库。",
            "results": [],
        }, ensure_ascii=False)
    except asyncio.TimeoutError:
        return json.dumps({
            "status": "timeout",
            "message": f"知识库检索超时（90秒）。请尝试缩小查询范围或稍后重试。",
            "results": [],
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"知识库检索异常: {str(e)}。请告知用户当前检索遇到问题，可用已有知识先回答。",
            "results": [],
        }, ensure_ascii=False)


# ── Tool 1b: search_attachment ──

@tool
async def search_attachment(query: str, top_k: int = 5) -> str:
    """搜索用户上传的附件内容。当需要查找用户上传文件中的具体信息时使用此工具。

    适用场景: 用户上传了PDF/Word/Excel等参考文档，需要从中提取特定信息。
    与 search_kb 的区别: search_kb 搜索预置知识库（标准法规），search_attachment 搜索用户上传文件。

    Args:
        query: 搜索查询。应包含具体关键词或问题。
        top_k: 返回结果数量，默认5条。

    Returns:
        JSON格式的搜索结果，每项包含匹配的文本片段、来源文件名和相关度评分。
    """
    import re as _re
    attachments = _current_attachments.get()
    if not attachments:
        return json.dumps({
            "status": "no_attachments",
            "message": "当前项目没有上传附件。请提示用户先上传相关文件，或使用 search_kb 检索知识库。",
            "results": [],
        }, ensure_ascii=False)

    # 在所有附件中搜索
    all_matches = []
    query_lower = query.lower()
    query_terms = query_lower.split()

    for att in attachments:
        full_text = att.get("full_text", "")
        if not full_text:
            continue

        filename = att.get("filename", "unknown")
        # 使用滑动窗口分割文本为段落（按双换行或单句分割）
        paragraphs = _re.split(r'\n\s*\n', full_text)
        if len(paragraphs) < 2:
            # 按句子分割
            paragraphs = _re.split(r'(?<=[。！？.!?])\s*', full_text)

        for para in paragraphs:
            para = para.strip()
            if len(para) < 10:
                continue

            para_lower = para.lower()
            # 计算相关度分数 (简单TF)
            score = 0
            for term in query_terms:
                count = para_lower.count(term)
                if count > 0:
                    score += count * (1.0 / len(query_terms))
            # 完整短语匹配加分
            if query_lower in para_lower:
                score += 2.0

            if score > 0:
                all_matches.append({
                    "content": para,
                    "source": filename,
                    "score": round(min(score / 3.0, 1.0), 3),
                })

    if not all_matches:
        # 尝试向量检索（如果附件已入库到uploads集合）
        try:
            from app.services.rag.vector_store import VectorStore
            store = VectorStore(collection_name="uploads")
            results = store.retrieve_hybrid(query=query, top_k=top_k, vector_weight=0.85)
            if results:
                formatted = []
                for r in results:
                    formatted.append({
                        "content": r.get("text", ""),
                        "source": r.get("source_file", "用户上传附件"),
                        "score": round(r.get("similarity", 0), 3),
                    })
                return json.dumps({
                    "status": "ok",
                    "query": query,
                    "count": len(formatted),
                    "source": "vector_search",
                    "results": formatted,
                }, ensure_ascii=False)
        except Exception:
            pass

        return json.dumps({
            "status": "no_match",
            "message": f'在已上传的{len(attachments)}个附件中未找到与"{query}"直接相关的内容。请尝试使用更通用的关键词，或告知用户当前附件中未包含此信息。',
            "results": [],
        }, ensure_ascii=False)

    # 按分数排序，去重
    all_matches.sort(key=lambda x: x["score"], reverse=True)
    seen = set()
    unique_matches = []
    for m in all_matches:
        key = m["content"][:100]
        if key not in seen:
            seen.add(key)
            unique_matches.append(m)
        if len(unique_matches) >= top_k:
            break

    return json.dumps({
        "status": "ok",
        "query": query,
        "count": len(unique_matches),
        "source": "attachment_text",
        "results": unique_matches,
    }, ensure_ascii=False)


# ── Tool 2: generate_section ──

@tool
async def generate_section(section_name: str, doc_type: str = "design_development_plan") -> str:
    """基于当前已确认的策划内容，生成指定文档类型中指定章节的初稿。

    生成前应确认: 该章节依赖的策划内容项是否都已确认。
    生成后会暂停等待用户确认——用户可以批准、要求修改、或重新生成。

    Args:
        section_name: 章节名称，如 "目的和范围"、"阶段划分"、"风险可接受性准则" 等。
        doc_type: 文档类型标识，如 "design_development_plan"、"risk_management_plan" 等。
                  从当前会话状态的 doc_type 字段获取，默认为 design_development_plan。

    Returns:
        生成的章节内容 (Markdown格式)。
    """
    try:
        from app.services.minimax import _call_minimax_api_raw, DOC_CHAPTERS, DEFAULT_CHAPTERS
        from app.services.doc_types import DOC_TYPE_LABELS
        from app.services.prompt_engineer import DOC_TYPE_SPECIFIC_PROMPTS

        doc_label = DOC_TYPE_LABELS.get(doc_type, "设计策划文档")

        # 查找章节定义 (用于更精准的生成指导)
        chapters = DOC_CHAPTERS.get(doc_type, DEFAULT_CHAPTERS)
        chapter_query = ""
        for ch in chapters:
            if ch.get("name") == section_name:
                chapter_query = ch.get("query", "")
                break

        # 文档类型专属专家提示词
        expert_prompt = DOC_TYPE_SPECIFIC_PROMPTS.get(doc_type, "")

        # 构建 system prompt: 专家角色 + 文档类型特定要求
        expert_section = f"\n\n# 本文档类型特定要求\n{expert_prompt}" if expert_prompt else ""
        system_prompt = f"""你是一位贴敷式胰岛素泵RA文档专家。请基于当前已确认的策划内容信息，
生成《{doc_label}》文档中「{section_name}」章节的初稿。{expert_section}

要求:
- 内容必须极其细致和具体，每个段落都要有实质性内容，不能只写框架标题
- 内容专业、完整，符合NMPA注册申报要求
- 所有标准条款引用必须有明确的条款号
- 使用Markdown格式，标题层级清晰 (##, ###)
- 技术参数要具体、可测量、有明确的数值范围
- 表格要填写完整，不能留"(描述)"或"待填写"等占位符
- 针对贴敷式胰岛素泵产品特性编写
- 生成内容的详细程度要像实际可用于注册申报的正式文档一样
- 用中文表述 (标准号和必要缩写除外)

## 输出结构要求
请按以下结构组织内容:
1. 首先用1-2段概述本节要点（含法规依据和产品适用性）
2. 然后逐个详细阐述每个关键要求（每个要求至少200字，包含法规条款原文引用、产品参数映射、实施建议）
3. 如涉及数据/参数对比，以表格形式呈现（至少3列）
4. 最后用1段总结本节的合规要点和与贴敷式胰岛素泵的关联性"""

        # RAG 检索: 用章节名 + 文档类型 + 章节内容要点作为查询
        rag_context = ""
        try:
            rag_query = f"{doc_label} {section_name}"
            if chapter_query:
                rag_query += f" {chapter_query[:200]}"
            rag_result = await search_kb.ainvoke({"query": rag_query, "top_k": 5})
            rag_data = json.loads(rag_result)
            if rag_data.get("status") == "ok" and rag_data.get("results"):
                lines = ["\n\n# 知识库参考资料（必须优先依据以下内容编写）:"]
                for j, r in enumerate(rag_data["results"], 1):
                    lines.append(
                        f"\n[参考{j}] 来源: {r['source']} (相关度:{r['score']})\n{r['content']}"
                    )
                rag_context = "".join(lines)
        except Exception:
            pass

        # 构建 user prompt: 章节特定查询
        query_hint = f"\n\n# 本章节内容要点\n请重点覆盖以下方面: {chapter_query}" if chapter_query else ""
        user_prompt = f"请生成「{section_name}」章节的{doc_label}文档内容。内容包括该章节应覆盖的所有要求项、适用的法规标准依据、以及建议的具体参数/验收标准。{query_hint}{rag_context}"

        def _do_generate():
            return _call_minimax_api_raw(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.3,
                max_tokens=8192,
            )

        response = await asyncio.wait_for(
            asyncio.to_thread(_do_generate),
            timeout=300.0,
        )

        if response:
            return f"[章节: {section_name}]\n\n{response}"
        else:
            return f"[错误] 无法生成「{section_name}」章节。MiniMax API 返回空结果，请稍后重试。"

    except asyncio.TimeoutError:
        return f"[错误] 生成「{section_name}」章节超时（300秒）。请尝试缩小章节范围或稍后重试。"
    except ImportError:
        return f"[错误] 生成服务暂时不可用。请稍后重试。"
    except Exception as e:
        return f"[错误] 生成「{section_name}」时发生异常: {str(e)}。请稍后重试。"


# ── Tool 3: revise_section ──（见下方）

# ── Tool 4: build_docx ──

# 内存中的文档存储 (download_id → bytes)
_docx_store: dict = {}


def _store_docx(file_bytes: bytes, filename: str) -> str:
    """将生成的docx存入内存，返回download_id"""
    import uuid
    download_id = str(uuid.uuid4())[:8]
    _docx_store[download_id] = {"bytes": file_bytes, "filename": filename}
    return download_id


def _get_docx(download_id: str) -> dict | None:
    """从内存中取出docx（不删除，允许多次下载）"""
    return _docx_store.get(download_id)


@tool
async def build_docx(doc_type: str = "", product_name: str = "", markdown: str = "") -> str:
    """将已生成的Markdown文档内容构建为Word (.docx) 文件并提供下载。

    调用时机: 在所有章节生成完毕、用户确认内容无误后调用。
    调用后前端会自动弹出下载按钮。
    当 markdown 参数为空时，自动使用已生成的章节内容。

    Args:
        doc_type: 文档类型，如 "design_input"、"risk_management_report" 等。为空时自动从上下文获取。
        product_name: 产品名称，用于生成文件名。为空时自动从上下文获取。
        markdown: 完整的Markdown格式文档内容。为空时自动使用已生成的所有章节。

    Returns:
        JSON格式的结果，包含 download_id 和文件名。
    """
    try:
        from app.services.template import TemplateService
        from app.services.doc_types import DOC_TYPE_LABELS

        # 从上下文填充缺失参数
        if not doc_type:
            doc_type = _current_doc_type.get()
        if not product_name:
            product_name = _current_product_name.get()
        if not markdown:
            markdown = _current_generated_markdown.get()

        if not markdown:
            return json.dumps({
                "status": "error",
                "message": "没有已生成的文档内容。请先生成至少一个章节后再导出。",
            }, ensure_ascii=False)

        def _do_build():
            template_service = TemplateService()
            doc = template_service.load_template(doc_type)
            doc = template_service.fill_template(
                doc=doc,
                content=markdown,
                product_name=product_name,
                doc_type=doc_type,
            )
            file_bytes = template_service.document_to_bytes(doc)
            return file_bytes

        file_bytes = await asyncio.wait_for(
            asyncio.to_thread(_do_build),
            timeout=60.0,
        )

        label = DOC_TYPE_LABELS.get(doc_type, doc_type)
        filename = f"{product_name}_{label}.docx"

        download_id = _store_docx(file_bytes, filename)

        return json.dumps({
            "status": "ok",
            "download_id": download_id,
            "filename": filename,
            "size_bytes": len(file_bytes),
            "message": f"文档「{filename}」已生成，点击下载按钮即可获取。",
        }, ensure_ascii=False)

    except asyncio.TimeoutError:
        return json.dumps({
            "status": "error",
            "message": "文档构建超时（60秒）。请稍后重试。",
        }, ensure_ascii=False)
    except ImportError as e:
        return json.dumps({
            "status": "error",
            "message": f"Word模板服务不可用: {str(e)}",
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"文档构建失败: {str(e)}",
        }, ensure_ascii=False)

@tool
async def revise_section(section_name: str, instruction: str, doc_type: str = "design_development_plan") -> str:
    """根据用户指令修改指定文档类型的指定章节。

    在现有内容基础上修改，不推翻重写。
    修改后展示变更摘要。

    Args:
        section_name: 要修改的章节名称。
        instruction: 用户的具体修改指令，如"将阶段划分从5个阶段调整为7个阶段"。
        doc_type: 文档类型标识，如 "design_development_plan"、"risk_management_plan" 等。
                  从当前会话状态的 doc_type 字段获取，默认为 design_development_plan。

    Returns:
        修改后的章节内容 (Markdown格式)。
    """
    try:
        from app.services.minimax import _call_minimax_api_raw
        from app.services.doc_types import DOC_TYPE_LABELS
        from app.services.prompt_engineer import DOC_TYPE_SPECIFIC_PROMPTS

        doc_label = DOC_TYPE_LABELS.get(doc_type, "设计策划文档")
        expert_prompt = DOC_TYPE_SPECIFIC_PROMPTS.get(doc_type, "")
        expert_section = f"\n\n# 本文档类型要求\n{expert_prompt}" if expert_prompt else ""

        system_prompt = f"""你是一位贴敷式胰岛素泵RA文档专家。用户要求修改《{doc_label}》文档中「{section_name}」章节。{expert_section}

修改规则:
- 仅修改用户指定的内容，保持其他部分不变
- 保持Markdown格式和标题层级
- 如果修改影响了其他章节的参数/引用，在回复末尾用"⚠️ 关联影响:"标注
- 用中文回复"""

        user_prompt = f"""请修改《{doc_label}》的「{section_name}」章节，修改指令: {instruction}

请在修改后:
1. 输出修改后的完整章节内容
2. 在末尾用"📝 修改摘要:"列出具体变更点"""

        def _do_revise():
            return _call_minimax_api_raw(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.3,
                max_tokens=8192,
            )

        response = await asyncio.wait_for(
            asyncio.to_thread(_do_revise),
            timeout=180.0,
        )

        if response:
            return f"[已修改: {section_name}]\n\n{response}"
        else:
            return f"[错误] 无法修改「{section_name}」章节。请稍后重试。"

    except asyncio.TimeoutError:
        return f"[错误] 修改「{section_name}」章节超时（180秒）。请稍后重试。"
    except ImportError:
        return f"[错误] 修订服务暂时不可用。请稍后重试。"
    except Exception as e:
        return f"[错误] 修订「{section_name}」时发生异常: {str(e)}。请稍后重试。"


# ═══════════════════════════════════════════════════════
# 多代理协作工具 (Option A: Subagents Pattern)
# 将子代理包装为工具，主代理通过工具调用驱动子代理
# ═══════════════════════════════════════════════════════

# 子代理实例（懒加载）
_outline_agent = None
_chapter_agent = None

# Semaphore 控制并行 LLM 调用数，防止 API 限流
_llm_semaphore = asyncio.Semaphore(4)


def _get_outline_agent():
    global _outline_agent
    if _outline_agent is None:
        from app.services.subagents import create_outline_agent
        _outline_agent = create_outline_agent()
    return _outline_agent


def _get_chapter_agent():
    global _chapter_agent
    if _chapter_agent is None:
        from app.services.subagents import create_chapter_agent
        _chapter_agent = create_chapter_agent()
    return _chapter_agent


def _extract_json(text: str) -> str:
    """从子代理输出中提取 JSON 字符串（去除可能的 markdown 代码块标记）"""
    import re
    text = text.strip()
    # 尝试匹配 ```json ... ``` 包裹
    match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
    if match:
        return match.group(1).strip()
    # 尝试匹配裸 JSON 对象 { ... }
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        return match.group(0).strip()
    return text


@tool
async def design_outline(
    doc_type: str = "design_development_plan",
    product_name: str = "贴敷式胰岛素泵",
    special_requirements: str = "",
) -> str:
    """设计文档框架（章节结构）。

    调用子代理A（框架设计师），生成完整的文档章节大纲，包含每章的标题、描述、小节列表。
    应在以下时机调用:
    - 开始生成新文档前（用户尚未确认框架时）
    - 用户要求调整框架结构时

    Args:
        doc_type: 文档类型标识 (如 "design_development_plan", "risk_management_plan" 等)
        product_name: 产品名称
        special_requirements: 用户对框架的特殊要求（可选），如"增加FDA申报内容"

    Returns:
        JSON字符串，包含完整的章节框架结构，每章有 title, description, subsections 等字段。
        格式: {"doc_title": "...", "chapters": [{"id": 1, "title": "...", ...}]}
    """
    agent = _get_outline_agent()

    # ── 强制前置 RAG 检索 ──
    rag_context = ""
    try:
        rag_result = await search_kb.ainvoke({
            "query": f"{doc_type} 章节结构 标准要求",
            "top_k": 5,
        })
        rag_data = json.loads(rag_result)
        if rag_data.get("status") == "ok" and rag_data.get("results"):
            rag_parts = []
            for i, r in enumerate(rag_data["results"], 1):
                rag_parts.append(
                    f"[参考{i}] 来源: {r['source']} (相关度: {r['score']})\n{r['content']}"
                )
            rag_context = "\n\n".join(rag_parts)
            print(f"[agent_tools] RAG for outline: {len(rag_data['results'])} results")
    except Exception as e:
        print(f"[agent_tools] RAG failed for outline: {e}")

    prompt = f"""请为以下文档设计章节框架:

文档类型: {doc_type}
产品名称: {product_name}"""
    if special_requirements:
        prompt += f"\n特殊要求: {special_requirements}"

    if rag_context:
        prompt += f"""

## 知识库参考资料（必须优先依据以下内容设计框架）
{rag_context}"""

    prompt += "\n\n请严格按照 JSON 格式输出，不要包含任何 JSON 之外的解释文字。"

    try:
        async with _llm_semaphore:
            result = await agent.ainvoke({
                "messages": [{"role": "user", "content": prompt}]
            })
        content = result["messages"][-1].content
        content = _extract_json(content)
        return content
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"框架设计失败: {str(e)}",
        }, ensure_ascii=False)


def _find_chapter_subsections(outline_json: str, chapter_name: str) -> list[dict]:
    """从框架JSON中查找指定章节的小节列表。

    支持模糊匹配：去除"第X章"前缀后比较，也支持章节标题包含关系。
    """
    import re
    if not outline_json or not chapter_name:
        return []
    try:
        data = json.loads(outline_json)
        chapters = data.get("chapters", [])
    except (json.JSONDecodeError, TypeError):
        return []

    # 标准化：去除"第X章"前缀
    name_clean = re.sub(r'^第[一二三四五六七八九十\d]+章\s*', '', chapter_name).strip()

    for ch in chapters:
        title = ch.get("title", "")
        title_clean = re.sub(r'^第[一二三四五六七八九十\d]+章\s*', '', title).strip()
        # 精确匹配、包含匹配
        if (name_clean == title_clean
                or name_clean in title_clean
                or title_clean in name_clean
                or chapter_name == title
                or chapter_name in title
                or title in chapter_name):
            return ch.get("subsections", [])
    return []


@tool
async def write_chapter(
    chapter_name: str,
    outline_json: str,
    doc_type: str = "design_development_plan",
) -> str:
    """编写指定章节的完整内容（逐小节生成）。

    从框架中提取本章的小节列表，每个小节独立调用LLM生成内容，
    最后按小节顺序组装为完整章节。

    可并行调用——主代理在一次回复中同时发起多个 write_chapter 调用，
    系统会自动并行执行它们。

    应在以下时机调用:
    - 框架已确认，需要生成某章节内容时
    - 一次可同时对多个章节发起调用（每个章节一次调用）

    Args:
        chapter_name: 章节名称，如 "目的和范围"、"风险管理计划"
        outline_json: 完整的文档框架JSON（由 design_outline 工具生成）。
                      必须包含所有章节信息，让子代理了解全局结构后再写单章。
        doc_type: 文档类型标识

    Returns:
        Markdown格式的章节完整内容，可直接拼接进最终文档。
    """

    # ── 获取 doc_type 的可读标签 ──
    from app.services.minimax import _call_minimax_api_raw
    doc_label = doc_type
    expert_prompt = ""
    try:
        from app.services.doc_types import DOC_TYPE_LABELS
        from app.services.prompt_engineer import DOC_TYPE_SPECIFIC_PROMPTS
        doc_label = DOC_TYPE_LABELS.get(doc_type, doc_type)
        expert_prompt = DOC_TYPE_SPECIFIC_PROMPTS.get(doc_type, "")
    except ImportError:
        pass
    expert_section = f"\n\n{expert_prompt}" if expert_prompt else ""

    # 轻量去重: 对同一小节的检索结果做 Jaccard 相似度去重
    def _dedup_results(results: list, threshold: float = 0.6) -> list:
        if len(results) <= 1:
            return results
        kept = []
        for r in results:
            dup = False
            r_words = set(r.get("content", "")[:200].split())
            if not r_words:
                kept.append(r)
                continue
            for k in kept:
                k_words = set(k.get("content", "")[:200].split())
                if not k_words:
                    continue
                intersection = len(r_words & k_words)
                union = len(r_words | k_words)
                jaccard = intersection / union if union > 0 else 0
                if jaccard > threshold:
                    dup = True
                    break
            if not dup:
                kept.append(r)
        return kept

    # ── 强制前置 RAG 检索（逐小节粒度）──
    # 从框架中提取本章的小节列表，每个小节单独检索
    subsections = _find_chapter_subsections(outline_json, chapter_name)
    sub_rag_map = {}  # sub_title -> rag_context_string

    if subsections:
        _rag_sem = asyncio.Semaphore(6)
        async def _search_subsection(sub: dict):
            sub_title = sub.get("title", "")
            content_points = sub.get("content_points", [])
            query = f"{doc_type} {chapter_name} {sub_title}"
            if content_points:
                query += f" {' '.join(content_points[:3])}"
            try:
                async with _rag_sem:
                    r = await search_kb.ainvoke({"query": query, "top_k": 5})
                data = json.loads(r)
                if data.get("status") == "ok" and data.get("results"):
                    return (sub_title, data["results"])
            except Exception:
                pass
            return (sub_title, [])

        tasks = [asyncio.create_task(_search_subsection(s)) for s in subsections]
        await asyncio.gather(*tasks)

        for task in tasks:
            sub_title, results = task.result()
            if results:
                results = _dedup_results(results)
                lines = [f"### 小节「{sub_title}」参考资料:"]
                for j, r in enumerate(results, 1):
                    lines.append(
                        f"  [源] {r['source']} (相关度:{r['score']})\n  {r['content']}"
                    )
                sub_rag_map[sub_title] = "\n".join(lines)
            else:
                sub_rag_map[sub_title] = ""

        rag_hit_count = sum(1 for v in sub_rag_map.values() if v)
        print(f"[agent_tools] Subsection RAG for '{chapter_name}': "
              f"{rag_hit_count}/{len(subsections)} subsections matched")

        # ── 逐小节生成内容 ──
        async def _gen_subsection(sub: dict) -> tuple:
            sub_title = sub.get("title", "")
            content_points = sub.get("content_points", [])
            rag = sub_rag_map.get(sub_title, "")

            points_hint = ""
            if content_points:
                points_hint = "\n请重点覆盖以下内容要点:\n" + "\n".join(
                    f"- {p}" for p in content_points
                )

            system_prompt = (
                f"你是一位贴敷式胰岛素泵RA文档专家。"
                f"请编写《{doc_label}》文档中「{chapter_name}」章节下「{sub_title}」小节的内容。"
                f"{expert_section}\n\n"
                f"要求:\n"
                f"- 内容必须极其细致和具体，每个段落都要有实质性内容，不能只写框架标题\n"
                f"- 内容专业、完整，符合NMPA注册申报要求\n"
                f"- 所有标准条款引用必须有明确的条款号\n"
                f"- 使用Markdown格式\n"
                f"- 技术参数要具体、可测量、有明确的数值范围\n"
                f"- 表格要填写完整，不能留\"(描述)\"或\"待填写\"等占位符\n"
                f"- 针对贴敷式胰岛素泵产品特性编写\n"
                f"- 生成内容的详细程度要像实际可用于注册申报的正式文档一样\n"
                f"- 用中文表述 (标准号和必要缩写除外)\n"
                f"- 只生成本小节内容，不要添加章节标题（如 ## 或 ###）\n"
                f"\n"
                f"## 输出结构要求\n"
                f"请按以下结构组织内容:\n"
                f"1. 首先用1-2段概述本小节要点（含法规依据和产品适用性）\n"
                f"2. 然后逐个详细阐述每个关键要求（每个要求至少200字，包含法规条款原文引用、产品参数映射、实施建议）\n"
                f"3. 如涉及数据/参数对比，以表格形式呈现（至少3列）\n"
                f"4. 最后用1段总结本小节的合规要点和与贴敷式胰岛素泵的关联性"
            )

            user_prompt = (
                f"请编写「{chapter_name}」→「{sub_title}」小节的内容。"
                f"{points_hint}\n\n"
                f"文档框架参考:\n{outline_json}"
            )

            if rag:
                user_prompt += (
                    f"\n\n知识库参考资料（必须优先依据以下内容编写）:\n{rag}"
                )

            try:
                async with _llm_semaphore:
                    response = await asyncio.wait_for(
                        asyncio.to_thread(
                            _call_minimax_api_raw,
                            system_prompt=system_prompt,
                            user_prompt=user_prompt,
                            temperature=0.3,
                            max_tokens=8192,
                        ),
                        timeout=300.0,
                    )
                if response:
                    return (sub_title, response)
                else:
                    return (sub_title, "[错误] 无法生成小节内容。")
            except asyncio.TimeoutError:
                return (sub_title, "[错误] 生成小节超时（300秒）。")
            except Exception as e:
                return (sub_title, f"[错误] 生成小节异常: {str(e)}")

        gen_tasks = [asyncio.create_task(_gen_subsection(s)) for s in subsections]
        await asyncio.gather(*gen_tasks)

        # 组装章节完整内容
        parts = [f"## {chapter_name}\n\n"]
        for task in gen_tasks:
            sub_title, content = task.result()
            parts.append(f"### {sub_title}\n\n{content}\n\n")
        full_content = "".join(parts)

        # 统计小节生成结果
        success_count = sum(1 for t in gen_tasks if not str(t.result()[1]).startswith("[错误]"))
        subsection_names = [s.get("title", "") for s in subsections]

        # 完整内容通过 contextvar 旁路传给 _after_tools_node，
        # 工具返回值只包含简短摘要，避免大段内容进入 LLM 对话历史
        _pending_chapter_contents[chapter_name] = full_content

        print(f"[agent_tools] write_chapter '{chapter_name}': "
              f"{success_count}/{len(subsections)} subsections generated")
        return json.dumps({
            "status": "ok",
            "chapter_name": chapter_name,
            "subsections_count": len(subsections),
            "success_count": success_count,
            "subsections": subsection_names,
            "preview": full_content[:300] + ("..." if len(full_content) > 300 else ""),
        }, ensure_ascii=False)

    else:
        # ── 回退：框架无小节信息时，整体生成章节 ──
        rag_context = ""
        try:
            rag_result = await search_kb.ainvoke({
                "query": f"{doc_type} {chapter_name}",
                "top_k": 5,
            })
            rag_data = json.loads(rag_result)
            if rag_data.get("status") == "ok" and rag_data.get("results"):
                rag_parts = []
                for i, r in enumerate(rag_data["results"], 1):
                    rag_parts.append(
                        f"[参考{i}] 来源: {r['source']} (相关度: {r['score']})\n{r['content']}"
                    )
                rag_context = "\n\n".join(rag_parts)
                print(f"[agent_tools] Fallback RAG for '{chapter_name}': "
                      f"{len(rag_data['results'])} results")
        except Exception as e:
            print(f"[agent_tools] RAG failed for '{chapter_name}': {e}")

        system_prompt = (
            f"你是一位贴敷式胰岛素泵RA文档专家。"
            f"请编写《{doc_label}》文档中「{chapter_name}」章节的完整内容。"
            f"{expert_section}\n\n"
            f"要求:\n"
            f"- 内容必须极其细致和具体，每个段落都要有实质性内容，不能只写框架标题\n"
            f"- 内容专业、完整，符合NMPA注册申报要求\n"
            f"- 所有标准条款引用必须有明确的条款号\n"
            f"- 使用Markdown格式，标题层级清晰 (### 用于小节)\n"
            f"- 技术参数要具体、可测量、有明确的数值范围\n"
            f"- 表格要填写完整，不能留\"(描述)\"或\"待填写\"等占位符\n"
            f"- 针对贴敷式胰岛素泵产品特性编写\n"
            f"- 生成内容的详细程度要像实际可用于注册申报的正式文档一样\n"
            f"- 用中文表述 (标准号和必要缩写除外)\n"
            f"\n"
            f"## 输出结构要求\n"
            f"请按以下结构组织内容:\n"
            f"1. 首先用1-2段概述本章要点（含法规依据和产品适用性）\n"
            f"2. 然后逐个详细阐述每个关键要求（每个要求至少200字，包含法规条款原文引用、产品参数映射、实施建议）\n"
            f"3. 如涉及数据/参数对比，以表格形式呈现（至少3列）\n"
            f"4. 最后用1段总结本章的合规要点和与贴敷式胰岛素泵的关联性"
        )

        user_prompt = (
            f"请编写「{chapter_name}」章节的完整内容。\n\n"
            f"文档框架:\n{outline_json}"
        )

        if rag_context:
            user_prompt += f"\n\n知识库参考资料:\n{rag_context}"

        user_prompt += (
            f"\n\n注意:\n"
            f"- 只输出本章内容，不要输出其他章节\n"
            f"- 第一行以 `## {chapter_name}` 开头"
        )

        try:
            async with _llm_semaphore:
                response = await asyncio.wait_for(
                    asyncio.to_thread(
                        _call_minimax_api_raw,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        temperature=0.3,
                        max_tokens=8192,
                    ),
                    timeout=300.0,
                )
            if response:
                # 完整内容通过 dict 旁路传递，返回值仅含摘要
                _pending_chapter_contents[chapter_name] = response
                return json.dumps({
                    "status": "ok",
                    "chapter_name": chapter_name,
                    "subsections_count": 0,
                    "success_count": 1,
                    "subsections": [],
                    "preview": response[:300] + ("..." if len(response) > 300 else ""),
                }, ensure_ascii=False)
            else:
                return f"[错误] 章节「{chapter_name}」编写失败: API 返回空结果。"
        except asyncio.TimeoutError:
            return f"[错误] 章节「{chapter_name}」编写超时（300秒）。"
        except Exception as e:
            return f"[错误] 章节「{chapter_name}」编写失败: {str(e)}。请重试。"


@tool
async def update_outline(
    outline_json: str,
    instruction: str,
) -> str:
    """根据用户指令修改文档框架。

    当用户要求调整章节结构（增删章节、调整顺序、修改标题）时调用。
    如果修改幅度很小（如仅改一个章节标题），主代理可以直接修改JSON，
    不需要调用此工具。

    Args:
        outline_json: 当前完整的文档框架JSON
        instruction: 用户的修改指令，如"删除第3章，将第5章移到第2章"

    Returns:
        修改后的完整框架JSON
    """
    agent = _get_outline_agent()

    prompt = f"""当前框架如下:
```json
{outline_json}
```

用户要求: {instruction}

请输出修改后的完整 JSON 框架。保持一样的格式和字段，只修改用户要求的部分。
只输出 JSON，不要包含其他文字。"""

    try:
        async with _llm_semaphore:
            result = await agent.ainvoke({
                "messages": [{"role": "user", "content": prompt}]
            })
        content = result["messages"][-1].content
        content = _extract_json(content)
        return content
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"框架更新失败: {str(e)}",
        }, ensure_ascii=False)


# ── Tool 7: web_search ──

@tool
async def web_search(query: str, doc_type: str = "design_development_plan") -> str:
    """搜索互联网获取医疗器械法规标准、技术文献等最新信息。

    与 search_kb 的区别: search_kb 搜索本地预置知识库，web_search 搜索互联网最新内容。
    当本地知识库找不到需要的信息，或需要查询最新法规动态时使用此工具。

    Args:
        query: 搜索查询关键词。应为具体的标准号、法规名或技术问题。
        doc_type: 文档类型标识，用于优化搜索策略。

    Returns:
        JSON格式的搜索结果，包含网页摘要和相关法规信息。
    """
    import concurrent.futures

    web_info = ""
    search_method = "none"

    # 从上下文获取当前产品名称，避免硬编码
    product_name = _current_product_name.get() or "贴敷式胰岛素泵"

    # 优先使用 Claude Agent SDK 搜索（更智能，质量更高）
    try:
        from app.services.agent_search import SyncAgentSearchService
        agent_search = SyncAgentSearchService()
        if agent_search.available:
            web_info, _ = agent_search.search_regulations(
                chapter_name=query,
                product_type=product_name,
                max_results=3,
                enable_deep_scrape=True,
                enable_file_download=False,
                doc_type=doc_type,
            )
            if web_info:
                search_method = "agent_sdk"
                print(f"[agent_tools] web_search: {len(web_info)} chars via Agent SDK")
    except Exception as e:
        print(f"[agent_tools] Agent SDK search failed: {e}")

    # Agent SDK 不可用或失败时回退到 Playwright
    if not web_info:
        try:
            from app.services.web_search import SyncWebSearchService
            playwright_search = SyncWebSearchService()
            if playwright_search.playwright_available:
                web_info, _ = playwright_search.search_regulations(
                    chapter_name=query,
                    product_type=product_name,
                    max_results=3,
                    enable_deep_scrape=True,
                    enable_file_download=False,
                    doc_type=doc_type,
                )
                if web_info:
                    search_method = "playwright"
                    print(f"[agent_tools] web_search: {len(web_info)} chars via Playwright")
        except Exception as e:
            print(f"[agent_tools] Playwright search failed: {e}")

    if not web_info:
        return json.dumps({
            "status": "no_results",
            "message": f'未找到与"{query}"相关的网络信息。请尝试使用不同关键词，或用 search_kb 检索本地知识库。',
            "results": [],
        }, ensure_ascii=False)

    print(f"[agent_tools] web_search: {len(web_info)} chars via {search_method}")
    return json.dumps({
        "status": "ok",
        "query": query,
        "search_method": search_method,
        "content": web_info[:3000],
    }, ensure_ascii=False)


# ── Tool 8: analyze_document_structure ──

@tool
async def analyze_document_structure(file_id: str = "") -> str:
    """分析用户上传文档的章节结构和内容概要。

    将文档全文直接送给大模型，让大模型自主识别章节标题、层级关系和内容摘要。

    当用户询问以下问题时必须调用此工具:
    - "这个文档有哪些章节？"
    - "第三章讲了什么？"
    - "帮我梳理一下这个文档的结构"
    - 任何涉及文档章节、结构、内容概要的问题

    Args:
        file_id: 可选，指定要分析的附件file_id。为空时分析所有已上传的附件。

    Returns:
        JSON格式的章节结构，包含每章标题、层级、内容摘要。
    """
    from app.services.minimax import _call_minimax_api_raw

    attachments = _current_attachments.get()
    if not attachments:
        return json.dumps({
            "status": "no_attachments",
            "message": "当前没有上传附件。请先上传Word或PDF文档。",
            "structures": [],
        }, ensure_ascii=False)

    # 筛选目标附件
    target_attachments = attachments
    if file_id:
        target_attachments = [a for a in attachments if a.get("file_id") == file_id]
        if not target_attachments:
            return json.dumps({
                "status": "not_found",
                "message": f"未找到 file_id={file_id} 的附件。",
                "structures": [],
            }, ensure_ascii=False)

    all_structures = []

    for att in target_attachments:
        filename = att.get("filename", "unknown")
        full_text = att.get("full_text", "")

        if not full_text:
            all_structures.append({
                "filename": filename,
                "file_id": att.get("file_id", ""),
                "status": "no_text",
                "message": "该文件无文本内容可供分析",
            })
            continue

        # 截取前 12000 字符送给 LLM（兼顾 token 消耗与覆盖范围）
        text_sample = full_text[:12000]

        system_prompt = """你是一位文档结构分析专家，专精于医疗器械注册文档的章节结构分析。

请分析以下文档全文，直接识别其章节结构。你需要:
1. 识别所有章节标题及其正确的层级（1-4级）
2. 修正不规范的编号（如"一."→"一、"、"1 概述"→"1. 概述"）
3. 对每个章节，用一句话概括其核心内容
4. 按文档中的出现顺序排列

输出必须为严格JSON格式，不要包含任何其他文字。"""

        user_prompt = f"""请分析文档「{filename}」的章节结构。

文档全文:
{text_sample}

请输出以下JSON格式:
```json
{{
  "document_title": "文档标题（从内容推断）",
  "chapters": [
    {{
      "level": 1,
      "title": "章节标题",
      "summary": "一句话概括本章核心内容（不超过40字）"
    }}
  ]
}}
```

要求:
- level: 1=章, 2=节, 3=小节, 4=子小节
- summary: 用中文，不超过40字
- 章节按文档中的出现顺序排列
- 如果文档有目录（TOC），优先参照目录结构"""

        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    _call_minimax_api_raw,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=0.2,
                    max_tokens=8192,
                ),
                timeout=180.0,
            )

            if response:
                # 提取 JSON
                import re as _re
                json_text = response.strip()
                match = _re.search(r'```(?:json)?\s*([\s\S]*?)```', json_text)
                if match:
                    json_text = match.group(1).strip()
                else:
                    match = _re.search(r'\{[\s\S]*\}', json_text)
                    if match:
                        json_text = match.group(0).strip()

                try:
                    parsed = json.loads(json_text)
                    parsed["filename"] = filename
                    parsed["file_id"] = att.get("file_id", "")
                    parsed["status"] = "ok"
                    all_structures.append(parsed)
                except json.JSONDecodeError:
                    all_structures.append({
                        "filename": filename,
                        "file_id": att.get("file_id", ""),
                        "status": "parse_error",
                        "raw_response": response[:1000],
                    })
            else:
                all_structures.append({
                    "filename": filename,
                    "file_id": att.get("file_id", ""),
                    "status": "empty_response",
                    "message": "LLM 返回空结果",
                })

        except asyncio.TimeoutError:
            all_structures.append({
                "filename": filename,
                "file_id": att.get("file_id", ""),
                "status": "timeout",
                "message": "章节分析超时（180秒）",
            })
        except Exception as e:
            all_structures.append({
                "filename": filename,
                "file_id": att.get("file_id", ""),
                "status": "error",
                "message": f"章节分析异常: {str(e)}",
            })

    return json.dumps({
        "status": "ok",
        "analyzed_count": len(all_structures),
        "structures": all_structures,
    }, ensure_ascii=False)


# ── Tool 9: ingest_attachment_to_kb ──

@tool
async def ingest_attachment_to_kb(file_id: str = "") -> str:
    """将用户上传的附件文档转为向量写入主知识库，使其可被 search_kb 检索。

    调用时机:
    - 用户说"把这个文档导入知识库"
    - 用户说"记住这个文档的内容"
    - 用户上传了重要参考资料，希望后续生成文档时能引用
    - Agent 判断附件内容对后续工作有价值时，主动建议导入

    Args:
        file_id: 可选，指定要导入的附件file_id。为空时导入所有已上传的附件。

    Returns:
        JSON格式的导入结果，包含每个文件的chunk数量和导入状态。
    """
    from app.services.rag.ingest import chunk_text
    from app.services.rag.vector_store import VectorStore

    attachments = _current_attachments.get()
    if not attachments:
        return json.dumps({
            "status": "no_attachments",
            "message": "当前没有上传附件。请先上传Word或PDF文档。",
            "results": [],
        }, ensure_ascii=False)

    # 筛选目标附件
    target_attachments = attachments
    if file_id:
        target_attachments = [a for a in attachments if a.get("file_id") == file_id]
        if not target_attachments:
            return json.dumps({
                "status": "not_found",
                "message": f"未找到 file_id={file_id} 的附件。",
                "results": [],
            }, ensure_ascii=False)

    results = []

    for att in target_attachments:
        filename = att.get("filename", "unknown")
        full_text = att.get("full_text", "")

        if not full_text:
            results.append({
                "filename": filename,
                "file_id": att.get("file_id", ""),
                "status": "no_text",
                "message": "该文件无文本内容，无法导入",
            })
            continue

        # 按段落切分（full_text 已包含 ## 标题标记）
        paragraphs = []
        section_title = ""
        for line in full_text.split("\n"):
            line = line.strip()
            if not line:
                continue
            if line.startswith("## "):
                section_title = line.lstrip("#").strip()
            else:
                paragraphs.append((section_title, line))

        if not paragraphs:
            paragraphs = [("", full_text)]

        # 复用现有的分块函数
        chunks = chunk_text(paragraphs)

        # 标记来源
        for i, chunk in enumerate(chunks):
            chunk["doc_type"] = "agent_attachment"
            chunk["source_file"] = filename
            chunk["chunk_id"] = f"att_{att.get('file_id', '?')}_{i}"

        # 写入主知识库
        try:
            vector_store = VectorStore(collection_name="insulin_pump_kb")
            vector_store.add_chunks(chunks)
            # 使 BM25 缓存失效
            VectorStore.invalidate_bm25_cache()

            results.append({
                "filename": filename,
                "file_id": att.get("file_id", ""),
                "status": "ok",
                "chunk_count": len(chunks),
                "message": f"「{filename}」已导入知识库，{len(chunks)} 个分块。后续可通过 search_kb 检索到此文档内容。",
            })
            print(f"[agent_tools] Ingested '{filename}' → insulin_pump_kb ({len(chunks)} chunks)")
        except Exception as e:
            results.append({
                "filename": filename,
                "file_id": att.get("file_id", ""),
                "status": "error",
                "message": f"导入失败: {str(e)}",
            })

    success_count = sum(1 for r in results if r["status"] == "ok")
    return json.dumps({
        "status": "ok",
        "total": len(results),
        "success_count": success_count,
        "results": results,
    }, ensure_ascii=False)


# ── 工具列表导出 ──

PHASE1_TOOLS = [
    search_kb,
    search_attachment,
    web_search,
    analyze_document_structure,
    ingest_attachment_to_kb,
    generate_section,
    revise_section,
    build_docx,
    design_outline,
    write_chapter,
    update_outline,
]
