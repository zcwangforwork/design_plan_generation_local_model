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

# 产品画像上下文（供 generate_search_query 等工具读取）
_current_product_classification: contextvars.ContextVar[str] = contextvars.ContextVar(
    'product_classification', default=''
)
_current_product_intended_use: contextvars.ContextVar[str] = contextvars.ContextVar(
    'product_intended_use', default=''
)
_current_confirmed_standards: contextvars.ContextVar[list] = contextvars.ContextVar(
    'confirmed_standards', default=[]
)

# write_chapter 完整内容旁路: 工具返回简短摘要，完整内容通过此字典
# 传递给 _after_tools_node，避免大段内容进入 LLM 对话历史。
# 使用模块级 dict 而非 contextvar，避免 LangGraph 异步节点切换时
# contextvar 丢失导致回退到 JSON 摘要被写入文档。
_pending_chapter_contents: dict = {}  # {chapter_name: full_content}


def set_current_doc_context(
    doc_type: str,
    product_name: str,
    markdown: str,
    product_classification: str = '',
    product_intended_use: str = '',
    confirmed_standards: list = None,
) -> None:
    """由 agent_engine 在每轮开始前调用，同步当前文档上下文"""
    _current_doc_type.set(doc_type)
    _current_product_name.set(product_name)
    _current_generated_markdown.set(markdown)
    _current_product_classification.set(product_classification or '')
    _current_product_intended_use.set(product_intended_use or '')
    _current_confirmed_standards.set(confirmed_standards or [])


def set_current_attachments(attachments: list[dict]) -> None:
    """由 agent_engine 在每轮开始前调用，同步当前附件列表"""
    _current_attachments.set(attachments or [])


def get_pending_chapter_content(chapter_name: str) -> dict:
    """读取并删除 write_chapter 写入的指定章节完整内容"""
    return {"chapter_name": chapter_name, "full_content": _pending_chapter_contents.pop(chapter_name, "")}


# ── Tool 1: search_kb ──

@tool
async def search_kb(query: str, top_k: int = 5, use_rerank: bool = True) -> str:
    """检索贴敷式胰岛素泵知识库 (标准、法规、技术文档、测试报告等)。

    当需要查找具体标准条款、技术参数限值、法规要求、同类产品数据时，必须先调用此工具。
    不要在未检索的情况下编造标准条款号和具体限值。

    Args:
        query: 搜索查询关键词。应为具体的标准号、参数名或技术问题。
        top_k: 返回结果数量，默认5条。
        use_rerank: 是否启用Cross-Encoder精排（默认开启，提升精确率）。
                    两阶段检索：粗召回30条→BGE-Reranker精排→Top-K。

    Returns:
        JSON格式的检索结果列表，每项包含 content, source, score。
    """
    try:
        from app.services.rag.vector_store import VectorStore

        def _do_retrieve():
            store = VectorStore()
            if use_rerank:
                return store.retrieve_and_rerank(
                    query=query,
                    top_k=top_k,
                    candidate_pool_size=30,
                    vector_weight=0.85,
                )
            else:
                return store.retrieve_hybrid(
                    query=query,
                    top_k=top_k,
                    vector_weight=0.85,
                )

        # 精排增加30秒容错（粗召回90s + 精排30s）
        timeout_sec = 120.0 if use_rerank else 90.0
        results = await asyncio.wait_for(
            asyncio.to_thread(_do_retrieve),
            timeout=timeout_sec,
        )

        if not results:
            return json.dumps({
                "status": "no_results",
                "message": f'未找到与"{query}"直接相关的知识库内容。请用已有知识回答，并告知用户此为基于经验的建议，建议用户自行查证最新标准。',
                "results": [],
            }, ensure_ascii=False)

        formatted = []
        for r in results:
            # 优先使用精排分数，回退到相似度分数
            score = r.get("rerank_score", r.get("similarity", 0))
            formatted.append({
                "content": r.get("text", ""),
                "source": r.get("source_file", "未知来源"),
                "score": round(score, 3),
            })

        return json.dumps({
            "status": "ok",
            "query": query,
            "count": len(formatted),
            "reranked": use_rerank,
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


# ── Tool 1a: generate_search_query ──

async def _generate_search_queries(
    chapter_name: str,
    sub_title: str = "",
    content_points: list = None,
    intent: str = "",
    num_queries: int = 3,
) -> list[str]:
    """
    内部函数：调用 LLM 基于产品上下文+章节信息生成针对性检索查询词。

    供 generate_search_query 工具和 write_chapter/generate_section/design_outline 内部复用。

    Args:
        chapter_name: 章节名（如"设计开发阶段划分"）
        sub_title: 小节名（可选，逐小节检索时传入）
        content_points: 内容要点列表（可选）
        intent: Agent 描述的查询意图（可选，如"查找IEC 62304软件C级测试要求"）
        num_queries: 生成的查询词数量

    Returns:
        查询词列表，如 ["IEC 62304 Class C 软件单元测试要求", ...]
    """
    from app.services.minimax import _call_minimax_api_raw

    # 收集产品上下文
    product_name = _current_product_name.get() or "贴敷式胰岛素泵"
    product_class = _current_product_classification.get()
    intended_use = _current_product_intended_use.get()
    standards = _current_confirmed_standards.get()
    doc_type = _current_doc_type.get()

    context_lines = [f"- 产品名称: {product_name}"]
    if product_class:
        context_lines.append(f"- 分类: {product_class}")
    if intended_use:
        context_lines.append(f"- 预期用途: {intended_use}")
    if standards:
        context_lines.append(f"- 已确认适用标准: {', '.join(standards[:10])}")
    if doc_type:
        context_lines.append(f"- 目标文档类型: {doc_type}")
    context_block = "\n".join(context_lines)

    target = chapter_name
    if sub_title:
        target = f"{chapter_name} → {sub_title}"

    points_hint = ""
    if content_points:
        points_hint = "\n本节内容要点:\n" + "\n".join(f"- {p}" for p in content_points[:5])

    intent_hint = ""
    if intent:
        intent_hint = f"\nAgent查询意图: {intent}"

    system_prompt = f"""你是医疗器械法规标准检索专家。基于以下产品上下文和章节信息，生成 {num_queries} 条精准的知识库检索查询词。

## 产品上下文
{context_block}

## 当前检索目标
章节: {target}{points_hint}{intent_hint}

## 任务
生成 {num_queries} 条检索查询词，每条查询词应：
1. 包含具体的标准号、参数名、技术术语或法规条款号（不要泛泛而谈）
2. 与产品上下文紧密结合（如软件C级、贴敷式胰岛素泵、闭环控制等）
3. 覆盖不同检索维度（如：标准要求类、测试方法类、限值参数类）

## 输出格式（严格JSON）
{{
  "queries": ["查询词1", "查询词2", "查询词3"]
}}

只输出JSON，不要包含其他文字或markdown代码块标记。"""

    try:
        response = await asyncio.wait_for(
            asyncio.to_thread(
                _call_minimax_api_raw,
                system_prompt=system_prompt,
                user_prompt=f"请生成 {num_queries} 条针对「{target}」的检索查询词。",
                temperature=0.2,
                max_tokens=1024,
            ),
            timeout=60.0,
        )
        if not response:
            return []

        # 提取JSON
        import re
        json_text = response.strip()
        match = re.search(r'```(?:json)?\s*([\s\S]*?)```', json_text)
        if match:
            json_text = match.group(1).strip()
        else:
            match = re.search(r'\{[\s\S]*\}', json_text)
            if match:
                json_text = match.group(0).strip()

        parsed = json.loads(json_text)
        queries = parsed.get("queries", [])
        return [str(q).strip() for q in queries if str(q).strip()][:num_queries]
    except Exception as e:
        print(f"[generate_search_query] LLM生成查询失败: {e}")
        # 回退到简单拼接
        fallback = " ".join(filter(None, [doc_type, chapter_name, sub_title]))
        if content_points:
            fallback += " " + " ".join(content_points[:3])
        return [fallback] if fallback else []


@tool
async def generate_search_query(
    chapter_name: str,
    intent: str = "",
    num_queries: int = 3,
) -> str:
    """基于当前产品上下文和章节信息，生成精准的知识库检索查询词。

    调用此工具后，会返回 2-4 条针对性查询词，每条覆盖不同检索维度
    （标准要求/测试方法/限值参数）。然后用这些查询词分别调用 search_kb 检索。

    何时调用:
    - 开始生成新章节前（替代凭直觉拼凑查询词）
    - 需要查找具体标准条款但不确定用什么关键词时
    - 用户描述模糊的查询意图时（如"查一下软件测试要求"）

    Args:
        chapter_name: 章节名或主题，如"设计开发阶段划分"、"软件验证"
        intent: 查询意图描述（可选），如"查找IEC 62304 C级软件单元测试的具体要求和通过准则"
        num_queries: 生成的查询词数量，默认3条

    Returns:
        JSON格式的查询词列表，供后续 search_kb 使用。
    """
    queries = await _generate_search_queries(
        chapter_name=chapter_name,
        intent=intent,
        num_queries=num_queries,
    )

    if not queries:
        return json.dumps({
            "status": "error",
            "message": "查询词生成失败，请直接用章节名或标准号调用 search_kb。",
            "queries": [],
        }, ensure_ascii=False)

    return json.dumps({
        "status": "ok",
        "chapter_name": chapter_name,
        "intent": intent,
        "queries": queries,
        "usage_hint": "请用上述每条查询词分别调用 search_kb，合并结果后取最相关的作为生成参考。",
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

        # RAG 检索: 用 LLM 基于产品上下文+章节信息生成针对性查询词
        rag_context = ""
        try:
            queries = await _generate_search_queries(
                chapter_name=section_name,
                intent=chapter_query[:200] if chapter_query else "",
                num_queries=3,
            )
            if not queries:
                # 回退到原硬编码拼接
                fallback = f"{doc_label} {section_name}"
                if chapter_query:
                    fallback += f" {chapter_query[:200]}"
                queries = [fallback]

            # 对每条查询词调用 search_kb，合并去重结果
            merged_results = []
            seen_keys = set()
            for q in queries:
                rag_result = await search_kb.ainvoke({"query": q, "top_k": 5})
                rag_data = json.loads(rag_result)
                if rag_data.get("status") == "ok" and rag_data.get("results"):
                    for item in rag_data["results"]:
                        key = item.get("content", "")[:100]
                        if key not in seen_keys:
                            seen_keys.add(key)
                            merged_results.append(item)

            if merged_results:
                lines = ["\n\n# 知识库参考资料（必须优先依据以下内容编写）:"]
                for j, r in enumerate(merged_results, 1):
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

    在现有内容基础上修改，保持与首次生成相同的详细程度和专业深度。
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
        expert_section = f"\n\n{expert_prompt}" if expert_prompt else ""

        # ── 获取当前章节内容 ──
        current_section_content = ""
        current_markdown = _current_generated_markdown.get()
        if current_markdown:
            import re
            pattern = rf'^## \s*{re.escape(section_name)}\s*$'
            lines = current_markdown.split('\n')
            in_section = False
            section_lines = []
            for line in lines:
                if re.match(pattern, line.strip()):
                    in_section = True
                    section_lines.append(line)
                elif in_section and re.match(r'^## ', line.strip()):
                    break
                elif in_section:
                    section_lines.append(line)
            if section_lines:
                current_section_content = '\n'.join(section_lines)

        # ── RAG 检索：获取相关知识库参考资料 ──
        rag_context = ""
        try:
            rag_query = f"{doc_label} {section_name} {instruction}"
            rag_result = await search_kb.ainvoke({"query": rag_query, "top_k": 5})
            rag_data = json.loads(rag_result)
            if rag_data.get("status") == "ok" and rag_data.get("results"):
                lines = ["\n\n# 知识库参考资料（必须优先依据以下内容进行修改）:"]
                for j, r in enumerate(rag_data["results"], 1):
                    lines.append(
                        f"\n[参考{j}] 来源: {r['source']} (相关度:{r['score']})\n{r['content']}"
                    )
                rag_context = "".join(lines)
        except Exception:
            pass

        # ── 构建高质量修订 prompt（复用 write_chapter 的详细质量要求）──
        system_prompt = f"""你是一位贴敷式胰岛素泵RA文档专家。用户要求修改《{doc_label}》文档中「{section_name}」章节。{expert_section}

要求:
- 内容必须极其细致和具体，每个段落都要有实质性内容，不能只写框架标题
- 内容专业、完整，符合NMPA注册申报要求
- 所有标准条款引用必须有明确的条款号
- 使用Markdown格式，标题层级清晰 (##, ###)
- 技术参数要具体、可测量、有明确的数值范围
- 表格要填写完整，不能留"(描述)"或"待填写"等占位符
- 针对贴敷式胰岛素泵产品特性编写
- 修改后内容的详细程度要与首次生成的其他章节保持一致，像实际可用于注册申报的正式文档一样
- 用中文表述 (标准号和必要缩写除外)

修改规则:
- 根据用户指令修改内容，但保持修改后章节的详细程度和专业深度不低于原章节
- 如果用户指令要求添加新内容，应展开详细描述（每个要点至少200字），而非只加一句话
- 保持Markdown格式和标题层级
- 如果修改影响了其他章节的参数/引用，在回复末尾用"⚠️ 关联影响:"标注
- 用中文回复

## 输出结构要求
请按以下结构组织修改后的内容:
1. 首先用1-2段概述本节要点（含法规依据和产品适用性）
2. 然后逐个详细阐述每个关键要求（每个要求至少200字，包含法规条款原文引用、产品参数映射、实施建议）
3. 如涉及数据/参数对比，以表格形式呈现（至少3列）
4. 最后用1段总结本节的合规要点和与贴敷式胰岛素泵的关联性"""

        user_prompt = f"""请修改《{doc_label}》的「{section_name}」章节，修改指令: {instruction}

请在修改后:
1. 输出修改后的完整章节内容（保持与首次生成相同的详细程度）
2. 在末尾用"📝 修改摘要:"列出具体变更点"""

        # 注入当前章节内容和 RAG 上下文
        if current_section_content:
            user_prompt += f"\n\n# 当前章节内容（请在此基础上修改）:\n{current_section_content}"
        if rag_context:
            user_prompt += f"\n{rag_context}"

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
    # 用 LLM 基于产品上下文+章节信息生成针对性查询词，替代硬编码 query
    rag_context = ""
    try:
        queries = await _generate_search_queries(
            chapter_name=f"{doc_type} 章节结构设计",
            intent="查找该文档类型的标准章节结构要求和适用法规",
            num_queries=3,
        )
        if not queries:
            queries = [f"{doc_type} 章节结构 标准要求"]
        merged_results = []
        seen_keys = set()
        for q in queries:
            rag_result = await search_kb.ainvoke({"query": q, "top_k": 5})
            rag_data = json.loads(rag_result)
            if rag_data.get("status") == "ok" and rag_data.get("results"):
                for item in rag_data["results"]:
                    key = item.get("content", "")[:100]
                    if key not in seen_keys:
                        seen_keys.add(key)
                        merged_results.append(item)
        if merged_results:
            rag_parts = [
                f"[参考{i}] 来源: {r['source']} (相关度: {r['score']})\n{r['content']}"
                for i, r in enumerate(merged_results, 1)
            ]
            rag_context = "\n\n".join(rag_parts)
            print(f"[agent_tools] RAG for outline: {len(merged_results)} results (from {len(queries)} queries)")
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
    """从框架JSON中查找指定章节的小节列表（支持4级层级 schema）。

    支持两种 schema:
    - 旧 schema: chapters[].subsections[] (2级，仅 ###)
    - 新 schema: chapters[].sections[].subsections[].sub_subsections[] (4级)

    返回扁平化的 leaf 小节列表，每个元素:
    {
        "title": "X.Y.Z 标题",         # 不含 markdown 标记
        "level": 3 | 4 | 5,              # ### / #### / #####
        "content_points": [...],          # 该小节的要点
        "parent_path": "X > X.Y",         # 父级标题路径（用于上下文）
        "markdown_prefix": "### " | "#### " | "##### ",
    }

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

    target_chapter = None
    for ch in chapters:
        title = ch.get("title", "")
        title_clean = re.sub(r'^第[一二三四五六七八九十\d]+章\s*', '', title).strip()
        if (name_clean == title_clean
                or name_clean in title_clean
                or title_clean in name_clean
                or chapter_name == title
                or chapter_name in title
                or title in chapter_name):
            target_chapter = ch
            break

    if not target_chapter:
        return []

    # 收集 leaf 小节
    leaf_subs = []

    def _add_leaf(title: str, level: int, content_points: list, parent_path: str):
        prefix_map = {3: "### ", 4: "#### ", 5: "##### "}
        leaf_subs.append({
            "title": title,
            "level": level,
            "content_points": content_points or [],
            "parent_path": parent_path,
            "markdown_prefix": prefix_map.get(level, "### "),
        })

    # 新 schema: chapters[].sections[].subsections[].sub_subsections[]
    sections = target_chapter.get("sections", [])
    if sections:
        for sec in sections:
            sec_title = sec.get("title", "")
            sec_path = sec_title
            subsects = sec.get("subsections", [])
            for sub in subsects:
                sub_title = sub.get("title", "")
                sub_path = f"{sec_path} > {sub_title}" if sec_path else sub_title
                sub_subsects = sub.get("sub_subsections", [])
                if sub_subsects:
                    # 有子小节，sub 是父级，leaf 在 sub_subsections
                    for ssub in sub_subsects:
                        ssub_title = ssub.get("title", "")
                        ssub_path = f"{sub_path} > {ssub_title}" if sub_path else ssub_title
                        _add_leaf(ssub_title, 5, ssub.get("content_points", []), ssub_path)
                else:
                    # 无子小节，sub 本身就是 leaf (level 4)
                    _add_leaf(sub_title, 4, sub.get("content_points", []), sub_path)
        return leaf_subs

    # 旧 schema: chapters[].subsections[]
    old_subsects = target_chapter.get("subsections", [])
    for sub in old_subsects:
        sub_title = sub.get("title", "")
        _add_leaf(sub_title, 3, sub.get("content_points", []), "")
    return leaf_subs

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
            # 用 LLM 基于产品上下文+章节信息生成针对性查询词（替代硬编码拼接）
            queries = await _generate_search_queries(
                chapter_name=chapter_name,
                sub_title=sub_title,
                content_points=content_points,
                num_queries=3,
            )
            if not queries:
                # 回退到简单拼接
                fallback = " ".join(filter(None, [doc_type, chapter_name, sub_title]))
                if content_points:
                    fallback += " " + " ".join(content_points[:3])
                queries = [fallback] if fallback else []

            # 对每条查询词调用 search_kb，合并去重结果
            merged_results = []
            seen_sources = set()
            try:
                async with _rag_sem:
                    for q in queries:
                        r = await search_kb.ainvoke({"query": q, "top_k": 5})
                        data = json.loads(r)
                        if data.get("status") == "ok" and data.get("results"):
                            for item in data["results"]:
                                # 按内容前100字符去重
                                key = item.get("content", "")[:100]
                                if key not in seen_sources:
                                    seen_sources.add(key)
                                    merged_results.append(item)
            except Exception:
                pass
            return (sub_title, merged_results)

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

            parent_path = sub.get("parent_path", "")
            level = sub.get("level", 3)
            markdown_prefix = sub.get("markdown_prefix", "### ")

            context_hint = ""
            if parent_path:
                context_hint = f"（层级位置：{parent_path}）"

            system_prompt = (
                f"你是一位贴敷式胰岛素泵RA文档专家。"
                f"请编写《{doc_label}》文档中「{chapter_name}」章节下「{sub_title}」小节的内容{context_hint}。"
                f"{expert_section}\n\n"
                f"## 写作风格（参考《产品技术要求》和《软件需求规范》：精简+短句）\n"
                f"\n"
                f"要求:\n"
                f"- **每个要点 1-3 句话，单句 30-80 字，绝对不写长段落**\n"
                f"- 大量使用 `- 项目符号` 列表项表达并列规则/参数\n"
                f"- 每个要点风格如 \"支持X功能\"/\"参数：Y\"/\"协议：Z\"\n"
                f"- 涉及多个数值参数时用小表格呈现（2-10 行 × 2-6 列），不要写大表格\n"
                f"- 所有标准条款引用必须有明确的条款号\n"
                f"- 技术参数要具体、可测量、有明确的数值范围\n"
                f"- 表格要填写完整，不能留\"(描述)\"或\"待填写\"等占位符\n"
                f"- 针对贴敷式胰岛素泵产品特性编写\n"
                f"- 用中文表述 (标准号和必要缩写除外)\n"
                f"- 只生成本小节正文，不要添加章节标题（如 ## 或 ###）\n"
                f"\n"
                f"## 禁止事项\n"
                f"- **禁止写概述段、引入段、铺垫段**\n"
                f"- **禁止写总结段、归纳段、结尾段**\n"
                f"- **禁止写\"本小节将介绍...\"/\"综上所述...\"等过渡句**\n"
                f"- **禁止把同一要点展开成完整段落**\n"
                f"\n"
                f"## 输出结构示例（参考《产品技术要求》风格）\n"
                f"直接写要点/列表/小表格，第一行就是实质内容（不是标题）。\n"
                f"\n"
                f"## 字数约束\n"
                f"本小节总字数控制在 200-500 字之间。**宁少勿多**。"
            )

            user_prompt = (
                f"请编写「{chapter_name}」->「{sub_title}」小节的内容。"
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
                            max_tokens=2048,
                        ),
                        timeout=300.0,
                    )
                if response:
                    return (sub_title, markdown_prefix, response)
                else:
                    return (sub_title, markdown_prefix, "[错误] 无法生成小节内容。")
            except asyncio.TimeoutError:
                return (sub_title, markdown_prefix, "[错误] 生成小节超时（300秒）。")
            except Exception as e:
                return (sub_title, markdown_prefix, f"[错误] 生成小节异常: {str(e)}")

        gen_tasks = [asyncio.create_task(_gen_subsection(s)) for s in subsections]
        await asyncio.gather(*gen_tasks)

        # 组装章节完整内容（按 level 使用 ### / #### / #####）
        parts = [f"## {chapter_name}\n\n"]
        for task in gen_tasks:
            sub_title, markdown_prefix, content = task.result()
            parts.append(f"{markdown_prefix}{sub_title}\n\n{content}\n\n")
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
        # 用 LLM 生成针对性查询词，替代硬编码 query
        rag_context = ""
        try:
            queries = await _generate_search_queries(
                chapter_name=chapter_name,
                intent=f"查找 {doc_type} 文档中 {chapter_name} 章节的标准要求和法规依据",
                num_queries=3,
            )
            if not queries:
                queries = [f"{doc_type} {chapter_name}"]
            merged_results = []
            seen_keys = set()
            for q in queries:
                rag_result = await search_kb.ainvoke({"query": q, "top_k": 5})
                rag_data = json.loads(rag_result)
                if rag_data.get("status") == "ok" and rag_data.get("results"):
                    for item in rag_data["results"]:
                        key = item.get("content", "")[:100]
                        if key not in seen_keys:
                            seen_keys.add(key)
                            merged_results.append(item)
            if merged_results:
                rag_parts = [
                    f"[参考{i}] 来源: {r['source']} (相关度: {r['score']})\n{r['content']}"
                    for i, r in enumerate(merged_results, 1)
                ]
                rag_context = "\n\n".join(rag_parts)
                print(f"[agent_tools] Fallback RAG for '{chapter_name}': "
                      f"{len(merged_results)} results (from {len(queries)} queries)")
        except Exception as e:
            print(f"[agent_tools] RAG failed for '{chapter_name}': {e}")

        system_prompt = (
            f"你是一位贴敷式胰岛素泵RA文档专家。"
            f"请编写《{doc_label}》文档中「{chapter_name}」章节的完整内容。"
            f"{expert_section}\n\n"
            f"## 写作风格（参考《产品技术要求》和《软件需求规范》：精简+短句+多层结构）\n"
            f"\n"
            f"要求:\n"
            f"- **每个要点 1-3 句话，单句 30-80 字，绝对不写长段落**\n"
            f"- 大量使用 `- 项目符号` 列表项表达并列规则/参数\n"
            f"- 每个要点风格如 \"支持X功能\"/\"参数：Y\"/\"协议：Z\"\n"
            f"- 涉及多个数值参数时用小表格呈现（2-10 行 × 2-6 列）\n"
            f"- 使用 4 级层级 Markdown: ## (章) -> ### (节) -> #### (小节) -> ##### (子小节)\n"
            f"- 所有标准条款引用必须有明确的条款号\n"
            f"- 技术参数要具体、可测量、有明确的数值范围\n"
            f"- 表格要填写完整，不能留\"(描述)\"或\"待填写\"等占位符\n"
            f"- 针对贴敷式胰岛素泵产品特性编写\n"
            f"- 用中文表述 (标准号和必要缩写除外)\n"
            f"\n"
            f"## 禁止事项\n"
            f"- **禁止写概述段、引入段、铺垫段**\n"
            f"- **禁止写总结段、归纳段、结尾段**\n"
            f"- **禁止把同一要点展开成完整段落**\n"
            f"\n"
            f"## 字数约束\n"
            f"整章总字数控制在 800-2000 字之间。**宁少勿多**。"
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
                        max_tokens=4096,
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


# ── Tool 6b: summarize_section ──
# 小节精简工具：对已生成章节中的每个 ### 小节内容做 LLM 精简，
# 用精简后内容替换原小节。支持字数模式（按原字数比例分配预算）和比例模式。
# 完整内容通过 _pending_chapter_contents 旁路传递，避免大段内容进入 LLM 对话历史。

# 摘要子代理实例（懒加载）
_summary_agent = None


def _get_summary_agent():
    """懒加载摘要子代理实例"""
    global _summary_agent
    if _summary_agent is None:
        from app.services.subagents import create_summary_agent
        _summary_agent = create_summary_agent()
    return _summary_agent


def _split_chapter_into_subsections(chapter_content: str) -> list[dict]:
    """将章节内容按 ### / #### / ##### 小节拆分（支持4级层级）

    Args:
        chapter_content: 章节完整内容，以 `## {章节名}` 开头

    Returns:
        list of {"header": "## 章节名", "subsections": [{"title": "### 小节标题", "body": "小节正文"}]}
        每遇到 ### / #### / ##### 开头的行即开启一个新小节（保持向后兼容：
        仅含 ### 的旧文档行为不变；含 #### / ##### 的新4级结构会被拆分为更细粒度小节）
        若无任何小节标题，subsections 为空列表（调用方应走整章节精简回退路径）
    """
    if not chapter_content:
        return [{"header": "", "subsections": []}]

    lines = chapter_content.split("\n")
    header_lines = []      # ## 章节标题及其后的空行
    subsections = []

    # 提取章节标题（## 开头）部分
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith("## ") and not stripped.startswith("### "):
            # 章节标题行
            header_lines.append(line)
            i += 1
            # 收集章节标题后的空行（直到第一个 ### / #### / ##### 或正文开始）
            while i < len(lines):
                s = lines[i].strip()
                if s == "":
                    header_lines.append(lines[i])
                    i += 1
                elif s.startswith("### ") or s.startswith("#### ") or s.startswith("##### "):
                    break
                else:
                    # 章节标题后直接是正文（无小节），整体作为单一小节
                    break
            break
        else:
            # 非 ## 开头（异常情况），作为 header 保留
            header_lines.append(line)
            i += 1

    # 收集 ### / #### / ##### 小节
    body_lines = lines[i:]
    current_title = None
    current_body = []

    for line in body_lines:
        stripped = line.strip()
        is_subsection_heading = (
            stripped.startswith("### ")
            or stripped.startswith("#### ")
            or stripped.startswith("##### ")
        )

        if is_subsection_heading:
            # 新小节开始，保存前一小节
            if current_title is not None:
                subsections.append({
                    "title": current_title,
                    "body": "\n".join(current_body).strip(),
                })
            current_title = stripped
            current_body = []
        else:
            if current_title is not None:
                current_body.append(line)
            else:
                # ### 之前的非空行（已在 header 收集过空行），忽略
                pass

    # 收集最后一个小节
    if current_title is not None:
        subsections.append({
            "title": current_title,
            "body": "\n".join(current_body).strip(),
        })

    return [{
        "header": "\n".join(header_lines).rstrip(),
        "subsections": subsections,
    }]


def _count_chinese_chars(text: str) -> int:
    """统计有效字符数（中文+英文字符，排除空白和markdown标记）

    用于估算小节内容长度，作为字数预算分配依据。
    """
    if not text:
        return 0
    import re
    # 去除 markdown 标记符号和空白
    cleaned = re.sub(r'[#*_`>\-\[\]\(\)\|]', '', text)
    cleaned = re.sub(r'\s+', '', cleaned)
    return len(cleaned)


def _truncate_at_boundary(text: str, max_chars: int) -> str:
    """按句子/段落边界截断文本，使字数不超过 max_chars 的 1.05 倍。

    优先在句子边界（。！？；\\n）截断，保留表格和代码块完整性。
    用于 LLM 精简后仍超目标时的硬截断兜底。
    修复：原 1.15× 上限偏高，比例模式下应更紧凑。
    """
    if not text or max_chars <= 0:
        return text

    current = _count_chinese_chars(text)
    if current <= max_chars * 1.05:
        return text  # 已经达标

    # 逐句累加，在不超过 max_chars * 1.05 的前提下找最佳截断点
    import re
    sentences = re.split(r'(?<=[。！？；\n])', text)

    result_parts = []
    accumulated = 0
    limit = int(max_chars * 1.05)

    for sent in sentences:
        if not sent:
            continue
        sent_chars = _count_chinese_chars(sent)
        if accumulated + sent_chars > limit and result_parts:
            break
        result_parts.append(sent)
        accumulated += sent_chars

    if not result_parts:
        # 所有句子都太长，硬截断
        return text[:max_chars * 2]

    return "".join(result_parts).rstrip()


async def _summarize_one_subsection(
    sub_title: str,
    sub_body: str,
    target_chars: int,
    chapter_name: str,
    doc_label: str,
    is_ratio_mode: bool = False,
    aggressive: bool = False,
) -> tuple[str, str, dict]:
    """对单个小节调用 LLM 精简

    Args:
        sub_title: 小节标题行（含 ### 前缀）
        sub_body: 小节正文（不含标题行）
        target_chars: 目标字数
        chapter_name: 所属章节名（用于 prompt 上下文）
        doc_label: 文档类型标签
        is_ratio_mode: 是否为比例模式（影响 max_tokens 计算，避免小目标时 LLM 输出过大）
        aggressive: 是否为激进模式（二次精简时启用，更严格的 prompt 和 max_tokens）

    Returns:
        (sub_title, 精简后正文, 统计信息dict)
        失败时返回 (sub_title, 原sub_body, {"status": "error", ...})
    """
    from app.services.minimax import _call_minimax_api_raw

    orig_chars = _count_chinese_chars(sub_body)
    # 比例化保底：下限 40（避免被压成 0），但不超过原文 90%（避免扩展破坏比例）
    # 修复固定 80 字保底对短小节导致 ratio 反向扩展的问题
    if target_chars < 40:
        target_chars = min(40, int(orig_chars * 0.9)) if orig_chars > 0 else 40
    hard_limit = int(target_chars * 1.15)  # 硬上限：目标+15%（修复：原 1.15×→1.3× 区间太宽）

    system_prompt = f"""你是医疗器械注册文档的小节内容精简专家。

## 任务
精简以下小节内容，**严格控制在 {target_chars} 字以内，最多不超过 {hard_limit} 字**。
本小节属于《{doc_label}》文档「{chapter_name}」章节。

## 精简规则

### 必须保留（不可删除/篡改）
- 所有法规标准条款号（如 "ISO 13485 §7.3.2"、"GB 9706.224-2021 第4章"）
- 所有具体技术参数和数值（如 "0.05 U/h"、"IPX8"、"3-7天"）
- 表格中的数据行（保留表格，可精简表格周围说明文字）
- 核心结论和合规判定语句
- 关键术语首次出现时的定义

### 可以精简
- 重复表述的同一观点（合并为一句）
- 过度展开的背景介绍（压缩为一句）
- 冗长的过渡句和铺垫（删除）
- 同一标准的多条引用（合并为一条带多个条款号）
- 非关键的示例和说明性文字

### 禁止
- 编造原文没有的数据、条款号或参数
- 删除任何法规标准引用
- 改变技术参数的数值或单位
- 增加原文没有的新观点或新结论

## 输出格式
- 直接输出精简后的 Markdown 正文
- 不要输出小节标题（### XXX），只输出正文
- 不要输出任何解释、前言、总结
- 保留原有的 Markdown 格式（表格、列表、加粗等）""" + (
    """

## ⚠️ 紧急要求：必须严格控制字数
你之前的精简尝试未达标，现在必须**更激进地**精简。
- 大胆合并同义句、删除所有过渡性描述、削减举例说明
- 表格只保留表头和必要数据行，删除解释列
- 法规条款号可保留但删除其后的展开说明
- 再次严格控制在 {target_chars} 字以内（最多 {hard_limit} 字）"""
    if aggressive else ""
)

    user_prompt = f"""请精简以下小节内容（必须控制在 {target_chars} 字以内，最多 {hard_limit} 字）：

{sub_body}"""

    try:
        # qwen3.5:122b 模型存在 thinking tokens 占用：实测 num_predict < 16384 时
        # 模型会消耗所有 tokens 但 content 为空，done_reason="length"。
        # 因此下限和上限都至少 16384，否则空响应 → 精简失败。
        # 比例模式小目标时：max_tok 应 = target*3 (中文字 1字≈1.5 token) + 16384 thinking 预算
        # 非比例模式：max_tok 应 = target*3 + 16384 thinking 预算
        if is_ratio_mode:
            max_tok = min(max(int(target_chars * 3), 16384), 16384)
        else:
            max_tok = min(max(target_chars * 3, 16384), 16384)
        async with _llm_semaphore:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    _call_minimax_api_raw,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=0.2 if not aggressive else 0.1,
                    max_tokens=max_tok,
                ),
                timeout=180.0,
            )

        if not response:
            print(f"[summarize_section] LLM空响应: 小节='{sub_title}', "
                  f"目标={target_chars}字, 原文={orig_chars}字 "
                  f"(检查 Ollama {os.getenv('OLLAMA_BASE_URL', 'http://localhost:11435')} "
                  f"模型 {os.getenv('OLLAMA_MODEL', 'qwen3.5:122b')} 是否正常)")
            return (sub_title, sub_body, {
                "status": "error", "reason": "LLM空响应",
                "orig_chars": orig_chars, "new_chars": orig_chars,
            })

        # 清理响应：移除可能误加的 ### 标题行
        cleaned = response.strip()
        # 若 LLM 误加了 ### 开头，移除第一行
        if cleaned.startswith("### "):
            lines = cleaned.split("\n", 1)
            cleaned = lines[1].strip() if len(lines) > 1 else cleaned

        new_chars = _count_chinese_chars(cleaned)

        # 迭代精简：若结果超目标 15%，最多重试 2 轮（共 3 轮），每轮更激进
        # 修复：原 1.2× 阈值过宽，比例模式下允许 20% 偏差导致总比例偏离目标
        max_rounds = 3
        for round_idx in range(1, max_rounds):
            if new_chars <= target_chars * 1.15:
                break  # 已达标（目标+15%以内）
            print(f"[summarize_section] 第{round_idx}轮精简不足: "
                  f"{new_chars}字 > 目标{target_chars}字×1.15={int(target_chars*1.15)}字，重试")
            retry_user_prompt = f"""上一轮精简后为 {new_chars} 字，仍超出目标 {target_chars} 字。
请**更激进地**精简以下内容，必须控制在 {target_chars} 字以内（最多 {hard_limit} 字）。
大胆删除重复表述、冗长背景、过渡句，只保留核心结论、法规条款号和技术参数：

{cleaned}"""
            try:
                async with _llm_semaphore:
                    retry_response = await asyncio.wait_for(
                        asyncio.to_thread(
                            _call_minimax_api_raw,
                            system_prompt=system_prompt,
                            user_prompt=retry_user_prompt,
                            temperature=0.1,
                            max_tokens=max_tok,
                        ),
                        timeout=180.0,
                    )
                if retry_response:
                    retry_cleaned = retry_response.strip()
                    if retry_cleaned.startswith("### "):
                        lines = retry_cleaned.split("\n", 1)
                        retry_cleaned = lines[1].strip() if len(lines) > 1 else retry_cleaned
                    retry_chars = _count_chinese_chars(retry_cleaned)
                    # 只在比上一轮更好的时候接受
                    if retry_chars < new_chars:
                        cleaned = retry_cleaned
                        new_chars = retry_chars
            except Exception as e:
                print(f"[summarize_section] 第{round_idx}轮重试失败: {e}")
                break

        # 硬截断兜底：3 轮后仍超目标 15%，按句子边界截断
        # 修复：原阈值 1.3× 偏高，会让最终结果超出目标
        if new_chars > target_chars * 1.15:
            truncated = _truncate_at_boundary(cleaned, target_chars)
            if truncated and _count_chinese_chars(truncated) < new_chars:
                print(f"[summarize_section] 硬截断兜底: {new_chars} -> "
                      f"{_count_chinese_chars(truncated)} 字")
                cleaned = truncated
                new_chars = _count_chinese_chars(cleaned)

        return (sub_title, cleaned, {
            "status": "ok",
            "orig_chars": orig_chars,
            "new_chars": new_chars,
            "target_chars": target_chars,
        })

    except asyncio.TimeoutError:
        print(f"[summarize_section] LLM超时(180秒): 小节='{sub_title}', 目标={target_chars}字")
        return (sub_title, sub_body, {
            "status": "error", "reason": "LLM超时（180秒）",
            "orig_chars": orig_chars, "new_chars": orig_chars,
        })
    except Exception as e:
        print(f"[summarize_section] LLM调用异常: 小节='{sub_title}', "
              f"错误={type(e).__name__}: {e}")
        return (sub_title, sub_body, {
            "status": "error", "reason": str(e),
            "orig_chars": orig_chars, "new_chars": orig_chars,
        })


@tool
async def summarize_section(
    section_name: str,
    mode: str = "ratio",
    target: float = 0.5,
    doc_type: str = "",
) -> str:
    """对已生成章节中的每个小节内容进行精简，用精简后内容替换原小节。

    支持两种模式：
    - 字数模式 (mode="words"): target 为目标字数（int），按各小节原字数比例分配预算
    - 比例模式 (mode="ratio"): target 为压缩比例（float 0.1~1.0），每小节按原字数×比例精简

    精简由 LLM 完成，严格保留法规条款号、技术参数、表格数据和核心结论。
    单个小节精简失败时保留原文，不影响其他小节。

    调用时机:
    - 用户说"精简/总结/压缩 XX 章节内容"时
    - 用户要求"把这一章缩短到 XXX 字"时
    - 文档生成完成后用户要求缩短整体内容时

    Args:
        section_name: 要精简的章节名称（必须已存在于 generated_sections 中）
        mode: 精简模式，"words"（字数模式）或 "ratio"（比例模式）
        target: 目标值。mode="words" 时为 int 字数；mode="ratio" 时为 float 比例（0.1~1.0）
        doc_type: 文档类型标识，用于prompt上下文。为空时自动从上下文获取。

    Returns:
        JSON格式结果，包含精简前后字数对比、各小节状态。完整精简内容通过旁路传递给
        _after_tools_node，不进入LLM对话历史。
    """
    try:
        from app.services.doc_types import DOC_TYPE_LABELS

        # 从上下文获取 doc_type 和当前 generated_sections
        if not doc_type:
            doc_type = _current_doc_type.get()
        doc_label = DOC_TYPE_LABELS.get(doc_type, doc_type) if doc_type else "设计策划文档"

        # 参数校验
        if mode not in ("words", "ratio"):
            return json.dumps({
                "status": "error",
                "message": f'参数 mode 必须为 "words" 或 "ratio"，当前为 "{mode}"',
            }, ensure_ascii=False)

        if mode == "ratio":
            if not (0.1 <= float(target) <= 1.0):
                return json.dumps({
                    "status": "error",
                    "message": f"比例模式下 target 应在 0.1~1.0 之间，当前为 {target}",
                }, ensure_ascii=False)
        else:
            if int(target) < 100:
                return json.dumps({
                    "status": "error",
                    "message": f"字数模式下 target 应不小于 100 字，当前为 {target}",
                }, ensure_ascii=False)

        # 读取当前章节内容
        current_markdown = _current_generated_markdown.get()
        if not current_markdown:
            return json.dumps({
                "status": "error",
                "message": "当前没有已生成的文档内容。请先生成至少一个章节。",
            }, ensure_ascii=False)

        # 从完整 markdown 中提取指定章节内容
        # 章节内容格式：# {章节名}\n\n{章节正文}\n\n
        # 其中章节正文以 ## {章节名} 开头
        import re
        chapter_pattern = rf'^# \s*{re.escape(section_name)}\s*$'
        lines = current_markdown.split("\n")
        in_chapter = False
        chapter_lines = []
        for line in lines:
            if re.match(chapter_pattern, line.strip()):
                in_chapter = True
                continue  # 跳过 # 章节名 这一行（generated_sections 存的是 ## 开头的内容）
            elif in_chapter and re.match(r'^# ', line.strip()):
                # 遇到下一个 # 章节，结束
                break
            elif in_chapter:
                chapter_lines.append(line)

        chapter_content = "\n".join(chapter_lines).strip()
        if not chapter_content:
            return json.dumps({
                "status": "error",
                "message": f'未找到章节「{section_name}」。请确认章节名称正确，且该章节已生成。',
                "available_sections": list(_get_section_names_from_markdown(current_markdown)),
            }, ensure_ascii=False)

        # 拆分小节
        parsed = _split_chapter_into_subsections(chapter_content)
        if not parsed or not parsed[0]["subsections"]:
            # 回退：无 ### 小节，整体作为一个小节精简
            header = chapter_content.split("\n", 1)[0] if "\n" in chapter_content else ""
            subsections = [{"title": "### 概述", "body": chapter_content}]
            parsed = [{"header": header, "subsections": subsections}]
            print(f"[summarize_section] No ### subsections found in '{section_name}', "
                  f"treating whole chapter as single subsection")

        subsections = parsed[0]["subsections"]
        header = parsed[0]["header"]
        sub_count = len(subsections)

        # 计算每个小节的目标字数
        orig_chars_list = [_count_chinese_chars(s["body"]) for s in subsections]
        orig_total = sum(orig_chars_list) or 1  # 避免除0

        targets = []
        ratio = float(target) if mode == "ratio" else None
        if mode == "ratio":
            # 比例化保底：下限 40（避免小节被压成 0），但不超过原文 90%（防扩展破坏比例）
            # 修复：原固定 80 字保底对短小节导致 ratio 反向扩展（如 50 字小节 ratio=0.5 → 实际 160%）
            for oc in orig_chars_list:
                raw_target = int(oc * ratio)
                if raw_target < 40 and oc > 0:
                    targets.append(min(40, int(oc * 0.9)))
                else:
                    targets.append(max(raw_target, 40))
        else:  # words
            total_target = int(target)
            for oc in orig_chars_list:
                # 按原字数比例分配，保底80字
                allocated = max(int(total_target * oc / orig_total), 80)
                targets.append(allocated)

        print(f"[summarize_section] '{section_name}': {sub_count} subsections, "
              f"mode={mode}, target={target}, orig_total={orig_total}")

        # 并行精简各小节
        gen_tasks = [
            asyncio.create_task(_summarize_one_subsection(
                sub_title=s["title"],
                sub_body=s["body"],
                target_chars=targets[i],
                chapter_name=section_name,
                doc_label=doc_label,
                is_ratio_mode=(mode == "ratio"),
            ))
            for i, s in enumerate(subsections)
        ]
        await asyncio.gather(*gen_tasks)

        # 组装精简后章节内容（暂存以便失败再平衡）
        assembled = []  # [(sub_title, new_body, stat_dict, subsection_ref)]
        new_total = 0
        for i, task in enumerate(gen_tasks):
            sub_title, new_body, stat = task.result()
            new_total += stat["new_chars"]
            if stat["status"] == "ok":
                assembled.append((sub_title, new_body, {
                    "title": sub_title,
                    "orig_chars": stat["orig_chars"],
                    "new_chars": stat["new_chars"],
                    "target_chars": stat.get("target_chars", targets[i]),
                    "status": "ok",
                }, subsections[i]))
            else:
                # 失败保留原文
                assembled.append((sub_title, subsections[i]["body"], {
                    "title": sub_title,
                    "orig_chars": stat["orig_chars"],
                    "new_chars": stat["orig_chars"],
                    "target_chars": targets[i],
                    "status": "error",
                    "reason": stat.get("reason", "unknown"),
                }, subsections[i]))

        # ────────── 比例模式失败再平衡 ──────────
        # 修复：失败小节和超目标小节会拉高总比例，超出目标 15% 时触发二次精简
        if mode == "ratio" and ratio is not None and new_total > orig_total * ratio * 1.15:
            need_retry = [
                (i, t, b, s) for i, (t, b, s, _) in enumerate(assembled)
                if s.get("status") == "ok" and s.get("new_chars", 0) > s.get("target_chars", 0) * 1.15
            ]
            if need_retry:
                print(f"[summarize_section] 比例模式触发再平衡: "
                      f"new_total={new_total}, orig_total={orig_total}, "
                      f"当前比例={new_total/orig_total:.1%} > 目标 {ratio:.1%}×1.15="
                      f"{ratio*1.15:.1%}, 二次精简 {len(need_retry)} 节")
                # 收集"原文"和"上次精简结果"用于二次精简
                retry_tasks = [
                    asyncio.create_task(_summarize_one_subsection(
                        sub_title=t,
                        sub_body=b,  # 用上次精简结果继续精简（已经是较短版本）
                        target_chars=s.get("target_chars", targets[i]),
                        chapter_name=section_name,
                        doc_label=doc_label,
                        is_ratio_mode=True,
                        aggressive=True,  # 启用激进模式
                    ))
                    for i, t, b, s in need_retry
                ]
                await asyncio.gather(*retry_tasks)
                # 合并二次精简结果
                for (i, t, _, s), task in zip(need_retry, retry_tasks):
                    sub_title, new_body, stat = task.result()
                    if stat["status"] == "ok":
                        assembled[i] = (sub_title, new_body, {
                            "title": sub_title,
                            "orig_chars": s.get("orig_chars", stat["orig_chars"]),
                            "new_chars": stat["new_chars"],
                            "target_chars": stat.get("target_chars", s.get("target_chars")),
                            "status": "ok",
                        }, subsections[i])
                # 重新计算 new_total
                new_total = sum(item[2].get("new_chars", 0) for item in assembled)
                print(f"[summarize_section] 再平衡完成: new_total={new_total}, "
                      f"新比例={new_total/orig_total:.1%}")

        # 最终组装 parts / sub_stats
        parts = [header + "\n\n"] if header else []
        success_count = 0
        failed_count = 0
        sub_stats = []
        for sub_title, new_body, stat, _ in assembled:
            if stat.get("status") == "ok":
                success_count += 1
            else:
                failed_count += 1
            parts.append(f"{sub_title}\n\n{new_body}\n\n")
            sub_stats.append(stat)

        full_content = "".join(parts).rstrip() + "\n"

        # 通过旁路传递完整精简后内容（避免进入LLM对话历史）
        _pending_chapter_contents[section_name] = full_content

        print(f"[summarize_section] '{section_name}': "
              f"{success_count}/{sub_count} subsections summarized, "
              f"{orig_total} -> {new_total} chars "
              f"({(new_total/orig_total*100) if orig_total else 0:.1f}%)")

        return json.dumps({
            "status": "ok",
            "section_name": section_name,
            "mode": mode,
            "target": target,
            "subsections_count": sub_count,
            "success_count": success_count,
            "failed_count": failed_count,
            "orig_total_chars": orig_total,
            "new_total_chars": new_total,
            "compression_ratio": round(new_total / orig_total, 3) if orig_total else 0,
            "subsections": sub_stats,
        }, ensure_ascii=False)

    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"精简章节「{section_name}」时异常: {str(e)}",
        }, ensure_ascii=False)


def _get_section_names_from_markdown(markdown: str) -> list[str]:
    """从完整 markdown 中提取所有 # 一级章节名"""
    import re
    if not markdown:
        return []
    names = []
    for line in markdown.split("\n"):
        m = re.match(r'^#\s+(.+?)\s*$', line)
        if m:
            names.append(m.group(1).strip())
    return names


@tool
async def summarize_document(
    mode: str = "ratio",
    target: float = 0.5,
    doc_type: str = "",
) -> str:
    """对已生成的所有章节进行批量精简（每章每小节分别精简）。

    内部循环调用 summarize_section 处理每个章节。失败章节不影响其他章节。

    调用时机:
    - 用户说"精简整个文档"/"压缩整篇文档"时
    - 用户要求"把全文缩短到 XXXX 字"时

    Args:
        mode: 精简模式，"words"（字数模式）或 "ratio"（比例模式）
        target: 目标值。words 模式为总目标字数（按章节小节数比例分配）；
                ratio 模式为压缩比例（0.1~1.0）
        doc_type: 文档类型标识，为空时自动从上下文获取

    Returns:
        JSON格式结果，包含每个章节的精简状态汇总。
    """
    try:
        current_markdown = _current_generated_markdown.get()
        if not current_markdown:
            return json.dumps({
                "status": "error",
                "message": "当前没有已生成的文档内容。请先生成至少一个章节。",
            }, ensure_ascii=False)

        section_names = _get_section_names_from_markdown(current_markdown)
        if not section_names:
            return json.dumps({
                "status": "error",
                "message": "未找到任何章节。请先生成文档。",
            }, ensure_ascii=False)

        # 字数模式下，按章节数平均分配总字数预算
        section_targets = []
        if mode == "words":
            total_target = int(target)
            per_section = max(int(total_target / len(section_names)), 200)
            for name in section_names:
                section_targets.append((name, per_section))
        else:
            for name in section_names:
                section_targets.append((name, target))

        # 顺序处理各章节（避免并发过多导致API限流）
        results = []
        success_count = 0
        for name, t in section_targets:
            summary_result = await summarize_section.ainvoke({
                "section_name": name,
                "mode": mode,
                "target": t,
                "doc_type": doc_type,
            })
            try:
                data = json.loads(summary_result)
                if data.get("status") == "ok":
                    success_count += 1
                results.append({
                    "section_name": name,
                    "status": data.get("status", "error"),
                    "orig_chars": data.get("orig_total_chars", 0),
                    "new_chars": data.get("new_total_chars", 0),
                    "subsections_count": data.get("subsections_count", 0),
                    "success_count": data.get("success_count", 0),
                    "failed_count": data.get("failed_count", 0),
                    "message": data.get("message", "") if data.get("status") != "ok" else "",
                })
            except json.JSONDecodeError:
                results.append({
                    "section_name": name,
                    "status": "error",
                    "message": "工具返回非JSON",
                })

        return json.dumps({
            "status": "ok" if success_count > 0 else "error",
            "mode": mode,
            "target": target,
            "total_sections": len(section_names),
            "success_count": success_count,
            "failed_count": len(section_names) - success_count,
            "results": results,
        }, ensure_ascii=False)

    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"批量精简异常: {str(e)}",
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


# ── Tool 8b: outline_from_attachment ──

@tool
async def outline_from_attachment(
    file_id: str = "",
    doc_type: str = "design_development_plan",
    product_name: str = "贴敷式胰岛素泵",
) -> str:
    """基于上传附件的章节结构生成文档框架（附件优先路径）。

    当用户上传参考文档并希望按附件结构生成新文档时调用。
    输出与 design_outline 完全兼容的格式，后续可直接调用 write_chapter 逐章生成。

    与 design_outline 的区别:
    - design_outline 由 LLM 自主设计框架（基于标准法规）
    - outline_from_attachment 复用附件原有章节结构，LLM 仅补全小节和内容要点

    何时用: 用户上传了附件 + 明确选择"按附件结构生成"时。
    降级: 附件无可识别章节时返回 error，Agent 应回退到 design_outline。

    Args:
        file_id: 可选，指定要参照的附件 file_id。为空时自动取第一个有章节结构的附件。
        doc_type: 目标文档类型标识，用于调整小节以适配目标文档类型特有维度。
        product_name: 产品名称，用于生成 doc_title。

    Returns:
        JSON 字符串，格式与 design_outline 一致:
        {"doc_title": "...", "chapters": [{"id", "title", "description",
         "key_standards", "subsections": [{"title", "content_points"}]}]}
        失败时返回 {"status": "error", "message": "..."}
    """
    from app.services.minimax import _call_minimax_api_raw
    from app.services.doc_types import DOC_TYPE_LABELS

    doc_label = DOC_TYPE_LABELS.get(doc_type, doc_type)

    # Step 1: 从 _current_attachments 取目标附件
    attachments = _current_attachments.get()
    if not attachments:
        return json.dumps({
            "status": "error",
            "message": "当前没有上传附件。请先上传参考文档，或改用 design_outline 自主设计框架。",
        }, ensure_ascii=False)

    att = None
    if file_id:
        att = next((a for a in attachments if a.get("file_id") == file_id), None)
    if not att:
        # 取第一个有 full_text 的附件
        att = next((a for a in attachments if a.get("full_text")), None)
    if not att:
        return json.dumps({
            "status": "error",
            "message": "未找到可用的附件（所有附件均无文本内容）。请改用 design_outline。",
        }, ensure_ascii=False)

    filename = att.get("filename", "unknown")
    full_text = att.get("full_text", "")
    if not full_text:
        return json.dumps({
            "status": "error",
            "message": f"附件「{filename}」无文本内容（可能是空文件或提取失败）。请改用 design_outline。",
        }, ensure_ascii=False)

    # Step 2: 截取附件全文送给 LLM
    # 使用更大窗口（50000 字符，约覆盖 3-5 万字文档），
    # 避免短窗口（如 analyze_document_structure 的 12000）导致长文档章节识别不完整
    text_sample = full_text[:50000]
    if len(full_text) > 50000:
        print(f"[agent_tools] outline_from_attachment: full_text truncated "
              f"({len(full_text)} -> 50000 chars) for '{filename}'")

    # Step 3: 一步 LLM 调用 — 直接从附件全文识别章节结构并补全 subsections + content_points
    # 不再依赖 analyze_document_structure 的两步流程，避免中间格式转换损失和短窗口截断
    system_prompt = f"""你是医疗器械注册文档框架分析专家。请分析附件全文，识别其完整的章节结构，
并输出与 design_outline 兼容的框架JSON。

## 任务
1. 通读附件全文，识别所有章节标题及其层级关系
2. 按文档中的出现顺序，提取每一章（level=1 的标题）
3. 为每章补全 subsections（小节）和 content_points（内容要点）
4. 小节从附件原文中识别（如 1.1、1.2 等子标题），无明确子标题时根据内容归纳

## 关键约束
- **必须保留附件原始章节顺序与标题**，不得增删章节或重命名
- 必须识别出附件的**所有**章节，不得遗漏（即使章节在文档末尾）
- 每个章节至少2个小节，最多4个小节
- 每个小节至少2条 content_points，最多4条
- content_points 应从附件全文中提炼具体要点，不得编造
- 目标文档类型为「{doc_label}」（doc_type={doc_type}），description 中可注明与该文档类型的关联
- description 控制在1-2句话，概括本章主要内容
- key_standards 从全文推断引用的标准，无则留空数组

## 章节识别策略
1. 优先识别明确的章节编号（如"1."、"第一章"、"一、"等）
2. 识别 Markdown 标题（# ## ### ####）
3. 识别 Word 样式标题（如"1.1 目的"）
4. 如有目录（TOC），优先参照目录
5. 合并连续无标题段落为"概述"小节

## 输出格式（严格JSON，无其他文字）
{{
  "doc_title": "附件推断的文档标题",
  "chapters": [{{
    "id": 1,
    "title": "第X章 章节标题（保留附件原标题）",
    "description": "本章主要内容概括",
    "key_standards": ["GB XXXX", "ISO XXXX"],
    "subsections": [
      {{"title": "X.1 小节标题", "content_points": ["要点1", "要点2"]}}
    ]
  }}]
}}

只输出 JSON 对象，不要包含 markdown 代码块标记（```）或任何其他文字。"""

    user_prompt = f"""请分析以下附件全文，识别完整的章节结构并补全 subsections 和 content_points。

## 附件文件名
{filename}

## 附件全文
{text_sample}"""

    try:
        response = await asyncio.wait_for(
            asyncio.to_thread(
                _call_minimax_api_raw,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.2,
                max_tokens=16384,
            ),
            timeout=300.0,
        )
    except asyncio.TimeoutError:
        return json.dumps({
            "status": "error",
            "message": "附件框架识别超时（300秒）。请稍后重试，或改用 design_outline。",
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"附件框架识别失败: {str(e)}。请改用 design_outline。",
        }, ensure_ascii=False)

    if not response:
        return json.dumps({
            "status": "error",
            "message": "LLM 返回空结果，附件框架识别失败。请改用 design_outline。",
        }, ensure_ascii=False)

    # 复用 _extract_json 提取并清理 JSON
    outline_str = _extract_json(response)

    # 校验输出可解析
    try:
        parsed = json.loads(outline_str)
        if not parsed.get("chapters"):
            raise ValueError("输出缺少 chapters 字段")
    except (json.JSONDecodeError, ValueError) as e:
        return json.dumps({
            "status": "error",
            "message": f"附件框架识别输出格式无效: {str(e)}。请改用 design_outline。",
        }, ensure_ascii=False)

    print(f"[agent_tools] outline_from_attachment: "
          f"{len(parsed.get('chapters', []))} chapters from attachment '{att.get('filename', '?')}'")
    return outline_str


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
    generate_search_query,
    generate_section,
    revise_section,
    build_docx,
    design_outline,
    outline_from_attachment,
    write_chapter,
    update_outline,
    summarize_section,
    summarize_document,
]
