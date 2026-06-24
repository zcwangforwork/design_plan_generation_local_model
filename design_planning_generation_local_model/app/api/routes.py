"""
API Routes - 文档生成接口（异步任务模式）+ 附件上传接口
"""

import uuid
import threading
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel, Field
from app.services.generator import DocumentGenerator
from app.services.doc_types import DOC_TYPE_LABELS, DOC_CATEGORIES, SUPPORTED_UPLOAD_FORMATS, MAX_UPLOAD_SIZE_BYTES
from app.services.attachment_service import (
    validate_upload, submit_extract_task, get_extract_status
)
from app.services.conversation import conversation_manager
import io
import os
import time
import zipfile
from typing import Optional, List
from urllib.parse import quote

router = APIRouter()


class GenerateRequest(BaseModel):
    """文档生成请求"""
    doc_type: str = Field(..., description="文档类型")
    product_name: str = Field(..., description="产品名称")
    product_type: str = Field(default="", description="产品类型，如：有源医疗器械")
    product_params: str = Field("", description="产品参数详情")
    file_ids: Optional[List[str]] = Field(None, description="已入库附件的file_id列表")
    attachment_content: Optional[str] = Field(None, description="临时附件的提取文本内容")


# 内存中的任务存储
tasks = {}  # task_id -> {status, progress, message, file_bytes, filename, created_at}


def _run_generation(task_id: str, doc_type: str, product_name: str, product_type: str, product_params: str, file_ids: Optional[List[str]] = None, attachment_content: Optional[str] = None):
    """在后台线程中执行文档生成"""
    try:
        tasks[task_id]["status"] = "generating"
        tasks[task_id]["progress"] = 5
        tasks[task_id]["message"] = "正在生成文档结构大纲..."

        # 进度回调：根据阶段更新任务状态
        def on_progress(phase, current, total, message):
            if phase == "outline":
                progress = 10
            elif phase == "rag":
                # RAG阶段占 10-30%
                progress = 10 + int(20 * current / max(total, 1))
            elif phase == "generate":
                # 生成阶段占 30-95%
                progress = 30 + int(65 * current / max(total, 1))
            else:
                progress = 10
            tasks[task_id]["progress"] = min(progress, 95)
            tasks[task_id]["message"] = message

        generator = DocumentGenerator(progress_callback=on_progress)

        # 同步生成文档（在线程中运行，不阻塞事件循环）
        import asyncio
        loop = asyncio.new_event_loop()
        try:
            file_bytes = loop.run_until_complete(
                generator.generate(
                    doc_type=doc_type,
                    product_name=product_name,
                    product_type=product_type,
                    product_params=product_params,
                    file_ids=file_ids,
                    attachment_content=attachment_content
                )
            )
        finally:
            loop.close()

        # 生成文件名
        label = DOC_TYPE_LABELS.get(doc_type, doc_type)
        filename = f"{product_name}_{label}.docx"

        # 创建会话，保存生成的内容供后续修订
        session_id = conversation_manager.create_session(
            doc_type=doc_type,
            product_name=product_name,
            product_type=product_type,
            product_params=product_params,
            doc_content=generator.last_generated_content
        )

        tasks[task_id]["status"] = "completed"
        tasks[task_id]["progress"] = 100
        tasks[task_id]["message"] = "文档生成完成！"
        tasks[task_id]["file_bytes"] = file_bytes
        tasks[task_id]["filename"] = filename
        tasks[task_id]["search_log"] = generator.search_log
        tasks[task_id]["timing_log"] = generator.timing_log
        tasks[task_id]["session_id"] = session_id
        tasks[task_id]["doc_content"] = generator.last_generated_content

    except Exception as e:
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["progress"] = 0
        tasks[task_id]["message"] = f"生成失败: {str(e)}"


@router.post("/generate")
async def generate_document(request: GenerateRequest):
    """
    提交文档生成任务（异步）

    立即返回任务ID，文档在后台生成，前端通过 /api/status/{task_id} 轮询进度
    """
    # 验证必填字段
    if not request.product_name.strip():
        raise HTTPException(status_code=400, detail="产品名称不能为空")

    # 创建任务
    task_id = str(uuid.uuid4())[:8]
    tasks[task_id] = {
        "status": "pending",
        "progress": 0,
        "message": "任务已创建，等待生成...",
        "file_bytes": None,
        "filename": "",
        "created_at": time.time()
    }

    # 启动后台线程
    thread = threading.Thread(
        target=_run_generation,
        args=(task_id, request.doc_type, request.product_name, request.product_type, request.product_params, request.file_ids, request.attachment_content),
        daemon=True
    )
    thread.start()

    return {"task_id": task_id, "status": "pending", "message": "任务已提交"}


@router.get("/status/{task_id}")
async def get_task_status(task_id: str):
    """查询任务状态"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="任务不存在")

    task = tasks[task_id]
    result = {
        "task_id": task_id,
        "status": task["status"],
        "progress": task["progress"],
        "message": task["message"],
        "filename": task.get("filename", ""),
        "search_log": task.get("search_log", []),
        "timing_log": task.get("timing_log", {})
    }
    # 生成完成时返回 session_id 供前端进入修订模式
    if task["status"] == "completed" and task.get("session_id"):
        result["session_id"] = task["session_id"]
    return result


@router.get("/download/{task_id}")
async def download_document(task_id: str):
    """下载生成的文档"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="任务不存在")

    task = tasks[task_id]

    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail="文档尚未生成完成")

    if not task["file_bytes"]:
        raise HTTPException(status_code=500, detail="文档数据为空")

    encoded_filename = quote(task["filename"])

    return StreamingResponse(
        io.BytesIO(task["file_bytes"]),
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={
            "Content-Disposition": f"attachment; filename*=UTF-8''{encoded_filename}"
        }
    )


# ==================== 数字员工交互式修订接口 ====================

class ReviseRequest(BaseModel):
    """文档修订请求"""
    session_id: str = Field(..., description="会话ID（从首次生成返回）")
    feedback: str = Field(..., description="用户修改意见")


@router.post("/revise")
async def revise_document(request: ReviseRequest):
    """
    基于用户反馈修订文档（数字员工交互模式）

    提交修改意见，返回修订后的文档
    """
    session = conversation_manager.get_session(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在或已过期")

    if not request.feedback.strip():
        raise HTTPException(status_code=400, detail="反馈意见不能为空")

    # 记录反馈并创建版本快照
    version_label = conversation_manager.add_feedback(request.session_id, request.feedback)

    # 在后台执行修订
    task_id = str(uuid.uuid4())[:8]
    tasks[task_id] = {
        "status": "revising",
        "progress": 0,
        "message": "正在根据反馈修订文档...",
        "file_bytes": None,
        "filename": "",
        "session_id": request.session_id,
        "created_at": time.time()
    }

    thread = threading.Thread(
        target=_run_revision,
        args=(task_id, request.session_id, request.feedback),
        daemon=True
    )
    thread.start()

    return {"task_id": task_id, "status": "revising", "message": "修订任务已提交", "version": version_label}


def _run_revision(task_id: str, session_id: str, feedback: str):
    """在后台线程中执行文档修订"""
    try:
        tasks[task_id]["status"] = "revising"
        tasks[task_id]["progress"] = 20
        tasks[task_id]["message"] = "正在根据反馈修订文档..."

        session = conversation_manager.get_session(session_id)
        if not session:
            tasks[task_id]["status"] = "failed"
            tasks[task_id]["message"] = "会话不存在"
            return

        generator = DocumentGenerator()

        import asyncio
        loop = asyncio.new_event_loop()
        try:
            file_bytes = loop.run_until_complete(
                generator.revise(
                    current_content=session.current_content,
                    feedback=feedback,
                    doc_type=session.doc_type,
                    product_name=session.product_name,
                    product_type=session.product_type,
                    product_params=session.product_params
                )
            )
        finally:
            loop.close()

        # 更新会话中的文档内容
        conversation_manager.update_content(session_id, generator.last_generated_content)

        # 将差异数据写入版本快照，供前端展示修订对比
        if generator.last_diff_data:
            conversation_manager.set_version_diff(session_id, generator.last_diff_data)

        label = DOC_TYPE_LABELS.get(session.doc_type, session.doc_type)
        filename = f"{session.product_name}_{label}_修订版.docx"

        tasks[task_id]["status"] = "completed"
        tasks[task_id]["progress"] = 100
        tasks[task_id]["message"] = "文档修订完成！"
        tasks[task_id]["file_bytes"] = file_bytes
        tasks[task_id]["filename"] = filename

    except Exception as e:
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["progress"] = 0
        tasks[task_id]["message"] = f"修订失败: {str(e)}"


@router.get("/session/{session_id}")
async def get_session(session_id: str):
    """获取数字员工会话状态"""
    session = conversation_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在或已过期")
    return session.to_dict()


@router.get("/session/{session_id}/download")
async def download_session_document(session_id: str):
    """
    下载会话当前的文档版本

    根据会话中的最新内容重新生成 Word 文件并返回
    """
    session = conversation_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在或已过期")

    from app.services.template import TemplateService
    template_service = TemplateService()

    doc = template_service.load_template(session.doc_type)
    doc = template_service.fill_template(
        doc=doc,
        content=session.current_content,
        product_name=session.product_name,
        doc_type=session.doc_type
    )
    file_bytes = template_service.document_to_bytes(doc)

    label = DOC_TYPE_LABELS.get(session.doc_type, session.doc_type)
    v = session.version_count
    filename = f"{session.product_name}_{label}_v{v}.docx"
    encoded_filename = quote(filename)

    return StreamingResponse(
        io.BytesIO(file_bytes),
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={
            "Content-Disposition": f"attachment; filename*=UTF-8''{encoded_filename}"
        }
    )


@router.get("/doc-types")
async def get_doc_types():
    """获取支持的文档类型列表 — 贴敷式胰岛素泵全生命周期94+种文档"""
    types = []
    for cat_key, cat_info in DOC_CATEGORIES.items():
        for doc_type in cat_info["types"]:
            label = DOC_TYPE_LABELS.get(doc_type, doc_type)
            types.append({
                "value": doc_type,
                "label": label,
                "category": cat_info["name"],
                "category_key": cat_key,
                "description": cat_info["description"]
            })
    return {
        "types": types,
        "categories": [
            {
                "key": cat_key,
                "name": cat_info["name"],
                "description": cat_info["description"],
                "icon": cat_info["icon"],
                "count": len(cat_info["types"])
            }
            for cat_key, cat_info in DOC_CATEGORIES.items()
        ]
    }


# ==================== 附件上传接口 ====================

@router.post("/upload")
async def upload_attachment(
    file: UploadFile = File(..., description="附件文件 (.docx/.pdf/.txt)"),
    persist: bool = Form(False, description="是否存入知识库供后续复用")
):
    """
    上传附件文档，提交后台提取任务

    返回 file_id，前端通过 GET /api/extract-status/{file_id} 轮询提取进度
    """
    # 验证格式和大小
    file_content = await file.read()
    is_valid, error_msg = validate_upload(file.filename, len(file_content))
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_msg)

    # 提交后台提取任务
    task_id = submit_extract_task(
        file_content=file_content,
        filename=file.filename,
        persist=persist,
        doc_type="unknown"
    )

    return {
        "file_id": task_id,
        "filename": file.filename,
        "status": "pending",
        "message": "文件已接收，正在后台提取文本..."
    }


@router.get("/extract-status/{file_id}")
async def get_extract_task_status(file_id: str):
    """查询附件提取任务状态"""
    status = get_extract_status(file_id)
    if status is None:
        raise HTTPException(status_code=404, detail="任务不存在或已过期")
    return status


@router.get("/debug/env")
async def debug_env():
    """调试端点 - 检查环境变量"""
    api_key = os.getenv("MINIMAX_API_KEY", "")
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11435")
    ollama_model = os.getenv("OLLAMA_MODEL", "qwen3.5:122b")
    ollama_embed = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    return {
        "api_key_set": bool(api_key),
        "ollama_base_url": ollama_url,
        "ollama_model": ollama_model,
        "ollama_embed_model": ollama_embed,
        "env_file_loaded": os.path.exists(".env"),
        "cwd": os.getcwd()
    }


@router.get("/debug/test-api")
async def test_api():
    """测试 MiniMax API 调用"""
    from app.services.minimax import MiniMaxService
    service = MiniMaxService()
    try:
        result = service.generate_content(
            doc_type="risk_management_report",
            product_name="测试产品",
            product_type="有源医疗器械",
            product_params="测试参数"
        )
        return {"success": True, "result": result[:500] if result else "empty"}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ═══════════════════════════════════════════════════════════════
# Agent API — 设计策划文档写作Agent (LangGraph)
# ═══════════════════════════════════════════════════════════════

from fastapi.responses import StreamingResponse
import json


@router.post("/agent/projects/{project_id}/messages")
async def agent_send_message(
    project_id: str,
    message: str = Form(...),
):
    """发送消息到Agent，返回SSE事件流

    事件类型:
    - token: LLM逐token输出 → 前端打字机效果
    - tool_start: 工具调用开始 → 前端展示图标
    - tool_end: 工具调用结束 → 前端展示结果预览
    - waiting_approval: HITL暂停 → 前端展示确认按钮
    - done: 流结束
    """
    from app.services.agent_engine import stream_agent_events
    from app.services.agent_state import create_initial_state

    async def event_stream():
        try:
            # 首次消息: 传入初始状态
            initial_state = create_initial_state()
            async for event in stream_agent_events(
                user_message=message,
                thread_id=project_id,
                initial_state=initial_state,
            ):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/agent/projects/{project_id}/resume")
async def agent_resume(
    project_id: str,
    decision: str = Form(...),
):
    """HITL暂停后恢复Agent执行

    Args:
        project_id: 项目ID
        decision: 用户决定
            - "approve" — 确认生成
            - "reject" — 跳过不生成
            - "edit:修改后的指令" — 用修改后的指令重新生成
    """
    from app.services.agent_engine import resume_agent

    async def event_stream():
        try:
            async for event in resume_agent(
                thread_id=project_id,
                decision=decision,
            ):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/agent/projects/{project_id}/auto-generate")
async def agent_auto_generate(
    project_id: str,
    product_name: str = Form(...),
    product_classification: str = Form("III类有源医疗器械"),
    product_intended_use: str = Form("用于糖尿病患者持续皮下输注胰岛素"),
    doc_type: str = Form("design_development_plan", description="设计策划阶段文档类型"),
):
    """一键自动生成: Agent按SOP全流程自动执行，无需用户逐步确认

    Agent会依次完成: 产品画像 → 标准/资料检索 → 策划内容采集 → 章节生成 → 导出.docx
    全程通过SSE推送进度，前端实时展示。

    支持 doc_type 参数指定要生成的文档类型，默认为 design_development_plan（项目开发计划书）。
    设计策划阶段支持的文档类型参见 DOC_CATEGORIES['design_planning']['types']。
    """
    from app.services.agent_engine import stream_agent_events
    from app.services.agent_state import create_initial_state
    from app.services.doc_types import DOC_TYPE_LABELS

    doc_label = DOC_TYPE_LABELS.get(doc_type, doc_type)

    # 构建一键生成指令 — 根据文档类型定制
    auto_prompt = f"""请按SOP流程自动完成《{doc_label}》文档的编写。全程无需等待用户确认，直接执行所有步骤。

产品信息:
- 产品名称: {product_name}
- 医疗器械分类: {product_classification}
- 预期用途: {product_intended_use}
- 目标文档类型: {doc_label}

请严格按以下顺序自动执行:

步骤1 — 记录以上产品画像信息
步骤2 — 调用 search_kb 检索贴敷式胰岛素泵与{doc_label}相关的策划依据和参考文档，列出后自动确认
步骤3 — 根据{doc_label}的文档特点，搜集所需策划内容并自动确认
步骤4 — 逐章调用 generate_section 生成{doc_label}的所有章节内容
步骤5 — 全部生成后调用 build_docx 导出Word文档

全程自动执行，不要询问用户，不要等待确认。"""

    async def event_stream():
        try:
            initial_state = create_initial_state()
            # 预填产品信息，设置自动模式
            initial_state["product_name"] = product_name
            initial_state["product_classification"] = product_classification
            initial_state["product_intended_use"] = product_intended_use
            initial_state["product_status"] = "confirmed"
            initial_state["auto_mode"] = True
            initial_state["doc_type"] = doc_type

            async for event in stream_agent_events(
                user_message=auto_prompt,
                thread_id=project_id,
                initial_state=initial_state,
            ):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/agent/projects/{project_id}/batch-generate")
async def agent_batch_generate(
    project_id: str,
    product_name: str = Form(...),
    product_classification: str = Form("III类有源医疗器械"),
    product_intended_use: str = Form("用于糖尿病患者持续皮下输注胰岛素"),
):
    """批量生成设计策划阶段全部文档（使用经典生成API）

    按DHF清单顺序依次生成设计策划阶段所有文档，
    每完成一个文档推送一条SSE事件，前端可实时展示进度。
    所有文档生成完毕后打包为ZIP下载。
    """
    from app.services.doc_types import DOC_CATEGORIES, DOC_TYPE_LABELS

    # 获取设计策划阶段全部文档类型
    planning_types = DOC_CATEGORIES["design_planning"]["types"]

    async def event_stream():
        generated_files = []  # [(filename, bytes), ...]
        total = len(planning_types)

        yield f"data: {json.dumps({'type': 'batch_start', 'total': total, 'docs': planning_types}, ensure_ascii=False)}\n\n"

        for idx, doc_type in enumerate(planning_types):
            doc_label = DOC_TYPE_LABELS.get(doc_type, doc_type)
            yield f"data: {json.dumps({'type': 'doc_start', 'index': idx + 1, 'total': total, 'doc_type': doc_type, 'label': doc_label}, ensure_ascii=False)}\n\n"

            try:
                generator = DocumentGenerator()
                import asyncio
                loop = asyncio.new_event_loop()
                try:
                    file_bytes = loop.run_until_complete(
                        generator.generate(
                            doc_type=doc_type,
                            product_name=product_name,
                            product_type=product_classification,
                            product_params=f"预期用途: {product_intended_use}",
                        )
                    )
                finally:
                    loop.close()

                filename = f"{product_name}_{doc_label}.docx"
                generated_files.append((filename, file_bytes))

                yield f"data: {json.dumps({'type': 'doc_done', 'index': idx + 1, 'total': total, 'doc_type': doc_type, 'label': doc_label, 'filename': filename, 'size_kb': len(file_bytes) // 1024}, ensure_ascii=False)}\n\n"

            except Exception as e:
                yield f"data: {json.dumps({'type': 'doc_error', 'index': idx + 1, 'total': total, 'doc_type': doc_type, 'label': doc_label, 'error': str(e)}, ensure_ascii=False)}\n\n"

        # 打包为ZIP
        if generated_files:
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                for fname, fbytes in generated_files:
                    zf.writestr(fname, fbytes)
            zip_buffer.seek(0)
            zip_bytes = zip_buffer.getvalue()

            # 存储ZIP供下载（使用project_id作为key）
            _batch_results[project_id] = {
                "zip_bytes": zip_bytes,
                "filename": f"{product_name}_设计策划阶段文档集.zip",
                "files": [f[0] for f in generated_files],
            }

            yield f"data: {json.dumps({'type': 'batch_done', 'total': total, 'completed': len(generated_files), 'download_id': project_id}, ensure_ascii=False)}\n\n"
        else:
            yield f"data: {json.dumps({'type': 'batch_error', 'message': '所有文档生成失败'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# 批量生成结果存储
_batch_results = {}


@router.get("/agent/batch-download/{project_id}")
async def agent_batch_download(project_id: str):
    """下载批量生成的ZIP文件"""
    if project_id not in _batch_results:
        raise HTTPException(status_code=404, detail="批量生成结果不存在或已过期")

    result = _batch_results[project_id]
    encoded_filename = quote(result["filename"])

    return StreamingResponse(
        io.BytesIO(result["zip_bytes"]),
        media_type="application/zip",
        headers={
            "Content-Disposition": f"attachment; filename*=UTF-8''{encoded_filename}"
        }
    )


@router.get("/agent/projects/{project_id}/state")
async def agent_get_state(project_id: str):
    """获取Agent当前状态快照 (供前端进度面板使用)

    返回PRD Section 4.3格式的状态JSON
    """
    from app.services.agent_engine import get_agent_state

    try:
        state = await get_agent_state(project_id)
        return {"success": True, "state": state}
    except Exception as e:
        # Agent可能尚未初始化或thread不存在
        from app.services.agent_state import create_initial_state, build_state_snapshot
        state = build_state_snapshot(create_initial_state())
        return {"success": True, "state": state, "note": f"使用默认状态 (Agent: {str(e)})"}


@router.get("/agent/projects/{project_id}/document")
async def agent_get_document(project_id: str):
    """获取Agent已生成文档的组装内容（供审阅页面使用）

    从Agent状态中读取 generated_sections 并组装为结构化JSON，
    包含产品信息、标准清单、各章节内容。
    """
    from app.services.agent_engine import get_agent

    agent = get_agent()
    config = {"configurable": {"thread_id": project_id}}

    try:
        state = await agent.aget_state(config)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"无法读取Agent状态: {str(e)}")

    if not state or not state.values:
        raise HTTPException(status_code=404, detail="项目不存在或尚未开始")

    values = state.values
    generated = values.get("generated_sections", {}) or {}
    product_name = values.get("product_name", "") or "未命名产品"

    # 章节排序: 先放文档信息类章节，再放技术领域章节
    priority_order = [
        "封面", "文档信息", "产品画像", "标准适用性清单",
        "性能要求", "安全要求", "软件要求", "硬件要求",
        "EMC要求", "电磁兼容性", "生物相容性", "包装要求",
        "标签要求", "网络安全", "可用性",
    ]
    ordered_sections = []
    remaining = dict(generated)
    for key in priority_order:
        if key in remaining:
            ordered_sections.append({"title": key, "content": remaining.pop(key)})
    # 剩余未排序章节
    for key in sorted(remaining.keys()):
        ordered_sections.append({"title": key, "content": remaining[key]})

    return {
        "success": True,
        "project_id": project_id,
        "product_name": product_name,
        "product": {
            "name": values.get("product_name"),
            "classification": values.get("product_classification"),
            "intended_use": values.get("product_intended_use"),
            "status": values.get("product_status", "not_started"),
        },
        "standards": {
            "confirmed": values.get("confirmed_standards", []),
            "candidate": values.get("candidate_standards", []),
            "status": values.get("standards_status", "not_started"),
        },
        "document_status": values.get("document_status", "not_started"),
        "sections": ordered_sections,
        "unresolved_items": values.get("unresolved_items", []),
    }


@router.get("/agent/projects/{project_id}/download")
async def agent_download_document(project_id: str):
    """直接从Agent状态组装文档并下载.docx（无需Agent参与）

    读取 generated_sections，组装为完整Markdown，
    通过TemplateService构建.docx并返回。
    """
    from app.services.agent_engine import get_agent
    from app.services.template import TemplateService

    agent = get_agent()
    config = {"configurable": {"thread_id": project_id}}

    try:
        state = await agent.aget_state(config)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"无法读取Agent状态: {str(e)}")

    if not state or not state.values:
        raise HTTPException(status_code=404, detail="项目不存在或尚未开始")

    values = state.values
    generated = values.get("generated_sections", {}) or {}
    product_name = values.get("product_name", "") or "贴敷式胰岛素泵"

    if not generated:
        raise HTTPException(status_code=400, detail="尚未生成任何章节，请先在Agent对话中生成文档")

    # 组装Markdown文档
    priority_order = [
        "封面", "文档信息", "产品画像", "标准适用性清单",
        "性能要求", "安全要求", "软件要求", "硬件要求",
        "EMC要求", "电磁兼容性", "生物相容性", "包装要求",
        "标签要求", "网络安全", "可用性",
    ]
    remaining = dict(generated)
    ordered_parts = []
    for key in priority_order:
        if key in remaining:
            ordered_parts.append(remaining.pop(key))
    for key in sorted(remaining.keys()):
        ordered_parts.append(remaining[key])

    full_markdown = "\n\n".join(ordered_parts)

    # 读取实际的 doc_type，非硬编码
    doc_type = values.get("doc_type", "design_development_plan")
    doc_label = DOC_TYPE_LABELS.get(doc_type, "设计策划文档")

    # 构建.docx
    template_service = TemplateService()
    doc = template_service.load_template(doc_type)
    doc = template_service.fill_template(
        doc=doc,
        content=full_markdown,
        product_name=product_name,
        doc_type=doc_type,
    )
    file_bytes = template_service.document_to_bytes(doc)

    filename = f"{product_name}_{doc_label}.docx"
    from urllib.parse import quote
    encoded_filename = quote(filename)

    return StreamingResponse(
        io.BytesIO(file_bytes),
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={
            "Content-Disposition": f"attachment; filename*=UTF-8''{encoded_filename}"
        }
    )


@router.get("/agent/download/{download_id}")
async def agent_download_docx(download_id: str):
    """下载Agent通过 build_docx 工具生成的Word文档"""
    from app.services.agent_tools import _get_docx

    docx_data = _get_docx(download_id)
    if not docx_data:
        raise HTTPException(status_code=404, detail="下载链接不存在或已过期，请重新生成文档")

    from urllib.parse import quote
    encoded_filename = quote(docx_data["filename"])

    return StreamingResponse(
        io.BytesIO(docx_data["bytes"]),
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={
            "Content-Disposition": f"attachment; filename*=UTF-8''{encoded_filename}"
        }
    )


# ═══════════════════════════════════════════════════════════════
# Agent 附件上传接口
# ═══════════════════════════════════════════════════════════════

@router.post("/agent/upload/{project_id}")
async def agent_upload_attachment(
    project_id: str,
    file: UploadFile = File(..., description="附件文件 (.pdf/.docx/.doc/.txt/.xlsx)"),
):
    """Agent模式下上传附件

    上传文件后自动提取文本，将附件信息（含全文）存入Agent状态，
    供Agent在后续对话中通过 search_attachment 工具检索使用。

    同时将文本写入向量库（uploads集合），支持混合检索。
    """
    from app.services.agent_engine import get_agent
    from app.services.agent_state import create_initial_state

    # 验证文件
    file_content = await file.read()
    is_valid, error_msg = validate_upload(file.filename, len(file_content))
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_msg)

    # 提交提取任务（persist=True 写入向量库）
    task_id = submit_extract_task(
        file_content=file_content,
        filename=file.filename,
        persist=True,
        doc_type="agent_attachment",
    )

    # 轮询等待提取完成（最长30秒）
    import asyncio
    for _ in range(60):
        status = get_extract_status(task_id)
        if status is None:
            raise HTTPException(status_code=500, detail="提取任务丢失")
        if status["status"] == "completed":
            break
        if status["status"] == "failed":
            raise HTTPException(status_code=500, detail=f"文件提取失败: {status.get('message', '未知错误')}")
        await asyncio.sleep(0.5)
    else:
        raise HTTPException(status_code=500, detail="文件提取超时，请重试")

    # 获取提取后的完整文本（从 extract_tasks 内存中读取，提取完成后需重新获取）
    from app.services.attachment_service import extract_tasks
    task = extract_tasks.get(task_id, {})
    full_text = task.get("full_text", "")
    preview = task.get("preview", "")
    char_count = task.get("char_count", 0)

    # 如果内存中的 full_text 已被清理，尝试从向量库重建
    if not full_text and char_count > 0:
        try:
            from app.services.rag.vector_store import VectorStore
            vs = VectorStore(collection_name="uploads")
            results = vs.collection.get(
                where={"file_id": task_id},
                include=["documents"]
            )
            if results and results.get("documents"):
                full_text = "\n\n".join(results["documents"])
                preview = full_text[:500] + ("..." if len(full_text) > 500 else "")
        except Exception:
            pass

    # 读取当前Agent状态，追加附件
    agent = get_agent()
    config = {"configurable": {"thread_id": project_id}}

    try:
        current_state = await agent.aget_state(config)
        state_values = dict(current_state.values) if current_state and current_state.values else dict(create_initial_state())
    except Exception:
        state_values = dict(create_initial_state())

    attachments = list(state_values.get("attachments", []) or [])
    # 限制full_text长度防止状态过大（向量库中仍有完整内容）
    MAX_TEXT_IN_STATE = 50000
    stored_text = full_text
    if full_text and len(full_text) > MAX_TEXT_IN_STATE:
        stored_text = full_text[:MAX_TEXT_IN_STATE] + f"\n\n... (文件共{char_count}字符，仅存储前{MAX_TEXT_IN_STATE}字符。完整内容可通过search_attachment工具从向量库检索)"
    attachments.append({
        "file_id": task_id,
        "filename": file.filename,
        "char_count": char_count,
        "preview": preview,
        "full_text": stored_text,
        "status": "completed",
    })

    # 更新Agent状态
    await agent.aupdate_state(config, {"attachments": attachments})

    return {
        "success": True,
        "file_id": task_id,
        "filename": file.filename,
        "char_count": char_count,
        "preview": preview,
        "message": f"文件「{file.filename}」已上传并提取完成 ({char_count} 字符)。Agent现在可以在对话中检索此文件内容。",
    }


@router.get("/agent/projects/{project_id}/attachments")
async def agent_list_attachments(project_id: str):
    """获取Agent项目已上传的附件列表"""
    from app.services.agent_engine import get_agent
    from app.services.agent_state import create_initial_state, build_state_snapshot

    agent = get_agent()
    config = {"configurable": {"thread_id": project_id}}

    try:
        current_state = await agent.aget_state(config)
        state_values = current_state.values if current_state and current_state.values else {}
    except Exception:
        state_values = {}

    attachments = state_values.get("attachments", []) or []

    return {
        "success": True,
        "project_id": project_id,
        "attachments": [
            {
                "file_id": a.get("file_id"),
                "filename": a.get("filename"),
                "char_count": a.get("char_count", 0),
                "preview": a.get("preview", ""),
                "status": a.get("status", "unknown"),
            }
            for a in attachments
        ],
    }


@router.delete("/agent/projects/{project_id}/attachments/{file_id}")
async def agent_delete_attachment(project_id: str, file_id: str):
    """删除Agent项目中的指定附件"""
    from app.services.agent_engine import get_agent
    from app.services.agent_state import create_initial_state

    agent = get_agent()
    config = {"configurable": {"thread_id": project_id}}

    try:
        current_state = await agent.aget_state(config)
        state_values = dict(current_state.values) if current_state and current_state.values else dict(create_initial_state())
    except Exception:
        raise HTTPException(status_code=404, detail="项目不存在")

    attachments = list(state_values.get("attachments", []) or [])
    original_count = len(attachments)
    attachments = [a for a in attachments if a.get("file_id") != file_id]

    if len(attachments) == original_count:
        raise HTTPException(status_code=404, detail="附件不存在")

    await agent.aupdate_state(config, {"attachments": attachments})

    return {
        "success": True,
        "message": "附件已删除",
        "remaining": len(attachments),
    }
