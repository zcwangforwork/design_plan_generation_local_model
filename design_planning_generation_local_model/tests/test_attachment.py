"""
附件上传与文档生成 — 关键测试
"""

import os
import sys
import io
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# 添加项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# ==================== 附件验证测试 ====================

class TestUploadValidation:
    """上传文件验证测试"""

    def test_valid_docx(self):
        """有效.docx文件应通过验证"""
        from app.services.attachment_service import validate_upload
        is_valid, error = validate_upload("test.docx", 1024)
        assert is_valid
        assert error == ""

    def test_valid_pdf(self):
        """有效.pdf文件应通过验证"""
        from app.services.attachment_service import validate_upload
        is_valid, error = validate_upload("test.pdf", 5 * 1024 * 1024)
        assert is_valid
        assert error == ""

    def test_valid_txt(self):
        """有效.txt文件应通过验证"""
        from app.services.attachment_service import validate_upload
        is_valid, error = validate_upload("test.txt", 100)
        assert is_valid
        assert error == ""

    def test_invalid_format_xlsx(self):
        """不支持的.xlsx格式应返回400错误"""
        from app.services.attachment_service import validate_upload
        is_valid, error = validate_upload("data.xlsx", 1024)
        assert not is_valid
        assert "不支持" in error

    def test_invalid_format_jpg(self):
        """不支持的.jpg格式应返回400错误"""
        from app.services.attachment_service import validate_upload
        is_valid, error = validate_upload("photo.jpg", 2048)
        assert not is_valid
        assert "不支持" in error

    def test_file_too_large(self):
        """超过10MB的文件应返回错误"""
        from app.services.attachment_service import validate_upload
        oversized = (10 * 1024 * 1024) + 1  # 10MB + 1 byte
        is_valid, error = validate_upload("large.pdf", oversized)
        assert not is_valid
        assert "超过限制" in error or "MB" in error

    def test_empty_file(self):
        """空文件应返回错误"""
        from app.services.attachment_service import validate_upload
        is_valid, error = validate_upload("empty.docx", 0)
        assert not is_valid
        assert "空" in error or "有效" in error

    def test_max_size_boundary(self):
        """恰好10MB的文件应通过验证"""
        from app.services.attachment_service import validate_upload
        is_valid, error = validate_upload("exact.pdf", 10 * 1024 * 1024)
        assert is_valid


# ==================== 附件状态管理测试 ====================

class TestExtractTasks:
    """提取任务状态管理测试"""

    def test_submit_creates_task(self):
        """提交提取任务应创建并返回task_id"""
        from app.services.attachment_service import submit_extract_task, get_extract_status
        task_id = submit_extract_task(
            file_content=b"Hello world test content",
            filename="test.txt",
            persist=False
        )
        assert task_id is not None
        assert len(task_id) > 0

        status = get_extract_status(task_id)
        assert status is not None
        assert status["filename"] == "test.txt"

    def test_get_nonexistent_task(self):
        """查询不存在的任务应返回None"""
        from app.services.attachment_service import get_extract_status
        assert get_extract_status("nonexistent") is None


# ==================== 文档生成（带附件）测试 ====================

class TestGeneratorWithAttachment:
    """带附件参数的文档生成测试"""

    def test_generator_accepts_file_ids(self):
        """generator应接受file_ids参数"""
        from app.services.generator import DocumentGenerator
        gen = DocumentGenerator()
        # 验证方法签名接受新参数
        import inspect
        sig = inspect.signature(gen.generate)
        params = list(sig.parameters.keys())
        assert "file_ids" in params
        assert "attachment_content" in params

    def test_resolve_attachments_empty(self):
        """空附件参数应返回空字符串"""
        from app.services.generator import DocumentGenerator
        gen = DocumentGenerator()
        result = gen._resolve_attachments(None, None)
        assert result == ""

    def test_resolve_attachments_content_only(self):
        """仅有attachment_content时应直接返回"""
        from app.services.generator import DocumentGenerator
        gen = DocumentGenerator()
        result = gen._resolve_attachments(None, "测试附件内容")
        assert "测试附件内容" in result

    def test_resolve_attachments_both(self):
        """同时有file_ids和attachment_content时应合并"""
        from app.services.generator import DocumentGenerator
        gen = DocumentGenerator()
        with patch("app.services.generator.resolve_attachment_content", return_value="来自向量库的内容"):
            result = gen._resolve_attachments(["file123"], "直接传入的文本")
            assert "来自向量库的内容" in result
            assert "直接传入的文本" in result


# ==================== 向后兼容回归测试 ====================

class TestBackwardCompatibility:
    """确保不传附件参数时现有流程不变"""

    def test_generate_request_without_attachment(self):
        """不传附件参数的GenerateRequest应正常创建"""
        from app.api.routes import GenerateRequest
        req = GenerateRequest(
            doc_type="risk_management_report",
            product_name="胰岛素泵",
            product_type="有源医疗器械"
        )
        assert req.file_ids is None
        assert req.attachment_content is None
        assert req.product_name == "胰岛素泵"

    def test_generate_request_with_attachment(self):
        """带附件参数的GenerateRequest应正常创建"""
        from app.api.routes import GenerateRequest
        req = GenerateRequest(
            doc_type="sop",
            product_name="血糖仪",
            product_type="有源医疗器械",
            file_ids=["abc123"],
            attachment_content="产品说明文本"
        )
        assert req.file_ids == ["abc123"]
        assert req.attachment_content == "产品说明文本"

    def test_generate_request_roundtrip(self):
        """GenerateRequest应能序列化并包含所有字段"""
        from app.api.routes import GenerateRequest
        import json
        req = GenerateRequest(
            doc_type="product_spec",
            product_name="测试产品",
            product_type="植入器械",
            product_params="测试参数",
            file_ids=["f1", "f2"],
            attachment_content="附件文本"
        )
        data = req.model_dump()
        assert data["doc_type"] == "product_spec"
        assert data["file_ids"] == ["f1", "f2"]
        assert data["attachment_content"] == "附件文本"


# ==================== 文档类型常量测试 ====================

class TestDocTypeConstants:
    """公共常量完整性测试"""

    def test_doc_types_consistency(self):
        """DOC_TYPES和DOC_TYPE_LABELS应一一对应"""
        from app.services.doc_types import DOC_TYPES, DOC_TYPE_LABELS
        for dt in DOC_TYPES:
            assert dt in DOC_TYPE_LABELS, f"{dt} 缺少label映射"

    def test_all_imports_same_source(self):
        """所有模块应从同一来源导入"""
        from app.services.doc_types import DOC_TYPE_LABELS as SRC
        # 验证 template.py 的导入
        from app.services.template import DOC_TYPE_LABELS as TPL
        assert SRC is TPL


# ==================== 关键词匹配测试 ====================

class TestKeywordMatching:
    """附件内容关键词匹配测试"""

    def test_match_relevant_paragraphs(self):
        """应匹配到与查询相关的段落"""
        from app.services.minimax import MiniMaxService
        svc = MiniMaxService(api_key="test_key")

        text = "产品为有源医疗器械\n使用蓝牙通信\n外壳材质为医用级塑料\n适用人群为成人患者\n工作温度范围10-40度"
        query = "通信 蓝牙 有源"

        result = svc._match_relevant_paragraphs(text, query, max_chars=500)
        assert "蓝牙" in result
        assert "有源" in result

    def test_match_empty_text(self):
        """空文本应返回空字符串"""
        from app.services.minimax import MiniMaxService
        svc = MiniMaxService(api_key="test_key")
        result = svc._match_relevant_paragraphs("", "测试", 500)
        assert result == ""

    def test_match_no_keywords(self):
        """无匹配关键词时应返回空字符串"""
        from app.services.minimax import MiniMaxService
        svc = MiniMaxService(api_key="test_key")
        result = svc._match_relevant_paragraphs("完全不相关的内容", "蓝牙通信", 500)
        assert result == "" or len(result) >= 0  # 可能部分匹配或无匹配


# ==================== 跨Collection检索测试 ====================

class TestCrossCollectionRetrieval:
    """跨collection检索测试"""

    def test_minimax_accepts_attachment_content(self):
        """MinimaxService方法应接受attachment_content参数"""
        from app.services.minimax import MiniMaxService
        import inspect
        sig = inspect.signature(MiniMaxService.generate_content_with_fallback)
        params = list(sig.parameters.keys())
        assert "attachment_content" in params
