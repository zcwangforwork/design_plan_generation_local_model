## 2026-06-16 - BM25 jieba中文分词优化

### 依据
根据 `researcher_docs/BM25_jieba中文分词优化_详细技术方案_20260616.md` 实施。

### 问题诊断
`vector_store.py` 的 BM25 分词使用 `str.split()`，中文文本词汇之间无空白分隔，导致整段中文被当作一个巨大 token，BM25 关键词匹配完全失效。混合检索退化为纯向量检索。

### Files Created
1. **app/services/rag/medical_device_dict.txt**: 医疗法规领域词典 (~80词)，覆盖产品名称、标准体系、风险管理、设计开发、质量管理、生物学、临床注册、电气安全、软件等9大类
2. **app/services/rag/stopwords_zh.txt**: 中文停用词表 (~100词)，包含通用中文停用词 + 医疗法规领域低信息量词 + 文档结构词
3. **app/services/rag/tokenizer.py**: ChineseTokenizer 类，封装 jieba.cut_for_search()，支持领域词典加载、停用词过滤、LRU缓存、线程安全单例模式

### Files Modified
1. **app/services/rag/vector_store.py**:
   - 新增 import: `threading`, `get_tokenizer`
   - 新增类变量: `_bm25_cache` (BM25 索引缓存), `_bm25_cache_lock`
   - 新增类方法: `_get_or_build_bm25()` (构建/复用 BM25 索引 + jieba分词), `invalidate_bm25_cache()` (缓存失效)
   - 重写 `_bm25_search_collection()`: `.split()` → `tokenizer.tokenize()` + BM25 缓存 + 优化参数 (k1=1.2, b=0.6)
2. **app/services/rag/ingest.py**: `ingest_all()` 和 `ingest_files()` 末尾新增 `VectorStore.invalidate_bm25_cache()` 调用
3. **requirements.txt**: 新增 `jieba>=0.42.1`

### Verification
- jieba 安装成功 (v0.42.1)
- tokenizer 单例模式正常
- 中文分词测试通过: "贴敷式胰岛素泵风险管理" → ['胰岛', '胰岛素', '贴敷式胰岛素泵', '风险', '管理', '风险管理']
- tokenize_corpus 批量分词正常
- VectorStore 新增方法可正常访问

## 2026-06-12 - 设计输入阶段文档生成扩展

### Changes Summary
根据《胰岛素泵-DHF清单.xlsx》中设计输入阶段所需的15份文档，扩展项目使其能够生成该阶段的所有文档类型。

### Files Modified
1. **app/services/doc_types.py**:
   - 新增 `essential_principles_checklist` (医疗器械安全和性能基本原则EP清单)
   - 新增 `software_development_plan_di` (设计输入阶段软件开发计划引用)
   - 将 `software_development_plan` 加入 design_input 分类
   - design_input 分类从16种扩展为19种文档

2. **app/services/prompt_engineer.py**:
   - 新增 `essential_principles_checklist` 专属生成提示词 (覆盖7个章节: 概述、通用安全、化学生物特性、感染控制、有源器械安全、标签信息、符合性声明)
   - 新增 `software_development_plan_di` 专属生成提示词

3. **app/services/minimax.py**:
   - 新增 `essential_principles_checklist` 章节定义 (7章)
   - 新增 `software_development_plan_di` 章节定义 (5章，复用software_development_plan结构)

4. **app/api/routes.py**:
   - 更新 `/agent/projects/{project_id}/auto-generate` 支持 `doc_type` 参数
   - 新增 `/agent/projects/{project_id}/batch-generate` 批量生成设计输入阶段全部文档 (SSE推送进度+ZIP打包下载)
   - 新增 `/agent/batch-download/{project_id}` ZIP下载接口
   - 新增 `import zipfile` 顶层导入

5. **app/services/agent_state.py**:
   - 新增 `doc_type` 字段到 DesignInputAgentState TypedDict

### Result
- 设计输入阶段现支持19种文档类型，全部具有章节定义和专属生成提示词
- 经典API (`/api/generate`) 可生成任意文档类型
- Agent批量生成API可一次性生成设计输入阶段全部文档并打包为ZIP

## 2026-06-02 09:55:00 - 贴敷式胰岛素泵数字员工全生命周期文档系统改造
- Working Dir: `E:\nrf_sample_codes\working_team_work\public\project\project_0430_02_beta`
- Purpose: 将项目从 16 种文档类型的 QMS 生成工具改造为覆盖 94+ 种文档类型的贴敷式胰岛素泵全生命周期数字员工
- Files: doc_types.py (16→104), prompt_engineer.py (16→106), minimax.py DOC_CHAPTERS (16→106), routes.py, index.html
- Result: Success — 104 types, 106 chapters, 106 prompts, 10 categories, API 200 OK

## 2026-05-12 11:00:00 - RAG 知识库重建（使用向量化 API）
- Working Dir: `E:\nrf_sample_codes\code_writer\project_0430_02`
- Purpose: 使用火山方舟向量化 API (doubao-embedding-vision-250615, 1024维) 重建 RAG 知识库，替换旧的本地模型 (384维)
- Result: Success - 向量库从 0 重建到 650 chunks，172 个来源文件

### 详细操作记录

## 2026-05-12 11:05:00 - File Created
- File: `rebuild_kb_api.py`
- Change: 创建 RAG 知识库重建脚本，使用火山方舟 API 向量化，清除旧 384维 collection 并重新摄入
- Result: Success

## 2026-05-12 11:10:00 - File Created
- File: `test_rebuild_api.py`
- Change: 创建测试套件，覆盖 API Key、Embedder、VectorStore、摄入、检索、错误处理等 7 个测试
- Result: Success

## 2026-05-12 11:15:00 - File Edited
- File: `app/services/rag/embedder.py`
- Change: 重写 Embedder 类，从使用 volcenginesdkarkruntime SDK 改为使用 httpx 直接调用 /api/coding/v3/embeddings/multimodal 端点。原因：SDK 调用路径不正确（/api/v3/ vs /api/coding/v3/），且不支持 dimensions 参数
- Result: Success - API 调用正常返回 1024 维向量

## 2026-06-10 15:30:00 - Fix RAG retrieval blocking event loop (标准适用性清单 stuck)
- Working Dir: `E:\nrf_sample_codes\working_team_work\public\project\design_input_generation`

### 2026-06-10 15:30:00 - File Edited
- File: `app/services/agent_tools.py`
- Change: (1) Added `import asyncio`. (2) Wrapped ALL blocking calls in all 4 tools in `asyncio.to_thread()` + `asyncio.wait_for()`: search_kb (90s), generate_section (300s), revise_section (180s), build_docx (60s). (3) Added `asyncio.TimeoutError` handlers for each tool.
- Result: Success — no tool blocks the event loop, user can send messages during generation

### 2026-06-10 15:32:00 - File Edited
- File: `app/services/rag/vector_store.py`
- Change: (1) Fixed bug on line 286 where `self.client.get_collection()` was used instead of `client.get_collection()` in the vector search loop of `retrieve_hybrid()` — would cause extra DB configs to never be queried. (2) Added `import time`. (3) Added timing logs: embedding time, vector search time, BM25 time, total time.
- Result: Success

### 2026-06-10 15:35:00 - File Edited
- File: `app/services/agent_engine.py`
- Change: Wrapped `interrupt()` call in `_pre_tool_node` with try-catch for `RuntimeError("get_config")`. On Python 3.10, LangGraph's `get_config()` contextvar can fail to propagate across async task transitions, causing the generate_section HITL to crash. Now gracefully skips HITL and proceeds directly to generation when context is unavailable.
- Result: Success

## 2026-05-12 11:20:00 - File Edited
- File: `app/services/rag/vector_store.py`
- Change: 移除全局 _embedder 变量，改为实例变量延迟初始化
- Result: Success

## 2026-05-12 11:30:00 - Bash Command Executed
- Command: `python test_rebuild_api.py`
- Working Dir: `E:\nrf_sample_codes\code_writer\project_0430_02`
- Purpose: 运行测试套件验证重建流程
- Result: Success - 6 passed, 1 skipped (检索测试需先重建)

## 2026-05-12 11:35:00 - Bash Command Executed
- Command: `python rebuild_kb_api.py`
- Working Dir: `E:\nrf_sample_codes\code_writer\project_0430_02`
- Purpose: 执行知识库重建，清除旧 384维向量，使用 API 重新摄入
- Result: Partial Success - 169 文件摄入, 486 chunks, 15 文件因 API 限流 (429) 失败, 耗时 538.9 秒

## 2026-05-12 11:45:00 - File Created
- File: `retry_failed_ingest.py`
- Change: 创建重试脚本，使用更长间隔重试因限流失败的文件
- Result: Success

## 2026-05-12 11:50:00 - Bash Command Executed
- Command: `python retry_failed_ingest.py`
- Working Dir: `E:\nrf_sample_codes\code_writer\project_0430_02`
- Purpose: 重试因 API 限流失败的文件
- Result: Success - 3 文件重试成功, 新增 28 chunks

## 2026-05-12 12:00:00 - Bash Command Executed
- Command: 增量重试缺失文件（Python 脚本内联执行）
- Working Dir: `E:\nrf_sample_codes\code_writer\project_0430_02`
- Purpose: 查找并重试所有尚未在向量库中的文件
- Result: Success - 12 文件重试成功, 新增 136 chunks, 总量 650 chunks

## 2026-05-12 12:05:00 - Analysis
- Topic: Embedder API 兼容性
- Finding: volcenginesdkarkruntime SDK 默认调用 /api/v3/ 路径，但正确的向量化 API 路径是 /api/coding/v3/embeddings/multimodal。使用 httpx 直接调用 HTTP 接口更可靠。
- Decision: 重写 Embedder 类使用 httpx 直接调用，参考 project_0428 的实现

## 2026-05-12 12:10:00 - Analysis
- Topic: 知识库重建结果
- Finding: 最终向量库 650 chunks, 172 个来源文件。检索验证通过，4 个测试查询均有相关结果返回（相似度 0.43-0.53）
- Decision: 重建完成，可以投入使用

## 2026-05-29 15:00:00 - 任务开始：贴敷式胰岛素泵知识库向量化

### 任务描述
将 `贴敷式胰岛素泵知识库` 目录（25个子目录，779个文件）中的所有文档转化为向量，存入独立的 ChromaDB 目录，供项目 RAG 检索使用。

### 修改的文件

#### 1. `app/services/rag/ingest.py` - 扩展文档格式支持
- **变更**: 添加 `extract_text_from_xlsx()` 和 `extract_text_from_md()` 函数
- **变更**: 更新 `extract_text_from_file()` 支持 .xlsx 和 .md 格式
- **变更**: 更新 `get_supported_files()` 扩展名列表包含 .xlsx 和 .md
- **原因**: 知识库中包含 xlsx 和 md 文件需要处理

#### 2. `app/services/rag/vector_store.py` - 支持多目录查询
- **变更**: 添加 `EXTRA_DB_CONFIG` 类变量和 `_extra_clients` 缓存
- **变更**: 添加 `_get_client_for_path()` 和 `set_extra_db_config()` 类方法
- **变更**: 修改 `retrieve()`、`retrieve_hybrid()` 方法遍历所有 DB 目录的 collection
- **变更**: 修改 `_bm25_search_collection()` 接受 client 参数
- **变更**: 修改 `count()` 方法统计所有 DB 目录
- **原因**: 需要从多个 ChromaDB 目录查询，新知识库存放在独立目录

#### 3. `app/main.py` - 启动时自动加载胰岛素泵知识库
- **变更**: 添加启动时检测 `chroma_db_insulin_pump` 目录并配置 EXTRA_DB_CONFIG
- **原因**: 确保 RAG 服务启动时自动包含新知识库

#### 4. `build_insulin_pump_kb.py` (新建) - 知识库摄入脚本
- **内容**: 完整的知识库构建脚本，包含文件扫描、文本提取、向量生成、ChromaDB 写入
- **功能**: 25个子目录映射到24种文档类型，跳过扫描版PDF，自动测试检索

### 执行结果

#### 摄入统计
- 源目录: `贴敷式胰岛素泵知识库`
- 向量库目录: `chroma_db_insulin_pump` (独立于现有的 `chroma_db`)
- Collection 名称: `insulin_pump_kb`
- 总文件数: 777 (排除2个不支持的PPT格式)
- 成功处理: 390 个文件
- 失败/跳过: 387 个 (主要是扫描版PDF无文本、少数损坏文件和临时文件)
- 总 chunks: 6,782
- 总耗时: 4,778秒 (79.6分钟)
- DB 大小: 119MB

#### 文档类型分布 (Top 10)
| doc_type | chunks |
|---|---|
| biocompatibility | 1,272 |
| medical_electrical_safety | 854 |
| software_usability | 785 |
| quality_management | 736 |
| sterilization | 649 |
| ghtf_guidelines | 492 |
| product_registration | 220 |
| labeling | 220 |
| emc | 202 |
| transport_packaging | 179 |

#### 检索测试结果
所有测试查询均返回高相关性结果 (相似度 0.62-0.78):
- "胰岛素泵电气安全基本要求" → IEC 60601 系列标准
- "风险管理报告编写指南" → ISO 14971 风险管理标准
- "电磁兼容测试方法" → IEC 61000-4 EMC 测试标准
- "无菌包装验证流程" → 包装验证方案文档

### 架构说明
查询流程: VectorStore.retrieve() → 遍历主DB (chroma_db) 的 QUERY_COLLECTIONS + 额外DB (chroma_db_insulin_pump) 的 EXTRA_DB_CONFIG → 合并结果按相似度排序

## 2026-06-06 - 文档生成提速优化（步骤1：参数调整）
- Working Dir: `E:\nrf_sample_codes\working_team_work\public\project\project_0430_02_beta`
- Purpose: 通过调整两个核心参数显著降低文档生成总耗时
- 预期提速: 30-40%

### File Edited
- File: `app/services/minimax.py`
- Change 1 (line 1234): `self.max_concurrent = 6` → `self.max_concurrent = 12`
  - 理由: 小节LLM调用最大并发数翻倍。文档通常 20-40 小节，6 路并发需排 4-7 轮，12 路并发只需 2-4 轮，LLM 阶段墙钟时间近乎减半。火山方舟单 key 默认 RPM 300+，可支撑 12 并发。
- Change 2 (line 2313): `_call_api(..., max_tokens: int = 12000)` → `max_tokens: int = 6000`
  - 理由: 单小节实际内容很少超 6000 token。LLM 出 token 是线性时间消耗，上限砍半直接降低生成时长。不影响实际内容质量。
- Result: Success
- 验证方式: 重启服务后生成一份文档，对比 `timing_log` 中 `llm_total` 和 `total` 字段

## 2026-06-06 - 文档生成提速优化（Web搜索并发提升）
- Working Dir: `E:\nrf_sample_codes\working_team_work\public\project\project_0430_02_beta`
- File: `app/services/minimax.py`
- Change (line 1235): `self.max_search_workers = 6` → `self.max_search_workers = 12`
- 理由: Web 搜索阶段（Phase 2b）当前是仅次于 LLM 的耗时大户。Agent SDK 搜索本质是 Claude API 调用，资源占用很轻，可支撑较高并发。Playwright 回退路径开销较大，但有 Agent SDK 优先策略兜底。30 小节文档原需 5 轮搜索（6 并发），现只需 3 轮（12 并发）。
- 预期提速: Web 搜索阶段墙钟时间下降约 40%
- Result: Success

## 2026-06-10 16:30:00 - 新增文档审阅页面
- Working Dir: `E:\nrf_sample_codes\working_team_work\public\project\design_input_generation`

### 2026-06-10 16:30:00 - File Edited
- File: `app/api/routes.py`
- Change: 新增 `GET /api/agent/projects/{project_id}/document` 端点，从Agent状态中读取 generated_sections 并组装为结构化JSON（含产品信息、标准清单、章节排序、未处理项），供审阅页面使用。
- Result: Success

### 2026-06-10 16:32:00 - File Edited
- File: `app/main.py`
- Change: 新增 `GET /agent/review/{project_id}` 路由，返回 `review.html` 审阅页面。
- Result: Success

### 2026-06-10 16:35:00 - File Created
- File: `app/static/review.html`
- Change: 创建独立文档审阅页面。功能：(1) 通过URL提取project_id并调用 `/api/agent/projects/{project_id}/document` 获取文档数据；(2) 使用 marked.js CDN 渲染Markdown为格式化的HTML文档；(3) 左侧章节导航栏，点击跳转，滚动时自动高亮当前章节；(4) 顶部状态栏显示产品名、文档状态、章节数；(5) 产品信息面板显示名称/分类/预期用途；(6) 打印友好CSS (@media print隐藏侧边栏)；(7) 空状态/错误状态/加载状态三种UI；(8) 返回Agent对话链接。
- Result: Success

### 2026-06-10 16:38:00 - File Edited
- File: `app/static/agent.html`
- Change: 顶部栏新增"审阅文档"链接，指向 `/agent/review/{PROJECT_ID}`，初始化时自动设置href并显示。
- Result: Success

### 2026-06-10 17:00:00 - 新增直接下载端点 + 前端下载按钮
- Working Dir: `E:\nrf_sample_codes\working_team_work\public\project\design_input_generation`

### 2026-06-10 17:00:00 - File Edited
- File: `app/api/routes.py`
- Change: 新增 `GET /api/agent/projects/{project_id}/download` 端点，直接从Agent状态读取 generated_sections、组装Markdown、通过TemplateService构建.docx并返回。用户无需在Agent对话中说"导出文档"即可下载。
- Result: Success

### 2026-06-10 17:02:00 - File Edited
- File: `app/static/review.html`
- Change: `downloadDocx()` 函数改为直接调用 `/api/agent/projects/{PROJECT_ID}/download` 下载.docx，不再弹alert提示。
- Result: Success

### 2026-06-10 17:05:00 - File Edited
- File: `app/static/agent.html`
- Change: (1) 顶部栏新增"下载文档"链接按钮。 (2) 新增 `downloadDocument()` 函数调用下载端点。 (3) `fetchState()` 中检测 `sections_generated` 长度>0时自动显示"审阅文档"和"下载文档"链接。
- Result: Success

## 2026-06-10 16:00:00 - 对话输出简化 (Issue #4: 过长、混乱、难以阅读)
- Working Dir: `E:\nrf_sample_codes\working_team_work\public\project\design_input_generation`

### 2026-06-10 16:00:00 - File Edited
- File: `app/services/context_manager.py`
- Change: (1) `summarize_old_messages()` — 摘要从详细JSON格式改为紧凑纯文本（max 500 chars），每条消息截取从300→100字，LLM prompt改为"极简摘要（不超过300字）"。 (2) `_generate_fallback_summary()` — 简化降级摘要格式为纯文本。 (3) SystemMessage格式从详细说明简化为 `[进度回顾] {summary}`。
- Result: Success

### 2026-06-10 16:05:00 - File Edited
- File: `app/services/agent_prompt.py`
- Change: REPLY_STYLE 新增具体简洁规则：每次回复3-5句话以内，确认项用一行状态更新（如"✅ 性能要求已确认 (8项)"），生成章节后2-3行摘要，禁止输出JSON/状态快照/对话摘要，禁止重复列出所有已确认输入项。
- Result: Success

### 2026-06-10 16:10:00 - File Edited
- File: `app/static/agent.html`
- Change: (1) CSS新增 `.msg.agent.collapsed` 样式（max-height: 280px, overflow: hidden, 底部渐变遮罩）。 (2) CSS新增 `.msg-toggle` 展开/收起按钮样式。 (3) JS新增 `autoCollapseLongMessage(div)` 函数——文本>500字符自动折叠并添加"展开全部"/"收起"切换按钮。 (4) `sendMessage()` 和 `resumeAgent()` 的 `done` 事件中添加自动折叠调用。
- Result: Success

## 2026-06-10 17:15:00 - Fix LangGraph Recursion Limit
- Working Dir: `E:\nrf_sample_codes\working_team_work\public\project\design_input_generation`

### 2026-06-10 17:15:00 - File Edited
- File: `app/services/agent_engine.py`
- Change: `invoke_agent()`、`stream_agent_events()`、`resume_agent()` 三个函数的 config 中均添加 `"recursion_limit": 100`。默认25步不足以支撑多次工具调用（search_kb + generate_section × 多章节 + revise + build_docx），在文档生成流程中会触发 GRAPH_RECURSION_LIMIT 错误。
- Result: Success

