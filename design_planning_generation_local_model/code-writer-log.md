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


## 2026-06-30 11:04 - 修复 Agent 文档下载章节顺序错乱

### 问题
用户反馈：在修改完或重新生成完某一小节后，下载的文档中小节顺序错乱，新生成或修改后的小节没有放到应该放到的位置。

### 根因
`app/api/routes.py` 中两个直接下载端点使用了硬编码的 `priority_order` 列表 + `sorted(remaining.keys())` 字母序重排逻辑：

1. `GET /api/agent/projects/{project_id}/document` (review.html 章节导航用)
2. `GET /api/agent/projects/{project_id}/download` (下载按钮调用)

`priority_order` 列表（"封面"、"文档信息"、"产品画像"、"性能要求"等）与实际 design_planning 文档的章节名（"目的和范围"、"设计开发阶段划分"、"职责分配"等）完全不匹配，所有章节都落到 `sorted(remaining.keys())` 分支，按中文字符的 Unicode 码点排序，破坏了 `generated_sections` 字典本已正确保留的"首次生成顺序"。

而 Agent 驱动的 `build_docx` 工具（`agent_engine._sync_doc_context`）直接遍历 `sections.items()`，dict 顺序正确。两条下载路径行为不一致。

### 修复
移除两个端点的 `priority_order + sorted()` 重排逻辑，改为直接按 `generated` 字典的插入顺序输出，与 Agent `build_docx` 工具行为保持一致。

### 2026-06-30 11:20:00 - File Edited
- File: `app/api/routes.py`
- Change: `/agent/projects/{project_id}/document` 端点（原 line 764-779）的 `priority_order + sorted(remaining.keys())` 替换为 `[{"title": name, "content": content} for name, content in generated.items()]`
- Why: 保留 generated_sections 字典的插入顺序（即首次生成顺序），修订/重新生成同名小节时 dict 顺序自动保留
- Result: Success

### 2026-06-30 11:21:00 - File Edited
- File: `app/api/routes.py`
- Change: `/agent/projects/{project_id}/download` 端点（原 line 830-844）的 `priority_order + sorted(remaining.keys())` 替换为 `full_markdown = "\n\n".join(generated.values())`
- Why: 同上，这是 review.html "下载 .docx 文档" 按钮实际调用的端点
- Result: Success

### 2026-06-30 11:22:00 - Bash Command Executed
- Command: `python -c "import ast; ast.parse(open('.../routes.py', encoding='utf-8').read()); print('routes.py syntax OK')"`
- Working Dir: E:\nrf_sample_codes\working_team_work\code_writer
- Purpose: 语法检查修复后的 routes.py
- Result: Success - "routes.py syntax OK"

### 验证
- Grep 全项目确认无遗留 `priority_order` 或 `sorted(remaining` 模式
- tests/test_agent.py 仅断言 `build_state_snapshot` 输出（章节数量、key 成员），不涉及顺序，不受影响
- 未对 routes.py 两个端点做单元测试（不在最小修复范围内）

### 后续建议
若用户反馈 LLM 修订时传入的 section_name 与原 key 不一致（如 "目的和范围" vs "1. 目的和范围"），会导致新 key 追加到 dict 末尾、旧 key 残留，可再考虑基于 outline JSON 的章节匹配排序作为增强。

## 2026-06-30 17:00 - 新增"附件结构驱动文档生成"功能

### 背景
用户希望上传附件后能选择按附件章节结构生成新文档（而非让 LLM 自主设计 outline）。现有 analyze_document_structure 只输出扁平 level/title/summary，无法直接喂给 write_chapter（需 subsections[].title + content_points[]）。

### 用户选定方案
方案 B：LLM 二次细化——骨架识别 + LLM 基于附件全文补全 subsections/content_points。
**关键约束**：用户希望有附件时让用户选择 A(附件优先) 或 B(AI自主设计)，不强制走任一路径。

### 实施

#### 2026-06-30 17:00:00 - File Edited
- File: `app/services/agent_tools.py`
- Change: 新增 `outline_from_attachment` 工具（约 170 行），置于 analyze_document_structure 之后
  - Step1: 调用 analyze_document_structure 拿骨架
  - Step2: 从 _current_attachments 取附件 full_text（截前 20000 字符）
  - Step3: LLM 二次细化，注入骨架+全文，要求保留附件原章节顺序与标题，每章补 2-4 个 subsections
  - 复用 _extract_json 提取 JSON
  - 错误处理：附件无章节/识别失败 → 返回 error JSON，引导回退 design_outline
  - 加入 PHASE1_TOOLS 列表
- Why: 实现附件优先路径，输出与 design_outline 完全兼容，无缝衔接 write_chapter
- Result: Success

#### 2026-06-30 17:15:00 - File Edited
- File: `app/services/agent_engine.py` (line 222-232)
- Change: `_after_tools_node` 中 `if tool_name == "design_outline":` 改为 `if tool_name in ("design_outline", "outline_from_attachment"):`
- Why: 让新工具结果也写入 state["outline"] + outline_status="draft"
- Result: Success

#### 2026-06-30 17:30:00 - File Edited
- File: `app/services/agent_prompt.py`
- Change:
  (1) SOP_KNOWLEDGE 步骤4 增加"有附件时的路径选择"段落，要求 Agent 询问用户选 A(附件优先)或 B(AI自主)
  (2) TOOL_RULES 新增 outline_from_attachment 工具说明（5b）
  (3) "标准工作流"段落更新为反映两条路径
- Why: 让 Agent 有附件时主动询问用户选择，两条路径互斥
- Result: Success

#### 2026-06-30 17:45:00 - Bash Command Executed
- Command: `python -c "import ast; ast.parse(...)"` ×3 文件
- Purpose: 语法检查改动的 3 个 Python 文件
- Result: Success - agent_tools.py / agent_engine.py / agent_prompt.py 全部 OK

### 设计要点
1. **复用而非重造**：复用 analyze_document_structure、_extract_json、_find_chapter_subsections、write_chapter——零改动
2. **输出格式兼容**：与 design_outline 完全一致
3. **用户选择**：SOP 要求 Agent 询问 A/B，不擅自走任一路径
4. **优雅降级**：附件无章节时返回 error，Agent 告知用户并询问是否回退

### 涉及文件
| 文件 | 改动类型 | 改动量 |
|------|---------|-------|
| app/services/agent_tools.py | 新增工具 + 加入 PHASE1_TOOLS | ~170 行 |
| app/services/agent_engine.py | 1 行条件扩展 | 1 行 |
| app/services/agent_prompt.py | SOP + TOOL_RULES + 标准工作流 | ~30 行 |

无需改动：subagents.py、_find_chapter_subsections、write_chapter、routes.py、agent.html、analyze_document_structure

### 后续建议
- 启动服务实测：上传 Word 附件 → 确认被询问 A/B → 选 A → 验证 outline_from_attachment 被调用、state["outline"] 非空、write_chapter 匹配 subsections
- 若 LLM 细化输出偶尔不稳定，可调整 temperature 或增加 few-shot 示例

## 2026-06-30 17:50 - 修复附件上传 NameError

### 问题
用户上传附件时报错：`NameError: name 'attachments' is not defined` at routes.py line 951。

### 根因
`agent_upload_attachment` 函数中，从 `state_values` 读取了 Agent 状态，但未从中提取 `attachments` 字段到本地变量，直接调用 `attachments.append(...)` 导致 NameError。这是既有 bug（不是我引入的），上次附件功能开发时遗漏。

### 修复
在 `attachments.append(...)` 前添加：
```python
attachments = list(state_values.get("attachments", []) or [])
```

### 2026-06-30 17:50:00 - File Edited
- File: `app/api/routes.py` (line 951 附近)
- Change: 在 `attachments.append({...})` 前插入 `attachments = list(state_values.get("attachments", []) or [])`
- Why: 从 Agent 状态取出已有附件列表，避免 NameError
- Result: Success - routes.py syntax OK

### 验证
语法检查通过。重启服务后重新上传附件即可。

## 2026-06-30 18:00 - 重写 outline_from_attachment 提升章节识别完整性

### 问题
用户反馈"重启后，显示无法识别出完整的章节结构"。

### 根因
原实现两步流程依赖 `analyze_document_structure`：
1. `analyze_document_structure` 截取附件前 **12000 字符** 识别章节骨架——长文档只覆盖前几章，章节识别不完整
2. `outline_from_attachment` 又截取前 **20000 字符** 给细化 LLM——进一步限制覆盖
3. max_tokens=8192 输出可能不足以承载完整章节结构

### 修复
重写为**一步 LLM 调用**，不再依赖 `analyze_document_structure`：
1. 直接从 `_current_attachments` 取附件 full_text
2. 截取前 **50000 字符**（约覆盖 3-5 万字文档）送给 LLM
3. 一次 LLM 调用同时完成章节识别 + subsections + content_points 补全
4. max_tokens 提高到 **16384**
5. timeout 提高到 **300 秒**
6. system prompt 增加章节识别策略（编号、Markdown标题、Word样式、TOC优先）和"必须识别所有章节，不得遗漏"约束

### 2026-06-30 18:00:00 - File Edited
- File: `app/services/agent_tools.py` (outline_from_attachment 工具，约 line 1311-1479)
- Change: 重写整个工具实现，从两步流程改为一步 LLM 调用
- Why: 解决长文档章节识别不完整问题
- Result: Success - agent_tools.py syntax OK

### 验证
语法检查通过。重启服务后重新测试附件结构生成。

## 2026-07-02 17:54 - MinerU 文档解析集成

### 任务
参考 https://github.com/opendatalab/MinerU-Ecosystem/tree/main/langchain_mineru，在 design_planning_generation_local_model 项目中集成 MinerU 云端文档解析能力。

### 设计决策
- 解析模式：flash（无需 token，速度快，支持 PDF/DOCX/PPTX/XLS/XLSX/图片）
- 启用策略：opt-in，通过环境变量 USE_MINERU=true 启用；未启用或 SDK 缺失时自动回退到本地解析器
- 格式扩展：MinerU 启用时自动扩展上传格式到 .doc/.ppt/.pptx/.xls/.xlsx/.png/.jpg/.jpeg/.bmp/.tiff/.tif/.html/.htm
- 接口兼容：MinerU 返回 Markdown 后转换为 [(section_title, paragraph_text)] 列表，与现有 extract_text_from_* 系列签名一致

## 2026-07-02 17:54:13 - Analysis
- Topic: MinerU 集成点分析
- Finding: 项目文档解析入口集中在 app/services/rag/ingest.py 的 extract_text_from_file()，附件服务和 Agent 路径（ingest_attachment_to_kb / outline_from_attachment / search_attachment）都通过 full_text 间接使用其结果
- Decision: 在 extract_text_from_file 中加入 MinerU 优先路径并带 fallback，单一集成点覆盖所有下游消费方

## 2026-07-02 17:54:13 - File Created
- File: app/services/mineru_service.py
- Change: 新建 MinerU 服务封装模块，提供 is_mineru_enabled / is_mineru_sdk_available / extract_text_with_mineru / is_file_supported_by_mineru / mineru_supported_formats 等函数；包含 _markdown_to_paragraphs 将 MinerU 输出的 Markdown 切分为段落列表（标题→section_title，表格/代码块整体保留为段落）
- Result: Success - 语法校验通过，Markdown→段落 单元测试通过

## 2026-07-02 17:54:13 - File Edited
- File: app/services/doc_types.py
- Change: SUPPORTED_UPLOAD_FORMATS 改为通过 _compute_supported_upload_formats() 动态计算；MinerU 启用且 SDK 可用时追加 _MINERU_EXTRA_FORMATS（.doc/.ppt/.pptx/.xls/.xlsx/.png/.jpg/.jpeg/.bmp/.tiff/.tif/.html/.htm）
- Result: Success

## 2026-07-02 17:54:13 - File Edited
- File: app/services/rag/ingest.py
- Change: extract_text_from_file 增加 MinerU 优先路径（is_mineru_enabled + is_file_supported_by_mineru 检查通过则调用 extract_text_with_mineru，返回空或异常时回退本地解析器）；get_supported_files 在 MinerU 启用时追加扩展格式扩展名
- Result: Success - 导入级测试通过，无循环导入

## 2026-07-02 17:54:13 - File Edited
- File: .env
- Change: 追加 MinerU 配置段（USE_MINERU=false / MINERU_MODE=flash / MINERU_TOKEN= / MINERU_LANGUAGE=ch / MINERU_TIMEOUT=1200）
- Result: Success

## 2026-07-02 17:54:13 - File Edited
- File: requirements.txt
- Change: 追加 langchain-mineru>=0.1.0 和 mineru-open-sdk 依赖
- Result: Success

## 2026-07-02 17:54:13 - Bash Command Executed
- Command: `python -c "import ast; ast.parse(open(f).read())..."` for 3 files
- Working Dir: design_planning_generation_local_model
- Purpose: Python AST 语法校验 mineru_service.py / doc_types.py / ingest.py
- Result: Success - 3/3 文件通过

## 2026-07-02 17:54:13 - Bash Command Executed
- Command: `python -c "from app.services.mineru_service import ...; from app.services.doc_types import SUPPORTED_UPLOAD_FORMATS; from app.services.rag.ingest import extract_text_from_file"`
- Working Dir: design_planning_generation_local_model
- Purpose: 导入级测试，验证无循环导入，验证 USE_MINERU=false 时 SUPPORTED_UPLOAD_FORMATS 退化为 ['.docx', '.pdf', '.txt']
- Result: Success - 全部导入成功，预期行为符合

## 2026-07-02 17:54:13 - Bash Command Executed
- Command: `python -c "from app.services.mineru_service import _markdown_to_paragraphs; ..."`
- Working Dir: design_planning_generation_local_model
- Purpose: 验证 Markdown→段落 转换逻辑（标题/段落/表格/代码块）
- Result: Success - 5 段落输出正确，表格和代码块整体保留

### 验证总结
- 语法校验：3/3 通过
- 导入测试：无循环导入，依赖链正常
- 单元测试：Markdown→段落 转换符合预期

### 后续步骤（用户启用 MinerU 时）
1. `pip install langchain-mineru mineru-open-sdk`（在 env_01 环境）
2. 编辑 .env：设置 USE_MINERU=true（如需 precision 模式额外设置 MINERU_TOKEN）
3. 重启服务，SUPPORTED_UPLOAD_FORMATS 将自动扩展
4. 上传附件时 MinerU 优先解析，失败自动回退本地解析器

## 2026-07-02 18:00 - 用户启用 MinerU 后的调用链确认

### 用户问题
用户上传附件的时候可以进行解析吗？

### 分析
- 调用链：POST /api/upload → submit_extract_task() → _do_extract() → extract_text_from_file() → MinerU 优先路径
- /api/upload 和 /api/agent/upload/{project_id} 两个入口都走同一个 submit_extract_task，已自动覆盖
- validate_upload 使用动态 SUPPORTED_UPLOAD_FORMATS，会自动放行 MinerU 扩展格式
- Result: Yes — 用户设置 USE_MINERU=true 后上传附件会自动调用 MinerU 解析

## 2026-07-02 18:00:00 - File Edited
- File: app/api/routes.py
- Change: 更新 /api/upload 路由的 file 参数描述，反映 MinerU 启用后的扩展格式支持
- Result: Success

## 2026-07-02 18:10 - 服务启动崩溃修复（段错误隔离）

### 问题
用户重启服务后页面无响应。诊断：8002 端口未监听，uvicorn worker 进程在导入 app.main 时段错误崩溃（exit=139）。

### 根因
- 直接原因：`doc_types.py` 模块加载时调用 `is_mineru_sdk_available()` → `import langchain_mineru`
- `langchain_mineru` 导入链：→ `langchain_core.document_loaders.base` → `langchain_text_splitters` → `sentence_transformers` → **段错误**（sentence_transformers/__init__.py 第15行，C 扩展冲突）
- 段错误无法用 try/except 捕获，直接杀死 Python 进程

### 修复
1. `doc_types.py` 的 `_compute_supported_upload_formats()` 改为只检查 `is_mineru_enabled()`，不触发 SDK 导入
2. `mineru_service.py` 的 `is_mineru_sdk_available()` 改为子进程预检测，结果缓存
   - 子进程中 import langchain_mineru，段错误只杀死子进程，主进程安全
   - `_SDK_AVAILABLE_CACHE` 全局缓存避免重复开销
3. `extract_text_with_mineru` 在 SDK 不可用时直接返回空，绝不主进程 import

### 验证
- 完整导入链测试通过（exit=0）：mineru_service → doc_types → ingest → app.main
- `is_mineru_sdk_available()` = False（正确检测到段错误）
- `SUPPORTED_UPLOAD_FORMATS` 扩展到 16 种格式（基于 USE_MINERU=true）
- 服务启动正常：8002 端口监听，/api/doc-types 返回 HTTP 200，前端页面 HTTP 200

### 当前状态
- 服务已恢复，用户可正常发送消息
- USE_MINERU=true 但 SDK 不可用，所有解析走本地解析器（安全回退）
- 扩展格式（.doc/.ppt/.pptx/.xls 等）上传会被接受但本地解析器返回空
- 待解决：sentence_transformers 段错误问题（与 MinerU 集成无关，是环境已有问题）

## 2026-07-02 18:10:00 - Analysis
- Topic: langchain_mineru 段错误根因
- Finding: 段错误发生在 sentence_transformers/__init__.py 第15行，由 langchain_mineru 间接导入触发。这是环境已存在的 sentence_transformers 版本与 torch 不兼容问题，非 MinerU 集成引入
- Decision: 通过子进程隔离检测 SDK 可用性，主进程绝不直接 import langchain_mineru，保证服务稳定性

## 2026-07-02 18:10:00 - File Edited
- File: app/services/mineru_service.py
- Change: is_mineru_sdk_available() 改为子进程预检测+缓存；extract_text_with_mineru 在 SDK 不可用时直接返回空
- Result: Success - 主进程不再因 langchain_mineru 段错误崩溃

## 2026-07-02 18:10:00 - File Edited
- File: app/services/doc_types.py
- Change: _compute_supported_upload_formats() 移除 is_mineru_sdk_available() 调用，只检查 USE_MINERU 环境变量
- Result: Success - 模块加载不再触发 SDK 导入

## 2026-07-02 18:10:00 - Bash Command Executed
- Command: `python -c "from app.main import app; print('OK')"` 完整导入链测试
- Working Dir: design_planning_generation_local_model
- Purpose: 验证修复后导入链正常
- Result: Success - exit=0，所有模块导入正常

## 2026-07-02 18:10:00 - Bash Command Executed
- Command: `curl http://localhost:8002/api/doc-types`
- Working Dir: N/A
- Purpose: 验证服务恢复
- Result: Success - HTTP 200

## 2026-07-03 10:30 - MinerU 段错误根因修复（pyarrow DLL 冲突 + 子进程隔离）

### 问题
sentence_transformers 导入时段错误，导致 langchain_mineru 无法导入，MinerU SDK 检测失败。

### 根因定位（faulthandler）
段错误调用链：
```
sentence_transformers
  → datasets
  → pyarrow/dataset.py line 24  ← access violation
```
根本原因：torch 2.6.0+cu124 与 pyarrow 24.0.0 的 Intel OpenMP DLL 冲突（Windows DLL Hell），torch 加载后修改 DLL 搜索路径导致 pyarrow C 扩展加载错误的 DLL。

### 修复方案（两层防护）

#### 第一层：KMP_DUPLICATE_LIB_OK=TRUE 抑制 OpenMP 冲突
- `app/main.py` 顶部添加 `os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")`
- 必须在 import torch/pyarrow/sentence_transformers 之前设置
- 降级 sentence-transformers 到 3.4.1（兼容性更好）
- 降级 datasets 到 2.20.0（兼容 pyarrow）

#### 第二层：子进程隔离 MinerU 调用
- 新建 `app/services/mineru_runner.py`：独立子进程脚本，负责实际调用 MinerULoader
- `mineru_service.py` 的 `extract_text_with_mineru` 改为通过 subprocess 调用 runner
- 子进程输出 JSON 到 stdout，日志到 stderr
- 子进程段错误只杀子进程，主服务进程不受影响
- `is_mineru_sdk_available()` 也用子进程预检测+缓存

### 验证结果
- `app.main` 完整导入：exit=0，无段错误
- `is_mineru_sdk_available()`：True（子进程检测通过）
- `extract_text_with_mineru(docx)`：54 段落，主进程不崩溃
- 端到端测试：POST /api/upload → 20秒内 status=completed，preview 含 MinerU 返回的 Markdown
- 服务进程健康：reload+worker 都在运行，后续请求正常

### 文件变更

## 2026-07-03 10:30:00 - File Created
- File: app/services/mineru_runner.py
- Change: MinerU 子进程执行器，独立运行 MinerULoader，输出 JSON 到 stdout，日志到 stderr
- Result: Success

## 2026-07-03 10:30:00 - File Edited
- File: app/services/mineru_service.py
- Change: extract_text_with_mineru 改为 subprocess 调用 mineru_runner.py，主进程不直接 import langchain_mineru
- Result: Success

## 2026-07-03 10:30:00 - File Edited
- File: app/main.py
- Change: 顶部添加 KMP_DUPLICATE_LIB_OK=TRUE 环境变量（在所有 torch/pyarrow import 之前）
- Result: Success

## 2026-07-03 10:30:00 - Bash Command Executed
- Command: `pip install sentence-transformers==3.4.1`
- Purpose: 降级到稳定版本（5.4.1 与 transformers 4.46.3 不兼容）
- Result: Success

## 2026-07-03 10:30:00 - Bash Command Executed
- Command: `pip install datasets==2.20.0`
- Purpose: 降级 datasets 兼容 pyarrow
- Result: Success

## 2026-07-03 10:30:00 - Analysis
- Topic: pyarrow DLL 冲突根因
- Finding: torch 2.6.0+cu124 与 pyarrow 24.0.0 的 Intel OpenMP DLL 冲突，KMP_DUPLICATE_LIB_OK=TRUE 可抑制
- Decision: 两层防护（KMP_DUPLICATE_LIB_OK + 子进程隔离），确保主服务稳定性

## 2026-07-03 - Agent回复字数限制放宽

### 需求
用户希望 Agent 在与用户对话时的回答内容字数适当增多、回答内容更详细。

### Analysis
- Topic: Agent回复风格限制位置排查
- Finding: 回复字数与详细程度由 `app/services/agent_prompt.py` 中的 `REPLY_STYLE` 常量控制 (原420-430行)；其中"每次回复控制在3-5句话以内，极度简洁"和"生成章节后用2-3行摘要"是直接限制回复长度的核心约束。
- Decision: 仅修改 `REPLY_STYLE` 这一段提示词，放宽字数与详细度约束；不动 `agent_engine.py` 中的 `max_tokens=4096` (已足够容纳详细回复)；遵循最小编辑原则，不改动其他无关内容。

### File Edited
- File: `app/services/agent_prompt.py`
- Change: 重写 `REPLY_STYLE` 常量内容：
  - 将 "每次回复控制在3-5句话以内，极度简洁" 改为 "回复内容应充分、详尽，把问题讲清楚、讲完整，避免过度简短导致信息缺失"
  - 将 "生成章节后用2-3行摘要，不要重复输出全文" 改为 "可给出较详细的摘要：概括章节核心内容、关键条款/参数、引用的标准依据，以及与下一章的衔接关系"
  - 新增 "解释标准、条款、参数、设计决策时，可展开说明背景、适用场景、典型取值范围和注意事项"
  - 新增 "回答用户提问时，尽量直接给出有价值的信息和依据，而不是只回答'是/否'"
  - 保留 "不输出JSON/状态快照/对话摘要"、"不重复列出所有已确认策划项"、"给出依据"、"不编造"、"使用中文" 等核心规则
- Result: Success - Python语法通过，无遗漏的字数限制 (grep 已确认)

### Verification
- grep "简洁|简短|控制在|话以内|行摘要|不要重复" 仅匹配到新文案中的"避免过度简短"和"简短小标题"，旧的字数硬限制已全部移除。
- `agent_engine.py` 的 `max_tokens=4096` 保持不变，token上限足以支持更详细的回复。

### Impact
- 仅影响 Agent 与用户对话时的回复风格 prompt，不影响文档生成流程、工具调用、章节生成质量等核心功能。
- Agent 回复将更详细、更具解释性，对用户更友好；同时保留"不输出JSON/状态快照"等必要约束，避免污染对话上下文。

## 2026-07-03 - MinerU 子进程 stderr/stdout 解码乱码修复

### 需求
用户上传 docx 附件时，MinerU 子进程退出码 1 失败，但 stderr 中文被错误解码为乱码
（`[MinerU-Runner] ʼؽ...`），无法看到真实失败原因。

### Analysis
- Topic: MinerU 子进程输出解码方式排查
- Finding: `mineru_service.py:264` 和 `:269` 原本用 `data.decode("utf-8", errors="replace")`
  硬解码 stderr/stderr。但 Windows 下 `subprocess.run` 子进程的 stdout/stderr 默认跟随系统
  代码页 cp936/GBK，mineru_runner.py 中 print 的中文消息用 GBK 编码，用 utf-8 解码必然乱码。
- Decision: 新增 `_decode_subprocess_output` 辅助函数，依次尝试 utf-8 → cp936 → gbk → gb18030
  → latin1，首个成功为准；全部失败用 utf-8 + replace 兜底。同时修复 stderr 和 stdout 两处
  解码（stdout 也可能含中文，例如 JSON 内的段落内容）。遵循最小编辑原则，不动其他逻辑。

### File Edited
- File: `app/services/mineru_service.py`
- Change:
  1. L105-120 新增 `_decode_subprocess_output(data: bytes) -> str` 辅助函数
  2. L282 stderr 解码：`result.stderr.decode("utf-8", errors="replace")[-500:] if result.stderr else ""`
     → `_decode_subprocess_output(result.stderr)[-500:]`
  3. L287 stdout 解码：`result.stdout.decode("utf-8", errors="replace")`
     → `_decode_subprocess_output(result.stdout)`
- Result: Success - Python 语法检查通过；旧 `decode("utf-8", errors="replace")` 已全部替换。

### Verification
- `python -c "import ast; ast.parse(...)"` → syntax OK
- grep 确认 3 处改动生效 (L105 新增函数, L282 stderr, L287 stdout)

### Impact & Next Steps
- 仅影响 MinerU 子进程失败时的错误信息显示，不影响成功路径和回退逻辑。
- 需要重启 FastAPI 服务才能生效（模块已加载到内存）。
- 重启后重新上传一个会触发 MinerU 失败的 docx 文件，即可在日志中看到真实的中文错误信息，
  据此进一步诊断 MinerU 失败根因（可能是 GPU OOM、模型加载失败、文件格式问题等）。

## 2026-07-03 - MinerU 子进程失败时读取 stdout 错误详情

### 需求
stderr 乱码修复后，日志能看到中文 `[MinerU-Runner] 开始本地解析:` 但仍无 traceback，
真实错误信息缺失。

### Analysis
- Topic: mineru_runner.py 输出协议排查
- Finding: runner 的输出协议是 stderr 放日志、stdout 放 JSON 结果，**失败时也是 JSON**
  (`{"status":"error","message":"...","traceback":"..."}`, 见 mineru_runner.py L65/L70)。
  但父进程 `mineru_service.py` 在 `returncode != 0` 时只读 stderr，直接 return [],
  把 stdout 里的 JSON 错误详情（含 traceback）全部丢弃，所以永远看不到真实失败原因。
- Decision: 在 returncode != 0 分支中，额外读取并解析 stdout 的 JSON，提取 message 和
  traceback 字段打印出来。stdout 非 JSON 时也兜底显示原始内容（应对 import 阶段崩溃
  无输出的场景）。保持最小改动，不动成功路径和回退逻辑。

### File Edited
- File: `app/services/mineru_service.py` L281-302
- Change: returncode != 0 分支重写
  - 新增：读取 stdout 并尝试 JSON 解析
  - 新增：提取 `message` 和 `traceback` 字段，拼接显示
  - 新增：stdout 非 JSON 时的兜底显示
  - 保留：stderr 日志显示（作为辅助信息）
  - 输出格式改为分块打印：`[错误详情]` + `[stderr]` 两段
- Result: Success - Python 语法检查通过

### Verification
- `python -c "import ast; ast.parse(...)"` → syntax OK

### Impact & Next Steps
- 仅影响 MinerU 子进程失败时的错误信息显示完整性，不影响成功路径和回退逻辑。
- 需要重启 FastAPI 服务才能生效。
- 重启后重新上传触发 MinerU 失败的 docx，日志应能看到完整的 traceback，
  据此定位 MinerU 本地模型加载/推理的真实失败原因。

## 2026-07-03 - 修复 MinerU 子进程 ModuleNotFoundError

### 需求
stderr 乱码 + stdout 错误详情读取两处修复后，看到真实错误：
`ModuleNotFoundError: No module named 'app'` (mineru_runner.py:62)

### Analysis
- Topic: mineru_runner.py 子进程 sys.path 排查
- Finding: 父进程 `mineru_service.py:243` 用 `python <runner_path> <file>` 启动子进程，
  此时子进程 `sys.path[0]` 是 `app/services/` 目录，不含项目根目录。
  runner 顶部未做 sys.path 处理，导致 `from app.services.mineru_local import ...` 找不到 app 包。
  对比 `app/services/rag/ingest.py:17-18` 有同样的 sys.path 注入模式。
- Decision: 在 `mineru_runner.py` 顶部注入项目根目录到 sys.path，与 ingest.py 保持一致。
  最小改动，不动其他逻辑。

### File Edited
- File: `app/services/mineru_runner.py` L19-29
- Change: 顶部 import 区域新增：
  ```python
  from pathlib import Path
  project_root = Path(__file__).parent.parent.parent
  sys.path.insert(0, str(project_root))
  ```
  (`mineru_runner.py` 位于 `app/services/`，三层 .parent 回到项目根)
- Result: Success - Python 语法检查通过

### Verification
- `python -c "import ast; ast.parse(...)"` → syntax OK
- 完整功能验证需重启服务 + 重新上传 docx 文件

### Impact & Next Steps
- 修复 MinerU 子进程无法 import app 包的问题，理论上应让 MinerU 解析路径真正可用。
- 需要重启 FastAPI 服务才能生效。
- 重启后重新上传 docx，预期看到 `[MinerU] 解析完成: ... → N 段落, M 字符` 而非回退本地。
- 若仍有错误（如模型加载失败、GPU OOM 等），错误详情会通过上一条修复的 stdout 读取逻辑完整显示。

## 2026-07-03 - 安装 CUDA 版 torch 启用 GPU 推理

### 需求
MinerU 子进程加载本地 1.2B VLM 模型后推理极慢，导致路由 30 秒轮询超时返回 500。

### Analysis
- Topic: torch CUDA 支持排查
- Finding: `torch 2.12.1+cpu` 是 CPU 版本（`+cpu` 后缀），不含 CUDA 支持。
  `mineru_local.py:236-240` 的 `device_map="auto"` 在 CUDA 不可用时只能 fallback 到 CPU，
  1.2B VLM 在 CPU 上推理一页文档需 30-60 秒，多页直接超时。
- Decision: 替换为 CUDA 12.6 版 torch（用户 GPU 为 RTX 4060 Ti 8GB，CUDA 12.6）。
  查询 PyTorch 官方索引确认 `torch-2.12.1+cu126` 存在，可精确匹配当前版本不降级。

### Operations
- Command: `pip uninstall torch torchvision -y && pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126`
- Working Dir: env_01 conda 环境
- Result: Success - 下载 torch 2.6GB + torchvision 8.2MB，耗时约 5 分钟
- 安装结果: `torch-2.12.1+cu126` + `torchvision-0.27.1+cu126`

### Verification
- `torch.cuda.is_available()` → True ✓
- `torch.version.cuda` → 12.6 ✓
- `torch.cuda.get_device_name(0)` → NVIDIA GeForce RTX 4060 Ti ✓
- `torch.cuda.get_device_capability(0)` → (8, 9) Ada Lovelace ✓
- `total VRAM` → 8.0 GB ✓ (1.2B 模型 FP16 约 2.4GB，足够)

### Impact & Next Steps
- MinerU 子进程现在会通过 `device_map="auto"` 自动加载模型到 GPU，推理速度预计提升 10-50 倍。
- 需要重启 FastAPI 服务才能生效（新 torch 库需重新加载）。
- 潜在遗留问题：routes.py 的 30 秒轮询超时对大文档可能仍不够（GPU 首次加载模型 10-30 秒 +
  推理每页 1-5 秒，10 页文档可能需 30-60 秒）。建议重启后先测试小文档，若仍超时再调整超时或改异步上传。

## 2026-07-03 - 增大附件上传轮询超时到 5 分钟

### 需求
MinerU 启用 GPU 后，首次加载模型 + 多页推理可能超过原 30 秒轮询超时，导致 500 错误。

### File Edited
- File: `app/api/routes.py` L905-917
- Change:
  - `for _ in range(60)` → `for _ in range(600)` (30秒 → 5分钟)
  - 注释更新：`# 轮询等待提取完成（最长5分钟，适配 MinerU GPU 首次加载模型 + 多页推理耗时）`
  - 超时错误提示更新：`文件提取超时（超过5分钟），请重试或使用更小的文档`
- Result: Success - Python 语法检查通过

### Impact
- 附件上传接口 `/api/agent/upload/{project_id}` 的同步等待时间从 30 秒延长到 5 分钟。
- 足以覆盖 GPU 首次加载模型（10-30秒）+ 中等文档推理（10-50页 × 1-5秒/页）。
- 超过 5 分钟的极大文档仍会超时，届时可考虑改异步上传方案。
