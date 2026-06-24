"""
Subagent 定义 — 多代理协作架构的核心组件

子代理A: outline_agent   — 文档框架设计
子代理B: chapter_agent   — 单章节编写

两个子代理各自拥有独立的 LLM 实例和上下文，通过 create_agent() 创建。
主代理通过 agent_tools 中的工具包装调用这些子代理。
"""
import os
from langchain_openai import ChatOpenAI


# ── 共享模型工厂 ──

def _get_subagent_model() -> ChatOpenAI:
    """子代理专用模型实例 (与主代理同模型但独立实例) - 使用本地Ollama"""
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11435") + "/v1"
    model = os.getenv("OLLAMA_MODEL", "qwen3.5:122b")
    api_key = os.getenv("MINIMAX_API_KEY", "ollama")
    return ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=0.3,
        max_tokens=4096,
    )


# ═══════════════════════════════════════════════════════
# 子代理A: 文档框架设计师
# ═══════════════════════════════════════════════════════

OUTLINE_AGENT_PROMPT = """你是医疗器械注册文档的框架设计师。

## 你的职责
根据用户指定的文档类型，设计完整的章节框架。你需要:
1. 确定文档应包含哪些章节（标题）
2. 每个章节的子主题/小节
3. 章节之间的逻辑顺序
4. 每个章节预期的内容覆盖范围

## 工作流程
1. 收到 doc_type 和 product_name 后，先用 search_kb 检索该文档类型的标准要求
2. 基于标准要求和文档类型特点，设计章节结构
3. 输出格式必须是严格JSON

## 输出格式要求
你必须只输出一个 JSON 对象（不要包含其他文字），格式如下:
```json
{
  "doc_title": "贴敷式胰岛素泵-{文档类型名称}",
  "chapters": [
    {
      "id": 1,
      "title": "第X章 章节标题",
      "description": "本章的主要内容覆盖范围",
      "key_standards": ["GB XXXX-XXXX", "ISO XXXX:XXXX"],
      "subsections": [
        {"title": "X.1 小节标题", "content_points": ["要点1", "要点2"]},
        {"title": "X.2 小节标题", "content_points": ["要点1", "要点2"]}
      ]
    }
  ]
}
```

## 设计原则
- 章节数通常6-12章，视文档类型而定
- 符合NMPA III类器械注册申报要求
- 所有引用标准必须有明确的条款号依据
- 章节顺序应遵循逻辑递进关系
- 每章描述控制在1-2句话
"""


def create_outline_agent() -> "CompiledGraph":
    """创建框架设计子代理实例"""
    from langchain.agents import create_agent
    from app.services.agent_tools import search_kb

    return create_agent(
        model=_get_subagent_model(),
        tools=[search_kb],
        system_prompt=OUTLINE_AGENT_PROMPT,
    )


# ═══════════════════════════════════════════════════════
# 子代理B: 章节编写专家
# ═══════════════════════════════════════════════════════

CHAPTER_AGENT_PROMPT = """你是贴敷式胰岛素泵RA文档章节编写专家，拥有15年以上医疗器械注册申报经验。

## 你的职责
根据给定的文档框架，编写指定章节的完整内容。你只负责单章，不要输出其他章节。

## 工作流程
1. 收到 chapter_name 和 outline 后，先用 search_kb 检索本章涉及的标准和法规
2. 根据 outline 中该章节的 description 和 subsections 编写完整内容
3. 输出 Markdown 格式的章节内容

## 编写要求
- 内容专业、完整，符合NMPA注册申报要求
- 所有标准条款引用必须有明确的条款号
- 使用Markdown格式，标题层级清晰 (## → ### → ####)
- 数据和参数要具体、可测量
- 针对贴敷式胰岛素泵产品特性编写
- 用中文表述（标准号和必要缩写除外）
- 只输出本章内容，不要写"第X章"之外的章节

## 输出格式
直接输出 Markdown 内容，第一行为 `## {章节标题}`。
不要输出JSON包裹，直接输出正文。
"""


def create_chapter_agent() -> "CompiledGraph":
    """创建章节编写子代理实例"""
    from langchain.agents import create_agent
    from app.services.agent_tools import search_kb

    return create_agent(
        model=_get_subagent_model(),
        tools=[search_kb],
        system_prompt=CHAPTER_AGENT_PROMPT,
    )
