"""
Ollama API Service - AI内容生成 (本地qwen3.5:122b)
"""

import os
import json
import time
import threading
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import Optional, List, Dict, Tuple

from app.services.doc_types import DOC_TYPE_LABELS
from app.services.prompt_engineer import DOC_TYPE_SPECIFIC_PROMPTS


# ── 模块级API调用函数 (供Agent Tools等模块使用) ──

def _call_minimax_api_raw(
    system_prompt: str = "",
    user_prompt: str = "",
    temperature: float = 0.3,
    max_tokens: int = 4096,
    timeout: tuple = (30, 180),
) -> str:
    """调用本地Ollama API (qwen3.5:122b) (支持system+user消息)

    供agent_tools.py、context_manager.py等模块直接调用，不依赖MiniMaxService实例。

    Args:
        system_prompt: 系统角色prompt
        user_prompt: 用户prompt
        temperature: 温度参数 (0-1)
        max_tokens: 最大生成token数
        timeout: (连接超时, 读取超时) 元组

    Returns:
        API返回的文本内容，失败返回空字符串
    """
    import os
    import json
    import time
    import requests

    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11435")
    model = os.getenv("OLLAMA_MODEL", "qwen3.5:122b")
    api_url = f"{base_url}/api/chat"

    headers = {
        "Content-Type": "application/json; charset=utf-8",
    }

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    payload = {
        "model": model,
        "messages": messages,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
        "stream": False,
    }

    last_error = None
    for attempt in range(3):
        try:
            response = requests.post(
                api_url,
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()
            result = response.json()
            # Ollama /api/chat 格式: {"message": {"content": "..."}}
            message = result.get("message", {})
            if message:
                return message.get("content", "")
            return ""

        except requests.exceptions.Timeout as e:
            last_error = e
            if attempt < 2:
                time.sleep(2 ** attempt)
                continue
        except requests.exceptions.RequestException as e:
            last_error = e
            if attempt < 2:
                time.sleep(2 ** attempt)
                continue
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            raise ValueError(f"解析API响应失败: {str(e)}")

    if last_error:
        print(f"[_call_minimax_api_raw] 3次尝试后仍失败: {last_error}")
    return ""

# RAG 组件 - 延迟导入避免启动时耗时
_rag_available = False
_vector_store = None
_rag_prompt_builder = None

# Web 搜索组件 - 延迟初始化
_web_search_available = None
_web_search_service = None

# 跨 collection 检索 — uploads collection 是否可用
_uploads_collection_available = None


def _try_init_web_search():
    """延迟初始化 Web 搜索组件"""
    global _web_search_available, _web_search_service
    if _web_search_available is not None:
        return _web_search_available
    try:
        from app.services.web_search import SyncWebSearchService
        _web_search_service = SyncWebSearchService()
        _web_search_available = _web_search_service.playwright_available
        return _web_search_available
    except ImportError:
        _web_search_available = False
        return False

def _try_init_rag():
    """延迟初始化 RAG 组件"""
    global _rag_available, _vector_store, _rag_prompt_builder
    if _rag_available:
        return True
    try:
        from app.services.rag.vector_store import VectorStore
        from app.services.rag.rag_prompt import build_rag_prompt_from_base
        _vector_store = VectorStore
        _rag_prompt_builder = build_rag_prompt_from_base
        _rag_available = True
        return True
    except ImportError:
        return False

# API配置 - 从环境变量读取（延迟读取，避免导入时.env未加载）
def _get_api_key():
    return os.getenv("MINIMAX_API_KEY", "")

def _get_ollama_base_url():
    return os.getenv("OLLAMA_BASE_URL", "http://localhost:11435")

def _get_ollama_model():
    return os.getenv("OLLAMA_MODEL", "qwen3.5:122b")

MINIMAX_API_URL = "https://ark.cn-beijing.volces.com/api/coding/v3/chat/completions"  # 保留兼容，不再使用

# 文档章节结构定义 - 分章节生成的基础
# 按贴敷式胰岛素泵9大生命周期阶段组织
DOC_CHAPTERS = {
    # ================= 一、设计策划阶段 =================
    "design_development_plan": [
        {"id": "ch1", "name": "目的和范围", "query": "设计开发计划 目的 范围 产品描述 贴敷式胰岛素泵"},
        {"id": "ch2", "name": "设计开发阶段划分", "query": "设计开发阶段 阶段划分 里程碑 软件C级 无菌器械"},
        {"id": "ch3", "name": "职责分配", "query": "设计开发职责 人员分配 资质要求 项目组织"},
        {"id": "ch4", "name": "设计开发活动安排", "query": "设计评审 设计验证 设计确认 时间安排 资源配置"},
        {"id": "ch5", "name": "技术资源配置", "query": "设备 软件工具 标准 法规 外部资源 测试设备"},
        {"id": "ch6", "name": "设计变更管理", "query": "设计变更 控制流程 审批权限 追溯性"},
    ],
    "product_requirements_spec": [
        {"id": "ch1", "name": "产品概述", "query": "贴敷式胰岛素泵 产品概述 预期用途 适用范围"},
        {"id": "ch2", "name": "功能需求", "query": "功能需求 输注模式 闭环控制 CGM联动 报警功能"},
        {"id": "ch3", "name": "性能需求", "query": "性能需求 输注精度 流量稳定性 阻塞检测 电池续航"},
        {"id": "ch4", "name": "安全需求", "query": "安全需求 故障检测 冗余设计 安全状态 电气安全"},
        {"id": "ch5", "name": "使用环境需求", "query": "使用环境 家庭护理 温度 湿度 防水 电磁环境"},
        {"id": "ch6", "name": "法规和标准需求", "query": "法规标准 GB9706 ISO13485 IEC62304 注册要求"},
    ],
    "risk_management_plan": [
        {"id": "ch1", "name": "目的和范围", "query": "风险管理目的 范围 适用性 ISO14971"},
        {"id": "ch2", "name": "产品描述", "query": "贴敷式胰岛素泵 结构组成 技术参数 工作原理"},
        {"id": "ch3", "name": "人员资格和职责", "query": "人员资格 职责分配 风险管理团队"},
        {"id": "ch4", "name": "风险可接受性准则", "query": "风险可接受准则 严重度 频度数 探测度 评分标准"},
        {"id": "ch5", "name": "风险管理活动计划", "query": "风险管理活动 FMEA FTA 计划时间表 评审安排"},
        {"id": "ch6", "name": "生产和生产后信息收集", "query": "生产后信息 投诉 不良事件 文献 信息收集方案"},
    ],
    "software_development_plan": [
        {"id": "ch1", "name": "目的和范围", "query": "软件计划 目的范围 IEC62304 安全等级C"},
        {"id": "ch2", "name": "软件开发生命周期模型", "query": "生命周期模型 V模型 敏捷 阶段划分"},
        {"id": "ch3", "name": "软件配置管理", "query": "配置管理 版本控制 基线管理 工具"},
        {"id": "ch4", "name": "软件验证计划", "query": "单元测试 集成测试 系统测试 验证策略"},
        {"id": "ch5", "name": "交付物清单", "query": "软件交付物 文档清单 IEC62304要求"},
    ],
    "usability_engineering_plan": [
        {"id": "ch1", "name": "目的和范围", "query": "可用性工程 目的范围 IEC62366 贴敷式胰岛素泵"},
        {"id": "ch2", "name": "用户画像和使用场景", "query": "用户画像 糖尿病患者 使用场景 家庭 外出 运动"},
        {"id": "ch3", "name": "关键任务识别", "query": "关键任务 贴敷 更换 剂量设定 报警响应"},
        {"id": "ch4", "name": "形成性评价计划", "query": "形成性评价 原型测试 认知走查 启发式评估"},
        {"id": "ch5", "name": "总结性评价计划", "query": "总结性评价 真实用户 模拟环境 任务完成率 错误率"},
    ],
    "biological_evaluation_plan": [
        {"id": "ch1", "name": "目的和范围", "query": "生物学评价 目的范围 ISO10993 接触性质"},
        {"id": "ch2", "name": "产品接触性质分析", "query": "皮肤接触 长期接触 输注管路 材料清单"},
        {"id": "ch3", "name": "试验项目确定", "query": "细胞毒性 刺激 致敏 血液相容性 试验项目"},
        {"id": "ch4", "name": "化学表征计划", "query": "化学表征 材料成分 溶出物 可沥滤物"},
        {"id": "ch5", "name": "评价时间表", "query": "时间安排 试验顺序 依赖关系"},
    ],
    "sterilization_validation_plan": [
        {"id": "ch1", "name": "目的和范围", "query": "灭菌验证 目的范围 EO灭菌 辐照灭菌 选择依据"},
        {"id": "ch2", "name": "灭菌方法选择", "query": "EO灭菌 辐照灭菌 材料兼容性 灭菌效果"},
        {"id": "ch3", "name": "验证策略", "query": "IQ OQ PQ 生物负载 灭菌周期开发"},
        {"id": "ch4", "name": "接受标准", "query": "SAL 10-6 生物指示剂 过程参数 残留限值"},
        {"id": "ch5", "name": "验证时间表", "query": "时间安排 资源 样品制备"},
    ],
    "packaging_validation_plan": [
        {"id": "ch1", "name": "目的和范围", "query": "包装验证 目的范围 ISO11607 无菌屏障"},
        {"id": "ch2", "name": "包装系统描述", "query": "初级包装 次级包装 运输包装 材料"},
        {"id": "ch3", "name": "验证测试项目", "query": "密封强度 完整性 运输模拟 老化测试"},
        {"id": "ch4", "name": "接受标准", "query": "密封强度限值 完整性标准 运输后完好率"},
        {"id": "ch5", "name": "验证时间表", "query": "时间安排 样品量 测试顺序"},
    ],
    "supplier_management_plan": [
        {"id": "ch1", "name": "目的和范围", "query": "供应商管理 目的范围 ISO13485 7.4"},
        {"id": "ch2", "name": "关键外购件识别", "query": "微电机 储药器 贴片胶带 电池 传感器 关键物料"},
        {"id": "ch3", "name": "供应商评价准则", "query": "评价准则 质量体系 生产能力 交付 价格"},
        {"id": "ch4", "name": "供应商审核计划", "query": "审核计划 审核频率 审核内容 审核报告"},
        {"id": "ch5", "name": "进货检验和绩效监控", "query": "进货检验 供应商绩效 不合格处理"},
    ],
    "regulatory_strategy_document": [
        {"id": "ch1", "name": "注册策略概述", "query": "注册策略 目标市场 NMPA FDA CE 注册路径选择"},
        {"id": "ch2", "name": "适用标准清单", "query": "适用标准 GB9706 ISO13485 IEC62304 标准清单"},
        {"id": "ch3", "name": "注册时间表", "query": "注册时间表 里程碑 关键节点 资料准备"},
        {"id": "ch4", "name": "UDI实施策略", "query": "UDI 器械标识 生产标识 赋码方案"},
        {"id": "ch5", "name": "风险和应对", "query": "注册风险 法规变更 测试失败 应对措施"},
    ],

    # ================= 二、设计输入阶段 =================
    "design_input": [
        {"id": "ch1", "name": "概述", "query": "设计输入 目的 范围 产品描述 贴敷式胰岛素泵"},
        {"id": "ch2", "name": "功能性能要求", "query": "功能要求 性能要求 输注精度 报警 通信 技术参数"},
        {"id": "ch3", "name": "法规标准要求", "query": "法规标准 强制标准 推荐标准 GB9706 ISO13485 IEC62304"},
        {"id": "ch4", "name": "风险管理要求", "query": "风险管理 风险控制措施 安全性要求 危害分析"},
        {"id": "ch5", "name": "使用环境要求", "query": "使用环境 家庭护理 温度 湿度 电磁 防水 跌落"},
        {"id": "ch6", "name": "设计输入评审", "query": "设计输入评审 评审记录 批准 可追溯性"},
    ],
    "software_requirements_spec": [
        {"id": "ch1", "name": "概述", "query": "软件需求 概述 IEC62304 安全等级C 贴敷式胰岛素泵"},
        {"id": "ch2", "name": "功能需求", "query": "输注控制 CGM联动 报警管理 数据记录 功能需求"},
        {"id": "ch3", "name": "性能需求", "query": "响应时间 数据精度 存储容量 处理速度"},
        {"id": "ch4", "name": "接口需求", "query": "BLE通信协议 App API 固件OTA 传感器接口"},
        {"id": "ch5", "name": "安全需求", "query": "故障检测 冗余校验 安全状态机 软件安全"},
        {"id": "ch6", "name": "软件需求评审", "query": "需求评审 可追溯性 验证方法"},
    ],
    "hardware_requirements_spec": [
        {"id": "ch1", "name": "概述", "query": "硬件需求 概述 系统架构 贴敷式胰岛素泵"},
        {"id": "ch2", "name": "电路设计需求", "query": "电机驱动 传感器接口 电源管理 MCU选型"},
        {"id": "ch3", "name": "机械结构需求", "query": "泵体尺寸 储药器接口 贴敷面设计 外壳材料"},
        {"id": "ch4", "name": "电池需求", "query": "电池容量 充放电管理 安全保护 续航时间"},
        {"id": "ch5", "name": "传感器需求", "query": "压力传感器 温度传感器 位置传感器 精度"},
    ],
    "ui_requirements_spec": [
        {"id": "ch1", "name": "概述", "query": "UI需求 概述 IEC62366 贴敷式胰岛素泵用户界面"},
        {"id": "ch2", "name": "设备端UI需求", "query": "LED指示 按钮操作 声音报警 振动反馈"},
        {"id": "ch3", "name": "App端UI需求", "query": "输注状态 剂量设定 报警信息 历史数据 图表"},
        {"id": "ch4", "name": "可访问性需求", "query": "视力障碍 老年人 大字体 高对比度 语音辅助"},
        {"id": "ch5", "name": "UI需求评审", "query": "UI评审 可用性 用户反馈"},
    ],
    "cybersecurity_requirements": [
        {"id": "ch1", "name": "概述", "query": "网络安全 概述 IEC81001 贴敷式胰岛素泵 无线通信"},
        {"id": "ch2", "name": "通信安全需求", "query": "BLE加密 数据完整性 中间人攻击防护 配对认证"},
        {"id": "ch3", "name": "数据安全需求", "query": "患者数据保护 存储加密 传输加密 访问控制"},
        {"id": "ch4", "name": "固件安全需求", "query": "固件签名校验 OTA安全更新 回滚保护 防篡改"},
        {"id": "ch5", "name": "安全事件响应需求", "query": "安全监控 漏洞管理 安全更新机制"},
    ],
    "labeling_ifu_requirements": [
        {"id": "ch1", "name": "概述", "query": "标签说明书需求 ISO20417 ISO15223 法规要求"},
        {"id": "ch2", "name": "标签符号需求", "query": "标签符号 灭菌标识 UDI 制造商信息 警告标识"},
        {"id": "ch3", "name": "说明书内容需求", "query": "适应症 禁忌症 使用步骤 警示 维护 技术参数"},
        {"id": "ch4", "name": "多语言需求", "query": "多语言 中文 英文 出口市场 翻译验证"},
    ],
    "essential_principles_checklist": [
        {"id": "ch1", "name": "概述", "query": "医疗器械安全性能基本原则 GHTF EP清单 NMPA 贴敷式胰岛素泵"},
        {"id": "ch2", "name": "通用安全和性能要求", "query": "医疗器械通用安全要求 基本性能 风险管理 临床评价 生物学评价"},
        {"id": "ch3", "name": "化学物理和生物学特性", "query": "材料特性 生物相容性 药物相容性 可沥滤物 贴敷式胰岛素泵"},
        {"id": "ch4", "name": "感染和微生物污染", "query": "无菌保证 微生物污染控制 EO灭菌 包装完整性"},
        {"id": "ch5", "name": "有源医疗器械安全", "query": "电气安全 GB9706 电磁兼容 报警系统 软件安全 网络安全"},
        {"id": "ch6", "name": "标签和使用信息", "query": "标签要求 使用说明书 符号标识 UDI 语言要求"},
        {"id": "ch7", "name": "符合性声明", "query": "符合性判定 适用条款 不适用条款理由 证据索引"},
    ],
    "software_development_plan_di": [
        {"id": "ch1", "name": "目的和范围", "query": "软件计划 目的范围 IEC62304 安全等级C"},
        {"id": "ch2", "name": "软件开发生命周期模型", "query": "生命周期模型 V模型 敏捷 阶段划分"},
        {"id": "ch3", "name": "软件配置管理", "query": "配置管理 版本控制 基线管理 工具"},
        {"id": "ch4", "name": "软件验证计划", "query": "单元测试 集成测试 系统测试 验证策略"},
        {"id": "ch5", "name": "交付物清单", "query": "软件交付物 文档清单 IEC62304要求"},
    ],

    # ================= 三、设计输出阶段 =================
    "product_drawings": [
        {"id": "ch1", "name": "3D模型", "query": "3D模型 泵体 储药器 贴敷底座 装配关系"},
        {"id": "ch2", "name": "2D工程图", "query": "工程图 尺寸公差 GD&T 关键尺寸 装配图"},
        {"id": "ch3", "name": "材料标注", "query": "材料 表面处理 颜色 供应商 规格"},
        {"id": "ch4", "name": "图纸审批记录", "query": "审批 版本 变更记录"},
    ],
    "bill_of_materials": [
        {"id": "ch1", "name": "BOM结构", "query": "BOM 层次结构 成品 组件 零件 原材料"},
        {"id": "ch2", "name": "物料明细", "query": "物料编码 名称 规格 数量 供应商 关键物料"},
        {"id": "ch3", "name": "关键物料标识", "query": "安全件 法规件 关键特性 可追溯性"},
        {"id": "ch4", "name": "BOM审批和变更", "query": "BOM审批 版本管理 变更历史"},
    ],
    "product_specification": [
        {"id": "ch1", "name": "产品概述", "query": "贴敷式胰岛素泵 产品规格 概述"},
        {"id": "ch2", "name": "物理规格", "query": "尺寸 重量 材料 外观 颜色"},
        {"id": "ch3", "name": "性能规格", "query": "输注精度 流量范围 基础率 大剂量 精度"},
        {"id": "ch4", "name": "电气规格", "query": "电池容量 工作电压 功耗 充电参数"},
        {"id": "ch5", "name": "环境规格", "query": "工作温度 存储温度 湿度 防水等级 防护等级"},
    ],
    "software_architecture_doc": [
        {"id": "ch1", "name": "架构概述", "query": "软件架构 IEC62304 安全等级C 系统架构"},
        {"id": "ch2", "name": "模块划分", "query": "模块 组件 输注控制 CGM通信 报警管理 UI"},
        {"id": "ch3", "name": "接口定义", "query": "API 数据流 控制流 模块间通信"},
        {"id": "ch4", "name": "安全架构", "query": "故障隔离 冗余设计 安全监控 安全状态"},
        {"id": "ch5", "name": "部署架构", "query": "嵌入式固件 移动App 云服务 部署方案"},
    ],
    "software_detailed_design": [
        {"id": "ch1", "name": "设计概述", "query": "详细设计 IEC62304 模块设计 概述"},
        {"id": "ch2", "name": "核心算法设计", "query": "输注控制算法 CGM联动算法 报警决策 状态机"},
        {"id": "ch3", "name": "数据结构定义", "query": "数据结构 数据库 日志存储 配置数据"},
        {"id": "ch4", "name": "接口详细设计", "query": "API规范 通信协议 数据格式 时序"},
        {"id": "ch5", "name": "异常处理设计", "query": "异常处理 故障检测 恢复机制 降级模式"},
    ],
    "hardware_design_doc": [
        {"id": "ch1", "name": "设计概述", "query": "硬件设计 概述 系统框图 贴敷式胰岛素泵"},
        {"id": "ch2", "name": "电路原理图", "query": "原理图 电机驱动 电源管理 MCU 传感器接口"},
        {"id": "ch3", "name": "PCB设计", "query": "PCB布局 布线 叠层 电磁兼容设计"},
        {"id": "ch4", "name": "元器件选型", "query": "元器件选型 计算 关键参数 供应商"},
        {"id": "ch5", "name": "热设计", "query": "热设计 功耗预算 温升分析 散热方案"},
    ],
    "firmware_design_doc": [
        {"id": "ch1", "name": "固件架构", "query": "固件架构 RTOS 任务划分 中断处理"},
        {"id": "ch2", "name": "电机控制设计", "query": "步进电机 微步控制 PWM 位置反馈"},
        {"id": "ch3", "name": "传感器数据采集", "query": "压力传感器 温度传感器 信号调理 ADC"},
        {"id": "ch4", "name": "低功耗设计", "query": "低功耗 睡眠模式 唤醒机制 功耗优化"},
        {"id": "ch5", "name": "故障检测和安全", "query": "故障检测 看门狗 安全状态 错误处理"},
    ],
    "industrial_design_doc": [
        {"id": "ch1", "name": "外观设计", "query": "外观设计 壳体造型 轮廓 曲面 贴敷式胰岛素泵"},
        {"id": "ch2", "name": "CMF设计", "query": "颜色 材质 表面处理 医疗级材料"},
        {"id": "ch3", "name": "人因工程", "query": "贴敷舒适度 更换便利性 按键布局 指示灯"},
        {"id": "ch4", "name": "佩戴体验设计", "query": "人体工学 皮肤贴合 运动自由度 隐蔽性"},
    ],
    "label_nameplate_artwork": [
        {"id": "ch1", "name": "产品标签设计", "query": "标签 型号 批号 有效期 灭菌标识 UDI 符号"},
        {"id": "ch2", "name": "包装标签设计", "query": "包装标签 运输条件 存储条件 条码"},
        {"id": "ch3", "name": "法规符合性", "query": "ISO15223 NMPA FDA 标签法规 符号 格式"},
    ],
    "packaging_design_artwork": [
        {"id": "ch1", "name": "初级包装设计", "query": "无菌屏障 吸塑盒 涂胶盖材 密封设计"},
        {"id": "ch2", "name": "次级包装设计", "query": "彩盒 印刷内容 产品信息 品牌元素"},
        {"id": "ch3", "name": "运输包装设计", "query": "外箱 缓冲材料 堆码标识 运输标识"},
    ],
    "instruction_for_use": [
        {"id": "ch1", "name": "产品信息", "query": "贴敷式胰岛素泵 产品名称 型号 生产企业 结构组成"},
        {"id": "ch2", "name": "适用范围", "query": "适用范围 适应症 禁忌症 适用人群"},
        {"id": "ch3", "name": "使用步骤", "query": "贴敷步骤 储药器填充 启动 输注设定 更换 废弃"},
        {"id": "ch4", "name": "警示和注意事项", "query": "警示 警告 注意事项 并发症 不良事件"},
        {"id": "ch5", "name": "维护和故障排除", "query": "清洁 存储 故障现象 处理方法 客服联系"},
        {"id": "ch6", "name": "技术参数和符号", "query": "技术参数 符号说明 EMC 无线 电磁兼容说明"},
    ],
    "manufacturing_process_draft": [
        {"id": "ch1", "name": "组装工艺流程", "query": "组装流程 泵体组装 储药器组装 贴敷底座 工艺流程"},
        {"id": "ch2", "name": "调试和校准", "query": "输注精度校准 通信测试 功能检查 整机调试"},
        {"id": "ch3", "name": "包装工艺流程", "query": "内包 外包 灭菌前准备 包装工序"},
        {"id": "ch4", "name": "关键工艺参数", "query": "关键参数 控制范围 监控方法 工艺能力"},
    ],

    # ================= 四、设计验证阶段 =================
    "design_verification_master_plan": [
        {"id": "ch1", "name": "目的和范围", "query": "设计验证 总计划 ISO13485 贴敷式胰岛素泵"},
        {"id": "ch2", "name": "验证矩阵", "query": "验证项目 验证方法 接收标准 可追溯性矩阵"},
        {"id": "ch3", "name": "验证时间表", "query": "时间安排 资源 样品量 优先顺序"},
        {"id": "ch4", "name": "验证管理", "query": "职责分配 不符合项处理 报告模板 审批流程"},
    ],
    "electrical_safety_test_report": [
        {"id": "ch1", "name": "测试概述", "query": "电气安全 GB9706.1 测试目的 测试样品"},
        {"id": "ch2", "name": "测试项目和方法", "query": "耐压 漏电流 接地阻抗 温升 绝缘电阻"},
        {"id": "ch3", "name": "测试数据和结果", "query": "测试数据 限值 判定 充电模式 电池模式"},
        {"id": "ch4", "name": "结论", "query": "测试结论 符合性声明 不符合项"},
    ],
    "emc_test_report": [
        {"id": "ch1", "name": "测试概述", "query": "EMC测试 YY9706.102 测试目的 测试配置"},
        {"id": "ch2", "name": "发射测试", "query": "辐射发射 传导发射 限值 测试数据"},
        {"id": "ch3", "name": "抗扰度测试", "query": "ESD 辐射抗扰度 传导抗扰度 EFT 浪涌 测试数据"},
        {"id": "ch4", "name": "特殊考量", "query": "BLE通信 EMC考量 无线共存"},
        {"id": "ch5", "name": "结论", "query": "测试结论 符合性 不符合项"},
    ],
    "alarm_system_test_report": [
        {"id": "ch1", "name": "测试概述", "query": "报警测试 YY9706.108 报警系统 贴敷式胰岛素泵"},
        {"id": "ch2", "name": "报警优先级测试", "query": "高优先级 阻塞 低电量 中优先级 低优先级"},
        {"id": "ch3", "name": "声光特性测试", "query": "声压级 频率 脉冲模式 指示灯特性"},
        {"id": "ch4", "name": "智能报警测试", "query": "报警延迟 报警确认 报警升级 报警日志"},
        {"id": "ch5", "name": "结论", "query": "测试结论 符合性"},
    ],
    "infusion_accuracy_test_report": [
        {"id": "ch1", "name": "测试概述", "query": "输注精度 GB9706.224 测试目的 测试装置"},
        {"id": "ch2", "name": "基础率精度测试", "query": "基础率 不同速率 误差 流量 称重法"},
        {"id": "ch3", "name": "大剂量精度测试", "query": "Bolus 不同剂量 误差 重复性"},
        {"id": "ch4", "name": "阻塞检测测试", "query": "阻塞检测 响应时间 不同背压 报警触发"},
        {"id": "ch5", "name": "长期稳定性测试", "query": "长时间运行 输注一致性 温度影响"},
        {"id": "ch6", "name": "结论", "query": "精度结论 符合性 不确定度分析"},
    ],
    "software_unit_test_report": [
        {"id": "ch1", "name": "测试概述", "query": "单元测试 IEC62304 测试策略 测试环境"},
        {"id": "ch2", "name": "测试用例和结果", "query": "测试用例 测试数据 期望输出 实际结果"},
        {"id": "ch3", "name": "代码覆盖率", "query": "语句覆盖 分支覆盖 MC/DC 覆盖率统计"},
        {"id": "ch4", "name": "安全单元测试", "query": "安全相关单元 边界条件 异常输入 深度测试"},
        {"id": "ch5", "name": "结论", "query": "测试结论 未覆盖项 残留风险"},
    ],
    "software_integration_test_report": [
        {"id": "ch1", "name": "测试概述", "query": "集成测试 IEC62304 集成策略 测试环境"},
        {"id": "ch2", "name": "模块接口测试", "query": "接口测试 数据传输 时序 协议 正确性"},
        {"id": "ch3", "name": "软硬件集成测试", "query": "传感器读取 电机控制 通信协议栈 集成"},
        {"id": "ch4", "name": "回归测试", "query": "回归测试 变更影响 自动化测试"},
        {"id": "ch5", "name": "结论", "query": "集成测试结论 接口问题 解决方案"},
    ],
    "software_system_test_report": [
        {"id": "ch1", "name": "测试概述", "query": "系统测试 IEC62304 黑盒测试 测试环境"},
        {"id": "ch2", "name": "功能测试", "query": "功能测试 需求覆盖 可追溯性矩阵 测试结果"},
        {"id": "ch3", "name": "性能测试", "query": "响应时间 并发处理 长时间运行 性能数据"},
        {"id": "ch4", "name": "异常测试", "query": "异常条件 边界条件 压力测试 容错性"},
        {"id": "ch5", "name": "结论", "query": "系统测试结论 发布建议"},
    ],
    "battery_performance_test_report": [
        {"id": "ch1", "name": "测试概述", "query": "电池测试 IEC62133 UL1642 测试目的"},
        {"id": "ch2", "name": "续航测试", "query": "续航时间 不同模式 基础率 大剂量 蓝牙"},
        {"id": "ch3", "name": "充放电测试", "query": "充电循环 容量衰减 充电效率 放电特性"},
        {"id": "ch4", "name": "安全测试", "query": "过充 过放 短路 高温 安全保护"},
        {"id": "ch5", "name": "低电量测试", "query": "低电量报警 自动关机 数据保存"},
        {"id": "ch6", "name": "结论", "query": "电池测试结论 寿命预估"},
    ],
    "ip_rating_test_report": [
        {"id": "ch1", "name": "测试概述", "query": "IP防护 IEC60529 防水防尘 测试条件"},
        {"id": "ch2", "name": "测试执行", "query": "测试方法 测试装置 持续时间 试验过程"},
        {"id": "ch3", "name": "测试结果", "query": "进水检查 功能检查 外观检查 测试数据"},
        {"id": "ch4", "name": "结论", "query": "IP等级结论 符合性"},
    ],
    "environmental_reliability_test": [
        {"id": "ch1", "name": "测试概述", "query": "环境试验 GB/T14710 测试目的 测试项目"},
        {"id": "ch2", "name": "气候环境测试", "query": "低温 高温 湿热 温度循环 温度冲击"},
        {"id": "ch3", "name": "机械环境测试", "query": "振动 跌落 冲击 碰撞"},
        {"id": "ch4", "name": "测试结果汇总", "query": "测试前后对比 功能检查 外观检查"},
        {"id": "ch5", "name": "结论", "query": "环境可靠性结论"},
    ],
    "package_transport_test_report": [
        {"id": "ch1", "name": "测试概述", "query": "运输包装 GB/T4857 ASTM D4169 测试方案"},
        {"id": "ch2", "name": "跌落测试", "query": "自由跌落 不同高度 不同方向 角 棱 面"},
        {"id": "ch3", "name": "堆码和振动测试", "query": "堆码 压缩 随机振动 正弦振动"},
        {"id": "ch4", "name": "测试后检查", "query": "包装完整性 产品功能 外观 密封"},
        {"id": "ch5", "name": "结论", "query": "运输包装结论"},
    ],
    "seal_integrity_test_report": [
        {"id": "ch1", "name": "测试概述", "query": "密封完整性 ISO11607 YY/T0681 测试目的"},
        {"id": "ch2", "name": "物理测试", "query": "染色渗透 爆破强度 拉伸测试"},
        {"id": "ch3", "name": "无损测试", "query": "真空衰减 压力衰减 目视检查"},
        {"id": "ch4", "name": "灭菌前后对比", "query": "灭菌前 灭菌后 老化后 密封对比"},
        {"id": "ch5", "name": "结论", "query": "密封完整性结论 接受标准"},
    ],
    "material_characterization_report": [
        {"id": "ch1", "name": "测试概述", "query": "材料表征 ISO10993-18 测试目的 材料清单"},
        {"id": "ch2", "name": "物理特性", "query": "拉伸强度 硬度 密度 熔融指数"},
        {"id": "ch3", "name": "化学特性", "query": "成分分析 溶出物 可沥滤物 残留单体"},
        {"id": "ch4", "name": "特殊材料测试", "query": "贴片胶带 粘性 透气性 皮肤相容性"},
        {"id": "ch5", "name": "结论", "query": "材料表征结论 毒理学评估"},
    ],
    "accelerated_aging_test_report": [
        {"id": "ch1", "name": "测试概述", "query": "加速老化 ASTM F1980 YY/T0681.1 老化方案"},
        {"id": "ch2", "name": "老化条件", "query": "温度 湿度 时间 加速因子 Q10"},
        {"id": "ch3", "name": "老化后性能测试", "query": "输注精度 密封完整性 电池 外观 功能"},
        {"id": "ch4", "name": "有效期确定", "query": "有效期结论 安全系数 实时老化关联"},
        {"id": "ch5", "name": "结论", "query": "加速老化结论"},
    ],
    "sensor_calibration_validation": [
        {"id": "ch1", "name": "验证概述", "query": "传感器校准 压力传感器 位置传感器 温度传感器"},
        {"id": "ch2", "name": "校准方法", "query": "校准方法 标准器 校准环境 校准频率"},
        {"id": "ch3", "name": "校准数据", "query": "校准曲线 线性度 重复性 迟滞 漂移"},
        {"id": "ch4", "name": "结论", "query": "校准验证结论 不确定度"},
    ],
    "cybersecurity_test_report": [
        {"id": "ch1", "name": "测试概述", "query": "网络安全测试 IEC81001 测试范围 测试方法"},
        {"id": "ch2", "name": "渗透测试", "query": "BLE渗透 App渗透 固件渗透 渗透结果"},
        {"id": "ch3", "name": "加密验证", "query": "通信加密 存储加密 密钥管理 证书"},
        {"id": "ch4", "name": "固件安全测试", "query": "固件签名 防篡改 安全启动 OTA安全"},
        {"id": "ch5", "name": "漏洞评估", "query": "漏洞扫描 风险评估 缓解措施 残留风险"},
        {"id": "ch6", "name": "结论", "query": "网络安全测试结论"},
    ],

    # ================= 五、设计确认阶段 =================
    "design_validation_master_plan": [
        {"id": "ch1", "name": "目的和范围", "query": "设计确认 总计划 ISO13485 确认活动总览"},
        {"id": "ch2", "name": "可用性确认计划", "query": "可用性 总结性评价 真实用户 模拟场景"},
        {"id": "ch3", "name": "临床确认计划", "query": "临床评价 临床试验 CER 临床数据"},
        {"id": "ch4", "name": "生物学和灭菌确认", "query": "生物学评价 灭菌确认 SAL 残留"},
        {"id": "ch5", "name": "确认时间表", "query": "时间安排 资源 依赖关系"},
    ],
    "summative_usability_report": [
        {"id": "ch1", "name": "报告概述", "query": "总结性可用性 IEC62366 测试目的 测试场景"},
        {"id": "ch2", "name": "测试方法", "query": "参与者 测试场景 任务列表 数据收集"},
        {"id": "ch3", "name": "测试结果", "query": "任务完成率 错误率 时间 满意度 根本原因"},
        {"id": "ch4", "name": "分析和改进", "query": "使用错误分析 设计改进 人因工程"},
        {"id": "ch5", "name": "结论", "query": "可用性结论 安全性 有效性"},
    ],
    "clinical_evaluation_report": [
        {"id": "ch1", "name": "评价概述", "query": "临床评价 MEDDEV ISO14155 评价范围"},
        {"id": "ch2", "name": "文献检索和评价", "query": "文献检索策略 文献筛选 质量评价 数据提取"},
        {"id": "ch3", "name": "临床数据分析", "query": "安全性 有效性 性能 等效器械对比"},
        {"id": "ch4", "name": "受益-风险评价", "query": "受益评估 风险评估 综合结论"},
        {"id": "ch5", "name": "上市后要求", "query": "PMCF计划 数据缺口 后续研究"},
        {"id": "ch6", "name": "结论", "query": "临床评价结论 符合性声明"},
    ],
    "biological_evaluation_report": [
        {"id": "ch1", "name": "评价概述", "query": "生物学评价 ISO10993-1 接触性质 材料清单"},
        {"id": "ch2", "name": "化学表征总结", "query": "材料成分 溶出物 可沥滤物 化学表征"},
        {"id": "ch3", "name": "生物学试验总结", "query": "细胞毒性 刺激 致敏 血液相容性 试验结果"},
        {"id": "ch4", "name": "毒理学评估", "query": "毒理学 安全限值 暴露量 风险评估"},
        {"id": "ch5", "name": "结论", "query": "生物学评价结论 生物学安全性"},
    ],
    "cytotoxicity_test_report": [
        {"id": "ch1", "name": "测试概述", "query": "细胞毒性 ISO10993-5 试验目的 材料"},
        {"id": "ch2", "name": "试验方法", "query": "浸提液制备 细胞系 培养条件 评价方法"},
        {"id": "ch3", "name": "试验结果", "query": "定性结果 定量结果 细胞活力 形态"},
        {"id": "ch4", "name": "结论", "query": "细胞毒性结论 符合性"},
    ],
    "skin_irritation_sensitization": [
        {"id": "ch1", "name": "测试概述", "query": "皮肤刺激 致敏 ISO10993-10 测试目的"},
        {"id": "ch2", "name": "皮肤刺激试验", "query": "刺激试验 方法 动物模型 评分系统 结果"},
        {"id": "ch3", "name": "致敏试验", "query": "致敏试验 GPMT LLNA 方法 结果"},
        {"id": "ch4", "name": "结论", "query": "皮肤刺激和致敏结论"},
    ],
    "blood_compatibility_test_report": [
        {"id": "ch1", "name": "测试概述", "query": "血液相容性 ISO10993-4 测试目的 适用性"},
        {"id": "ch2", "name": "溶血试验", "query": "溶血率 直接接触 间接接触 试验结果"},
        {"id": "ch3", "name": "凝血试验", "query": "PTT PT 凝血激活 血小板激活"},
        {"id": "ch4", "name": "结论", "query": "血液相容性结论"},
    ],
    "sal_validation_report": [
        {"id": "ch1", "name": "验证概述", "query": "SAL验证 ISO11135 ISO11137 灭菌方法"},
        {"id": "ch2", "name": "IQ/OQ", "query": "安装确认 运行确认 设备验证"},
        {"id": "ch3", "name": "PQ", "query": "性能确认 生物负载 半周期法 SAL 10-6"},
        {"id": "ch4", "name": "结论", "query": "SAL验证结论 灭菌保证水平"},
    ],
    "sterilization_residue_test_report": [
        {"id": "ch1", "name": "测试概述", "query": "灭菌残留 ISO10993-7 EO残留 残留测试"},
        {"id": "ch2", "name": "EO残留测试", "query": "EO残留 ECH残留 测试方法 气相色谱"},
        {"id": "ch3", "name": "残留限值评估", "query": "安全限值 ISO10993-7 暴露评估"},
        {"id": "ch4", "name": "结论", "query": "残留测试结论"},
    ],
    "process_validation_iq_oq_pq": [
        {"id": "ch1", "name": "验证概述", "query": "过程确认 IQ OQ PQ 关键工艺"},
        {"id": "ch2", "name": "安装确认(IQ)", "query": "设备安装 校准 文件 公用设施"},
        {"id": "ch3", "name": "运行确认(OQ)", "query": "工艺参数 操作范围 挑战测试 最差条件"},
        {"id": "ch4", "name": "性能确认(PQ)", "query": "连续批次 一致性 能力指数 稳定性"},
        {"id": "ch5", "name": "结论", "query": "过程确认结论"},
    ],
    "cleaning_validation_report": [
        {"id": "ch1", "name": "验证概述", "query": "清洗验证 ISO15883 清洗目的"},
        {"id": "ch2", "name": "清洗工艺", "query": "清洗剂 温度 时间 方式 参数"},
        {"id": "ch3", "name": "验证结果", "query": "残留限值 粒子计数 微生物 内毒素"},
        {"id": "ch4", "name": "结论", "query": "清洗验证结论"},
    ],

    # ================= 六、设计转化阶段 =================
    "design_transfer_plan": [
        {"id": "ch1", "name": "目的和范围", "query": "设计转化 ISO13485 7.3.8 转化计划"},
        {"id": "ch2", "name": "交接条件", "query": "交接条件 检查清单 研发输出 生产接收"},
        {"id": "ch3", "name": "人员培训", "query": "培训计划 培训内容 考核 人员资质"},
        {"id": "ch4", "name": "设备和工装", "query": "生产设备 工装 模具 检验设备 采购"},
        {"id": "ch5", "name": "供应商准备", "query": "供应商确认 物料准备 首批供应"},
    ],
    "design_transfer_report": [
        {"id": "ch1", "name": "转化概述", "query": "设计转化 报告 转化过程 时间节点"},
        {"id": "ch2", "name": "输出转化确认", "query": "图纸 BOM 工艺文件 检验规范 转化确认"},
        {"id": "ch3", "name": "过程验证状态", "query": "IQ OQ PQ 验证状态 工艺能力"},
        {"id": "ch4", "name": "首批生产评估", "query": "首批生产 良率 问题 改进"},
        {"id": "ch5", "name": "结论", "query": "设计转化结论 审批"},
    ],
    "device_master_record": [
        {"id": "ch1", "name": "DMR概述", "query": "DMR ISO13485 4.2.3 产品主记录"},
        {"id": "ch2", "name": "产品规范", "query": "规格书 图纸 BOM 标签 包装规范"},
        {"id": "ch3", "name": "生产工艺规范", "query": "组装SOP 包装SOP 灭菌规范 工艺参数"},
        {"id": "ch4", "name": "检验规范", "query": "进货检验 过程检验 成品检验 接收标准"},
        {"id": "ch5", "name": "版本管理", "query": "版本 变更历史 分发控制"},
    ],
    "manufacturing_sop": [
        {"id": "ch1", "name": "目的和范围", "query": "生产工艺SOP 目的 范围 引用文件"},
        {"id": "ch2", "name": "职责和资质", "query": "职责 岗位 资质要求 培训要求"},
        {"id": "ch3", "name": "组装操作步骤", "query": "组装 泵体 储药器 电池 贴敷底座 操作步骤"},
        {"id": "ch4", "name": "灌装和灭菌准备", "query": "灌装 灭菌前准备 装载 参数设置"},
        {"id": "ch5", "name": "质量控制点", "query": "关键控制点 检查项目 参数监控 记录"},
        {"id": "ch6", "name": "安全注意事项", "query": "安全 个人防护 设备安全 环境安全"},
    ],
    "inspection_sop": [
        {"id": "ch1", "name": "目的和范围", "query": "检验SOP 目的 范围 引用标准"},
        {"id": "ch2", "name": "进货检验", "query": "电机 传感器 储药器 胶带 电池 检验项目 抽样"},
        {"id": "ch3", "name": "过程检验", "query": "组装检验 功能测试 通信测试 外观检查"},
        {"id": "ch4", "name": "成品检验", "query": "最终性能 输注精度 密封 包装 标签 接收标准"},
        {"id": "ch5", "name": "检验设备和记录", "query": "检验设备 校准 记录表单 填写要求"},
    ],
    "equipment_operation_procedures": [
        {"id": "ch1", "name": "概述", "query": "设备操作规程 范围 适用设备"},
        {"id": "ch2", "name": "生产设备操作", "query": "点胶机 焊接 组装工装 操作步骤 维护"},
        {"id": "ch3", "name": "检验设备操作", "query": "流量测试台 拉力机 泄漏仪 操作步骤 校准"},
        {"id": "ch4", "name": "安全操作", "query": "安全操作 紧急停机 防护 事故处理"},
    ],
    "work_environment_control_doc": [
        {"id": "ch1", "name": "目的和范围", "query": "工作环境 ISO13485 6.4 洁净室 控制"},
        {"id": "ch2", "name": "洁净室管理", "query": "洁净室 级别 温湿度 压差 粒子数 微生物"},
        {"id": "ch3", "name": "ESD控制", "query": "ESD 接地 电离器 腕带 服装"},
        {"id": "ch4", "name": "人员卫生", "query": "更衣 洗手 健康 行为规范"},
        {"id": "ch5", "name": "监测记录", "query": "环境监测 频率 方法 记录 超标处理"},
    ],

    # ================= 七、生产制造阶段 =================
    "batch_production_record": [
        {"id": "ch1", "name": "批次信息", "query": "批次 批号 生产日期 产量 操作人员"},
        {"id": "ch2", "name": "工序操作记录", "query": "组装 灌装 包装 各工序 操作记录 参数"},
        {"id": "ch3", "name": "物料使用记录", "query": "物料编码 批号 用量 追溯"},
        {"id": "ch4", "name": "设备和偏差记录", "query": "设备编号 运行参数 偏差 异常处理"},
    ],
    "batch_inspection_record": [
        {"id": "ch1", "name": "批次信息", "query": "批次 批号 检验日期 检验人员"},
        {"id": "ch2", "name": "进货检验", "query": "原材料 外购件 检验项目 数据 判定"},
        {"id": "ch3", "name": "过程检验", "query": "各工序 检验数据 判定"},
        {"id": "ch4", "name": "成品检验", "query": "性能 外观 包装 检验数据 最终判定"},
    ],
    "device_history_record": [
        {"id": "ch1", "name": "DHR概述", "query": "DHR ISO13485 4.2.5 批次汇总"},
        {"id": "ch2", "name": "生产记录汇总", "query": "BPR 关键工序 参数 操作人"},
        {"id": "ch3", "name": "检验记录汇总", "query": "进货 过程 成品 检验结果 判定"},
        {"id": "ch4", "name": "灭菌和UDI", "query": "灭菌记录 UDI清单 放行批准"},
    ],
    "incoming_inspection_records": [
        {"id": "ch1", "name": "记录概述", "query": "进货检验 记录 检验计划 抽样方案"},
        {"id": "ch2", "name": "检验记录明细", "query": "物料 批次 检验项目 数据 接收标准 判定"},
        {"id": "ch3", "name": "不合格品处理", "query": "不合格物料 数量 处理方式 供方通知"},
    ],
    "supplier_audit_reports": [
        {"id": "ch1", "name": "审核概述", "query": "供应商审核 目的 范围 审核组 审核计划"},
        {"id": "ch2", "name": "审核发现", "query": "检查表 符合项 不符合项 观察项 证据"},
        {"id": "ch3", "name": "审核结论", "query": "审核结论 供应商评级 整改要求 跟进"},
    ],
    "calibration_records": [
        {"id": "ch1", "name": "校准计划", "query": "校准计划 设备清单 校准周期 责任"},
        {"id": "ch2", "name": "校准证书", "query": "校准结果 不确定度 有效期 标准器"},
        {"id": "ch3", "name": "不合格处理", "query": "校准不合格 追溯 影响评估 纠正措施"},
    ],
    "environmental_monitoring_records": [
        {"id": "ch1", "name": "监测计划", "query": "环境监测 计划 频率 监测点 项目"},
        {"id": "ch2", "name": "监测数据", "query": "温度 湿度 压差 粒子 微生物 数据记录"},
        {"id": "ch3", "name": "超标处理", "query": "超标 调查 纠正措施 再监测"},
    ],
    "nonconforming_product_records": [
        {"id": "ch1", "name": "不合格品记录", "query": "不合格品 描述 发现 数量 位置 时间"},
        {"id": "ch2", "name": "评审和处置", "query": "评审结论 让步 返工 报废 处置 责任人"},
        {"id": "ch3", "name": "根本原因和CAPA", "query": "根本原因 纠正措施 预防措施 验证"},
    ],
    "sterilization_process_records": [
        {"id": "ch1", "name": "灭菌批次记录", "query": "灭菌批次 产品批号 灭菌日期 操作人员"},
        {"id": "ch2", "name": "灭菌参数", "query": "温度 EO浓度 时间 湿度 压力 剂量"},
        {"id": "ch3", "name": "指示剂结果", "query": "生物指示剂 化学指示剂 结果 判定"},
        {"id": "ch4", "name": "放行", "query": "灭菌放行 参数审核 批准"},
    ],
    "udi_assignment_records": [
        {"id": "ch1", "name": "UDI概述", "query": "UDI 赋码 YY/T1630 方案"},
        {"id": "ch2", "name": "UDI分配记录", "query": "UDI-DI UDI-PI 批次对应 码值"},
        {"id": "ch3", "name": "赋码质量", "query": "条码 二维码 打印质量 校验 抽查"},
    ],

    # ================= 八、注册申报阶段 =================
    "registration_dossier": [
        {"id": "ch1", "name": "注册申请概述", "query": "注册申请 产品 申请人 注册类型 III类"},
        {"id": "ch2", "name": "产品描述", "query": "贴敷式胰岛素泵 结构组成 原理 规格型号"},
        {"id": "ch3", "name": "性能和安全评价", "query": "性能指标 安全性 风险管理 验证确认"},
        {"id": "ch4", "name": "临床评价资料", "query": "临床评价 CER 临床试验 文献"},
        {"id": "ch5", "name": "质量体系文件", "query": "QMS 体系证明 生产许可证"},
        {"id": "ch6", "name": "标签和说明书", "query": "标签 说明书 IFU 符合性声明"},
    ],
    "product_technical_requirements": [
        {"id": "ch1", "name": "型号规格", "query": "型号 规格 结构组成 贴敷式胰岛素泵"},
        {"id": "ch2", "name": "性能指标", "query": "输注精度 报警 防水 通信 电气安全 性能指标"},
        {"id": "ch3", "name": "检验方法", "query": "检验方法 仪器 步骤 判定"},
        {"id": "ch4", "name": "标志包装运输贮存", "query": "标志 标签 包装 运输条件 贮存条件"},
    ],
    "risk_management_report": [
        {"id": "ch1", "name": "概述", "query": "贴敷式胰岛素泵 风险管理 概述 ISO14971"},
        {"id": "ch2", "name": "风险分析", "query": "危害识别 风险分析 安全性特征 贴敷式胰岛素泵"},
        {"id": "ch3", "name": "风险评价", "query": "风险评价 RPN 严重度 频度数 探测度"},
        {"id": "ch4", "name": "风险控制", "query": "风险控制措施 验证 实施"},
        {"id": "ch5", "name": "综合剩余风险评价", "query": "剩余风险 综合评价 受益-风险"},
        {"id": "ch6", "name": "风险管理评审结论", "query": "评审结论 批准 文档"},
    ],
    "essential_safety_conformity": [
        {"id": "ch1", "name": "声明概述", "query": "基本安全 基本性能 符合性声明 GB9706.1"},
        {"id": "ch2", "name": "逐条符合性检查", "query": "条款 要求 符合性 证据 说明"},
        {"id": "ch3", "name": "不适用条款说明", "query": "不适用条款 理由 合理性"},
        {"id": "ch4", "name": "结论", "query": "符合性结论 声明"},
    ],
    "software_version_description": [
        {"id": "ch1", "name": "版本概述", "query": "软件版本 版本号 发布日期 平台"},
        {"id": "ch2", "name": "版本内容", "query": "功能 变更 修复 新增"},
        {"id": "ch3", "name": "已知问题和限制", "query": "已知问题 限制 使用注意"},
        {"id": "ch4", "name": "软件配置清单", "query": "组件 版本 依赖 配置"},
    ],
    "cybersecurity_disclosure": [
        {"id": "ch1", "name": "概述", "query": "网络安全 披露 范围 设备描述"},
        {"id": "ch2", "name": "风险管理总结", "query": "威胁模型 风险评估 漏洞 缓解"},
        {"id": "ch3", "name": "安全更新策略", "query": "补丁管理 更新机制 验证"},
        {"id": "ch4", "name": "结论", "query": "网络安全结论 符合性"},
    ],
    "clinical_evaluation_report_reg": [
        {"id": "ch1", "name": "评价概述", "query": "临床评价 注册 MEDDEV 评价范围"},
        {"id": "ch2", "name": "文献和临床数据", "query": "文献检索 临床数据 等效器械"},
        {"id": "ch3", "name": "受益-风险评价", "query": "受益 风险 综合评价"},
        {"id": "ch4", "name": "结论", "query": "临床评价结论 安全性 有效性"},
    ],
    "ifu_labeling_for_registration": [
        {"id": "ch1", "name": "说明书审核稿", "query": "说明书 内容 使用步骤 警示 符号"},
        {"id": "ch2", "name": "标签审核稿", "query": "标签 UDI 灭菌标识 法规符号"},
        {"id": "ch3", "name": "法规符合性", "query": "NMPA FDA CE 符合性 多语言"},
    ],
    "declaration_of_conformity": [
        {"id": "ch1", "name": "声明信息", "query": "DoC EU MDR 制造商 产品 公告机构"},
        {"id": "ch2", "name": "符合的法规标准", "query": "法规 标准 清单 符合性"},
        {"id": "ch3", "name": "签署", "query": "签署人 职位 日期 签名"},
    ],
    "qms_certificate_proof": [
        {"id": "ch1", "name": "体系证书", "query": "ISO13485 证书 范围 有效期 认证机构"},
        {"id": "ch2", "name": "GMP证明", "query": "GMP检查 通过证明 检查日期"},
        {"id": "ch3", "name": "其他证书", "query": "MDSAP CE 其他体系认证"},
    ],

    # ================= 九、上市后阶段 =================
    "pms_plan": [
        {"id": "ch1", "name": "目的和范围", "query": "PMS Plan EU MDR Art.84 ISO TR 20416"},
        {"id": "ch2", "name": "数据收集方法", "query": "投诉 不良事件 文献 注册数据 社交媒体 数据源"},
        {"id": "ch3", "name": "数据分析和评价", "query": "数据分析 趋势 统计方法 评价准则"},
        {"id": "ch4", "name": "报告机制", "query": "PMS报告 编制周期 触发条件 分发"},
    ],
    "pmcf_plan": [
        {"id": "ch1", "name": "目的和范围", "query": "PMCF Plan EU MDR Annex XIV 临床跟踪"},
        {"id": "ch2", "name": "数据缺口识别", "query": "上市前 临床数据 不足 缺口 待补充"},
        {"id": "ch3", "name": "PMCF活动设计", "query": "临床研究 问卷 文献 注册数据 活动设计"},
        {"id": "ch4", "name": "时间表和职责", "query": "时间表 责任人 里程碑"},
    ],
    "complaint_handling_records": [
        {"id": "ch1", "name": "投诉登记", "query": "投诉人 产品信息 问题描述 日期 批号"},
        {"id": "ch2", "name": "调查过程", "query": "原因分析 检测 批次追溯 调查结论"},
        {"id": "ch3", "name": "回复和纠正", "query": "回复 纠正措施 监管报告 关闭"},
    ],
    "adverse_event_reports": [
        {"id": "ch1", "name": "不良事件描述", "query": "不良事件 类型 严重程度 器械关联性"},
        {"id": "ch2", "name": "产品信息", "query": "型号 批号 UDI 使用情况"},
        {"id": "ch3", "name": "处理措施", "query": "纠正措施 监管报告 时间表 跟踪"},
    ],
    "periodic_safety_update_report": [
        {"id": "ch1", "name": "报告概述", "query": "PSUR EU MDR Art.86 报告期 范围"},
        {"id": "ch2", "name": "安全性数据汇总", "query": "不良事件 投诉 文献 趋势 分析"},
        {"id": "ch3", "name": "受益-风险更新", "query": "受益 风险 综合评价 更新"},
        {"id": "ch4", "name": "结论和行动", "query": "问题识别 建议 行动 与既往比较"},
    ],
    "capa_records": [
        {"id": "ch1", "name": "CAPA来源", "query": "CAPA来源 投诉 不合格 审核 管理评审"},
        {"id": "ch2", "name": "根本原因分析", "query": "根本原因 5Why 鱼骨图 因果分析"},
        {"id": "ch3", "name": "纠正和预防措施", "query": "纠正措施 预防措施 实施计划 责任人"},
        {"id": "ch4", "name": "有效性验证", "query": "有效性验证 验证方法 验证结果"},
        {"id": "ch5", "name": "CAPA关闭", "query": "CAPA关闭 批准 日期 归档"},
    ],
    "change_control_records": [
        {"id": "ch1", "name": "变更描述", "query": "变更 描述 原因 变更前后状态"},
        {"id": "ch2", "name": "影响评估", "query": "安全 性能 法规 影响 风险评估"},
        {"id": "ch3", "name": "变更验证", "query": "验证 确认 测试 文档更新"},
        {"id": "ch4", "name": "审批和实施", "query": "审批 实施 追溯 变更通知"},
    ],
    "recall_field_safety_notice": [
        {"id": "ch1", "name": "召回决策", "query": "召回 决策 风险 等级 受影响 范围"},
        {"id": "ch2", "name": "召回行动计划", "query": "通知方式 回收流程 时间计划"},
        {"id": "ch3", "name": "监管报告", "query": "监管机构 报告 NMPA FDA 报告内容"},
        {"id": "ch4", "name": "召回归档", "query": "召回效果 评估 关闭 报告"},
    ],
    "management_review_report": [
        {"id": "ch1", "name": "评审概述", "query": "管理评审 ISO13485 5.6 评审周期"},
        {"id": "ch2", "name": "评审输入", "query": "质量目标 审核 CAPA 投诉 变更 改进"},
        {"id": "ch3", "name": "评审输出", "query": "改进措施 资源需求 质量方针修订"},
        {"id": "ch4", "name": "评审结论", "query": "体系有效性 适宜性 充分性 结论"},
    ],
    "internal_audit_reports": [
        {"id": "ch1", "name": "审核概述", "query": "内部审核 ISO13485 8.2.4 审核计划"},
        {"id": "ch2", "name": "审核发现", "query": "符合项 不符合项 观察项 改进机会"},
        {"id": "ch3", "name": "审核结论", "query": "体系有效性 不符合项 纠正措施 跟踪"},
    ],
    "pms_report": [
        {"id": "ch1", "name": "报告概述", "query": "PMS Report EU MDR Art.85 报告期"},
        {"id": "ch2", "name": "数据汇总和分析", "query": "PMS数据 汇总 趋势 统计 分析"},
        {"id": "ch3", "name": "受益-风险结论", "query": "受益 风险 综合评价 更新"},
        {"id": "ch4", "name": "后续行动", "query": "发现 建议 后续行动"},
    ],

    # ================= 保留原有兼容类型 =================
    "fmea_analysis": [
        {"id": "ch1", "name": "FMEA概述", "query": "FMEA概述 目的 范围 方法 贴敷式胰岛素泵"},
        {"id": "ch2", "name": "严重度频度数探测度评价准则", "query": "严重度 频度数 探测度 评价准则 评分标准"},
        {"id": "ch3", "name": "FMEA分析表", "query": "FMEA分析表 失效模式 失效效应 子系统"},
        {"id": "ch4", "name": "高风险项目改进措施", "query": "高风险 改进措施 RPN 降低 验证"},
        {"id": "ch5", "name": "FMEA分析结论", "query": "FMEA结论 风险 可接受性 建议"},
    ],
    "design_output": [
        {"id": "ch1", "name": "概述", "query": "设计输出 目的 范围 概述 ISO13485"},
        {"id": "ch2", "name": "产品图纸和规范", "query": "图纸 规范 技术文档 物料清单 BOM"},
        {"id": "ch3", "name": "生产工艺文件", "query": "生产工艺 作业指导书 SOP 检验规程"},
        {"id": "ch4", "name": "包装和标识规范", "query": "包装规范 标识 标签 IFU"},
        {"id": "ch5", "name": "设计输出评审", "query": "设计输出评审 评审记录 批准 可追溯"},
    ],
    "design_review": [
        {"id": "ch1", "name": "评审概述", "query": "设计评审 目的 范围 评审阶段"},
        {"id": "ch2", "name": "评审内容", "query": "评审内容 设计输入 设计输出 风险评估"},
        {"id": "ch3", "name": "评审参与人员", "query": "评审人员 资格 职责 签字"},
        {"id": "ch4", "name": "评审发现和建议", "query": "评审发现 问题 改进建议"},
        {"id": "ch5", "name": "评审结论和跟踪", "query": "评审结论 整改措施 跟踪验证"},
    ],
    "design_verification": [
        {"id": "ch1", "name": "验证概述", "query": "设计验证 目的 范围 验证计划"},
        {"id": "ch2", "name": "验证方法", "query": "验证方法 检验 测试 对比 计算"},
        {"id": "ch3", "name": "验证项目和结果", "query": "验证项目 测试数据 结果判定"},
        {"id": "ch4", "name": "不符合项处理", "query": "不符合项 原因分析 纠正措施"},
        {"id": "ch5", "name": "验证结论", "query": "验证结论 批准 记录"},
    ],
    "design_validation": [
        {"id": "ch1", "name": "确认概述", "query": "设计确认 目的 范围 确认计划"},
        {"id": "ch2", "name": "临床评价", "query": "临床评价 临床试验 临床数据 CER"},
        {"id": "ch3", "name": "模拟使用测试", "query": "模拟使用 用户测试 可用性评价"},
        {"id": "ch4", "name": "确认结果分析", "query": "确认结果 数据分析 有效性评价"},
        {"id": "ch5", "name": "确认结论", "query": "确认结论 批准 记录"},
    ],
    "design_change": [
        {"id": "ch1", "name": "变更概述", "query": "设计变更 变更描述 变更原因"},
        {"id": "ch2", "name": "变更影响评估", "query": "变更影响 风险评估 产品影响"},
        {"id": "ch3", "name": "变更验证计划", "query": "变更验证 验证方法 验证内容"},
        {"id": "ch4", "name": "变更审批", "query": "变更审批 审批人 审批意见"},
        {"id": "ch5", "name": "变更实施记录", "query": "变更实施 实施记录 追溯性"},
    ],
    "design_history_file": [
        {"id": "ch1", "name": "DHF目录", "query": "设计历史文件 目录 文件清单"},
        {"id": "ch2", "name": "设计开发计划记录", "query": "设计计划 记录 批准"},
        {"id": "ch3", "name": "设计输入输出记录", "query": "设计输入 设计输出 记录"},
        {"id": "ch4", "name": "设计评审验证确认记录", "query": "设计评审 验证 确认 记录"},
        {"id": "ch5", "name": "设计变更记录", "query": "设计变更 变更记录 追溯"},
        {"id": "ch6", "name": "DHF管理", "query": "DHF管理 归档 保存 查阅"},
    ],
    "risk_acceptance_criteria": [
        {"id": "ch1", "name": "概述", "query": "风险可接受准则 ISO14971 概述 范围"},
        {"id": "ch2", "name": "评分标准", "query": "严重度 频度数 探测度 评分标准 定义"},
        {"id": "ch3", "name": "RPN阈值", "query": "RPN 阈值 设定 依据"},
        {"id": "ch4", "name": "风险处理原则", "query": "风险等级 处理 原则 接受 控制 转移"},
        {"id": "ch5", "name": "批准", "query": "准则批准 审批 记录"},
    ],
    "periodic_risk_evaluation": [
        {"id": "ch1", "name": "概述", "query": "定期风险评价 目的 范围 评价周期"},
        {"id": "ch2", "name": "风险信息汇总", "query": "投诉 不良事件 文献 新的危害"},
        {"id": "ch3", "name": "风险再评价", "query": "新风险识别 已识别风险 再评价"},
        {"id": "ch4", "name": "控制措施有效性", "query": "风险控制 有效性 不足 改进"},
        {"id": "ch5", "name": "结论", "query": "定期风险评价结论 建议"},
    ],
    "sop": [
        {"id": "ch1", "name": "目的和范围", "query": "SOP目的 范围 引用文件"},
        {"id": "ch2", "name": "职责", "query": "职责 部门 人员资质"},
        {"id": "ch3", "name": "操作步骤", "query": "操作步骤 工艺流程 质量控制点"},
        {"id": "ch4", "name": "安全注意事项", "query": "安全注意事项 防护措施 应急处理"},
    ],
    "product_spec": [
        {"id": "ch1", "name": "型号规格", "query": "型号 规格 结构组成 贴敷式胰岛素泵"},
        {"id": "ch2", "name": "性能指标", "query": "性能指标 输注精度 物理性能 电气性能"},
        {"id": "ch3", "name": "检验方法", "query": "检验方法 检验规则 判定标准"},
        {"id": "ch4", "name": "标志包装运输贮存", "query": "标志 包装 运输 贮存"},
    ],
    "instruction": [
        {"id": "ch1", "name": "产品信息", "query": "贴敷式胰岛素泵 产品名称 型号 生产企业"},
        {"id": "ch2", "name": "适用范围", "query": "适用范围 适应症 禁忌症"},
        {"id": "ch3", "name": "使用方法", "query": "使用方法 操作步骤 贴敷 更换 剂量设定"},
        {"id": "ch4", "name": "注意事项", "query": "注意事项 警示 警告 不良反应"},
        {"id": "ch5", "name": "维护保养", "query": "维护 保养 故障排除 客服信息"},
    ],

    # ================= DHF清单扩充（55种新文档类型） =================
    "market_research_product_definition": [
        {"id": "ch1", "name": "临床需求和市场分析", "query": "贴敷式胰岛素泵 糖尿病发病率 胰岛素泵市场规模 患者需求分析 竞品分析"},
        {"id": "ch2", "name": "目标用户群体定义", "query": "糖尿病患者 胰岛素泵用户 用户画像 1型糖尿病 2型糖尿病 胰岛素依赖"},
        {"id": "ch3", "name": "产品定义与预期用途", "query": "贴敷式胰岛素泵 产品定义 预期用途 适用范围 适应症 禁忌症"},
        {"id": "ch4", "name": "产品关键特性定义", "query": "输注精度 无管路设计 BLE通信 闭环控制 贴敷方式 储药器容量"},
        {"id": "ch5", "name": "竞争产品对比分析", "query": "Omnipod Medtronic Tandem 竞品分析 差异化优势 市场定位"},
        {"id": "ch6", "name": "法规和市场准入分析", "query": "NMPA三类 FDA 510k MDR 注册路径 医保政策 市场准入策略"},
    ],
    "project_feasibility_study": [
        {"id": "ch1", "name": "项目背景和目的", "query": "贴敷式胰岛素泵 项目背景 研发目的 战略意义"},
        {"id": "ch2", "name": "技术可行性分析", "query": "胰岛素泵技术 微电机输注 闭环算法 BLE通信 技术成熟度 技术风险"},
        {"id": "ch3", "name": "商业可行性分析", "query": "市场规模 目标用户 预期售价 成本分析 投资回报 商业模式"},
        {"id": "ch4", "name": "合规可行性分析", "query": "NMPA注册 III类医疗器械 法规要求 标准合规 注册路径"},
        {"id": "ch5", "name": "资源和能力评估", "query": "研发团队 设备资源 生产能力 供应链 知识产权"},
        {"id": "ch6", "name": "风险评估与结论", "query": "项目风险 技术风险 市场风险 法规风险 可行性结论 建议"},
    ],
    "patent_analysis_report": [
        {"id": "ch1", "name": "检索范围和策略", "query": "胰岛素泵 专利检索 数据库 检索策略 IPC分类 关键词"},
        {"id": "ch2", "name": "核心专利分析", "query": "贴敷式胰岛素泵 输注机构 贴敷方式 闭环控制 核心专利 专利权人"},
        {"id": "ch3", "name": "专利布局分析", "query": "胰岛素泵专利 技术领域 地域分布 专利权人 申请趋势"},
        {"id": "ch4", "name": "专利风险分析", "query": "专利侵权风险 FTO分析 重点专利解读 规避设计建议"},
        {"id": "ch5", "name": "专利战略建议", "query": "专利申请策略 技术空白点 可专利性分析 知识产权布局"},
    ],
    "project_approval_review": [
        {"id": "ch1", "name": "项目概述和背景", "query": "贴敷式胰岛素泵 项目名称 项目编号 立项背景 战略意义"},
        {"id": "ch2", "name": "项目目标与范围", "query": "项目目标 研发范围 产品规格 项目边界 交付物"},
        {"id": "ch3", "name": "项目团队和资源", "query": "项目组织 团队成员 职责分配 资源配置 预算"},
        {"id": "ch4", "name": "项目计划和里程碑", "query": "项目计划 阶段划分 里程碑 时间表 关键路径"},
        {"id": "ch5", "name": "风险评估与应对", "query": "项目风险识别 技术风险 资源风险 市场风险 应对预案"},
        {"id": "ch6", "name": "评审结论和决议", "query": "评审意见 评审结论 审批人 批准日期 后续要求"},
    ],
    "user_needs_specification": [
        {"id": "ch1", "name": "预期用途和使用场景", "query": "贴敷式胰岛素泵 预期用途 临床需求 使用场景 目标用户 糖尿病患者"},
        {"id": "ch2", "name": "患者需求分析", "query": "胰岛素治疗 患者体验 贴敷舒适性 操作便捷性 剂量精度 疼痛管理"},
        {"id": "ch3", "name": "临床用户需求", "query": "医护人员需求 剂量设定 数据管理 报警系统 维护保养 培训需求"},
        {"id": "ch4", "name": "使用环境需求", "query": "家庭使用 外出携带 运动场景 防水需求 温度范围 电磁环境"},
        {"id": "ch5", "name": "用户界面和交互需求", "query": "App界面 蓝牙连接 糖数据展示 胰岛素输注记录 报警交互"},
        {"id": "ch6", "name": "法规和标准用户需求", "query": "YY/T 9706.106-2021 可用性工程 IEC 62366 人因工程 用户需求法规"},
    ],
    "preliminary_risk_analysis": [
        {"id": "ch1", "name": "分析目的和范围", "query": "初步风险分析 目的 范围 适用标准 ISO 14971 YY/T 1437-2023"},
        {"id": "ch2", "name": "产品初步描述", "query": "贴敷式胰岛素泵 初步结构 工作原理 预期用途 安全特征"},
        {"id": "ch3", "name": "已知和可预见的危害识别", "query": "输注过量 输注不足 贴敷脱落 漏液 过敏 感染 电磁干扰 BLE断连"},
        {"id": "ch4", "name": "初步风险估计和评价", "query": "风险严重度 发生概率 风险评价 可接受性初步判断"},
        {"id": "ch5", "name": "后续风险管理计划建议", "query": "风险管理计划 需要进一步分析的危害 设计阶段风险控制方向"},
    ],
    "product_risk_analysis_matrix": [
        {"id": "ch1", "name": "总表说明", "query": "风险分析 管理总表 说明 使用方法 更新频率 ISO 14971 GB/T 42062"},
        {"id": "ch2", "name": "危害识别汇总", "query": "贴敷式胰岛素泵 危害清单 电气 机械 生物 软件 可用性 环境"},
        {"id": "ch3", "name": "风险评估矩阵", "query": "风险矩阵 严重度等级 发生概率 可探测度 风险评估评分"},
        {"id": "ch4", "name": "风险控制措施汇总", "query": "风险控制措施 设计控制 防护措施 安全信息 验证方法"},
        {"id": "ch5", "name": "残余风险评价", "query": "残余风险评估 风险/收益分析 可接受性判断"},
        {"id": "ch6", "name": "风险管理追踪", "query": "风险编号 控制措施状态 验证状态 责任人 完成时限"},
    ],
    "cybersecurity_risk_analysis_matrix": [
        {"id": "ch1", "name": "总表范围和说明", "query": "网络安全 风险分析 管理总表 适用范围 YY/T 1843-2022"},
        {"id": "ch2", "name": "资产识别和威胁建模", "query": "贴敷式胰岛素泵 网络资产 通信接口 BLE USB 威胁建模 STRIDE"},
        {"id": "ch3", "name": "网络安全漏洞分析", "query": "漏洞识别 BLE安全 固件安全 App安全 数据安全 隐私保护"},
        {"id": "ch4", "name": "网络安全风险评估", "query": "漏洞严重度 CVSS评分 可利用性 影响分析 风险等级"},
        {"id": "ch5", "name": "安全控制措施汇总", "query": "安全控制 加密 认证 访问控制 安全启动 安全更新 事件响应"},
        {"id": "ch6", "name": "网络安全风险追踪", "query": "风险编号 控制措施 验证状态 残余风险 责任人 更新记录"},
    ],
    "software_config_management_plan": [
        {"id": "ch1", "name": "目的和范围", "query": "软件配置管理 目的 范围 IEC 62304 安全等级C"},
        {"id": "ch2", "name": "配置项识别", "query": "软件配置项 固件 App 测试代码 配置工具 文档 基线定义"},
        {"id": "ch3", "name": "版本控制和基线管理", "query": "版本控制 Git 分支策略 基线管理 标签管理 发布管理"},
        {"id": "ch4", "name": "变更控制和审批", "query": "变更控制流程 变更请求 影响分析 审批权限 变更实施"},
        {"id": "ch5", "name": "配置状态报告和审计", "query": "配置状态报告 配置审计 追溯性 合规性检查"},
    ],
    "structural_design_requirements": [
        {"id": "ch1", "name": "结构设计概述", "query": "贴敷式胰岛素泵 结构组成 外形尺寸 重量限制 整体布局"},
        {"id": "ch2", "name": "药囊组件需求", "query": "储药器 药囊材料 胰岛素相容性 密封性 容量规格 透明度"},
        {"id": "ch3", "name": "动力组件需求", "query": "微电机 传动机构 推注机构 精度要求 可靠性 噪音控制"},
        {"id": "ch4", "name": "输注组件需求", "query": "输注管路 输注针 连接器 防漏设计 生物相容性 插入深度"},
        {"id": "ch5", "name": "壳体组件需求", "query": "上壳 底壳 材料选择 结构强度 防水密封 人机交互界面"},
        {"id": "ch6", "name": "附件组件需求", "query": "背胶 离型纸 注射器 敷贴器 包装固定结构"},
    ],
    "packaging_labeling_requirements": [
        {"id": "ch1", "name": "包装设计需求概述", "query": "贴敷式胰岛素泵 包装需求 法规要求 ISO 11607 无菌包装"},
        {"id": "ch2", "name": "初包装需求", "query": "吸塑盒 特卫强(Tyvek) 无菌屏障 密封强度 剥离性能"},
        {"id": "ch3", "name": "中包装和外包装需求", "query": "包装盒 瓦楞纸箱 缓冲设计 堆码强度 运输要求"},
        {"id": "ch4", "name": "标签需求", "query": "产品标签 灭菌标签 UDI标签 追溯性标签 标签耐久性 可读性"},
        {"id": "ch5", "name": "灭菌包装需求", "query": "EO灭菌适应性 辐照灭菌适应性 灭菌指示物 灭菌后包装完整性"},
        {"id": "ch6", "name": "标识和说明书需求", "query": "符号标识 警示标识 使用说明 法规标识要求 GB 9706.1"},
    ],
    "product_rtm": [
        {"id": "ch1", "name": "追溯矩阵说明", "query": "需求追溯矩阵 目的 范围 维护方法 关联关系说明"},
        {"id": "ch2", "name": "用户需求到设计输入追溯", "query": "用户需求ID 设计输入ID 追溯关系 满足性评价 追溯缺口"},
        {"id": "ch3", "name": "设计输入到系统需求追溯", "query": "设计输入ID 系统需求ID 硬件需求 软件需求 结构需求"},
        {"id": "ch4", "name": "需求覆盖度分析", "query": "需求覆盖度 未满足需求 冗余需求 需求完整性评估"},
        {"id": "ch5", "name": "追溯矩阵维护和管理", "query": "RTM维护流程 变更管理 版本控制 评审记录"},
    ],
    "software_rtm": [
        {"id": "ch1", "name": "软件追溯矩阵说明", "query": "软件追溯 IEC 62304 追溯关系 软件需求 软件架构 软件测试"},
        {"id": "ch2", "name": "软件需求到架构追溯", "query": "软件需求ID 软件架构模块 功能分解 接口定义"},
        {"id": "ch3", "name": "软件架构到详细设计追溯", "query": "架构模块ID 详细设计单元 函数/类 实现映射"},
        {"id": "ch4", "name": "详细设计到测试追溯", "query": "设计单元ID 单元测试用例 集成测试用例 覆盖率分析"},
        {"id": "ch5", "name": "软件需求到风险控制追溯", "query": "软件需求 风险控制措施 安全需求追溯 IEC 62304 §7.3"},
    ],
    "cybersecurity_traceability_matrix": [
        {"id": "ch1", "name": "网络安全追溯说明", "query": "网络安全追溯 YY/T 1843-2022 追溯范围 维护方法"},
        {"id": "ch2", "name": "安全需求到设计追溯", "query": "安全需求ID 安全设计措施 加密实现 认证机制 访问控制"},
        {"id": "ch3", "name": "安全设计到测试追溯", "query": "安全设计ID 安全测试用例 渗透测试 漏洞扫描 验证结果"},
        {"id": "ch4", "name": "漏洞到修复追溯", "query": "漏洞编号 修复措施 验证测试 残余风险评估"},
    ],
    "hardware_design_plan": [
        {"id": "ch1", "name": "硬件设计概述", "query": "贴敷式胰岛素泵 硬件架构 电路板 底软 整体方案 设计目标"},
        {"id": "ch2", "name": "电源管理设计", "query": "锂电池 电源管理芯片 充电电路 电量检测 低功耗设计 续航优化"},
        {"id": "ch3", "name": "电机驱动和控制设计", "query": "微电机选型 驱动电路 位置检测 电流监控 堵转检测 保护电路"},
        {"id": "ch4", "name": "传感器接口设计", "query": "压力传感器 温度传感器 气泡检测 流量检测 信号调理 ADC采样"},
        {"id": "ch5", "name": "通信模块设计", "query": "BLE蓝牙 天线设计 射频性能 通信协议 数据完整性"},
        {"id": "ch6", "name": "PCB布局和EMC设计", "query": "PCB布局 电磁兼容 信号完整性 接地设计 屏蔽设计"},
    ],
    "structural_design_plan": [
        {"id": "ch1", "name": "结构设计总方案", "query": "贴敷式胰岛素泵 整体结构方案 总体布局 外形尺寸 装配关系"},
        {"id": "ch2", "name": "药囊组件结构设计", "query": "储药器结构 活塞密封 药囊接口 容量设计 材料选择"},
        {"id": "ch3", "name": "动力组件结构设计", "query": "电机安装 传动齿轮 推注螺杆 减速机构 定位精度"},
        {"id": "ch4", "name": "壳体组件结构设计", "query": "上壳结构 底壳结构 卡扣设计 防水密封 人机工程"},
        {"id": "ch5", "name": "输注组件结构设计", "query": "输注针 针座 弹簧 触发机构 回缩机构 针尖保护"},
        {"id": "ch6", "name": "模具和成型方案", "query": "注塑模具 材料选择 脱模斜度 壁厚设计 分型面 工艺方案"},
    ],
    "software_coding_standard": [
        {"id": "ch1", "name": "编码规范总则", "query": "软件编码规范 IEC 62304 安全等级C 编码标准 适用范围"},
        {"id": "ch2", "name": "嵌入式固件编码规范", "query": "C语言编码规范 MISRA-C 命名规则 代码结构 注释规范 内存管理"},
        {"id": "ch3", "name": "移动端App编码规范", "query": "Java/Kotlin/Swift编码规范 移动端最佳实践 API设计 线程安全"},
        {"id": "ch4", "name": "代码审查和静态分析", "query": "代码审查流程 静态分析工具 代码质量度量 缺陷密度 复杂性控制"},
        {"id": "ch5", "name": "软件安全编码规范", "query": "安全编码 缓冲区溢出 输入验证 加密实现 密钥管理 安全日志"},
    ],
    "packaging_labeling_design_plan": [
        {"id": "ch1", "name": "包装设计总体方案", "query": "贴敷式胰岛素泵 包装总体方案 设计目标 法规依据 ISO 11607"},
        {"id": "ch2", "name": "初包装详细设计", "query": "吸塑盒设计 尺寸 材料 Tyvek盖材 热封参数 密封验证"},
        {"id": "ch3", "name": "中包装和外包装设计", "query": "彩盒设计 缓冲结构 堆码设计 运输包装 瓦楞纸箱选型"},
        {"id": "ch4", "name": "标签设计", "query": "产品标签 布局设计 字体 符号 UDI码 一维码/二维码 色标"},
        {"id": "ch5", "name": "灭菌包装设计", "query": "EO穿透性 残留排空 辐照适应性 包装完整性设计 灭菌指示物"},
        {"id": "ch6", "name": "包装工艺设计", "query": "包装装配流程 热封工艺 贴标工艺 检漏工艺 包装线设计"},
    ],
    "primary_packaging_material_report": [
        {"id": "ch1", "name": "初包装概述和要求", "query": "贴敷式胰岛素泵 初包装 定义 功能 法规要求 ISO 11607"},
        {"id": "ch2", "name": "材料筛选和评估", "query": "PETG PET APET 吸塑盒材料 Tyvek 1073B 2FS 盖材 材料比较"},
        {"id": "ch3", "name": "灭菌适应性确认", "query": "EO灭菌材料适应性 EO穿透性 残留吸附 辐照灭菌 剂量耐受 "},
        {"id": "ch4", "name": "无菌屏障能力验证", "query": "密封强度 微生物屏障 完整性测试 染料渗透 气泡泄漏测试"},
        {"id": "ch5", "name": "材料确认和结论", "query": "材料确认结果 供应商信息 规格参数 结论和建议"},
    ],
    "performance_research_records": [
        {"id": "ch1", "name": "性能研究概述", "query": "贴敷式胰岛素泵 性能研究 目的 范围 研究计划"},
        {"id": "ch2", "name": "输注性能研究", "query": "输注精度 基础率 大剂量 流量稳定性 输注误差 影响因子"},
        {"id": "ch3", "name": "结构性能研究", "query": "结构强度 跌落测试 贴敷力 剥离力 插拔力 疲劳测试"},
        {"id": "ch4", "name": "电子性能研究", "query": "射频性能 天线效率 功耗测试 电池特性 信号完整性"},
        {"id": "ch5", "name": "设计版本和打样记录", "query": "设计版本历史 打样编号 测试数据 改进记录 结论"},
    ],
    "inspection_method_validation": [
        {"id": "ch1", "name": "验证概述和范围", "query": "检验方法学验证 目的 范围 适用检验项目 法规依据"},
        {"id": "ch2", "name": "初始污染菌检验方法验证", "query": "初始污染菌 检验方法 回收率 重复性 精密度 检出限"},
        {"id": "ch3", "name": "无菌检验方法验证", "query": "无菌检验 直接接种法 薄膜过滤法 方法适用性 阳性对照"},
        {"id": "ch4", "name": "细菌内毒素检验方法验证", "query": "细菌内毒素 凝胶法 光度法 干扰试验 灵敏度验证"},
        {"id": "ch5", "name": "验证结果和结论", "query": "验证结果汇总 方法适用性 操作规范 结论和建议"},
    ],
    "material_specification_drawing": [
        {"id": "ch1", "name": "物料概述和分类", "query": "贴敷式胰岛素泵 物料清单 原辅包材分类 物料编码规则"},
        {"id": "ch2", "name": "原材料规格", "query": "树脂材料 硅胶材料 金属材料 胶带材料 规格参数 供应商信息"},
        {"id": "ch3", "name": "辅料规格", "query": "润滑剂 密封圈 粘合剂 焊料 清洁剂 规格和用量"},
        {"id": "ch4", "name": "包装材料规格", "query": "吸塑盒 Tyvek 彩盒 标签 说明书 灭菌袋 规格参数"},
        {"id": "ch5", "name": "物料图纸和验收标准", "query": "物料图纸清单 关键尺寸 公差 验收标准 抽样方案"},
    ],
    "process_flow_diagram": [
        {"id": "ch1", "name": "工艺概述和总流程", "query": "贴敷式胰岛素泵 生产工艺总流程 从进料到成品 关键控制点"},
        {"id": "ch2", "name": "组装工艺流程", "query": "组件预装 药囊组装 动力组装 输注组件 壳体组装 总装流程"},
        {"id": "ch3", "name": "关键工序详细流程", "query": "焊接工序 热封工序 灭菌工序 检漏工序 特殊工序参数"},
        {"id": "ch4", "name": "检验节点和控制", "query": "来料检验 过程检验 成品检验 检验节点 判定标准"},
        {"id": "ch5", "name": "包装工艺流程", "query": "内包装 外包装 贴标 赋码 装箱 托盘 入库流程"},
    ],
    "tooling_drawing_acceptance": [
        {"id": "ch1", "name": "工装概述和清单", "query": "贴敷式胰岛素泵 工装清单 工装编号 用途分类 设计依据"},
        {"id": "ch2", "name": "注塑模具图纸", "query": "壳体模具 药囊模具 零件模具 模具图纸 材料 热处理"},
        {"id": "ch3", "name": "装配工装图纸", "query": "组装夹具 热封夹具 焊接夹具 测试工装 图纸和规格"},
        {"id": "ch4", "name": "检测工装图纸", "query": "检漏工装 密封测试工装 流量测试工装 尺寸检具"},
        {"id": "ch5", "name": "工装验收记录", "query": "工装验收 尺寸检验 试模报告 功能验证 验收结论 使用批准"},
    ],
    "approved_supplier_list": [
        {"id": "ch1", "name": "供应商管理概述", "query": "贴敷式胰岛素泵 供应商管理 管理策略 ISO 13485 §7.4"},
        {"id": "ch2", "name": "原材料供应商清单", "query": "树脂供应商 硅胶供应商 金属件供应商 胶带供应商 评估状态"},
        {"id": "ch3", "name": "组件供应商清单", "query": "电机供应商 传感器供应商 电池供应商 电路板供应商 BLE模块供应商"},
        {"id": "ch4", "name": "包装材料供应商清单", "query": "吸塑盒供应商 Tyvek供应商 标签供应商 印刷供应商"},
        {"id": "ch5", "name": "供应商评估和分级", "query": "供应商评估 审核结果 质量评级 供货能力 合作协议 合格状态"},
    ],
    "design_output_checklist": [
        {"id": "ch1", "name": "设计输出清单说明", "query": "设计输出 清单说明 管理目的 使用范围 ISO 13485 §7.3"},
        {"id": "ch2", "name": "采购相关信息", "query": "物料清单 规格书 合格供应商 采购文件 来料检验标准"},
        {"id": "ch3", "name": "生产相关信息", "query": "工艺文件 作业指导书 设备清单 工装清单 生产环境要求"},
        {"id": "ch4", "name": "检验相关信息", "query": "检验规范 检验方法 抽样方案 接收标准 检验设备"},
        {"id": "ch5", "name": "使用和服务相关信息", "query": "说明书 标签 安装指南 维护手册 售后配件清单"},
        {"id": "ch6", "name": "产品技术要求", "query": "产品技术要求 性能指标 安全指标 标准符合性 注册要求"},
    ],
    "performance_verification_plan": [
        {"id": "ch1", "name": "验证目的和范围", "query": "性能验证 验证目的 适用范围 验证策略 法规依据"},
        {"id": "ch2", "name": "输注功能验证方案", "query": "基础率输注验证 大剂量输注验证 输注速率稳定性 输注精度验证方法"},
        {"id": "ch3", "name": "报警功能验证方案", "query": "阻塞报警 气泡报警 低电量报警 无 insulin 报警 脱针报警"},
        {"id": "ch4", "name": "电池和防水验证方案", "query": "电池续航测试 防水IP等级验证 防水密封性测试"},
        {"id": "ch5", "name": "环境适应性验证方案", "query": "高低温 湿热 振动 跌落测试 电磁环境适应性"},
        {"id": "ch6", "name": "验证进度和资源配置", "query": "验证时间表 样品数量 设备和人员 验收标准"},
    ],
    "performance_verification_report": [
        {"id": "ch1", "name": "验证概要", "query": "性能验证 验证概要 验证依据 验证范围 验证结论概述"},
        {"id": "ch2", "name": "输注功能验证结果", "query": "输注精度数据 基础率 大剂量 流量稳定性 数据统计 结论"},
        {"id": "ch3", "name": "报警功能验证结果", "query": "阻塞报警 气泡报警 低电量报警 脱针报警 各项报警验证数据"},
        {"id": "ch4", "name": "电池和防水验证结果", "query": "电池容量 功耗 续航时间 防水等级 密封性测试数据"},
        {"id": "ch5", "name": "环境适应性验证结果", "query": "高低温 湿热 振动 跌落 测试数据和结论"},
        {"id": "ch6", "name": "综合结论和改进建议", "query": "总体验证结论 不合格项 改进建议 后续验证计划"},
    ],
    "software_unit_test_plan": [
        {"id": "ch1", "name": "测试目的和范围", "query": "软件单元测试 测试目的 范围 IEC 62304 安全等级C"},
        {"id": "ch2", "name": "测试策略和方法", "query": "白盒测试 黑盒测试 边界值 等价类 语句覆盖 分支覆盖"},
        {"id": "ch3", "name": "测试用例设计", "query": "固件模块 电机控制模块 传感器模块 通信模块 报警模块 测试用例"},
        {"id": "ch4", "name": "测试环境和工具", "query": "单元测试框架 CppUTest Google Test 覆盖率工具 模拟器"},
        {"id": "ch5", "name": "测试进度和资源", "query": "测试计划 里程碑 人员和设备 测试完成准则"},
    ],
    "software_integration_test_plan": [
        {"id": "ch1", "name": "测试目的和范围", "query": "软件集成测试 测试目的 范围 IEC 62304 §5.7 集成策略"},
        {"id": "ch2", "name": "集成测试策略", "query": "自顶向下 自底向上 大爆炸 渐进式集成 接口测试 数据流测试"},
        {"id": "ch3", "name": "测试用例设计", "query": "模块接口 MCU与BLE 固件与App 传感器接口 通信协议 数据同步"},
        {"id": "ch4", "name": "测试环境和工具", "query": "集成测试环境 HIL测试 通信测试工具 日志分析 自动化测试"},
        {"id": "ch5", "name": "测试进度和缺陷管理", "query": "测试计划 缺陷管理流程 严重度分级 修复验证 完成准则"},
    ],
    "software_system_test_plan": [
        {"id": "ch1", "name": "测试目的和范围", "query": "软件系统测试 测试目的 范围 IEC 62304 §5.8 系统级验证"},
        {"id": "ch2", "name": "功能测试设计", "query": "输注功能测试 BLE通信测试 报警逻辑测试 UI测试 数据管理测试"},
        {"id": "ch3", "name": "非功能测试设计", "query": "性能测试 压力测试 稳定性测试 安全性测试 兼容性测试"},
        {"id": "ch4", "name": "端到端场景测试", "query": "用户场景 贴敷-输注-更换场景 异常场景 恢复场景 断电场景"},
        {"id": "ch5", "name": "测试进度和风险评估", "query": "测试计划 风险分析 回归测试策略 完成准则 交付标准"},
    ],
    "software_quality_test_plan": [
        {"id": "ch1", "name": "测试目的和范围", "query": "软件质量测试 测试目的 范围 质量属性 合规要求"},
        {"id": "ch2", "name": "可靠性测试", "query": "MTBF 故障注入 容错测试 恢复测试 长时间运行测试"},
        {"id": "ch3", "name": "可维护性测试", "query": "代码复杂度 可读性 文档完整性 版本管理 变更影响分析"},
        {"id": "ch4", "name": "安全性和隐私测试", "query": "静态代码分析 漏洞扫描 渗透测试 数据加密 权限控制"},
        {"id": "ch5", "name": "测试进度和工具", "query": "测试计划 质量度量指标 测试工具链 完成准则"},
    ],
    "software_quality_test_report": [
        {"id": "ch1", "name": "测试概要", "query": "软件质量测试 测试概要 测试范围 测试结论概述"},
        {"id": "ch2", "name": "可靠性测试结果", "query": "MTBF数据 故障注入结果 容错测试 异常恢复测试 稳定性数据"},
        {"id": "ch3", "name": "可维护性测试结果", "query": "代码复杂度分析 覆盖率报告 代码审查结果 文档完整性"},
        {"id": "ch4", "name": "安全性测试结果", "query": "静态分析结果 漏洞扫描报告 渗透测试发现 数据安全验证"},
        {"id": "ch5", "name": "综合结论和改进", "query": "整体质量评价 不符合项 改进建议 后续质量计划"},
    ],
    "cybersecurity_test_plan": [
        {"id": "ch1", "name": "测试目的和范围", "query": "网络安全测试 测试目的 范围 YY/T 1843-2022 安全目标"},
        {"id": "ch2", "name": "BLE安全测试", "query": "BLE 配对 加密 中间人攻击 重放攻击 DoS攻击 嗅探测试"},
        {"id": "ch3", "name": "App安全测试", "query": "App渗透测试 数据存储安全 传输安全 认证安全 会话管理"},
        {"id": "ch4", "name": "固件安全测试", "query": "固件逆向 安全启动 固件签名 调试接口 敏感信息泄露"},
        {"id": "ch5", "name": "测试进度和报告", "query": "测试计划 漏洞分级 修复优先级 回归测试 测试交付物"},
    ],
    "software_interface_security_test_plan": [
        {"id": "ch1", "name": "测试目的和范围", "query": "软件接口安全测试 测试目的 接口范围 安全威胁模型"},
        {"id": "ch2", "name": "BLE接口安全测试", "query": "BLE GATT安全 服务发现 特征值访问 配对认证 数据加密传输"},
        {"id": "ch3", "name": "App-云端接口安全测试", "query": "API安全 认证鉴权 TLS/HTTPS 数据完整性 重放攻击防护"},
        {"id": "ch4", "name": "内部接口安全测试", "query": "MCU-BLE模块接口 固件-App接口 传感器接口 调试接口安全"},
        {"id": "ch5", "name": "测试进度", "query": "测试计划 测试用例 测试环境 风险评估 完成准则"},
    ],
    "software_interface_security_test_report": [
        {"id": "ch1", "name": "测试概要", "query": "软件接口安全测试 测试概要 测试范围 测试结论概述"},
        {"id": "ch2", "name": "BLE接口安全测试结果", "query": "GATT安全验证结果 配对测试 加密验证 漏洞发现 修复状态"},
        {"id": "ch3", "name": "App-云端接口安全测试结果", "query": "API安全测试结果 认证验证 TLS配置验证 漏洞和修复"},
        {"id": "ch4", "name": "内部接口安全测试结果", "query": "MCU-BLE接口 MCU-传感器接口 固件-App接口 调试接口验证"},
        {"id": "ch5", "name": "结论和建议", "query": "总体安全评估 残余风险 改进建议 后续安全测试计划"},
    ],
    "packaging_verification_plan": [
        {"id": "ch1", "name": "验证目的和范围", "query": "包装验证 标识验证 验证目的 范围 ISO 11607 法规要求"},
        {"id": "ch2", "name": "初包装完整性验证", "query": "密封强度 染料渗透 气泡泄漏 微生物屏障 加速老化后完整性"},
        {"id": "ch3", "name": "标签和标识验证", "query": "标签耐久性 可读性 耐擦拭 耐溶剂 UDI码可读性 符号准确性"},
        {"id": "ch4", "name": "运输包装验证", "query": "ISTA包装运输测试 跌落 振动 堆码 温湿度适应性"},
        {"id": "ch5", "name": "验证进度和样品", "query": "验证计划 样品量 抽样方案 判定标准 验证时间表"},
    ],
    "packaging_verification_report": [
        {"id": "ch1", "name": "验证概要", "query": "包装验证 标识验证 验证概要 验证依据 结论概述"},
        {"id": "ch2", "name": "初包装完整性验证结果", "query": "密封强度数据 染料渗透结果 微生物屏障 加速老化 数据和结论"},
        {"id": "ch3", "name": "标签和标识验证结果", "query": "标签耐久性 可读性测试 UDI码可读性 符号检查 数据"},
        {"id": "ch4", "name": "运输包装验证结果", "query": "ISTA测试数据 跌落 振动 堆码 产品完好性 结论"},
        {"id": "ch5", "name": "综合结论和改进", "query": "验证总体结论 不合格项 改进措施 复测结果"},
    ],
    "service_life_verification_plan": [
        {"id": "ch1", "name": "验证目的和范围", "query": "使用期限验证 验证目的 范围 使用期限定义 法规要求"},
        {"id": "ch2", "name": "使用期限评估方法", "query": "使用期限评估 有效期验证方法 加速老化 实时老化 设计评审"},
        {"id": "ch3", "name": "加速老化验证方案", "query": "加速老化 温度 湿度 老化因子 Q10方法 等效时间 老化条件"},
        {"id": "ch4", "name": "性能指标测试方案", "query": "老化后性能 输注精度 密封性 材料性能 电性能 生物相容性"},
        {"id": "ch5", "name": "验证进度和样品", "query": "样品分组 测试节点 时间安排 判定标准"},
    ],
    "service_life_verification_report": [
        {"id": "ch1", "name": "验证概要", "query": "使用期限验证 验证概要 验证方法 验证条件 结论概述"},
        {"id": "ch2", "name": "加速老化过程", "query": "老化条件 温度 湿度 老化时间 等效时间 过程记录"},
        {"id": "ch3", "name": "老化后性能验证结果", "query": "输注精度 密封完整性 材料性能 电性能 粘附性 测试数据"},
        {"id": "ch4", "name": "结论和有效期声明", "query": "使用期限结论 有效期声明 安全裕度 证据支持"},
    ],
    "shelf_life_verification_plan": [
        {"id": "ch1", "name": "验证目的和范围", "query": "货架有效期 验证目的 范围 定义 法规要求 ISO 11607"},
        {"id": "ch2", "name": "加速老化方案", "query": "加速老化 Q10 ASTM F1980 温度条件 湿度条件 等效时间计算"},
        {"id": "ch3", "name": "实时老化方案", "query": "实时老化 常温贮存 定期检测 时间节点 检测项目"},
        {"id": "ch4", "name": "检测项目和判定标准", "query": "包装完整性 产品性能 灭菌保持 生物相容性 每项检测标准"},
        {"id": "ch5", "name": "样品管理和验证进度", "query": "样品数量 分组 贮存条件 检测时间表"},
    ],
    "shelf_life_verification_report": [
        {"id": "ch1", "name": "验证概要", "query": "货架有效期验证 验证概要 验证方法 验证条件"},
        {"id": "ch2", "name": "加速老化验证结果", "query": "加速老化数据 各时间节点 包装完整性 产品性能 趋势分析"},
        {"id": "ch3", "name": "实时老化验证结果", "query": "实时老化数据 已完成的时间节点 当前结果 趋势分析"},
        {"id": "ch4", "name": "综合结论和有效期", "query": "货架有效期结论 加速与实时数据对比 有效期声明 证据总结"},
    ],
    "transport_verification_plan": [
        {"id": "ch1", "name": "验证目的和范围", "query": "包装运输验证 验证目的 范围 ISTA标准 运输条件"},
        {"id": "ch2", "name": "运输危害分析", "query": "运输危害 振动 冲击 跌落 堆码 温湿度 气压变化 搬运"},
        {"id": "ch3", "name": "测试方法设计", "query": "ISTA 2A ISTA 3A 振动测试 跌落测试 堆码测试 温湿度测试"},
        {"id": "ch4", "name": "验收标准和判定", "query": "产品完好性 包装完好性 功能性 密封性 外观 各项判定标准"},
        {"id": "ch5", "name": "验证进度和样品", "query": "样品数量 测试顺序 测试时间表 设备需求"},
    ],
    "leachables_test_plan": [
        {"id": "ch1", "name": "测试目的和范围", "query": "可沥滤物 测试目的 范围 法规依据 ISO 10993-18"},
        {"id": "ch2", "name": "EO残留测试方案", "query": "环氧乙烷EO 2-氯乙醇ECH 乙二醇EG 残留标准 GB/T 16886.7"},
        {"id": "ch3", "name": "分析方法验证", "query": "气相色谱法 GC方法验证 检出限 定量限 线性 精密度 回收率"},
        {"id": "ch4", "name": "样品制备和测试流程", "query": "样品浸提 浸提条件 浸提液处理 进样分析 数据处理"},
        {"id": "ch5", "name": "测试计划和判定标准", "query": "测试时间表 样品量 判定标准 法规限值"},
    ],
    "leachables_test_report": [
        {"id": "ch1", "name": "测试概要", "query": "可沥滤物测试 测试概要 测试方法 测试条件 结论概述"},
        {"id": "ch2", "name": "EO残留测试结果", "query": "EO残留量 各样品数据 统计分析 与限值对比"},
        {"id": "ch3", "name": "ECH和EG残留测试结果", "query": "2-氯乙醇残留 乙二醇残留 各样品数据 与限值对比"},
        {"id": "ch4", "name": "分析方法验证结果", "query": "方法验证数据 专属性 线性 精密度 回收率 系统适用性"},
        {"id": "ch5", "name": "综合结论", "query": "总体结论 是否合格 是否需要进一步处理 建议"},
    ],
    "biocompatibility_drug_compatibility_report": [
        {"id": "ch1", "name": "试验概要", "query": "生物相容性 药液相容性 试验概要 试验依据 ISO 10993系列标准"},
        {"id": "ch2", "name": "生物相容性试验结果", "query": "细胞毒性 皮肤刺激 致敏 血液相容性 各试验结果汇总"},
        {"id": "ch3", "name": "药液相容性试验结果", "query": "胰岛素相容性 材料与胰岛素接触 降解产物 吸附性 效价变化"},
        {"id": "ch4", "name": "输注针/胶布专项", "query": "输注针材料 不锈钢 塑料 胶布压敏胶 与皮肤接触的生物相容性"},
        {"id": "ch5", "name": "综合结论", "query": "生物安全性评价 药液安全性 残余风险 适用性结论"},
    ],
    "safety_emc_reliability_test_report": [
        {"id": "ch1", "name": "检测概要", "query": "强制检测 检测概要 检测项目 检测标准 检测机构"},
        {"id": "ch2", "name": "电气安全检测结果", "query": "漏电流 绝缘电阻 接地电阻 耐压测试 电气间隙 爬电距离 GB 9706.1"},
        {"id": "ch3", "name": "电磁兼容检测结果", "query": "EMC测试 辐射发射 传导发射 抗扰度 ESD 符合YY 9706.102-2021"},
        {"id": "ch4", "name": "环境可靠性检测结果", "query": "高温 低温 湿热 振动 跌落 IP防护等级 各项数据"},
        {"id": "ch5", "name": "综合结论", "query": "各项检测结论 符合性声明 不符合项 整改措施"},
    ],
    "registration_type_test_report": [
        {"id": "ch1", "name": "注册检验概述", "query": "注册检验 检验目的 检验范围 检验机构 检验标准"},
        {"id": "ch2", "name": "全性能检验结果", "query": "注册检验 全性能 性能指标 检验数据 标准符合性"},
        {"id": "ch3", "name": "EMC和安规检验结果", "query": "电气安全 EMC 电磁兼容 安规检验 限定值 判定"},
        {"id": "ch4", "name": "生物学检验结果", "query": "生物学评价 检验数据 标准符合性"},
        {"id": "ch5", "name": "综合结论和意见", "query": "注册检验总结论 产品符合性 整改项 注册建议"},
    ],
    "process_validation_plan": [
        {"id": "ch1", "name": "验证目的和范围", "query": "工艺验证 验证目的 范围 验证策略 法规要求 GMP"},
        {"id": "ch2", "name": "生产工艺描述", "query": "贴敷式胰岛素泵 主要工艺步骤 关键工序 特殊工序 工艺流程图"},
        {"id": "ch3", "name": "关键工艺参数识别", "query": "温度 压力 时间 速度 关键质量属性 工艺参数与质量关联"},
        {"id": "ch4", "name": "验证方案设计", "query": "IQ安装确认 OQ运行确认 PQ性能确认 样品量 抽样方案"},
        {"id": "ch5", "name": "验证进度和资源配置", "query": "验证时间表 人员安排 设备需求 文件计划"},
    ],
    "sterilization_validation_protocol": [
        {"id": "ch1", "name": "灭菌确认目的和范围", "query": "灭菌确认 目的 范围 灭菌方法 EO灭菌/辐照 法规依据"},
        {"id": "ch2", "name": "产品灭菌适应性", "query": "贴敷式胰岛素泵 材料 结构 灭菌适应性 灭菌负载配置"},
        {"id": "ch3", "name": "灭菌周期开发", "query": "EO灭菌参数 温度 湿度 气体浓度 暴露时间 预处理条件 解析时间"},
        {"id": "ch4", "name": "灭菌性能确认(PQ)", "query": "物理性能确认 微生物性能确认 BI/BI 挑战 无菌保证水平SAL"},
        {"id": "ch5", "name": "确认进度和判定标准", "query": "样品量 测试节点 判定标准 SAL≤10^-6 文件输出"},
    ],
    "sterilization_validation_report": [
        {"id": "ch1", "name": "确认概要", "query": "灭菌确认 确认概要 灭菌方法 确认范围 结论概述"},
        {"id": "ch2", "name": "灭菌周期参数确认", "query": "EO灭菌参数 温度 湿度 浓度 时间 预处理 解析 过程数据"},
        {"id": "ch3", "name": "物理性能确认结果", "query": "温度分布 湿度分布 气体浓度分布 产品温度曲线 数据"},
        {"id": "ch4", "name": "微生物性能确认结果", "query": "BI测试结果 无菌测试结果 SAL验证 阳性对照 数据汇总"},
        {"id": "ch5", "name": "结论和灭菌工艺确认", "query": "灭菌确认结论 灭菌工艺参数 常规监控要求 再确认计划"},
    ],
    "clinical_trial_plan": [
        {"id": "ch1", "name": "试验目的和设计", "query": "临床试验 试验目的 试验类型 试验设计 随机对照 多中心"},
        {"id": "ch2", "name": "受试者选择", "query": "糖尿病 胰岛素依赖 入选标准 排除标准 样本量 样本量计算"},
        {"id": "ch3", "name": "试验流程和方法", "query": "试验流程 筛选期 导入期 治疗期 随访期 评估指标 数据采集"},
        {"id": "ch4", "name": "安全性和有效性评估", "query": "主要终点 次要终点 安全性评价 不良事件 血糖控制 HbA1c"},
        {"id": "ch5", "name": "统计分析和质量控制", "query": "统计方法 样本量估算 数据管理 伦理审查 知情同意"},
    ],
    "clinical_trial_report": [
        {"id": "ch1", "name": "试验概要", "query": "临床试验 试验概要 试验目的 试验设计 试验机构"},
        {"id": "ch2", "name": "受试者基线特征", "query": "受试者 人口学特征 疾病基线 入组完成情况 脱落分析"},
        {"id": "ch3", "name": "有效性评价结果", "query": "HbA1c变化 TIR(时间区间) 血糖变异性 胰岛素用量 主要终点"},
        {"id": "ch4", "name": "安全性评价结果", "query": "不良事件 严重不良事件 器械相关事件 低血糖 高血糖 皮肤反应"},
        {"id": "ch5", "name": "讨论和结论", "query": "试验结论 产品性能和安全性 临床价值 与竞品比较 局限性"},
    ],
    "usability_test_plan": [
        {"id": "ch1", "name": "测试目的和范围", "query": "可用性测试 测试目的 范围 IEC 62366-1 形成性/总结性评价"},
        {"id": "ch2", "name": "用户群体和测试场景", "query": "糖尿病患者 照护者 医护人员 使用场景 关键任务定义"},
        {"id": "ch3", "name": "关键任务识别", "query": "设备贴敷 更换 剂量设定 报警响应 BLE配对 App操作 故障处理"},
        {"id": "ch4", "name": "测试方法和流程", "query": "模拟使用测试 认知走查 启发式评估 访谈 观察 绩效测量"},
        {"id": "ch5", "name": "测试进度和数据收集", "query": "测试时间表 参与者招募 测试环境 数据收集表 成功/失败标准"},
    ],
    "usability_test_report": [
        {"id": "ch1", "name": "测试概要", "query": "可用性测试 测试概要 测试方法 参与者信息 测试条件"},
        {"id": "ch2", "name": "关键任务测试结果", "query": "贴敷操作 更换操作 剂量设定 报警响应 BLE配对 App操作 每项结果"},
        {"id": "ch3", "name": "使用错误分析", "query": "使用错误 原因分析 潜在危害 风险等级 分类统计"},
        {"id": "ch4", "name": "用户反馈汇总", "query": "满意度 易用性 学习曲线 改进建议 用户访谈摘要"},
        {"id": "ch5", "name": "结论和改进建议", "query": "可用性评价结论 可用性问题 设计改进建议 是否满足可用性要求"},
    ],

}

# 默认章节（未定义的文档类型使用）
DEFAULT_CHAPTERS = [
    {"id": "ch1", "name": "概述", "query": "概述 产品简介"},
    {"id": "ch2", "name": "主要内容", "query": "主要内容 详细描述"},
    {"id": "ch3", "name": "结论", "query": "结论 总结"},
]


class MiniMaxService:
    """本地Ollama大模型调用服务 (qwen3.5:122b)"""

    _use_rag_default = False  # 默认禁用 RAG（避免模型问题影响启动）

    def __init__(self, api_key: Optional[str] = None, use_rag: bool = True):
        """
        初始化服务

        Args:
            api_key: 不再使用，保留兼容
            use_rag: 是否启用 RAG（检索增强生成），需要先运行 ingest 建立向量库
        """
        self.base_url = _get_ollama_base_url()
        self.model = _get_ollama_model()
        self.api_url = f"{self.base_url}/api/chat"
        self.api_key = api_key or _get_api_key()  # 保留兼容，不再校验
        self.use_rag = use_rag and _try_init_rag()
        self.use_web_search = _try_init_web_search()      # Playwright 直接搜索
        self.search_log = []  # 记录每个章节使用的搜索方式
        self._search_log_lock = threading.Lock()  # 线程安全锁
        self.max_concurrent = 4  # 本地模型并发数（122b大模型显存有限，降低并发）
        self.max_search_workers = 12  # 小节Web搜索最大并发数
        self.progress_callback = None  # 进度回调函数: callback(phase, current, total, message)
        self.timing_log = {}  # 计时日志: {"outline": float, "rag_total": float, "search_total": float, "sections": [...], "chapters": [...], "total": float}

    def generate_content(
        self,
        doc_type: str,
        product_name: str,
        product_type: str,
        product_params: str = "",
        chapter_mode: bool = True
    ) -> str:
        """
        调用本地Ollama API生成文档内容

        Args:
            doc_type: 文档类型
            product_name: 产品名称
            product_type: 产品类型
            product_params: 产品参数
            chapter_mode: 是否使用分章节生成模式（默认True）
        """
        if not self.model:
            raise ValueError("OLLAMA_MODEL 未设置，请在 .env 中配置")

        # 优先使用分章节生成模式
        if chapter_mode:
            return self._generate_by_chapters(
                doc_type, product_name, product_type, product_params
            )

        # 降级到单次生成
        template = self._get_prompt_template(doc_type)
        prompt = template.format(
            product_name=str(product_name),
            product_type=str(product_type),
            product_params=str(product_params) if product_params else "无特殊参数"
        )
        return self._call_api(prompt)

    def _get_prompt_template(self, doc_type: str) -> str:
        """获取详细Prompt模板 - 增强版（全中文要求）"""
        templates = {
            "risk_management_report": """
你是一位资深的医疗器械风险管理专家，持有ISO 14971内审员资质和多年医疗器械行业经验。请根据以下产品信息和参考范文，生成{chapter}章节内容。

【产品信息】
- 产品名称：{product_name}
- 产品类型：{product_type}
- 产品参数：{product_params}

【重要要求】
请参考上方的【参考范文】和【相关法规标准】部分（如果提供），严格按照参考范文的详细程度和格式生成{chapter}的完整内容。

特别注意：
0. 【强制要求】所有内容必须使用中文，不要使用任何英文单词或短语，除了标准号（如GB 42062-2022、ISO 14971:2019等）中必须包含的部分
1. 内容必须极其细致和具体，每个段落都要有实质性内容，不能只写框架标题
2. 风险分析要覆盖所有可能的危害，包括但不限于：能量危害、生物学危害、环境危害、使用危害等
3. 每个风险点都要有具体的分析、评价和控制措施
4. 技术参数要具体、可测量、有明确的数值范围
5. 表格要填写完整，不能留"（描述）"或"待填写"等占位符
6. 标准号和条款引用要准确，如GB 42062-2022、ISO 14971:2019等
7. 严重度、频度数、可探测度的评价要有理有据
8. 风险控制措施要具体可行，有明确的实施方法和验证要求
9. 剩余风险评价要有明确的结论和依据
10. 使用专业、规范的医疗器械行业术语

生成内容的详细程度要像实际可用于注册申报的正式文档一样。
""",
            "risk_management_plan": """
你是一位资深的医疗器械风险管理专家，持有ISO 14971内审员资质。请根据以下产品信息和参考范文，生成{chapter}章节内容。

【产品信息】
- 产品名称：{product_name}
- 产品类型：{product_type}
- 产品参数：{product_params}

【重要要求】
请参考上方的【参考范文】和【相关法规标准】部分（如果提供），严格按照参考范文的详细程度生成{chapter}的完整内容。

特别注意：
0. 【强制要求】所有内容必须使用中文，不要使用任何英文单词或短语，除了标准号中必须包含的部分
1. 内容必须极其细致和具体，不能只写框架
2. 人员职责要明确到具体岗位，资质要求要详细
3. 风险可接受准则要有明确的判定方法和数值标准
4. 风险管理活动计划要有具体的时间安排、责任人、输出文件
5. 生产和生产后信息收集要有具体的方法、频次、记录要求
6. 引用标准要准确，包括标准号和版本号
7. 每个条款都要有实质性内容，不能用笼统的概括
""",
            "fmea_analysis": """
你是一位资深的医疗器械FMEA分析专家，精通ISO 14971风险管理和GB 42062标准。请根据以下产品信息和参考范文，生成{chapter}章节内容。

【产品信息】
- 产品名称：{product_name}
- 产品类型：{product_type}
- 产品参数：{product_params}

【重要要求】
请参考上方的【参考范文】和【相关法规标准】部分（如果提供），严格按照参考范文的详细程度生成{chapter}的完整内容。

特别注意：
0. 【强制要求】所有内容必须使用中文，不要使用任何英文单词或短语，除了标准号和缩写（如FMEA、RPN等）中必须包含的部分
1. FMEA表格要极其详细，每个失效模式都要有完整的分析
2. 失效原因要分析到根本原因，不能只停留在表面
3. 失效后果要从多个维度分析：对患者的影响、对操作者的影响、对设备的影响等
4. 严重度、频度数、探测度的评分要有明确依据
5. 建议的改进措施要具体可行，有责任人和完成时间
6. 风险优先级数(RPN)计算要准确
7. 高风险项目要有详细的改进计划和验证方法
8. 每个条目都要有实质性内容，不能留空或用占位符
""",
            "design_development_plan": """
你是一位资深的医疗器械研发项目经理，精通ISO 13485和《医疗器械生产质量管理规范》关于设计开发的要求。请根据以下产品信息和参考范文，生成{chapter}章节内容。

【产品信息】
- 产品名称：{product_name}
- 产品类型：{product_type}
- 产品参数：{product_params}

【重要要求】
请参考上方的【参考范文】和【相关法规标准】部分（如果提供），严格按照参考范文的详细程度生成{chapter}的完整内容。

特别注意：
0. 【强制要求】所有内容必须使用中文，不要使用任何英文单词或短语，除了标准号和缩写（如DHF、DMR等）中必须包含的部分
1. 设计开发阶段划分要清晰，每个阶段都要有明确的输入、输出和验收标准
2. 职责分配要具体到岗位和人员，包括项目经理、研发工程师、测试工程师、QA等
3. 设计评审、验证、确认活动要有明确的时间点、参与人员和输出文件要求
4. 技术资源配置要详细，包括所需的设备、软件、标准、法规等
5. 设计变更管理流程要完整，包括变更申请、评估、审批、实施、验证等环节
6. 时间安排要合理，考虑各阶段的依赖关系和风险缓冲
7. 每个条款都要有实质性内容，不能笼统概括
8. 引用的标准和法规要准确，如ISO 13485、《医疗器械生产质量管理规范》等
""",
            "design_input": """
你是一位资深的医疗器械系统工程师，负责制定产品设计输入要求。请根据以下产品信息和参考范文，生成{chapter}章节内容。

【产品信息】
- 产品名称：{product_name}
- 产品类型：{product_type}
- 产品参数：{product_params}

【重要要求】
请参考上方的【参考范文】和【相关法规标准】部分（如果提供），严格按照参考范文的详细程度生成{chapter}的完整内容。

特别注意：
0. 【强制要求】所有内容必须使用中文，不要使用任何英文单词或短语，除了标准号中必须包含的部分
1. 性能要求要具体、可测量，每个指标都要有明确的数值范围和测试方法
2. 法规标准要求要列出所有适用的强制标准和推荐标准，包括标准号和版本
3. 风险管理要求要与风险管理报告中的风险控制措施相对应
4. 用户需求要从临床使用角度详细描述，包括预期用途、使用场景、用户群体等
5. 设计输入评审记录要完整，包括评审人员、评审意见、修改情况等
6. 所有要求都要明确、无歧义，能够作为设计输出的依据
7. 表格要完整填写，每个单元格都要有具体内容
""",
            "design_output": """
你是一位资深的医疗器械研发工程师，负责编制设计输出文件。请根据以下产品信息和参考范文，生成{chapter}章节内容。

【产品信息】
- 产品名称：{product_name}
- 产品类型：{product_type}
- 产品参数：{product_params}

【重要要求】
请参考上方的【参考范文】和【相关法规标准】部分（如果提供），严格按照参考范文的详细程度生成{chapter}的完整内容。

特别注意：
0. 【强制要求】所有内容必须使用中文，不要使用任何英文单词或短语，除了标准号中必须包含的部分
1. 产品图纸和规范要详细，包括零件图、装配图、技术规范、物料清单等
2. 生产工艺文件要包括工艺流程、作业指导书、检验规程等，具有可操作性
3. 包装和标识规范要符合法规要求，包括包装材料、标签内容、使用说明书等
4. 设计输出文件要与设计输入一一对应，能够证明满足设计输入要求
5. 文件编号和版本管理要规范，包括文件编号规则、版本历史、审批记录等
6. 设计输出评审记录要完整，包括评审人员、评审意见、修改情况等
7. 每个条款都要有实质性内容，不能笼统概括
""",
            "design_review": """
你是一位资深的医疗器械质量经理，负责组织和记录设计评审活动。请根据以下产品信息和参考范文，生成{chapter}章节内容。

【产品信息】
- 产品名称：{product_name}
- 产品类型：{product_type}
- 产品参数：{product_params}

【重要要求】
请参考上方的【参考范文】和【相关法规标准】部分（如果提供），严格按照参考范文的详细程度生成{chapter}的完整内容。

特别注意：
0. 【强制要求】所有内容必须使用中文，不要使用任何英文单词或短语，除了标准号中必须包含的部分
1. 评审概述要说明评审的阶段、目的、范围和依据
2. 评审内容要详细列出评审的所有项目，包括设计输入、设计输出、风险评估等
3. 评审参与人员要列出姓名、部门、资格、职责，并预留签字栏
4. 评审发现和建议要详细记录发现的问题、风险点和改进建议
5. 评审结论要明确，包括是否通过、需要整改的内容、整改期限等
6. 整改措施跟踪要记录整改情况、验证结果、验证人员等
7. 表格要完整填写，每个单元格都要有具体内容
""",
            "design_verification": """
你是一位资深的医疗器械验证工程师，负责设计验证活动的实施和记录。请根据以下产品信息和参考范文，生成{chapter}章节内容。

【产品信息】
- 产品名称：{product_name}
- 产品类型：{product_type}
- 产品参数：{product_params}

【重要要求】
请参考上方的【参考范文】和【相关法规标准】部分（如果提供），严格按照参考范文的详细程度生成{chapter}的完整内容。

特别注意：
0. 【强制要求】所有内容必须使用中文，不要使用任何英文单词或短语，除了标准号中必须包含的部分
1. 验证概述要说明验证目的、范围、依据和验证计划
2. 验证方法要详细描述每种验证方法，包括检验、测试、对比、计算等
3. 验证项目和结果要与设计输入一一对应，列出测试数据、结果判定
4. 测试记录要完整，包括测试条件、测试设备、测试人员、测试时间等
5. 不符合项处理要记录不符合项描述、原因分析、纠正措施、验证结果
6. 验证结论要明确说明设计输出是否满足设计输入要求
7. 批准记录要完整，包括批准人、批准日期、批准意见
8. 表格要完整填写，每个单元格都要有具体内容
""",
            "design_validation": """
你是一位资深的医疗器械临床评价专家，负责设计确认活动的实施和记录。请根据以下产品信息和参考范文，生成{chapter}章节内容。

【产品信息】
- 产品名称：{product_name}
- 产品类型：{product_type}
- 产品参数：{product_params}

【重要要求】
请参考上方的【参考范文】和【相关法规标准】部分（如果提供），严格按照参考范文的详细程度生成{chapter}的完整内容。

特别注意：
0. 【强制要求】所有内容必须使用中文，不要使用任何英文单词或短语，除了标准号中必须包含的部分
1. 确认概述要说明确认目的、范围、依据和确认计划
2. 临床评价要详细描述临床试验设计、受试者选择、评价指标、统计分析等
3. 模拟使用测试要描述测试环境、测试人员、测试流程、评价方法
4. 确认结果分析要详细分析测试数据，评价产品的有效性和安全性
5. 确认结论要明确说明产品是否满足预期用途和用户需求
6. 批准记录要完整，包括批准人、批准日期、批准意见
7. 每个条款都要有实质性内容，不能笼统概括
""",
            "design_change": """
你是一位资深的医疗器械变更控制专员，负责设计变更的管理和记录。请根据以下产品信息和参考范文，生成{chapter}章节内容。

【产品信息】
- 产品名称：{product_name}
- 产品类型：{product_type}
- 产品参数：{product_params}

【重要要求】
请参考上方的【参考范文】和【相关法规标准】部分（如果提供），严格按照参考范文的详细程度生成{chapter}的完整内容。

特别注意：
0. 【强制要求】所有内容必须使用中文，不要使用任何英文单词或短语，除了标准号中必须包含的部分
1. 变更概述要详细描述变更内容、变更原因、变更发起部门和人员
2. 变更影响评估要分析变更对产品安全性、有效性、风险管理、注册申报等的影响
3. 变更验证计划要明确验证方法、验证内容、验证标准、验证时间
4. 变更审批要记录各级审批人员的意见和签字
5. 变更实施记录要记录实施过程、实施人员、实施时间、相关文件更新情况
6. 变更追溯性要说明变更涉及的文件、批次、产品范围
7. 表格要完整填写，每个单元格都要有具体内容
""",
            "design_history_file": """
你是一位资深的医疗器械DHF管理人员，负责设计历史文件的整理和归档。请根据以下产品信息和参考范文，生成{chapter}章节内容。

【产品信息】
- 产品名称：{product_name}
- 产品类型：{product_type}
- 产品参数：{product_params}

【重要要求】
请参考上方的【参考范文】和【相关法规标准】部分（如果提供），严格按照参考范文的详细程度生成{chapter}的完整内容。

特别注意：
0. 【强制要求】所有内容必须使用中文，不要使用任何英文单词或短语，除了标准号和缩写（如DHF、DMR等）中必须包含的部分
1. DHF目录要完整列出所有设计历史文件，包括文件编号、版本、日期、编制人等
2. 设计开发计划记录要包括计划文件、更新记录、审批记录
3. 设计输入输出记录要包括输入文件、输出文件、评审记录
4. 设计评审验证确认记录要包括各次评审、验证、确认的完整记录
5. 设计变更记录要包括所有变更的申请、评估、审批、实施、验证记录
6. DHF管理要说明归档、保存期限、查阅权限、借阅记录等要求
7. 表格要完整填写，每个单元格都要有具体内容
""",
            "product_spec": """
你是一位资深的医疗器械注册工程师，精通《医疗器械注册与备案管理办法》和各类医疗器械标准。请根据以下产品信息和参考范文，生成{chapter}章节内容。

【产品信息】
- 产品名称：{product_name}
- 产品类型：{product_type}
- 产品参数：{product_params}

【重要要求】
请参考上方的【参考范文】和【相关法规标准】部分（如果提供），严格按照参考范文的详细程度生成{chapter}的完整内容。

特别注意：
0. 【强制要求】所有内容必须使用中文，不要使用任何英文单词或短语，除了标准号中必须包含的部分
1. 技术参数必须极其详细，有具体的数值范围、精度要求、测试方法
2. 性能指标要分条列出，每条都要有明确的要求和对应的检验方法
3. 引用的国家标准、行业标准要准确，包括标准号和年代号
4. 检验方法要具体，包括使用的仪器设备、操作步骤、判定标准
5. 标志、包装、运输、贮存要求要详细，有具体的条件和期限
6. 所有内容都要像正式注册申报资料一样专业、规范
7. 表格要完整填写，每个单元格都要有具体内容
""",
            "instruction": """
你是一位资深的医疗器械文档工程师，精通《医疗器械说明书和标签管理规定》和各类医疗器械说明书编写规范。请根据以下产品信息和参考范文，生成{chapter}章节内容。

【产品信息】
- 产品名称：{product_name}
- 产品类型：{product_type}
- 产品参数：{product_params}

【重要要求】
请参考上方的【参考范文】和【相关法规标准】部分（如果提供），严格按照参考范文的详细程度生成{chapter}的完整内容。

特别注意：
0. 【强制要求】所有内容必须使用中文，不要使用任何英文单词或短语，除了标准号中必须包含的部分
1. 内容要极其细致，像正式的产品说明书一样完整
2. 使用方法要分步骤描述，每个步骤都要有具体的操作说明
3. 注意事项要全面，包括禁忌症、警告、提示等
4. 维护保养要有具体的方法、频次、检查项目
5. 故障排除要列出常见故障、原因分析、处理方法
6. 使用的语言要通俗易懂，但又要专业规范
7. 每个章节都要有实质性内容，不能简略
""",
            "sop": """
你是一位资深的医疗器械质量体系工程师，精通ISO 13485:2016和《医疗器械生产质量管理规范》，有多年编写SOP的实际经验。请根据以下产品信息和参考范文，生成{chapter}章节内容。

【产品信息】
- 产品名称：{product_name}
- 产品类型：{product_type}
- 产品参数：{product_params}

【重要要求】
请参考上方的【参考范文】和【相关法规标准】部分（如果提供），严格按照参考范文的详细程度生成{chapter}的完整内容。

特别注意：
0. 【强制要求】所有内容必须使用中文，不要使用任何英文单词或短语，除了标准号和缩写（如SOP、QA等）中必须包含的部分
1. 内容必须极其详细和具体，像正式执行的SOP一样具有可操作性
2. 每个操作步骤都要详细描述，包括使用的设备、工具、材料、操作要点、注意事项
3. 职责分工要明确到具体岗位，如操作人员、复核人员、QA人员等
4. 质量控制点要有明确的检验方法、验收标准、记录要求
5. 安全注意事项要具体，包括个人防护、设备安全、环境安全等
6. 引用的文件和记录表单要明确，如文件名、编号、版本
7. 异常情况处理要有具体的流程和责任人
8. 使用规范的SOP语言，表述准确、清晰、无歧义
9. 每个条款都要有实质性内容，不能笼统概括

生成内容的详细程度要像车间实际使用的作业指导书一样，能够指导操作人员完成每一步工作。
""",
        }
        # 为没有专属模板的文档类型提供通用模板，避免回退到风险管理报告
        fallback = templates.get(doc_type)
        if not fallback:
            doc_label = DOC_TYPE_LABELS.get(doc_type, doc_type)
            # 尝试从 prompt_engineer 获取该文档类型的专属提示词
            specific_prompt = DOC_TYPE_SPECIFIC_PROMPTS.get(doc_type, "")
            if specific_prompt:
                specific_section = f"""
【本文档类型专属要求】
{specific_prompt}
"""
            else:
                specific_section = ""

            fallback = f"""
你是一位资深的医疗器械文档编写专家，精通ISO 13485质量管理体系、医疗器械法规和贴敷式胰岛素泵产品特性。请根据以下产品信息和参考范文，生成{{chapter}}章节内容。

【产品信息】
- 产品名称：{{product_name}}
- 产品类型：{{product_type}}
- 产品参数：{{product_params}}
{specific_section}
【重要要求】
请参考上方的【参考范文】和【相关法规标准】部分（如果提供），严格按照参考范文的详细程度和格式生成{{chapter}}的完整内容。

特别注意：
0. 【强制要求】所有内容必须使用中文，不要使用任何英文单词或短语，除了标准号（如GB 42062-2022、ISO 14971:2019等）中必须包含的部分
1. 内容必须极其细致和具体，每个段落都要有实质性内容，不能只写框架标题
2. 技术参数要具体、可测量、有明确的数值范围
3. 表格要填写完整，不能留"（描述）"或"待填写"等占位符
4. 使用专业、规范的医疗器械行业术语
5. 每个章节都要有实质性内容，不能简略
6. 格式上严格参照参考范文的标题层级、表格样式、段落组织

你当前正在生成的是：{doc_label}。请确保生成内容与文档类型严格匹配。
"""
        return fallback

    def _rag_retrieve_for_chapter(
        self,
        chapter_query: str,
        doc_type: str
    ) -> Tuple[list, list]:
        """
        为单个章节执行 RAG 检索（线程安全）

        Returns:
            (chunks, uploads_chunks) — chunks 已合并 uploads_chunks
        """
        # 1. 检查 uploads collection
        uploads_chunks = []
        uploads_has_data = False
        try:
            from app.services.rag.vector_store import VectorStore
            uploads_vs = VectorStore(collection_name="uploads")
            if uploads_vs.count() > 0:
                uploads_has_data = True
                uploads_chunks = uploads_vs.retrieve_hybrid(
                    doc_type=doc_type,
                    query=chapter_query,
                    top_k=5,
                    similarity_threshold=0.3,
                    vector_weight=0.7
                )
                if uploads_chunks:
                    print(f"    附件检索: 从uploads collection检索到 {len(uploads_chunks)} 条相关段落")
        except Exception:
            pass

        # 2. 主知识库检索
        main_k = 13 if not uploads_has_data else 8
        chunks = []
        if self.use_rag:
            for attempt in range(2):
                try:
                    vector_store = _vector_store()
                    chunks = vector_store.retrieve_hybrid(
                        doc_type=doc_type,
                        query=chapter_query,
                        top_k=main_k,
                        similarity_threshold=0.3,
                        vector_weight=0.7
                    )
                    break
                except Exception as e:
                    if attempt == 0:
                        print(f"    RAG检索首次失败，重试中: {e}")
                        continue
                    print(f"    [WARNING] RAG检索最终失败，回退到无RAG模式: {e}")

        # 合并
        chunks.extend(uploads_chunks)
        return chunks, uploads_chunks

    def _search_for_chapter(
        self,
        chapter_name: str,
        product_type: str,
        product_params: str,
        doc_type: str
    ) -> Tuple[str, list, str]:
        """
        为单个章节执行 Web 搜索（线程安全）

        Returns:
            (web_info, downloaded_files, search_method)
        """
        web_info = ""
        downloaded_files = []
        search_method = "none"

        if self.use_web_search and _web_search_service:
            try:
                web_info, downloaded_files = _web_search_service.search_regulations(
                    chapter_name=chapter_name,
                    product_type=product_type,
                    product_params=product_params,
                    max_results=3,
                    enable_deep_scrape=True,
                    enable_file_download=True,
                    doc_type=doc_type
                )
                if web_info:
                    print(f"    Web搜索(Playwright) [{chapter_name}]: 获取到 {len(web_info)} 字符")
                    search_method = "playwright"
                if downloaded_files:
                    print(f"    下载文件 [{chapter_name}]: {len(downloaded_files)} 个")
                    self._add_files_to_knowledge_base(downloaded_files, doc_type)
            except Exception as e:
                print(f"    [WARNING] Web搜索失败 [{chapter_name}]: {e}")

        # 线程安全写入 search_log
        with self._search_log_lock:
            self.search_log.append({
                "chapter": chapter_name,
                "method": search_method
            })

        return web_info, downloaded_files, search_method

    def _build_chapter_prompt(
        self,
        index: int,
        chapter_name: str,
        chapter_query: str,
        chunks: list,
        uploads_chunks: list,
        web_info: str,
        doc_type: str,
        product_name: str,
        product_type: str,
        product_params: str,
        attachment_content: str
    ) -> str:
        """为单个章节构建增强后的 prompt（纯函数，线程安全）"""
        # 构建基础 prompt
        template = self._get_prompt_template(doc_type)
        base_prompt = template.format(
            product_name=str(product_name),
            product_type=str(product_type),
            product_params=str(product_params) if product_params else "无特殊参数",
            chapter=chapter_name
        )

        # 注入 RAG 上下文
        all_chunks = list(chunks) if chunks else []
        if uploads_chunks:
            all_chunks.extend(uploads_chunks)
        if all_chunks and self.use_rag and _rag_prompt_builder:
            enhanced_prompt = _rag_prompt_builder(
                base_prompt=base_prompt,
                doc_type=doc_type,
                product_name=product_name,
                product_type=product_type,
                product_params=product_params,
                retrieved_chunks=all_chunks
            )
            print(f"    RAG增强 [{chapter_name}]: {len(all_chunks)} 条段落")
        elif all_chunks:
            chunk_texts = "\n---\n".join(
                c.get("text", "") for c in all_chunks[:5]
            )
            enhanced_prompt = base_prompt + "\n\n【参考文档片段】\n" + chunk_texts
            print(f"    直接注入 [{chapter_name}]: {len(all_chunks)} 条段落")
        else:
            enhanced_prompt = base_prompt

        # 注入附件内容 — 第一章全文注入，后续章节关键词匹配
        if attachment_content:
            if index == 1:
                enhanced_prompt = enhanced_prompt.rstrip() + "\n\n" + \
                    "【附件文档 - 产品背景信息】\n" + \
                    "以下内容来自用户上传的产品相关文档，请充分利用这些信息生成内容：\n" + \
                    attachment_content[:3000] + "\n"
            else:
                relevant = self._match_relevant_paragraphs(attachment_content, chapter_query, max_chars=1500)
                if relevant:
                    enhanced_prompt = enhanced_prompt.rstrip() + "\n\n" + \
                        "【附件文档 - 相关段落】\n" + relevant + "\n"

        # 注入 Web 搜索上下文
        if web_info:
            enhanced_prompt = enhanced_prompt.rstrip() + "\n\n" + "【相关法规标准 - 来自网络搜索】\n" + web_info + "\n"

        return enhanced_prompt

    def _build_section_prompt(
        self,
        index: int,
        chapter_name: str,
        section_name: str,
        section_query: str,
        chunks: list,
        uploads_chunks: list,
        web_info: str,
        doc_type: str,
        product_name: str,
        product_type: str,
        product_params: str,
        attachment_content: str,
        total_sections: int = 1
    ) -> str:
        """为单个小节构建增强后的 prompt（纯函数，线程安全）"""
        doc_name = DOC_TYPE_LABELS.get(doc_type, doc_type)

        # 构建小节专用 prompt
        section_prompt = f"""你是一位资深的医疗器械注册文档编写专家，拥有15年以上医疗器械行业经验。请根据以下信息，生成文档中指定小节的详细内容。

【文档类型】{doc_name}
【产品名称】{product_name}
【产品类型】{product_type}
【产品参数】{product_params if product_params else '无特殊参数'}

【当前章节】{chapter_name}
【当前小节】{section_name}

【重要要求】
请参考上方的【参考范文】和【相关法规标准】部分（如果提供），生成"{chapter_name}"章节中"{section_name}"小节的完整详细内容。

特别注意：
0. 【强制要求】所有内容必须使用中文，不要使用任何英文单词或短语，除了标准号（如GB 9706.1-2020、ISO 14971:2019等）和必要缩写中必须包含的部分
1. 内容必须极其细致和具体，每个段落都要有实质性内容，不能只写框架标题
2. 技术参数要具体、可测量、有明确的数值范围
3. 表格要填写完整，不能留"（描述）"或"待填写"等占位符
4. 标准号和条款引用要准确
5. 使用专业、规范的医疗器械行业术语
6. 只生成本小节的内容，不要重复其他小节的内容，不要添加章节标题
7. 生成内容的详细程度要像实际可用于注册申报的正式文档一样
"""

        # 注入 RAG 上下文
        all_chunks = list(chunks) if chunks else []
        if uploads_chunks:
            all_chunks.extend(uploads_chunks)
        if all_chunks and self.use_rag and _rag_prompt_builder:
            enhanced_prompt = _rag_prompt_builder(
                base_prompt=section_prompt,
                doc_type=doc_type,
                product_name=product_name,
                product_type=product_type,
                product_params=product_params,
                retrieved_chunks=all_chunks
            )
            print(f"    RAG增强 [{chapter_name}-{section_name}]: {len(all_chunks)} 条段落")
        elif all_chunks:
            chunk_texts = "\n---\n".join(
                c.get("text", "") for c in all_chunks[:5]
            )
            enhanced_prompt = section_prompt + "\n\n【参考文档片段】\n" + chunk_texts
            print(f"    直接注入 [{chapter_name}-{section_name}]: {len(all_chunks)} 条段落")
        else:
            enhanced_prompt = section_prompt

        # 注入附件内容 — 第一个小节全文注入，后续小节关键词匹配
        if attachment_content:
            if index == 1:
                enhanced_prompt = enhanced_prompt.rstrip() + "\n\n" + \
                    "【附件文档 - 产品背景信息】\n" + \
                    "以下内容来自用户上传的产品相关文档，请充分利用这些信息生成内容：\n" + \
                    attachment_content[:3000] + "\n"
            else:
                relevant = self._match_relevant_paragraphs(attachment_content, section_query, max_chars=1500)
                if relevant:
                    enhanced_prompt = enhanced_prompt.rstrip() + "\n\n" + \
                        "【附件文档 - 相关段落】\n" + relevant + "\n"

        # 注入 Web 搜索上下文
        if web_info:
            enhanced_prompt = enhanced_prompt.rstrip() + "\n\n" + "【相关法规标准 - 来自网络搜索】\n" + web_info + "\n"

        return enhanced_prompt

    def _generate_outline(
        self,
        doc_type: str,
        product_name: str,
        product_type: str,
        product_params: str,
        chapters: list,
        doc_name: str
    ) -> list:
        """
        第一阶段：生成文档结构大纲（章节 → 小节）

        调用 LLM 根据已有章节定义，为每个章节生成小节列表。
        返回格式: [{"chapter_id": "ch1", "chapter_name": "概述", "sections": [{"id": "s1.1", "name": "目的", "query": "..."}, ...]}, ...]
        """
        # 构建章节描述
        chapter_descriptions = []
        for ch in chapters:
            chapter_descriptions.append(f"- 第{chapters.index(ch)+1}章: {ch['name']}（检索关键词: {ch['query']}）")

        outline_prompt = f"""你是一位资深的医疗器械注册文档编写专家。请为以下文档生成详细的小节结构大纲。

【文档类型】{doc_name}
【产品名称】{product_name}
【产品类型】{product_type}
【产品参数】{product_params if product_params else '无特殊参数'}

【已有章节结构】
{chr(10).join(chapter_descriptions)}

【任务要求】
请为上述每一章生成详细的小节（section）列表。每个小节应有明确且具体的名称和检索关键词。
要求：
1. 每个章节至少包含2-4个小节，内容粒度要细致
2. 小节名称要具体、有实质性内容指向，不能只是"概述"等笼统标题
3. query 字段是用于知识库检索的关键词，应包含该小节的核心专业术语
4. 小节之间逻辑连贯，覆盖该章节应有的全部内容
5. 贴敷式胰岛素泵是III类有源植入医疗器械，需要严格遵循 ISO 13485、ISO 14971、IEC 62304、IEC 62366、GB 9706.1 等标准

【输出格式】严格返回如下JSON，不要包含任何其他文字或markdown标记：
```json
[
  {{
    "chapter_id": "ch1",
    "chapter_name": "目的和范围",
    "sections": [
      {{"id": "s1.1", "name": "文档目的", "query": "文档目的 编制依据 法规要求"}},
      {{"id": "s1.2", "name": "适用范围", "query": "适用范围 产品覆盖 贴敷式胰岛素泵"}},
      {{"id": "s1.3", "name": "术语和定义", "query": "术语定义 专业术语 缩略语"}}
    ]
  }}
]
```"""

        print(f"\n阶段1: 生成文档结构大纲...")
        outline_text = self._call_api(outline_prompt, max_tokens=4000)

        # 解析 JSON
        outline = self._parse_outline_json(outline_text, chapters)
        total_sections = sum(len(ch.get("sections", [])) for ch in outline)
        print(f"  大纲生成完成: {len(outline)} 章, {total_sections} 小节")

        return outline

    def _parse_outline_json(self, text: str, fallback_chapters: list) -> list:
        """解析 LLM 返回的大纲 JSON，失败时回退到无小节模式"""
        import re

        # 提取 JSON 块
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
        if json_match:
            json_str = json_match.group(1).strip()
        else:
            # 尝试直接解析整个文本
            json_str = text.strip()

        # 移除可能的 BOM 和前后空白
        json_str = json_str.lstrip('\ufeff').strip()

        try:
            outline = json.loads(json_str)
            if not isinstance(outline, list) or len(outline) == 0:
                raise ValueError("outline 不是有效的列表")

            # 验证结构完整性
            for ch in outline:
                if "chapter_name" not in ch or "sections" not in ch:
                    raise ValueError(f"章节结构不完整: {ch}")
                for sec in ch.get("sections", []):
                    if "name" not in sec or "query" not in sec:
                        raise ValueError(f"小节结构不完整: {sec}")

            return outline

        except (json.JSONDecodeError, ValueError) as e:
            print(f"  [WARNING] 大纲JSON解析失败 ({e})，回退到章节级生成")
            # 回退：每个章节作为一个整体小节
            result = []
            for i, ch in enumerate(fallback_chapters):
                result.append({
                    "chapter_id": ch.get("id", f"ch{i+1}"),
                    "chapter_name": ch["name"],
                    "sections": [
                        {"id": f"s{i+1}.1", "name": ch["name"], "query": ch["query"]}
                    ]
                })
            return result

    def _generate_by_chapters(
        self,
        doc_type: str,
        product_name: str,
        product_type: str,
        product_params: str = "",
        attachment_content: str = ""
    ) -> str:
        """
        分小节生成文档内容并汇总（两阶段生成）

        阶段1: 调用LLM生成文档结构大纲（章节 → 小节）
        阶段2: 逐小节执行 RAG检索 + Web搜索 + Prompt构建 + LLM生成
        最后: 按章节→小节顺序组装完整文档
        """
        if not self.model:
            raise ValueError("OLLAMA_MODEL 未设置，请在 .env 中配置")

        chapters = DOC_CHAPTERS.get(doc_type, DEFAULT_CHAPTERS)
        doc_name = DOC_TYPE_LABELS.get(doc_type, doc_type)

        print(f"开始分小节生成文档（共 {len(chapters)} 章）...")
        self.search_log = []  # 重置搜索日志
        self.timing_log = {
            "outline": 0.0,
            "rag_total": 0.0,
            "search_total": 0.0,
            "prompt_total": 0.0,
            "llm_total": 0.0,
            "sections": [],   # [{"chapter_idx", "chapter_name", "section_name", "rag_time", "llm_time"}]
            "chapters": [],   # [{"chapter_idx", "chapter_name", "section_count", "total_time"}]
            "total": 0.0,
        }
        _total_start = time.time()

        # 进度回调辅助
        def _progress(phase, current, total, message):
            if self.progress_callback:
                try:
                    self.progress_callback(phase, current, total, message)
                except Exception:
                    pass

        _progress("outline", 0, 1, "正在生成文档结构大纲...")

        # ========== 阶段1: 生成文档结构大纲 ==========
        _outline_start = time.time()
        outline = self._generate_outline(
            doc_type, product_name, product_type, product_params, chapters, doc_name
        )
        self.timing_log["outline"] = time.time() - _outline_start
        print(f"  [计时] 大纲生成: {self.timing_log['outline']:.2f}s")

        # 展平小节列表，保留章节信息
        all_sections = []
        for ch in outline:
            ch_idx = outline.index(ch) + 1
            for sec in ch.get("sections", []):
                all_sections.append({
                    "chapter_idx": ch_idx,
                    "chapter_name": ch["chapter_name"],
                    "section_id": sec.get("id", f"s{ch_idx}.{len(all_sections)+1}"),
                    "section_name": sec["name"],
                    "section_query": sec["query"]
                })

        total = len(all_sections)
        print(f"\n阶段2: 逐小节生成内容（共 {total} 小节）...")

        _progress("outline", 1, 1, f"大纲生成完成，共 {len(outline)} 章 {total} 小节，开始生成内容...")

        # 小节计时记录：section_id -> {rag_time, search_time, llm_time}
        section_timings = {sec['section_id']: {"rag_time": 0.0, "search_time": 0.0, "llm_time": 0.0}
                           for sec in all_sections}

        # ========== Phase 2a: 串行 RAG 检索（每个小节独立检索） ==========
        print(f"\nPhase 2a: 串行 RAG 检索（{total} 小节）...")
        _rag_phase_start = time.time()
        section_rag = {}  # section_id -> (chunks, uploads_chunks)
        for i, sec in enumerate(all_sections, 1):
            query_str = f"{sec['section_query']} {product_name} {product_type}"
            print(f"  RAG [{i}/{total}] {sec['chapter_name']} - {sec['section_name']}...")
            _t = time.time()
            section_rag[sec['section_id']] = self._rag_retrieve_for_chapter(query_str, doc_type)
            section_timings[sec['section_id']]["rag_time"] = time.time() - _t
            _progress("rag", i, total, f"RAG检索 [{i}/{total}] {sec['chapter_name']} - {sec['section_name']}")
        self.timing_log["rag_total"] = time.time() - _rag_phase_start
        print(f"  [计时] RAG 检索阶段总耗时: {self.timing_log['rag_total']:.2f}s")

        # ========== Phase 2b: 并行 Web 搜索（每个小节独立搜索） ==========
        print(f"\nPhase 2b: 并行 Web 搜索（{total} 小节，最多 {self.max_search_workers} 路并发）...")
        _search_phase_start = time.time()
        section_search = {}  # section_id -> (web_info, downloaded_files, search_method)
        _search_start_per_id = {}

        def _timed_search(sec_id, ch_name, sec_name):
            _start = time.time()
            try:
                result = self._search_for_chapter(
                    f"{ch_name}-{sec_name}",
                    product_type, product_params, doc_type
                )
                return result, time.time() - _start
            except Exception as e:
                return ("", [], f"error: {e}"), time.time() - _start

        with ThreadPoolExecutor(max_workers=self.max_search_workers) as executor:
            future_to_id = {}
            for sec in all_sections:
                future = executor.submit(
                    _timed_search, sec['section_id'], sec['chapter_name'], sec['section_name']
                )
                future_to_id[future] = sec['section_id']

            for future in as_completed(future_to_id):
                sec_id = future_to_id[future]
                try:
                    result, elapsed = future.result()
                    section_search[sec_id] = result
                    section_timings[sec_id]["search_time"] = elapsed
                except Exception as e:
                    print(f"    [WARNING] 搜索线程异常 [{sec_id}]: {e}")
                    section_search[sec_id] = ("", [], "error")
        self.timing_log["search_total"] = time.time() - _search_phase_start
        print(f"  [计时] Web 搜索阶段总耗时（并行墙钟）: {self.timing_log['search_total']:.2f}s")

        # ========== Phase 2c: 串行构建 Prompt（每个小节） ==========
        print(f"\nPhase 2c: 构建 Prompt（{total} 小节）...")
        _prompt_phase_start = time.time()
        section_inputs = []

        for i, sec in enumerate(all_sections, 1):
            sec_id = sec['section_id']
            chunks, uploads_chunks = section_rag[sec_id]
            web_info, _, _ = section_search[sec_id]

            enhanced_prompt = self._build_section_prompt(
                index=i,
                chapter_name=sec['chapter_name'],
                section_name=sec['section_name'],
                section_query=sec['section_query'],
                chunks=chunks,
                uploads_chunks=uploads_chunks,
                web_info=web_info,
                doc_type=doc_type,
                product_name=product_name,
                product_type=product_type,
                product_params=product_params,
                attachment_content=attachment_content,
                total_sections=total
            )

            section_inputs.append((i, sec, enhanced_prompt))
            print(f"  [{i}/{total}] {sec['chapter_name']} - {sec['section_name']} Prompt 构建完成 ({len(enhanced_prompt)} 字符)")
        self.timing_log["prompt_total"] = time.time() - _prompt_phase_start
        print(f"  [计时] Prompt 构建阶段总耗时: {self.timing_log['prompt_total']:.2f}s")

        # ========== Phase 2d: 并行调用 LLM API（逐小节） ==========
        print(f"\nPhase 2d: 并行调用LLM API（{total} 小节，最多 {self.max_concurrent} 路并发）...")
        _llm_phase_start = time.time()

        results = {}  # index -> (content, error)
        completed_count = 0
        _progress_lock = threading.Lock()

        def _timed_call_api(prompt):
            _start = time.time()
            try:
                content = self._call_api(prompt)
                return content, time.time() - _start, None
            except Exception as e:
                return "", time.time() - _start, e

        with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
            future_to_idx = {
                executor.submit(_timed_call_api, prompt): (idx, sec)
                for idx, sec, prompt in section_inputs
            }
            for future in as_completed(future_to_idx):
                idx, sec = future_to_idx[future]
                try:
                    content, elapsed, err = future.result()
                    section_timings[sec['section_id']]["llm_time"] = elapsed
                    if err is not None:
                        results[idx] = ("", str(err))
                        print(f"  [{idx}/{total}] {sec['chapter_name']} - {sec['section_name']} 失败 ({elapsed:.2f}s): {err}")
                    else:
                        results[idx] = (content, None)
                        print(f"  [{idx}/{total}] {sec['chapter_name']} - {sec['section_name']} 完成 ({len(content)} 字符, {elapsed:.2f}s)")
                except Exception as e:
                    results[idx] = ("", str(e))
                    print(f"  [{idx}/{total}] {sec['chapter_name']} - {sec['section_name']} 异常: {e}")
                finally:
                    with _progress_lock:
                        completed_count += 1
                        _progress("generate", completed_count, total,
                                  f"内容生成 [{completed_count}/{total}] {sec['chapter_name']} - {sec['section_name']}")
        self.timing_log["llm_total"] = time.time() - _llm_phase_start
        print(f"  [计时] LLM 调用阶段总耗时（并行墙钟）: {self.timing_log['llm_total']:.2f}s")

        # ========== Phase 2e: 按章节→小节顺序组装文档 ==========
        full_document = []
        full_document.append(f"# {product_name} - {doc_name}\n\n")
        full_document.append(f"**产品信息：**\n")
        full_document.append(f"- 产品名称：{product_name}\n")
        full_document.append(f"- 产品类型：{product_type}\n")
        full_document.append(f"- 产品参数：{product_params if product_params else '无'}\n")
        if attachment_content:
            full_document.append(f"\n**附件材料：** 已提供产品相关参考文档\n")
        full_document.append("---\n\n")

        # 按章节分组
        current_chapter_idx = None
        for idx, sec, prompt in section_inputs:
            ch_idx = sec['chapter_idx']
            ch_name = sec['chapter_name']
            sec_name = sec['section_name']

            # 新章节开始
            if ch_idx != current_chapter_idx:
                current_chapter_idx = ch_idx
                full_document.append(f"## 第{ch_idx}章 {ch_name}\n\n")

            # 小节标题
            full_document.append(f"### {sec_name}\n\n")
            content, error = results.get(idx, ("", "未知错误"))
            if error:
                full_document.append(f"（内容生成失败: {error}）\n\n")
            else:
                full_document.append(content)
                full_document.append("\n\n")

        # ========== 计时汇总 ==========
        # 记录每个小节的耗时
        for sec in all_sections:
            t = section_timings[sec['section_id']]
            self.timing_log["sections"].append({
                "chapter_idx": sec['chapter_idx'],
                "chapter_name": sec['chapter_name'],
                "section_name": sec['section_name'],
                "rag_time": round(t["rag_time"], 2),
                "search_time": round(t["search_time"], 2),
                "llm_time": round(t["llm_time"], 2),
                "section_total": round(t["rag_time"] + t["search_time"] + t["llm_time"], 2),
            })

        # 按章节聚合：每章节耗时 = 该章节所有小节的 (rag + search + llm) 之和
        chapter_agg = {}  # ch_idx -> {chapter_name, section_count, rag, search, llm, total}
        for sec in all_sections:
            ch_idx = sec['chapter_idx']
            t = section_timings[sec['section_id']]
            if ch_idx not in chapter_agg:
                chapter_agg[ch_idx] = {
                    "chapter_idx": ch_idx,
                    "chapter_name": sec['chapter_name'],
                    "section_count": 0,
                    "rag_time": 0.0,
                    "search_time": 0.0,
                    "llm_time": 0.0,
                    "total_time": 0.0,
                }
            agg = chapter_agg[ch_idx]
            agg["section_count"] += 1
            agg["rag_time"] += t["rag_time"]
            agg["search_time"] += t["search_time"]
            agg["llm_time"] += t["llm_time"]
            agg["total_time"] += t["rag_time"] + t["search_time"] + t["llm_time"]

        for ch_idx in sorted(chapter_agg.keys()):
            agg = chapter_agg[ch_idx]
            self.timing_log["chapters"].append({
                "chapter_idx": agg["chapter_idx"],
                "chapter_name": agg["chapter_name"],
                "section_count": agg["section_count"],
                "rag_time": round(agg["rag_time"], 2),
                "search_time": round(agg["search_time"], 2),
                "llm_time": round(agg["llm_time"], 2),
                "total_time": round(agg["total_time"], 2),
            })

        self.timing_log["total"] = round(time.time() - _total_start, 2)
        self.timing_log["outline"] = round(self.timing_log["outline"], 2)
        self.timing_log["rag_total"] = round(self.timing_log["rag_total"], 2)
        self.timing_log["search_total"] = round(self.timing_log["search_total"], 2)
        self.timing_log["prompt_total"] = round(self.timing_log["prompt_total"], 2)
        self.timing_log["llm_total"] = round(self.timing_log["llm_total"], 2)

        # 打印汇总表
        print("\n" + "=" * 80)
        print("文档生成计时汇总")
        print("=" * 80)
        print(f"  大纲生成:       {self.timing_log['outline']:>8.2f}s")
        print(f"  RAG 检索阶段:   {self.timing_log['rag_total']:>8.2f}s")
        print(f"  Web 搜索阶段:   {self.timing_log['search_total']:>8.2f}s (并行墙钟)")
        print(f"  Prompt 构建:    {self.timing_log['prompt_total']:>8.2f}s")
        print(f"  LLM 调用阶段:   {self.timing_log['llm_total']:>8.2f}s (并行墙钟)")
        print("-" * 80)
        print(f"  {'章节':<40} {'小节数':>6} {'章节总耗时':>12}")
        print("-" * 80)
        for ch in self.timing_log["chapters"]:
            name = f"第{ch['chapter_idx']}章 {ch['chapter_name']}"
            if len(name) > 38:
                name = name[:37] + "…"
            print(f"  {name:<40} {ch['section_count']:>6} {ch['total_time']:>10.2f}s")
        print("-" * 80)
        print(f"  整篇文档总耗时: {self.timing_log['total']:>8.2f}s")
        print("=" * 80)

        print(f"\n文档生成完成！总计 {len(''.join(full_document))} 字符")
        return "".join(full_document)

    # 保留旧方法名作为别名，兼容现有调用
    generate_content_with_rag = _generate_by_chapters

    def _call_api(self, prompt: str, max_tokens: int = 6000) -> str:
        """内部方法：调用本地Ollama API（带重试机制）"""
        headers = {
            "Content-Type": "application/json; charset=utf-8"
        }

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "options": {
                "temperature": 0.7,
                "num_predict": max_tokens,
            },
            "stream": False,
        }

        last_error = None
        for attempt in range(3):
            try:
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    timeout=(30, 180)
                )
                response.raise_for_status()

                result = response.json()
                # Ollama /api/chat 格式: {"message": {"content": "..."}}
                message = result.get("message", {})
                if message:
                    return message.get("content", "")
                return ""

            except requests.exceptions.Timeout as e:
                last_error = TimeoutError(f"Ollama API 请求超时 (第{attempt + 1}次尝试)")
                if attempt < 2:
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                    continue
                raise last_error
            except requests.exceptions.RequestException as e:
                last_error = ConnectionError(f"Ollama API 请求失败 (第{attempt + 1}次尝试): {str(e)}")
                if attempt < 2:
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                    continue
                raise last_error
            except (KeyError, IndexError, json.JSONDecodeError) as e:
                raise ValueError(f"解析API响应失败: {str(e)}")

    def revise_content(
        self,
        current_content: str,
        feedback: str,
        doc_type: str,
        product_name: str,
        product_type: str,
        product_params: str = ""
    ):
        """
        基于用户反馈修订文档内容（增量修改模式）

        策略：
        1. 将文档按 ## 标题拆分为章节
        2. 对用户反馈中涉及的内容进行 RAG 检索，获取相关参考资料
        3. 将全部章节 + 用户反馈 + RAG 上下文发给 LLM
        4. LLM 返回需要修改的章节索引及修改后内容
        5. 在原文档中用修改后的章节替换对应位置
        6. 未修改的章节保持原样不变

        Returns:
            (revised_content: str, diff_data: dict|None)
        """
        if not self.model:
            raise ValueError("OLLAMA_MODEL 未设置，请在 .env 中配置")

        # 按 ## 标题拆分为章节
        sections = self._split_sections(current_content)
        if len(sections) <= 1:
            # 文档没有章节结构，使用简单全文修订
            result = self._revise_simple(current_content, feedback, doc_type,
                                         product_name, product_type, product_params)
            diff_data = self._compute_full_diff(current_content, result, feedback)
            return result, diff_data

        # ── RAG 检索：基于用户反馈获取相关知识库参考资料 ──
        rag_context = ""
        if self.use_rag:
            try:
                doc_label = DOC_TYPE_LABELS.get(doc_type, doc_type)
                rag_query = f"{doc_label} {feedback[:300]}"
                rag_chunks, _ = self._rag_retrieve_for_chapter(rag_query, doc_type)
                if rag_chunks:
                    lines = ["\n\n【参考范文 - 请严格参照此详细程度、格式和专业深度进行修改】"]
                    for j, chunk in enumerate(rag_chunks[:8], 1):
                        source = chunk.get("source_file", chunk.get("source", "未知"))
                        score = chunk.get("similarity", chunk.get("score", 0))
                        text = chunk.get("text", chunk.get("content", ""))
                        lines.append(
                            f"\n[参考{j}] 来源: {source} (相关度:{score:.2f})\n{text}"
                        )
                    rag_context = "".join(lines)
                    print(f"[revise_content] RAG检索到 {len(rag_chunks[:8])} 条参考资料")
            except Exception as e:
                print(f"[revise_content] RAG检索失败，继续无RAG模式: {e}")

        # 构建定位+修改的 Prompt
        revision_prompt = self._build_targeted_revision_prompt(
            sections=sections,
            feedback=feedback,
            doc_type=doc_type,
            product_name=product_name,
            product_type=product_type,
            product_params=product_params,
            rag_context=rag_context,
        )

        try:
            result = self._call_api(revision_prompt, max_tokens=8000)
            changes = self._parse_revision_result(result, len(sections))

            if not changes:
                # LLM 认为无需修改，返回原文档
                return current_content, None

            # 计算差异数据
            diff_data = self._compute_section_diffs(sections, changes)
            # 应用修改：用修订后的章节替换原章节
            revised = self._apply_section_changes(sections, changes)
            return revised, diff_data

        except Exception as e:
            # 增量修改失败时回退到简单全文修订
            try:
                result = self._revise_simple(current_content, feedback, doc_type,
                                             product_name, product_type, product_params)
                diff_data = self._compute_full_diff(current_content, result, feedback)
                return result, diff_data
            except Exception:
                return current_content + f"\n\n---\n> **修订失败**: {str(e)}\n", None

    def _split_sections(self, content: str) -> list:
        """将文档按 ## 标题拆分为章节列表，每个元素为 {title, body, start, end}"""
        import re
        # 匹配 ## 开头的标题行
        pattern = r'^## (.+)$'
        lines = content.split('\n')
        sections = []
        current_title = '_head'  # 第一个 ## 之前的内容
        current_start = 0
        current_lines = []

        for i, line in enumerate(lines):
            m = re.match(pattern, line.strip())
            if m:
                # 保存前一个章节
                if current_lines or current_title == '_head':
                    sections.append({
                        'index': len(sections),
                        'title': current_title,
                        'body': '\n'.join(current_lines),
                        'start_line': current_start,
                        'end_line': i - 1
                    })
                current_title = m.group(1).strip()
                current_start = i
                current_lines = [line]
            else:
                current_lines.append(line)

        # 最后一个章节
        if current_lines:
            sections.append({
                'index': len(sections),
                'title': current_title,
                'body': '\n'.join(current_lines),
                'start_line': current_start,
                'end_line': len(lines) - 1
            })

        return sections

    def _build_targeted_revision_prompt(
        self,
        sections: list,
        feedback: str,
        doc_type: str,
        product_name: str,
        product_type: str,
        product_params: str,
        rag_context: str = "",
    ) -> str:
        """构建定位式修订 Prompt — 让 LLM 只输出需要修改的章节"""
        doc_name = DOC_TYPE_LABELS.get(doc_type, doc_type)

        # 获取文档类型专属专家提示词
        expert_prompt = ""
        try:
            from app.services.prompt_engineer import DOC_TYPE_SPECIFIC_PROMPTS
            expert_prompt = DOC_TYPE_SPECIFIC_PROMPTS.get(doc_type, "")
        except ImportError:
            pass
        expert_section = f"\n\n{expert_prompt}" if expert_prompt else ""

        # 构建章节索引列表（只发送标题+摘要，节省 token）
        section_index = []
        for sec in sections:
            body_preview = sec['body'][:200].replace('\n', ' ').strip()
            section_index.append(f"[{sec['index']}] {sec['title']} — 内容预览: {body_preview}...")

        # 构建完整文档（供 LLM 定位）
        full_doc_parts = []
        for sec in sections:
            full_doc_parts.append(sec['body'] if sec['body'].startswith('##') else sec['body'])
        full_doc = '\n'.join(full_doc_parts)

        # 文档过长时使用摘要
        max_doc_len = 30000
        if len(full_doc) > max_doc_len:
            half = max_doc_len // 2
            full_doc = full_doc[:half] + "\n\n...（中间内容省略）...\n\n" + full_doc[-half:]

        prompt = f"""你是医疗器械注册文档编辑专家。你需要根据用户反馈，**对现有文档进行精准的局部修改**，同时保持修改后章节的详细程度和专业深度不低于原章节。{expert_section}

【产品信息】
- 产品名称：{product_name}
- 产品类型：{product_type}

【文档类型】{doc_name}

【修改质量要求】
- 内容必须极其细致和具体，每个段落都要有实质性内容
- 所有标准条款引用必须有明确的条款号
- 技术参数要具体、可测量、有明确的数值范围
- 表格要填写完整，不能留"(描述)"或"待填写"等占位符
- 修改后内容的详细程度要与首次生成的其他章节保持一致，像实际可用于注册申报的正式文档一样
- 如果用户指令要求添加新内容，应展开详细描述（每个要点至少200字），而非只加一句话

【章节索引】
{chr(10).join(section_index)}

【完整文档内容】（共 {len(sections)} 个章节）
{full_doc}
{rag_context}
【用户修改意见】
{feedback}

【任务说明】
请仔细阅读用户修改意见，找出文档中需要修改的章节，然后仅输出需要修改的章节的新内容。

【输出格式 — 严格按此格式】
对于每个需要修改的章节，按以下格式输出：

---SECTION: <章节索引号>---
<该章节修改后的完整内容（包含 ## 标题行），Markdown 格式>
---END---

如果某个章节不需要修改，则不要输出该章节。
如果用户意见不涉及任何具体章节修改，请只输出一行：NO_CHANGE

【修改原则】
1. 只在原章节内容基础上进行用户要求的修改，不要重写整个章节
2. 如果用户要求增加内容，在原章节内容的合适位置插入
3. 如果用户要求更正数据，只修改数据，保留周围文字不变
4. 未提及修改的章节不要输出
5. 修改后的章节保持原有的 Markdown 格式和层级结构

请输出："""
        return prompt

    def _parse_revision_result(self, result: str, section_count: int) -> dict:
        """解析 LLM 返回的修订结果，返回 {section_index: revised_body} 字典"""
        import re

        if not result or 'NO_CHANGE' in result.upper():
            return {}

        changes = {}
        pattern = r'---SECTION:\s*(\d+)\s*---\s*\n(.*?)\n---END---'
        matches = re.findall(pattern, result, re.DOTALL)

        for idx_str, new_body in matches:
            try:
                idx = int(idx_str)
                if 0 <= idx < section_count:
                    changes[idx] = new_body.strip()
            except ValueError:
                continue

        return changes

    def _apply_section_changes(self, sections: list, changes: dict) -> str:
        """将修改后的章节应用到原文档中"""
        result_lines = []
        all_lines = []
        # 重建完整行列表
        combined = []
        for sec in sections:
            combined.append(sec['body'])

        full_text = '\n'.join(combined)
        lines = full_text.split('\n')

        # 对每个 section，如果它在 changes 中，使用修改后的内容
        # 否则使用原始内容
        new_sections = []
        for sec in sections:
            if sec['index'] in changes:
                new_sections.append(changes[sec['index']])
            else:
                new_sections.append(sec['body'])

        return '\n'.join(new_sections)

    def _compute_section_diffs(self, sections: list, changes: dict) -> dict:
        """为被修改的章节生成左右对比HTML表格，返回可供前端渲染的差异数据"""
        import difflib

        section_diffs = {}
        for sec in sections:
            idx = sec['index']
            if idx in changes:
                old_text = sec['body']
                new_text = changes[idx]
                old_lines = old_text.splitlines(keepends=True)
                new_lines = new_text.splitlines(keepends=True)

                html_diff = difflib.HtmlDiff().make_table(
                    new_lines, old_lines,
                    fromdesc="当前版本",
                    todesc="原版",
                    context=True,
                    numlines=2
                )
                section_diffs[str(idx)] = {
                    "title": sec['title'],
                    "diff_html": html_diff
                }

        return {
            "mode": "sectional",
            "sections": section_diffs
        }

    def _compute_full_diff(self, old_content: str, new_content: str, feedback: str) -> dict:
        """为全文修订模式生成左右对比HTML表格"""
        import difflib

        old_lines = old_content.splitlines(keepends=True)
        new_lines = new_content.splitlines(keepends=True)
        html_diff = difflib.HtmlDiff().make_table(
            new_lines, old_lines,
            fromdesc="当前版本",
            todesc="原版",
            context=True,
            numlines=2
        )
        return {
            "mode": "full",
            "feedback": feedback[:80],
            "diff_html": html_diff
        }

    def _revise_simple(
        self,
        current_content: str,
        feedback: str,
        doc_type: str,
        product_name: str,
        product_type: str,
        product_params: str
    ) -> str:
        """简单全文修订（无章节结构时的回退方案）"""
        doc_name = DOC_TYPE_LABELS.get(doc_type, doc_type)

        # 获取文档类型专属专家提示词
        expert_prompt = ""
        try:
            from app.services.prompt_engineer import DOC_TYPE_SPECIFIC_PROMPTS
            expert_prompt = DOC_TYPE_SPECIFIC_PROMPTS.get(doc_type, "")
        except ImportError:
            pass
        expert_section = f"\n\n{expert_prompt}" if expert_prompt else ""

        prompt = f"""你是医疗器械注册文档编辑专家。请根据用户反馈修改以下文档，保持修改后内容的详细程度和专业深度不低于原文档。{expert_section}

【产品信息】
- 产品名称：{product_name}
- 产品类型：{product_type}

【当前文档】
{current_content[:15000]}

【用户修改意见】
{feedback}

【要求】
- 内容必须极其细致和具体，每个段落都要有实质性内容
- 所有标准条款引用必须有明确的条款号
- 技术参数要具体、可测量、有明确的数值范围
- 表格要填写完整，不能留"(描述)"或"待填写"等占位符
- 修改后内容的详细程度要与首次生成的其他章节保持一致
- 如果用户指令要求添加新内容，应展开详细描述（每个要点至少200字）
- 只修改用户意见指出的部分，其余内容原样保留
输出修改后的完整文档（Markdown 格式）。
在文档末尾添加：<!-- 修订说明: {feedback[:80]} -->

请输出："""
        revised = self._call_api(prompt, max_tokens=16000)
        return revised if revised and len(revised.strip()) >= 50 else current_content

    def generate_content_with_fallback(
        self,
        doc_type: str,
        product_name: str,
        product_type: str,
        product_params: str = "",
        attachment_content: str = ""
    ) -> str:
        """
        生成文档内容（分章节生成模式，统一入口）

        优先使用分章节生成，当API不可用时降级为占位符
        """
        try:
            # 统一使用分章节生成模式
            return self._generate_by_chapters(
                doc_type, product_name, product_type, product_params, attachment_content
            )
        except Exception as e:
            return self._generate_placeholder(doc_type, product_name, product_type, product_params, str(e))

    def _generate_placeholder(
        self,
        doc_type: str,
        product_name: str,
        product_type: str,
        product_params: str,
        error: str
    ) -> str:
        """生成占位符内容（当API不可用时）"""
        return f"""# {product_name} - {DOC_TYPE_LABELS.get(doc_type, doc_type)}

**产品信息：**
- 产品名称：{product_name}
- 产品类型：{product_type}
- 产品参数：{product_params if product_params else '无'}

---

## 文档内容

> **注意：** 当前AI服务暂时不可用（{error}），请手动填写以下内容。

---
*由 QMS Document Generator 自动生成 | {doc_type}*
"""

    def _match_relevant_paragraphs(self, text: str, query: str, max_chars: int = 1500) -> str:
        """
        从文本中简单关键词匹配并返回与查询相关的段落

        Args:
            text: 完整文本
            query: 查询关键词
            max_chars: 返回最大字符数

        Returns:
            匹配到的相关段落文本
        """
        if not text or not query:
            return ""

        # 提取关键词（简单分词）
        keywords = [w.strip() for w in query.replace(" ", "").split("_") if w.strip()]
        if not keywords:
            return ""

        # 按段落分割
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
        if not paragraphs:
            return ""

        # 为每个段落打分
        scored = []
        for para in paragraphs:
            score = 0
            para_lower = para.lower()
            for kw in keywords:
                for char in kw:
                    if char in para_lower:
                        score += 1
            if score > 0:
                scored.append((score, para))

        # 按分数排序，取前几个
        scored.sort(key=lambda x: x[0], reverse=True)

        result = []
        total_chars = 0
        for _, para in scored:
            if total_chars + len(para) > max_chars:
                # 截断最后一段以不超过限制
                remaining = max_chars - total_chars
                if remaining > 50:
                    result.append(para[:remaining] + "...")
                break
            result.append(para)
            total_chars += len(para) + 2

        return "\n\n".join(result) if result else ""

    def _add_files_to_knowledge_base(self, file_paths: List[str], doc_type: str):
        """
        将下载的文件添加到知识库中

        Args:
            file_paths: 文件路径列表
            doc_type: 文档类型
        """
        if not file_paths:
            return

        try:
            from app.services.rag.vector_store import VectorStore
            from app.services.rag.ingest import ingest_files

            print(f"    [知识库] 正在将 {len(file_paths)} 个文件添加到知识库...")

            # 摄入文件到向量库
            result = ingest_files(
                file_paths=file_paths,
                collection_name="all",
                force_doc_type=doc_type
            )

            print(f"    [知识库] 添加完成: {result['processed_docs']} 文档, {result['total_chunks']} chunks")

            # 重新初始化RAG组件（可选）
            global _rag_available, _vector_store, _rag_prompt_builder
            _rag_available = False
            _vector_store = None
            _rag_prompt_builder = None
            # 下次使用时会重新初始化

        except Exception as e:
            print(f"    [知识库] 添加失败: {e}")

