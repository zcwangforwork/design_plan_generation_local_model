"""
Mermaid 流程图渲染服务

用 Playwright（同步 API）+ 本地 mermaid.min.js 将 mermaid 源码渲染为 PNG 图片，
供 template.py 在 Markdown → Word 转换时把 ```mermaid 代码块插入为真实图片。

设计要点:
- 使用 playwright.sync_api（同步版），因为 template._parse_and_fill 是同步函数，
  运行在 asyncio.to_thread 的独立线程中，无事件循环冲突。
- mermaid.js 来自本地 node_modules（npm install mermaid），离线可用，数据不出企业。
- 任何失败（缺依赖/语法错误/超时）都返回 None，由调用方降级为代码文本，不中断文档生成。
"""

import os

# node_modules 相对本文件定位到项目根: app/services/ -> 项目根
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_MERMAID_JS_CANDIDATES = [
    os.path.join(_PROJECT_ROOT, "node_modules", "mermaid", "dist", "mermaid.min.js"),
    os.path.join(_PROJECT_ROOT, "node_modules", "mermaid", "dist", "mermaid.js"),
]

# 页面模板：用占位符替换，避免 f-string 与 mermaid 源码中的 {} 冲突
_HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head><meta charset="utf-8"></head>
<body>
<pre class="mermaid">__CODE__</pre>
<script>__MERMAID_JS__</script>
<script>
(function () {
  try {
    mermaid.initialize({startOnLoad: false, theme: 'default', flowchart: {useMaxWidth: true, htmlLabels: true}});
    mermaid.run({querySelector: '.mermaid'}).then(function () {
      document.body.setAttribute('data-render', 'ok');
    }).catch(function (err) {
      document.body.setAttribute('data-render', 'error');
      document.body.setAttribute('data-error', String(err && err.message || err));
    });
  } catch (e) {
    document.body.setAttribute('data-render', 'error');
    document.body.setAttribute('data-error', String(e && e.message || e));
  }
})();
</script>
</body>
</html>
"""


def _find_mermaid_js() -> str | None:
    """返回本地 mermaid.min.js 的绝对路径；不存在返回 None。"""
    for path in _MERMAID_JS_CANDIDATES:
        if os.path.exists(path):
            return path
    return None


def render_mermaid_to_png(code: str, timeout_ms: int = 15000) -> bytes | None:
    """将 mermaid 源码渲染为 PNG 图片字节流。

    Args:
        code: mermaid 语法源码（不含 ```mermaid 围栏），如 "flowchart TD; A-->B;"。
        timeout_ms: 等待渲染完成的超时（毫秒）。

    Returns:
        PNG 图片 bytes；任何失败（缺 playwright / 缺 mermaid.js / 语法错误 / 超时）返回 None。
    """
    if not code or not code.strip():
        return None

    mermaid_js_path = _find_mermaid_js()
    if mermaid_js_path is None:
        return None

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return None

    try:
        with open(mermaid_js_path, encoding="utf-8") as f:
            mermaid_js = f.read()
    except OSError:
        return None

    html = (_HTML_TEMPLATE
            .replace("__CODE__", code.strip())
            .replace("__MERMAID_JS__", mermaid_js))

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            try:
                page = browser.new_page(
                    viewport={"width": 1400, "height": 900},
                    device_scale_factor=2,
                )
                page.set_content(html)
                # 等待渲染完成（成功或失败标志置位）
                page.wait_for_function(
                    "document.body.getAttribute('data-render') === 'ok' || "
                    "document.body.getAttribute('data-render') === 'error'",
                    timeout=timeout_ms,
                )
                if page.get_attribute("body", "data-render") != "ok":
                    return None
                svg = page.locator(".mermaid svg").first
                png = svg.screenshot(type="png")
                return png
            finally:
                browser.close()
    except Exception:
        return None
