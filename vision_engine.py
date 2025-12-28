from __future__ import annotations

"""
视觉理解模块（GLM-4V / Qwen-VL / Qwen-Omni）
--------------------------------
目标：把“用户上传到 out/{session_id}/ 的图片”转换为可注入到文本 LLM 的“视觉描述”。

实现策略：
- 读取本地图片文件
- （可选）用 Pillow 压缩/缩放，避免 base64 过大
- 以 OpenAI-compatible 的多模态 messages 格式调用：
  - 智谱 GLM-4V：ZHIPU_BASE_URL (chat/completions)
  - 千问 Qwen-VL / Qwen-Omni：DashScope compatible-mode chat/completions
    - 示例模型：qwen-vl-plus / qwen-omni-turbo / qwen3-omni-flash

注意：
- 该模块只负责“看图 -> 生成文字描述”，不负责最终回答。
- 后端可把返回的文字描述注入 Ask prompt / Agent data_context，从而实现“能看图”。
"""

import base64
import importlib
import mimetypes
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _truncate(s: str, n: int) -> str:
    t = str(s or "")
    return t if len(t) <= n else (t[:n] + "\n...(截断)")


def _load_image_bytes(image_path: str) -> Optional[Tuple[bytes, str]]:
    """
    image_path: 前端传来的相对路径："{session_id}/{filename}"
    实际文件位置：out/{image_path}
    """
    p = str(image_path or "").strip()
    if not p:
        return None

    fp = Path(p)
    if not str(fp).startswith("out"):
        fp = Path("out") / fp

    if not fp.exists() or not fp.is_file():
        return None

    try:
        data = fp.read_bytes()
    except Exception:
        return None

    mime = mimetypes.guess_type(fp.name)[0] or "image/png"
    return data, mime


def _shrink_image_bytes(
    data: bytes,
    mime: str,
    *,
    max_edge: int = 1024,
    jpeg_quality: int = 85,
) -> Tuple[bytes, str]:
    """
    尝试用 Pillow 缩放/压缩图片以降低 base64 体积。
    - 默认输出 JPEG（更小），mime 变为 image/jpeg
    - 若 Pillow 不可用或处理失败，则原样返回
    """
    try:
        from PIL import Image  # type: ignore
    except Exception:
        return data, mime

    try:
        img = Image.open(BytesIO(data))
    except Exception:
        return data, mime

    # 尝试处理 EXIF 方向
    try:
        from PIL import ImageOps  # type: ignore

        img = ImageOps.exif_transpose(img)
    except Exception:
        pass

    try:
        w, h = img.size
    except Exception:
        return data, mime

    if max(w, h) > int(max_edge):
        scale = float(max_edge) / float(max(w, h))
        new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
        try:
            # Pillow>=10
            img = img.resize(new_size, resample=Image.Resampling.LANCZOS)
        except Exception:
            img = img.resize(new_size)

    # JPEG 需要 RGB
    try:
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
    except Exception:
        return data, mime

    out = BytesIO()
    try:
        img.save(out, format="JPEG", quality=int(jpeg_quality), optimize=True)
        return out.getvalue(), "image/jpeg"
    except Exception:
        return data, mime


def _to_data_url(data: bytes, mime: str) -> str:
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"


def choose_vision_provider(api_keys: Dict[str, str], preference: str = "qwen") -> Optional[str]:
    """
    固定返回 "qwen"（使用 qwen-omni-turbo 作为唯一的视觉理解模型）
    """
    keys = api_keys or {}
    if bool(str(keys.get("qwen", "") or "").strip()):
        return "qwen"
    return None


def _extract_openai_compatible_text(obj: Any) -> Optional[str]:
    """
    尽量兼容 OpenAI-compatible 的返回格式。
    """
    try:
        text = obj["choices"][0]["message"]["content"]
        return str(text).strip() if text is not None else None
    except Exception:
        pass
    # 兜底：部分兼容网关可能返回 output_text
    try:
        text = obj.get("output_text")
        return str(text).strip() if text is not None else None
    except Exception:
        return None


def call_vision_chat(
    *,
    provider: str,
    api_key: str,
    url: str,
    model: str,
    user_query: str,
    image_data_url: str,
    timeout: int = 120,
    temperature: float = 0.2,
) -> Dict[str, Any]:
    """
    调用视觉模型返回文字描述（不做解析/结构化）。
    """
    try:
        requests = importlib.import_module("requests")
    except Exception as e:
        return {"ok": False, "error": "缺少 requests 依赖", "detail": str(e)}

    prompt = (
        "你是一名视觉理解助手。\n"
        "请结合【用户问题】全面理解图片内容，提取所有对回答有用的信息。\n"
        "要求：\n"
        "1) 用 3-6 句中文概括图片的整体内容和主题；\n"
        "2) **文字信息**：如果图片中有任何文字（标题、标签、说明、注释等），完整准确地抄录所有文字内容；\n"
        "3) **表格数据**：如果图片包含表格，完整列出所有表头、行、列的数据，保持原始格式和数值精度；\n"
        "4) **图表信息**：如果图片包含图表（折线图、柱状图、散点图等），描述图表类型、坐标轴标签、数据趋势、关键数值点；\n"
        "5) **标准/规范/限值**：如果图片包含标准、规范、要求、限值等信息，完整列出所有指标名称、符号（≥/≤/等）、数值、单位、等级划分等；\n"
        "6) **界面元素**：如果是界面截图，描述所有可见的按钮、字段、菜单、状态、错误提示等；\n"
        "7) **图像内容**：描述图片中的图像、图标、符号、颜色、布局等视觉元素；\n"
        "8) **其他信息**：提取任何其他可能对回答有用的信息（公式、代码片段、流程、关系等）；\n"
        "9) 确保信息完整准确，不要遗漏任何细节；不确定的内容明确标注；不要猜测或编造信息。\n\n"
        f"【用户问题】\n{_truncate(user_query, 1500)}\n"
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ],
        }
    ]

    payload = {
        "model": str(model),
        "messages": messages,
        "temperature": float(temperature),
    }
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    try:
        resp = requests.post(str(url), headers=headers, json=payload, timeout=int(timeout))
    except Exception as e:
        return {"ok": False, "error": "请求失败", "detail": str(e)}

    if resp.status_code != 200:
        return {"ok": False, "error": f"HTTP {resp.status_code}", "detail": _truncate(resp.text or "", 2000)}

    try:
        data = resp.json()
    except Exception:
        return {"ok": False, "error": "响应非JSON", "detail": _truncate(resp.text or "", 2000)}

    text = _extract_openai_compatible_text(data)
    if not text:
        return {"ok": False, "error": "模型未返回内容", "detail": _truncate(str(data), 2000)}

    return {"ok": True, "text": text, "raw": data}


def describe_images(
    *,
    image_paths: List[str],
    user_query: str,
    api_keys: Dict[str, str],
    preference: str,
    zhipu_url: str,
    qwen_url: str,
    zhipu_model: str = "glm-4v",  # 不再使用，保留以兼容接口
    qwen_model: str = "qwen-vl-plus",  # 不再使用，保留以兼容接口
    max_images: int = 3,
    max_edge: int = 1024,
    max_bytes: int = 2_000_000,
    timeout: int = 120,
) -> Dict[str, Any]:
    """
    批量生成图片视觉描述（统一选择一个 provider，以减少不确定性）。
    返回：
    {
      provider, model, results:[{path, ok, text?/error?...}], truncated, ts
    }
    """
    paths = [str(p) for p in (image_paths or []) if str(p or "").strip()]
    if not paths:
        return {"provider": None, "model": None, "results": [], "note": "no_images", "truncated": False, "ts": time.time()}

    provider = choose_vision_provider(api_keys or {}, "qwen")
    if not provider:
        return {"provider": None, "model": None, "results": [], "note": "missing_vision_key", "truncated": bool(len(paths) > max_images), "ts": time.time()}

    api_key = str((api_keys or {}).get(provider, "") or "").strip()
    if not api_key:
        return {"provider": None, "model": None, "results": [], "note": "missing_vision_key", "truncated": bool(len(paths) > max_images), "ts": time.time()}

    # 固定使用 qwen-omni-turbo
    url = str(qwen_url)
    model = "qwen-omni-turbo"

    max_n = max(0, int(max_images))
    truncated = len(paths) > max_n if max_n else True
    use_paths = paths[:max_n] if max_n else []

    results: List[Dict[str, Any]] = []
    for p in use_paths:
        # 尝试加载图片
        loaded = _load_image_bytes(p)
        if not loaded:
            # 检查文件路径是否存在，提供更详细的错误信息
            check_path = Path("out") / p if not str(p).startswith("out") else Path(p)
            file_exists = check_path.exists() if check_path else False
            err_msg = f"文件不存在或读取失败（检查路径: {check_path}, 文件存在: {file_exists}）"
            results.append({"path": p, "ok": False, "error": err_msg})
            continue
        raw, mime = loaded

        data = raw
        out_mime = mime
        if len(data) > int(max_bytes):
            data, out_mime = _shrink_image_bytes(data, mime, max_edge=int(max_edge), jpeg_quality=85)
        if len(data) > int(max_bytes):
            # 再次尝试更激进的压缩
            data2, out_mime2 = _shrink_image_bytes(data, out_mime, max_edge=max(512, int(int(max_edge) * 0.75)), jpeg_quality=70)
            if len(data2) < len(data):
                data, out_mime = data2, out_mime2

        data_url = _to_data_url(data, out_mime)
        call = call_vision_chat(
            provider=provider,
            api_key=api_key,
            url=url,
            model=model,
            user_query=user_query,
            image_data_url=data_url,
            timeout=int(timeout),
            temperature=0.2,
        )
        item = {"path": p, **call}
        if item.get("ok") and item.get("text"):
            item["text"] = _truncate(str(item["text"]), 3500)
        results.append(item)

    return {"provider": provider, "model": model, "results": results, "truncated": truncated, "ts": time.time()}


