import os
import json
import base64
import random
import textwrap
import requests
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field, model_validator
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont

try:
    from moviepy.editor import ImageSequenceClip
except ImportError:
    from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_PATH = Path(__file__).parent.resolve()
STATIC_PATH = BASE_PATH / "static"
OUTPUT_ROOT = Path(os.getenv("NOVEL_OUTPUT_ROOT", r"D:\每日小说")).expanduser().resolve()
FONT_PATH = str(BASE_PATH / "font.ttf")

os.makedirs(OUTPUT_ROOT, exist_ok=True)
HISTORY_FILE = OUTPUT_ROOT / "_history.json"

IS_CANCELLING = False
MAX_CONCURRENT_IMAGES = int(os.getenv("NOVEL_MAX_CONCURRENT", "3"))
history_lock = threading.Lock()
batch_counter_lock = threading.Lock()

# 出图类型常量
KIND_TEXT_SINGLE = "text_single"
KIND_LR_SPLIT = "lr"
KIND_TB_SPLIT = "tb"
KIND_SCROLL = "scroll"
KIND_POPUP_BG = "popup_bg"
_SQUARE_KINDS = {KIND_TEXT_SINGLE, KIND_LR_SPLIT, KIND_TB_SPLIT}

_MAX_IMAGE_PROMPT_HARD_CAP = 10**9 
_DEFAULT_MAX_IMAGE_PROMPT_CHARS = 10**9
MAX_CHAT_USER_PROMPT_CHARS = int(os.getenv("MAX_CHAT_USER_PROMPT_CHARS", "24000"))
MAX_CHAT_NOVEL_CHARS = int(os.getenv("MAX_CHAT_NOVEL_CHARS", "48000"))


def allocate_batch_id(output_root: Path) -> int:
    with batch_counter_lock:
        output_root.mkdir(parents=True, exist_ok=True)
        counter_path = output_root / "_batch_counter.json"
        n = 1
        if counter_path.exists():
            try:
                data = json.loads(counter_path.read_text(encoding="utf-8"))
                n = max(1, int(data.get("next", 1)))
            except Exception:
                n = 1
        batch_id = n
        counter_path.write_text(
            json.dumps({"next": batch_id + 1}, ensure_ascii=False),
            encoding="utf-8",
        )
        return batch_id


# --- History helpers ---
def save_to_history(prompt: str, image_url: str, batch_id: int, img_type: str):
    with history_lock:
        entries = []
        if HISTORY_FILE.exists():
            try:
                entries = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
            except Exception:
                entries = []
        entries.append({
            "prompt": prompt,
            "image_url": image_url,
            "batch_id": batch_id,
            "type": img_type,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })
        # Keep last 300 entries
        if len(entries) > 300:
            entries = entries[-300:]
        HISTORY_FILE.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")


def load_history(date: str = None) -> list:
    entries = []
    if HISTORY_FILE.exists():
        try:
            entries = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        except Exception:
            return []
    if date:
        entries = [e for e in entries if e.get("timestamp", "").startswith(date)]
    return entries


def get_history_dates() -> list:
    dates = set()
    if HISTORY_FILE.exists():
        try:
            entries = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        except Exception:
            return []
        for e in entries:
            ts = e.get("timestamp", "")
            if ts:
                dates.add(ts[:10])
    return sorted(dates, reverse=True)


def save_task_record(batch_dir: Path, batch_id: int, body: "GenerateRequest",
                     status: str, message: str, generated_images: list,
                     video_url: str, popup_urls: list, warnings: list, errors: list,
                     text_single_prompts: list = None, lr_prompts: list = None,
                     tb_prompts: list = None, scroll_prompts: list = None):
    record = {
        "batch_id": batch_id,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": status,
        "message": message,
        "params": {
            "text_single_count": body.text_single_count,
            "lr_split_count": body.lr_split_count,
            "tb_split_count": body.tb_split_count,
            "scroll_count": body.scroll_count,
            "popup_count": body.popup_count,
            "video_style": body.video_style,
            "compress_prompt_before_image": body.compress_prompt_before_image,
            "max_concurrent": body.max_concurrent or MAX_CONCURRENT_IMAGES,
        },
        "novel_content_snippet": (body.novel_content or "")[:200],
        "user_prompt": (body.prompt or "")[:500],
        "video_text": (body.video_text or "")[:200],
        "model_info": {
            "chat_model": body.chat_model_name,
            "image_model": body.image_model_name,
            "api_url": body.api_url,
        },
        "prompts_used": {
            "text_single": text_single_prompts or [],
            "lr_split": lr_prompts or [],
            "tb_split": tb_prompts or [],
            "scroll_visual": scroll_prompts or [],
        },
        "results": {
            "images": generated_images,
            "video": video_url,
            "popup_videos": popup_urls,
            "image_count": len([u for u in generated_images if not u.endswith(".mp4")]),
            "video_count": len(popup_urls) + (1 if video_url else 0),
        },
        "warnings": warnings,
        "errors": errors,
    }
    record_path = batch_dir / "task_record.json"
    record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")


class GenerateRequest(BaseModel):
    api_key: str = ""
    api_url: str = ""
    chat_model_name: str = ""
    image_model_name: str = ""
    novel_content: str = ""
    prompt: str = ""
    video_text: str = ""
    text_single_count: int = 0
    lr_split_count: int = 0
    tb_split_count: int = 0
    scroll_count: int = 0
    popup_count: int = 0
    video_style: str = "random"
    max_image_prompt_chars: Optional[int] = Field(default=None, ge=0)
    compress_prompt_before_image: bool = True
    max_concurrent: Optional[int] = Field(default=None, ge=1, le=10)

    @model_validator(mode="before")
    @classmethod
    def _legacy(cls, data):
        if not isinstance(data, dict):
            return data
        d = dict(data)
        if "text_single_count" not in d and "single_count" in d:
            d["text_single_count"] = d.get("single_count", 0)
        return d


class ManualRequest(BaseModel):
    api_key: str = ""
    api_url: str = ""
    image_model_name: str = ""
    prompt: str = ""
    ratio: str = "single"  # "single", "lr", "tb"
    count: int = Field(default=1, ge=1, le=20)


def run_manual_generation(body: "ManualRequest") -> dict:
    images: list = []
    errors: list = []
    warnings: list = []
    batch_id = allocate_batch_id(OUTPUT_ROOT)
    batch_dir = OUTPUT_ROOT / str(batch_id)
    os.makedirs(batch_dir, exist_ok=True)

    base_url = body.api_url.rstrip("/") + "/images/generations"
    headers = {"Authorization": f"Bearer {body.api_key}", "Content-Type": "application/json"}
    prompt = (body.prompt or "").strip()

    if not prompt:
        return {"images": [], "errors": ["请输入提示词"], "batch_id": batch_id}

    if not body.api_key.strip():
        return {"images": [], "errors": ["请配置 API 密钥"], "batch_id": batch_id}

    # Size for all manual ratios: 1024x1024
    size = "1024x1024"
    ratio_suffix = {KIND_TEXT_SINGLE: _TEXT_SINGLE_SUFFIX, KIND_LR_SPLIT: _LR_SPLIT_SUFFIX, KIND_TB_SPLIT: _TB_SPLIT_SUFFIX}.get(body.ratio, _TEXT_SINGLE_SUFFIX)
    full_prompt = f"{prompt}, {ratio_suffix}"

    seq = 0
    seq_lock = threading.Lock()

    def fetch_one(i):
        nonlocal seq
        if not body.api_key.strip():
            return None
        payload = {"model": body.image_model_name, "prompt": full_prompt, "size": size, "n": 1}
        try:
            r = requests.post(base_url, json=payload, headers=headers, timeout=120)
            if r.status_code >= 400:
                errors.append(f"第{i + 1}张 HTTP {r.status_code}: {_api_error_snippet(r)}")
                return None
            j = r.json()
            data = j.get("data")
            if not isinstance(data, list) or not data:
                errors.append(f"第{i + 1}张 无 data")
                return None
            item = data[0]
            img_bytes = None
            if isinstance(item, dict):
                if item.get("url"):
                    ir = requests.get(item["url"], timeout=120)
                    img_bytes = ir.content if ir.status_code < 400 else None
                elif item.get("b64_json"):
                    img_bytes = base64.b64decode(item["b64_json"])
            if not img_bytes:
                errors.append(f"第{i + 1}张 无法获取图片数据")
                return None
            with seq_lock:
                seq += 1
                s = seq
            fname = f"{batch_id}-manual-{s}.png"
            fpath = batch_dir / fname
            with open(fpath, "wb") as f:
                f.write(img_bytes)
            rel = f"/static/output/{batch_id}/{fname}"
            save_to_history(full_prompt, rel, batch_id, body.ratio)
            return rel
        except Exception as e:
            errors.append(f"第{i + 1}张 {e}")
            return None

    workers = min(MAX_CONCURRENT_IMAGES, body.count)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(fetch_one, i) for i in range(body.count)]
        for future in as_completed(futures):
            result = future.result()
            if result:
                images.append(result)

    return {
        "images": images,
        "errors": errors,
        "warnings": warnings,
        "batch_id": batch_id,
        "message": f"完成 {len(images)}/{body.count} 张",
    }


def clamp_text(text: str, max_chars: int) -> str:
    t = (text or "").strip()
    if len(t) <= max_chars:
        return t
    return t[: max_chars - 12].rstrip() + "\n…(已截断)"


def clamp_text_smart(text: str, max_chars: int) -> str:
    # 啥也不管，直接返回原句
    return (text or "").strip()


# 仅保留比例与分屏结构控制
# 其它风格、文字、镜头、逻辑、负面词全部交给用户提示词动态控制

_TEXT_SINGLE_SUFFIX = (
    "1:1 square composition"
)

_SCROLL_VISUAL_SUFFIX = (
    "9:16 vertical portrait composition"
)

_SPLIT_TEXT_OVERRIDE = ""

_LR_SPLIT_SUFFIX = (
    "VERTICAL LEFT-RIGHT SPLIT SCREEN ONLY, "
    "single straight vertical divider line in center, "
    "left panel and right panel side by side, "
    "forbidden top-bottom layout"
)

_TB_SPLIT_SUFFIX = (
    "HORIZONTAL TOP-BOTTOM SPLIT SCREEN ONLY, "
    "single straight horizontal divider line in center, "
    "top panel above bottom panel, "
    "forbidden left-right layout"
)


def finalize_square_prompt(kind: str, core: str, base_fallback: str) -> str:
    base = (core or "").strip() or (base_fallback or "").strip()

    if kind == "text_single":
        return f"{base}, {_TEXT_SINGLE_SUFFIX}"

    if kind == "lr":
        return f"{base}, {_LR_SPLIT_SUFFIX}"

    if kind == "tb":
        return f"{base}, {_TB_SPLIT_SUFFIX}"

    return base


def finalize_scroll_visual_prompt(core: str, base_fallback: str) -> str:
    base = (core or "").strip() or (base_fallback or "").strip()
    return f"{base}, {_SCROLL_VISUAL_SUFFIX}"


def _norm_prompt_list(x) -> List[str]:
    if not isinstance(x, list):
        return []
    return [str(i).strip() for i in x]


def _split_legacy_square_prompts(
    flat: List[str], text_single_count: int, lr_count: int, tb_count: int
) -> Tuple[List[str], List[str], List[str]]:
    need = text_single_count + lr_count + tb_count
    buf = [str(x).strip() for x in flat]
    while len(buf) < need:
        buf.append("")
    i = 0
    ts = buf[i : i + text_single_count]
    i += text_single_count
    lr = buf[i : i + lr_count]
    i += lr_count
    tb = buf[i : i + tb_count]
    return ts, lr, tb


def _api_error_snippet(r: requests.Response) -> str:
    try:
        j = r.json()
        if isinstance(j, dict):
            err = j.get("error")
            if isinstance(err, dict):
                return str(err.get("message") or err.get("code") or err)[:800]
            if err is not None:
                return str(err)[:800]
        return json.dumps(j, ensure_ascii=False)[:800]
    except Exception:
        return (r.text or "")[:800]


def request_image_prompt_plan(
    api_url: str,
    api_key: str,
    chat_model_name: str,
    novel_content: str,
    user_prompt: str,
    text_single_count: int,
    lr_split_count: int,
    tb_split_count: int,
    scroll_visual_count: int,
) -> Tuple[List[str], List[str], List[str], List[str]]:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    url = api_url.rstrip("/") + "/chat/completions"
    n_square = text_single_count + lr_split_count + tb_split_count
    system = (
        "你是欧美小说投流出图提示词策划。只输出严格 JSON，不要 markdown。\n"
        '{"text_single_prompts":[],"scroll_visual_prompts":[],"lr_split_prompts":[],"tb_split_prompts":[]}\n'
        f"- text_single_prompts：{text_single_count} 条，1:1 方图，底部单行英文叠字，其余区域无字。\n"
        f"- scroll_visual_prompts：{scroll_visual_count} 条，9:16 竖屏，画中严禁任何文字；"
        f"条数 = 「滚屏单图」+「弹屏视频」所需底图之和（每张弹屏各用一张独立竖屏底图）。\n"
        f"- lr_split_prompts：{lr_split_count} 条，必须是「左右分屏」：中间一条竖线，左半屏与右半屏并排；"
        "禁止做成上下堆叠；左/右/底三行英文文案。\n"
        f"- tb_split_prompts：{tb_split_count} 条，必须是「上下分屏」：中间一条横线，上半屏在上、下半屏在下；"
        "禁止做成左右并排；仅在横线处一行英文文案。\n"
        "两类分屏互斥，不得混用。若某类为 0 则数组为 []。\n"
        f"兼容：可用 scroll_prompts；可用 single_square_prompts；或 square_prompts 长度 {n_square} 按顺序切：带字单图→左右→上下。"
    )
    user = (
        f"用户绘图规则：\n{clamp_text(user_prompt, MAX_CHAT_USER_PROMPT_CHARS)}\n\n"
        f"小说节选：\n{clamp_text(novel_content, MAX_CHAT_NOVEL_CHARS)}\n\n"
        f"数量：text_single={text_single_count}, lr={lr_split_count}, tb={tb_split_count}, scroll={scroll_visual_count}"
    )
    payload = {
        "model": chat_model_name,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.7,
    }
    r = requests.post(url, json=payload, headers=headers, timeout=120)
    if r.status_code >= 400:
        raise RuntimeError(f"HTTP {r.status_code}: {_api_error_snippet(r)}")
    j = r.json()
    raw = j["choices"][0]["message"]["content"].strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1]
        if "```" in raw:
            raw = raw.rsplit("```", 1)[0].strip()
    data = json.loads(raw)
    scroll = _norm_prompt_list(data.get("scroll_visual_prompts")) or _norm_prompt_list(
        data.get("scroll_prompts")
    )
    ts = _norm_prompt_list(data.get("text_single_prompts")) or _norm_prompt_list(
        data.get("single_square_prompts")
    )
    lr = _norm_prompt_list(data.get("lr_split_prompts"))
    tb = _norm_prompt_list(data.get("tb_split_prompts"))
    legacy = data.get("square_prompts")
    if legacy and not ts and not lr and not tb:
        ts, lr, tb = _split_legacy_square_prompts(
            _norm_prompt_list(legacy), text_single_count, lr_split_count, tb_split_count
        )
    return ts, lr, tb, scroll


def compress_image_style_prompt(
    api_url: str,
    api_key: str,
    chat_model_name: str,
    long_prompt: str,
    target_chars: int,
) -> str:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    url = api_url.rstrip("/") + "/chat/completions"
    cap = max(600, target_chars - 40)
    system = (
        f"压缩下列绘图约束到少于 {cap} 字符，保留画风与关键禁止项，只输出正文。\n"
    )
    payload = {
        "model": chat_model_name,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": clamp_text(long_prompt, min(MAX_CHAT_USER_PROMPT_CHARS * 3, 120000))},
        ],
        "temperature": 0.2,
    }
    r = requests.post(url, json=payload, headers=headers, timeout=120)
    if r.status_code >= 400:
        raise RuntimeError(_api_error_snippet(r))
    raw = r.json()["choices"][0]["message"]["content"].strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1]
        if "```" in raw:
            raw = raw.rsplit("```", 1)[0].strip()
    return raw.strip()


def pad_prompts(prompts: List[str], n: int, filler: str) -> List[str]:
    out: List[str] = []
    for i in range(n):
        if i < len(prompts) and str(prompts[i]).strip():
            out.append(str(prompts[i]).strip())
        else:
            out.append(filler)
    return out


def draw_text_with_spacing(draw, text, position, font, fill, char_spacing):
    x, y = position
    for char in text:
        draw.text((x, y), char, font=font, fill=fill)
        char_w = draw.textlength(char, font=font)
        x += char_w + char_spacing


def wrap_text_precisely(draw_obj, text, font, max_width, char_spacing):
    paragraphs = text.split("\n")
    lines = []
    for p in paragraphs:
        if p.strip() == "" and p == "":
            lines.append("")
            continue
        words = p.split(" ")
        current_line = ""
        for word in words:
            test_line = word if current_line == "" else current_line + " " + word
            w = draw_obj.textlength(test_line, font=font) + char_spacing * max(0, len(test_line) - 1)
            if w <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)
    return lines


def pre_render_text(text, target_width, font_path, font_size, text_color, line_spacing, char_spacing, align="居中对齐"):
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception:
        font = ImageFont.load_default()
    test_draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    wrapped_lines = wrap_text_precisely(test_draw, text, font, target_width, char_spacing)
    line_h = font_size + line_spacing
    img_h = len(wrapped_lines) * line_h + 100
    text_canvas = Image.new("RGBA", (target_width, img_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(text_canvas)
    current_y = 0
    for line in wrapped_lines:
        if line == "":
            current_y += line_h
            continue
        line_width = draw.textlength(line, font=font) + (char_spacing * max(0, len(line) - 1))
        x = (target_width - line_width) // 2 if align == "居中对齐" else 0
        draw_text_with_spacing(draw, line, (x, current_y), font, text_color, char_spacing)
        current_y += line_h
    return text_canvas


def popup_style_params(style: str) -> Tuple[str, str, int, int]:
    s = (style or "random").strip().lower()
    if s == "white_bg":
        return "#000000", "#FFFFFF", 180, 420
    if s == "black_bg":
        return "#FFFFFF", "#000000", 150, 420
    if random.random() < 0.5:
        return "#000000", "#FFFFFF", 180, 420
    return "#FFFFFF", "#000000", 150, 420


def popup_scheme_tag(style: str) -> str:
    s = (style or "random").strip().lower()
    if s == "white_bg":
        return "popup_white_bar"
    if s == "black_bg":
        return "popup_black_bar"
    return "popup_random_bar"


def split_text_smartly(full_text: str, max_chars_per_line: int) -> List[str]:
    raw = " ".join((full_text or "").split())
    if not raw:
        return []
    sentences = [s.strip() + "." for s in raw.split(".") if s.strip()]
    final_segments: List[str] = []
    current_chunk = ""
    for sentence in sentences:
        test_chunk = (current_chunk + " " + sentence).strip() if current_chunk else sentence
        wrapped = textwrap.wrap(test_chunk, width=max_chars_per_line)
        line_count = len(wrapped)
        if line_count > 10:
            if current_chunk:
                final_segments.append(current_chunk)
                current_chunk = sentence
            else:
                final_segments.append(sentence)
                current_chunk = ""
        else:
            current_chunk = test_chunk
    if current_chunk:
        final_segments.append(current_chunk)
    return final_segments


def create_popup_frame(
    text: str,
    bg_image: Image.Image,
    font_path: str,
    font_size: int,
    text_color: str,
    bg_color_hex: str,
    opacity: int,
    line_spacing: int,
    char_spacing: int,
) -> Image.Image:
    img = bg_image.copy().convert("RGBA")
    w, h = img.size
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw_ov = ImageDraw.Draw(overlay)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception:
        font = ImageFont.load_default()
    effective_char_w = (font_size * 0.5) + char_spacing
    max_chars_per_line = max(8, int((w * 0.85) / max(effective_char_w, 1e-6)))
    wrapped_lines = textwrap.wrap(text, width=max_chars_per_line)
    if not wrapped_lines:
        wrapped_lines = [text[: max_chars_per_line * 3]]
    line_h = font_size + line_spacing
    box_h = len(wrapped_lines) * line_h + 60
    box_y = h * 0.6
    hx = bg_color_hex.lstrip("#")
    if len(hx) >= 6:
        bg_rgb = tuple(int(hx[i : i + 2], 16) for i in (0, 2, 4))
    else:
        bg_rgb = (0, 0, 0)
    draw_ov.rectangle([40, box_y, w - 40, box_y + box_h], fill=(*bg_rgb, opacity))
    img = Image.alpha_composite(img, overlay).convert("RGB")
    draw_final = ImageDraw.Draw(img)
    current_y = box_y + 30
    for line in wrapped_lines:
        line_width = sum(draw_final.textlength(c, font=font) for c in line) + (
            char_spacing * max(0, len(line) - 1)
        )
        start_x = (w - line_width) // 2
        draw_text_with_spacing(draw_final, line, (start_x, current_y), font, text_color, char_spacing)
        current_y += line_h
    return img


def create_popup_video_on_bg(
    text: str,
    bg_image: Image.Image,
    output_file: Path,
    video_style: str,
    font_path: str,
) -> Optional[str]:
    text = (text or "").strip()
    if not text:
        return None
    t_hex, bar_hex, opacity, wpm = popup_style_params(video_style)
    f_size = 45
    line_spacing = 8
    char_spacing = 0
    fps = 30
    bg = bg_image.convert("RGB")
    effective_char_w = (f_size * 0.5) + char_spacing
    w0, _ = bg.size
    max_chars_per_line = max(8, int((w0 * 0.85) / max(effective_char_w, 1e-6)))
    segments = split_text_smartly(text, max_chars_per_line)
    if not segments:
        segments = textwrap.wrap(text, width=max_chars_per_line) or [text]
    frames: List[np.ndarray] = []
    durations: List[float] = []
    for seg in segments:
        words = len(seg.split())
        sec = max(2.5, (words / wpm) * 60)
        frame_img = create_popup_frame(
            seg, bg, font_path, f_size, t_hex, bar_hex, opacity, line_spacing, char_spacing
        )
        frames.append(np.array(frame_img))
        durations.append(sec)
    if not frames:
        return None
    clip = None
    try:
        try:
            clip = ImageSequenceClip(frames, durations=durations)
        except TypeError:
            expanded: List[np.ndarray] = []
            for arr, d in zip(frames, durations):
                n = max(1, int(d * fps))
                for _ in range(n):
                    expanded.append(arr)
            clip = ImageSequenceClip(expanded, fps=fps)
        clip.write_videofile(str(output_file), fps=fps, codec="libx264", audio=False)
    finally:
        if clip is not None:
            clip.close()
    return popup_scheme_tag(video_style)


def run_full_generation(body: GenerateRequest) -> dict:
    global IS_CANCELLING
    IS_CANCELLING = False  # 开始新任务前重置标识

    warnings: List[str] = []
    errors: List[str] = []
    generated_images: List[str] = []
    video_url = ""
    video_source_paths: List[str] = []

    batch_id = allocate_batch_id(OUTPUT_ROOT)
    batch_dir = OUTPUT_ROOT / str(batch_id)
    batch_dir.mkdir(parents=True, exist_ok=True)

    total_square = body.text_single_count + body.lr_split_count + body.tb_split_count
    scroll_visual_total = body.scroll_count + body.popup_count
    total_needed = total_square + scroll_visual_total

    text_single_prompts: List[str] = []
    lr_prompts: List[str] = []
    tb_prompts: List[str] = []
    scroll_prompts: List[str] = []

    if body.api_key.strip() and body.api_url.strip() and total_needed > 0:
        try:
            text_single_prompts, lr_prompts, tb_prompts, scroll_prompts = request_image_prompt_plan(
                body.api_url,
                body.api_key,
                body.chat_model_name,
                body.novel_content,
                body.prompt,
                body.text_single_count,
                body.lr_split_count,
                body.tb_split_count,
                scroll_visual_total,
            )
        except Exception as e:
            warnings.append(f"任务清单 chat 失败，使用默认提示词兜底：{e}")
            text_single_prompts, lr_prompts, tb_prompts, scroll_prompts = [], [], [], []

    prompt_budget = 10**9
    style_src = (body.prompt or "").strip()

    if body.compress_prompt_before_image and len(style_src) > int(prompt_budget * 0.92):
        try:
            if body.api_key.strip():
                style_src = compress_image_style_prompt(
                    body.api_url,
                    body.api_key,
                    body.chat_model_name,
                    style_src,
                    prompt_budget,
                )
                warnings.append("已对过长绘图提示词做压缩后再出图。")
        except Exception as e:
            warnings.append(f"压缩跳过：{e}")

    base_style = clamp_text_smart(style_src, prompt_budget)
    scroll_base = clamp_text_smart(
        f"{base_style}, 9:16 vertical portrait, no text, no subtitles, no letters",
        prompt_budget,
    )

    text_single_prompts = pad_prompts(text_single_prompts, body.text_single_count, base_style)
    lr_prompts = pad_prompts(lr_prompts, body.lr_split_count, base_style)
    tb_prompts = pad_prompts(tb_prompts, body.tb_split_count, base_style)
    scroll_prompts = pad_prompts(scroll_prompts, scroll_visual_total, scroll_base)

    headers = {"Authorization": f"Bearer {body.api_key}", "Content-Type": "application/json"}
    base_url = body.api_url.rstrip("/") + "/images/generations"
    image_seq = 0
    trunc_warned = False
    seq_lock = threading.Lock()
    pending_history: list = []  # batch history writes for flush at end

    def fetch_image(prompt: str, size: str, label: str) -> Optional[str]:
        nonlocal trunc_warned, image_seq
        if IS_CANCELLING:
            return None
        if not body.api_key.strip():
            errors.append(f"{label}：未配置 API 密钥")
            return None
        raw_prompt = (prompt or "").strip()
        prompt_send = clamp_text_smart(raw_prompt, prompt_budget)
        if len(raw_prompt) > prompt_budget and not trunc_warned:
            warnings.append(f"提示词超过上限 {prompt_budget}，已首尾智能截断。")
            trunc_warned = True
        payload = {"model": body.image_model_name, "prompt": prompt_send, "size": size, "n": 1}
        try:
            r = requests.post(base_url, json=payload, headers=headers, timeout=120)
            if r.status_code >= 400:
                errors.append(f"{label} HTTP {r.status_code}: {_api_error_snippet(r)}")
                return None
            j = r.json()
            data = j.get("data")
            if not isinstance(data, list) or not data:
                errors.append(f"{label} 无 data：{json.dumps(j, ensure_ascii=False)[:400]}")
                return None
            item = data[0]
            img_bytes = None
            if isinstance(item, dict):
                if item.get("url"):
                    ir = requests.get(item["url"], timeout=120)
                    img_bytes = ir.content if ir.status_code < 400 else None
                elif item.get("b64_json"):
                    img_bytes = base64.b64decode(item["b64_json"])
            if not img_bytes:
                errors.append(f"{label} 无法获取图片数据")
                return None
            with seq_lock:
                image_seq += 1
                seq = image_seq
            fname = f"{batch_id}-{seq}.png"
            fpath = batch_dir / fname
            with open(fpath, "wb") as f:
                f.write(img_bytes)
            return fname
        except Exception as e:
            errors.append(f"{label} {e}")
            return None

    square_jobs: List[Tuple[str, int, str]] = []
    for i in range(body.text_single_count):
        square_jobs.append((KIND_TEXT_SINGLE, i, f"带文字单图{i + 1}"))
    for i in range(body.lr_split_count):
        square_jobs.append((KIND_LR_SPLIT, i, f"左右分屏{i + 1}"))
    for i in range(body.tb_split_count):
        square_jobs.append((KIND_TB_SPLIT, i, f"上下分屏{i + 1}"))


    if square_jobs and not IS_CANCELLING:
        def gen_square(job):
            kind, idx, lab = job
            if IS_CANCELLING:
                return None
            core = {KIND_TEXT_SINGLE: text_single_prompts, KIND_LR_SPLIT: lr_prompts, KIND_TB_SPLIT: tb_prompts}[kind][idx]
            final_p = finalize_square_prompt(kind, core, base_style)
            name = fetch_image(final_p, "1024x1024", lab)
            if name:
                rel = f"/static/output/{batch_id}/{name}"
                save_to_history(final_p, rel, batch_id, kind)
                return rel
            return None

        workers = min(body.max_concurrent or MAX_CONCURRENT_IMAGES, len(square_jobs))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(gen_square, job) for job in square_jobs]
            for future in as_completed(futures):
                if IS_CANCELLING:
                    break
                try:
                    result = future.result()
                except Exception as e:
                    errors.append(f"方图生成异常: {e}")
                    continue
                if result:
                    generated_images.append(result)

    scroll_png_paths: List[Path] = []

    if scroll_visual_total > 0 and not IS_CANCELLING:
        ordered_results = [None] * scroll_visual_total

        def gen_scroll(i):
            if IS_CANCELLING:
                return (i, None)
            p = finalize_scroll_visual_prompt(scroll_prompts[i], scroll_base)
            lab = f"滚屏单图{i + 1}" if i < body.scroll_count else f"弹屏底图{i - body.scroll_count + 1}"
            name = fetch_image(p, "768x1344", lab)
            if name:
                rel = f"/static/output/{batch_id}/{name}"
                img_type = "scroll" if i < body.scroll_count else "popup_bg"
                save_to_history(p, rel, batch_id, img_type)
                return (i, (rel, batch_dir / name))
            return (i, None)

        workers = min(body.max_concurrent or MAX_CONCURRENT_IMAGES, scroll_visual_total)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(gen_scroll, i): i for i in range(scroll_visual_total)}
            for future in as_completed(futures):
                if IS_CANCELLING:
                    break
                try:
                    idx, result = future.result()
                except Exception as ex:
                    errors.append(f"竖屏图生成异常: {ex}")
                    continue
                if result:
                    ordered_results[idx] = result
        # 重建为原始顺序，保证后续弹屏索引正确
        for r in ordered_results:
            if r is not None:
                rel, png_path = r
                generated_images.append(rel)
                scroll_png_paths.append(png_path)

    # 滚屏视频生成 (使用第一张滚屏底图)
    video_source_paths = [str(p) for p in scroll_png_paths[: body.scroll_count]]
    font_path = FONT_PATH if Path(FONT_PATH).exists() else "font.ttf"

    if not IS_CANCELLING and video_source_paths and body.video_text.strip():
        try:
            base_img = Image.open(video_source_paths[0]).convert("RGB")
            W, H = base_img.size
            n_shrink, fps, f_size, l_spacing, wpm = 1.23, 30, 46, 7, 360
            t_color, bg_color = "#FFFFFF", (0, 0, 0, 150)
            ov_w, ov_h = int(W / n_shrink), int(H / n_shrink)
            ov_x, ov_y = (W - ov_w) // 2, (H - ov_h) // 2
            text_canvas = pre_render_text(body.video_text, ov_w - 40, font_path, f_size, t_color, l_spacing, 0)
            text_h = text_canvas.size[1]
            overlay_pic = Image.new("RGBA", (ov_w, ov_h), bg_color)
            base_frame = base_img.copy().convert("RGBA")
            base_frame.paste(overlay_pic, (ov_x, ov_y), overlay_pic)
            y_start, y_end = int(ov_h / 2), ov_h - text_h - 30
            valid_words = len([w for w in body.video_text.split() if w.strip()])
            duration = max(3.0, (valid_words / wpm) * 60)
            frames = []
            scroll_frames = max(1, int(duration * fps))
            y_offsets = np.linspace(y_start, y_end, scroll_frames)

            def make_frame(y_off):
                box = Image.new("RGBA", (ov_w, ov_h), (0, 0, 0, 0))
                box.paste(text_canvas, (20, int(y_off)), text_canvas)
                out = base_frame.copy()
                out.paste(box, (ov_x, ov_y), box)
                rgb = out.convert("RGB")
                return np.asarray(rgb, dtype=np.uint8)

            start_img = make_frame(y_start)
            for _ in range(fps * 2):
                if IS_CANCELLING:
                    break
                frames.append(start_img)
            for y in y_offsets:
                if IS_CANCELLING:
                    break
                frames.append(make_frame(y))
            end_img = make_frame(y_end)
            for _ in range(fps * 3):
                if IS_CANCELLING:
                    break
                frames.append(end_img)

            if not IS_CANCELLING and frames:
                v_name = f"{batch_id}-scroll-video.mp4"
                v_path = batch_dir / v_name
                clip = ImageSequenceClip(frames, fps=fps)
                try:
                    clip.write_videofile(str(v_path), fps=fps, codec="libx264", audio=False)
                finally:
                    clip.close()
                video_url = f"/static/output/{batch_id}/{v_name}"
        except Exception as e:
            warnings.append(f"滚屏视频合成失败：{e}")

    # 弹屏视频生成
    popup_urls: List[str] = []
    if not IS_CANCELLING and body.popup_count > 0 and body.video_text.strip():
        for i in range(body.popup_count):
            if IS_CANCELLING:
                break
            bg_idx = body.scroll_count + i
            if bg_idx >= len(scroll_png_paths):
                continue
            try:
                bg_img = Image.open(scroll_png_paths[bg_idx]).convert("RGB")
                out = batch_dir / f"{batch_id}-popup-{i + 1}.mp4"
                tag = create_popup_video_on_bg(body.video_text, bg_img, out, body.video_style, font_path)
                if tag:
                    u = f"/static/output/{batch_id}/{out.name}"
                    generated_images.append(u)
                    popup_urls.append(u)
            except Exception as e:
                errors.append(f"弹屏视频{i + 1}：{e}")

    # 最终状态判定
    if IS_CANCELLING:
        status, message = "failed", "任务已被用户手动取消。"
    else:
        expected = total_square + scroll_visual_total
        got_png = image_seq
        if expected == 0 and body.popup_count == 0:
            status, message = "success", "未请求出图或弹屏。"
        elif got_png == 0 and expected > 0:
            status, message = "failed", "出图全部失败。"
        elif got_png < expected:
            status, message = "partial", f"图片完成 {got_png}/{expected}。"
        else:
            status, message = "success", "任务完成。"

    all_video_urls = [u for u in generated_images if u.endswith(".mp4")] + ([video_url] if video_url else [])
    dl_names = [Path(u).name for u in generated_images if u.endswith(".png")]
    if video_url:
        dl_names.append(f"{batch_id}-scroll-video.mp4")
    dl_names.extend(Path(u).name for u in popup_urls)

    try:
        save_task_record(
            batch_dir, batch_id, body, status, message,
            generated_images, video_url, popup_urls,
            warnings, errors,
            text_single_prompts, lr_prompts, tb_prompts, scroll_prompts,
        )
    except Exception as e:
        warnings.append(f"任务记录保存失败: {e}")

    return {
        "status": status,
        "batch_id": batch_id,
        "batch_folder": str(batch_id),
        "output_root": str(OUTPUT_ROOT),
        "images": [u for u in generated_images if not u.endswith(".mp4")],
        "videos": all_video_urls,
        "video": video_url,
        "popup_videos": popup_urls,
        "message": message,
        "warnings": warnings,
        "errors": errors,
        "download_filenames": dl_names,
    }


def run_full_generation_stream(body: GenerateRequest, queue: "asyncio.Queue", main_loop):
    global IS_CANCELLING
    IS_CANCELLING = False

    warnings: List[str] = []
    errors: List[str] = []
    generated_images: List[str] = []
    video_url = ""

    batch_id = allocate_batch_id(OUTPUT_ROOT)
    batch_dir = OUTPUT_ROOT / str(batch_id)
    batch_dir.mkdir(parents=True, exist_ok=True)

    total_square = body.text_single_count + body.lr_split_count + body.tb_split_count
    scroll_visual_total = body.scroll_count + body.popup_count
    total_needed = total_square + scroll_visual_total

    text_single_prompts: List[str] = []
    lr_prompts: List[str] = []
    tb_prompts: List[str] = []
    scroll_prompts: List[str] = []

    if body.api_key.strip() and body.api_url.strip() and total_needed > 0:
        try:
            text_single_prompts, lr_prompts, tb_prompts, scroll_prompts = request_image_prompt_plan(
                body.api_url, body.api_key, body.chat_model_name,
                body.novel_content, body.prompt,
                body.text_single_count, body.lr_split_count,
                body.tb_split_count, scroll_visual_total,
            )
        except Exception as e:
            warnings.append(f"???? chat ?????????????{e}")
            text_single_prompts, lr_prompts, tb_prompts, scroll_prompts = [], [], [], []

    prompt_budget = 10**9
    style_src = (body.prompt or "").strip()

    if body.compress_prompt_before_image and len(style_src) > int(prompt_budget * 0.92):
        try:
            if body.api_key.strip():
                style_src = compress_image_style_prompt(
                    body.api_url, body.api_key, body.chat_model_name,
                    style_src, prompt_budget,
                )
                warnings.append("?????????????????")
        except Exception as e:
            warnings.append(f"?????{e}")

    base_style = clamp_text_smart(style_src, prompt_budget)
    scroll_base = clamp_text_smart(
        f"{base_style}, 9:16 vertical portrait, no text, no subtitles, no letters",
        prompt_budget,
    )

    text_single_prompts = pad_prompts(text_single_prompts, body.text_single_count, base_style)
    lr_prompts = pad_prompts(lr_prompts, body.lr_split_count, base_style)
    tb_prompts = pad_prompts(tb_prompts, body.tb_split_count, base_style)
    scroll_prompts = pad_prompts(scroll_prompts, scroll_visual_total, scroll_base)

    headers = {"Authorization": f"Bearer {body.api_key}", "Content-Type": "application/json"}
    base_url = body.api_url.rstrip("/") + "/images/generations"
    image_seq = 0
    trunc_warned = False
    seq_lock = threading.Lock()
    pending_history: list = []  # batch history writes for flush at end

    def fetch_image(prompt: str, size: str, label: str) -> Optional[str]:
        nonlocal trunc_warned, image_seq
        if IS_CANCELLING:
            return None
        if not body.api_key.strip():
            errors.append(f"{label}: 未配置 API 密钥")
            return None
        raw_prompt = (prompt or "").strip()
        prompt_send = clamp_text_smart(raw_prompt, prompt_budget)
        if len(raw_prompt) > prompt_budget and not trunc_warned:
            warnings.append(f"提示词超过上限 {prompt_budget}，已首尾智能截断。")
            trunc_warned = True
        payload = {"model": body.image_model_name, "prompt": prompt_send, "size": size, "n": 1}
        try:
            r = requests.post(base_url, json=payload, headers=headers, timeout=120)
            if r.status_code >= 400:
                errors.append(f"{label} HTTP {r.status_code}: {_api_error_snippet(r)}")
                return None
            j = r.json()
            data = j.get("data")
            if not isinstance(data, list) or not data:
                errors.append(f"{label} 无 data：{json.dumps(j, ensure_ascii=False)[:400]}")
                return None
            item = data[0]
            img_bytes = None
            if isinstance(item, dict):
                if item.get("url"):
                    ir = requests.get(item["url"], timeout=120)
                    img_bytes = ir.content if ir.status_code < 400 else None
                elif item.get("b64_json"):
                    img_bytes = base64.b64decode(item["b64_json"])
            if not img_bytes:
                errors.append(f"{label} 无法获取图片数据")
                return None
            with seq_lock:
                image_seq += 1
                seq = image_seq
            fname = f"{batch_id}-{seq}.png"
            fpath = batch_dir / fname
            with open(fpath, "wb") as f:
                f.write(img_bytes)
            return fname
        except Exception as e:
            errors.append(f"{label}: {e}")
            return None

    square_jobs = []
    for i in range(body.text_single_count):
        square_jobs.append(("text_single", i, f"带文字单图{i + 1}"))
    for i in range(body.lr_split_count):
        square_jobs.append(("lr", i, f"左右分屏{i + 1}"))
    for i in range(body.tb_split_count):
        square_jobs.append(("tb", i, f"上下分屏{i + 1}"))

    def emit_event(evt_type: str, data: dict):
        asyncio.run_coroutine_threadsafe(queue.put({"event": evt_type, "data": data}), main_loop)

    # Emit initial status
    total_expected = total_square + scroll_visual_total
    emit_event("status", {"message": f"开始生成 {total_square} 张方图 + {scroll_visual_total} 张竖屏图", "batch_id": batch_id, "total": total_expected, "done": 0})

    # Generate square images concurrently
    if square_jobs and not IS_CANCELLING:
        def gen_square(job):
            kind, idx, lab = job
            if IS_CANCELLING:
                return None
            core = {KIND_TEXT_SINGLE: text_single_prompts, KIND_LR_SPLIT: lr_prompts, KIND_TB_SPLIT: tb_prompts}[kind][idx]
            final_p = finalize_square_prompt(kind, core, base_style)
            name = fetch_image(final_p, "1024x1024", lab)
            if name:
                rel_url = f"/static/output/{batch_id}/{name}"
                save_to_history(final_p, rel_url, batch_id, kind)
                return (rel_url, lab, kind)
            return None

        workers = min(body.max_concurrent or MAX_CONCURRENT_IMAGES, len(square_jobs))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(gen_square, job): job for job in square_jobs}
            for future in as_completed(futures):
                if IS_CANCELLING:
                    break
                try:
                    result = future.result()
                except Exception as e:
                    _, _, lab = futures[future]
                    errors.append(f"{lab}: {e}")
                    emit_event("error", {"message": f"{lab} 生成异常: {e}"})
                    continue
                if result:
                    rel_url, lab, kind = result
                    generated_images.append(rel_url)
                    with seq_lock:
                        cur = image_seq
                    emit_event("image", {
                        "url": rel_url, "label": lab, "type": kind,
                        "batch_id": batch_id, "done": cur, "total": total_expected,
                    })
                else:
                    _, _, lab = futures[future]
                    emit_event("error", {"message": f"{lab} 生成失败"})

    # Generate scroll vertical images concurrently
    scroll_png_paths: List[Path] = []
    if scroll_visual_total > 0 and not IS_CANCELLING:
        ordered_scroll = [None] * scroll_visual_total  # preserve original order

        def gen_scroll(i):
            if IS_CANCELLING:
                return (i, None)
            p = finalize_scroll_visual_prompt(scroll_prompts[i], scroll_base)
            lab = f"滚屏单图{i + 1}" if i < body.scroll_count else f"弹屏底图{i - body.scroll_count + 1}"
            name = fetch_image(p, "768x1344", lab)
            if name:
                rel_url = f"/static/output/{batch_id}/{name}"
                img_type = "scroll" if i < body.scroll_count else "popup_bg"
                save_to_history(p, rel_url, batch_id, img_type)
                return (i, (rel_url, lab, img_type, batch_dir / name))
            return (i, None)

        workers = min(body.max_concurrent or MAX_CONCURRENT_IMAGES, scroll_visual_total)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(gen_scroll, i): i for i in range(scroll_visual_total)}
            for future in as_completed(futures):
                if IS_CANCELLING:
                    break
                try:
                    idx, result = future.result()
                except Exception as e:
                    errors.append(f"竖屏图 {futures[future] + 1}: {e}")
                    emit_event("error", {"message": f"竖屏图 {futures[future] + 1} 生成异常: {e}"})
                    continue
                if result:
                    ordered_scroll[idx] = result
                    rel_url, lab, img_type, png_path = result
                    generated_images.append(rel_url)
                    with seq_lock:
                        cur = image_seq
                    emit_event("image", {
                        "url": rel_url, "label": lab, "type": img_type,
                        "batch_id": batch_id, "done": cur, "total": total_expected,
                    })
                else:
                    emit_event("error", {"message": f"竖屏图 {idx + 1} 生成失败"})
        # 重建为原始顺序，保证后续弹屏索引正确
        for r in ordered_scroll:
            if r is not None:
                scroll_png_paths.append(r[3])

    # Scroll video generation
    video_source_paths = [str(p) for p in scroll_png_paths[: body.scroll_count]]
    font_path = FONT_PATH if Path(FONT_PATH).exists() else "font.ttf"
    if not IS_CANCELLING and video_source_paths and body.video_text.strip():
        try:
            base_img = Image.open(video_source_paths[0]).convert("RGB")
            W, H = base_img.size
            n_shrink, fps, f_size, l_spacing, wpm = 1.23, 30, 46, 7, 360
            t_color, bg_color = "#FFFFFF", (0, 0, 0, 150)
            ov_w, ov_h = int(W / n_shrink), int(H / n_shrink)
            ov_x, ov_y = (W - ov_w) // 2, (H - ov_h) // 2
            text_canvas = pre_render_text(body.video_text, ov_w - 40, font_path, f_size, t_color, l_spacing, 0)
            text_h = text_canvas.size[1]
            overlay_pic = Image.new("RGBA", (ov_w, ov_h), bg_color)
            base_frame = base_img.copy().convert("RGBA")
            base_frame.paste(overlay_pic, (ov_x, ov_y), overlay_pic)
            y_start, y_end = int(ov_h / 2), ov_h - text_h - 30
            valid_words = len([w for w in body.video_text.split() if w.strip()])
            duration = max(3.0, (valid_words / wpm) * 60)
            frames = []
            scroll_frames = max(1, int(duration * fps))
            y_offsets = np.linspace(y_start, y_end, scroll_frames)

            def make_frame(y_off):
                box = Image.new("RGBA", (ov_w, ov_h), (0, 0, 0, 0))
                box.paste(text_canvas, (20, int(y_off)), text_canvas)
                out = base_frame.copy()
                out.paste(box, (ov_x, ov_y), box)
                rgb = out.convert("RGB")
                return np.asarray(rgb, dtype=np.uint8)

            start_img = make_frame(y_start)
            for _ in range(fps * 2):
                if IS_CANCELLING: break
                frames.append(start_img)
            for y in y_offsets:
                if IS_CANCELLING: break
                frames.append(make_frame(y))
            end_img = make_frame(y_end)
            for _ in range(fps * 3):
                if IS_CANCELLING: break
                frames.append(end_img)

            if not IS_CANCELLING and frames:
                v_name = f"{batch_id}-scroll-video.mp4"
                v_path = batch_dir / v_name
                clip = ImageSequenceClip(frames, fps=fps)
                try:
                    clip.write_videofile(str(v_path), fps=fps, codec="libx264", audio=False)
                finally:
                    clip.close()
                video_url = f"/static/output/{batch_id}/{v_name}"
                emit_event("video", {"url": video_url, "label": "滚屏视频", "batch_id": batch_id})
        except Exception as e:
            warnings.append(f"视频生成失败: {e}")

    # Popup video generation
    popup_urls: List[str] = []
    if not IS_CANCELLING and body.popup_count > 0 and body.video_text.strip():
        for i in range(body.popup_count):
            if IS_CANCELLING: break
            bg_idx = body.scroll_count + i
            if bg_idx >= len(scroll_png_paths): continue
            try:
                bg_img = Image.open(scroll_png_paths[bg_idx]).convert("RGB")
                out = batch_dir / f"{batch_id}-popup-{i + 1}.mp4"
                tag = create_popup_video_on_bg(body.video_text, bg_img, out, body.video_style, font_path)
                if tag:
                    u = f"/static/output/{batch_id}/{out.name}"
                    popup_urls.append(u)
                    emit_event("popup_video", {"url": u, "label": f"弹屏 {i + 1}", "batch_id": batch_id})
            except Exception as e:
                errors.append(f"弹屏{i + 1}失败: {e}")

    # Final status
    if IS_CANCELLING:
        status, message = "cancelled", "任务已取消"
    else:
        expected = total_square + scroll_visual_total
        got_png = image_seq
        if expected == 0 and body.popup_count == 0:
            status, message = "success", "无需生成任何素材"
        elif got_png == 0 and expected > 0:
            status, message = "failed", "所有素材生成失败"
        elif got_png < expected:
            status, message = "partial", f"部分完成 {got_png}/{expected} 张"
        else:
            status, message = "success", "生成成功"

    dl_names = [Path(u).name for u in generated_images if u.endswith(".png")]
    if video_url:
        dl_names.append(f"{batch_id}-scroll-video.mp4")
    dl_names.extend(Path(u).name for u in popup_urls)

    try:
        save_task_record(
            batch_dir, batch_id, body, status, message,
            generated_images, video_url, popup_urls,
            warnings, errors,
            text_single_prompts, lr_prompts, tb_prompts, scroll_prompts,
        )
    except Exception as e:
        warnings.append(f"任务记录保存失败: {e}")

    emit_event("done", {
        "status": status,
        "message": message,
        "batch_id": batch_id,
        "images": [u for u in generated_images if not u.endswith(".mp4")],
        "videos": [u for u in generated_images if u.endswith(".mp4")] + ([video_url] if video_url else []),
        "video": video_url,
        "popup_videos": popup_urls,
        "warnings": warnings,
        "errors": errors,
        "download_filenames": dl_names,
    })


@app.post("/api/generate-stream")
async def api_generate_stream(body: GenerateRequest):
    queue: asyncio.Queue = asyncio.Queue()
    main_loop = asyncio.get_running_loop()

    def run_in_thread():
        try:
            run_full_generation_stream(body, queue, main_loop)
        except Exception as e:
            asyncio.run_coroutine_threadsafe(queue.put({"event": "error", "data": {"message": str(e)}}), main_loop)
            asyncio.run_coroutine_threadsafe(queue.put({"event": "done", "data": {"status": "failed", "message": str(e)}}), main_loop)

    async def event_generator():
        thread = threading.Thread(target=run_in_thread, daemon=True)
        thread.start()
        while True:
            event = await queue.get()
            evt_type = event["event"]
            data = json.dumps(event.get("data", {}), ensure_ascii=False)
            yield f"event: {evt_type}\ndata: {data}\n\n"
            if evt_type == "done":
                break

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/api/history")
def api_history(date: str = ""):
    return load_history(date or None)


@app.get("/api/history/dates")
def api_history_dates():
    return get_history_dates()


@app.post("/api/history/clear")
def api_history_clear(date: str = ""):
    if not date:
        if HISTORY_FILE.exists():
            HISTORY_FILE.unlink()
        return {"status": "cleared"}
    # Clear only entries for a specific date
    entries = load_history()
    entries = [e for e in entries if not e.get("timestamp", "").startswith(date)]
    HISTORY_FILE.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"status": "cleared", "date": date}


@app.post("/api/generate-manual")
def api_generate_manual(body: "ManualRequest"):
    return run_manual_generation(body)


@app.post("/api/generate")
def api_generate(body: GenerateRequest):
    return run_full_generation(body)


@app.post("/generate")
def generate_alias(body: GenerateRequest):
    return run_full_generation(body)


@app.post("/api/cancel")
def api_cancel():
    global IS_CANCELLING
    IS_CANCELLING = True
    return {"status": "cancelling", "message": "正在尝试中断后续任务..."}


app.mount("/static/output", StaticFiles(directory=str(OUTPUT_ROOT)), name="novel_output")
app.mount("/static", StaticFiles(directory=str(STATIC_PATH), html=True), name="static")


@app.get("/")
def read_root():
    idx = STATIC_PATH / "index.html"
    if idx.exists():
        return FileResponse(idx, media_type="text/html; charset=utf-8")
    return {"error": "缺少 static/index.html"}


if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*50)
    print("【素材工厂服务已启动】")
    print("本地访问: http://127.0.0.1:8000/static/index.html")
    print("="*50 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)