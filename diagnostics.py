# 本程序仅用于立直麻将算法学习研究，禁止用于线上游戏对局辅助。

"""只写入本机的轻量诊断日志与低置信度截图。"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from tile_utils import tile_name_to_chinese


_LOG_LOCK = threading.Lock()
_MAX_LOG_BYTES = 2 * 1024 * 1024
_LOG_BACKUPS = 3
_MAX_CAPTURE_PAIRS = 12
_LAST_CAPTURE_AT = 0.0


def diagnostics_directory() -> Path:
    override = os.environ.get("MAHJONG_STUDY_DATA_DIR")
    if override:
        base = Path(override).expanduser().resolve()
    elif getattr(sys, "frozen", False):
        base = Path(
            os.environ.get("LOCALAPPDATA")
            or os.environ.get("APPDATA")
            or Path.home()
        ) / "MahjongStudyAnalyzer"
    else:
        base = Path(__file__).resolve().parent / "local_diagnostics"
    path = base / "diagnostics"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _rotate_log(path: Path) -> None:
    try:
        if not path.exists() or path.stat().st_size < _MAX_LOG_BYTES:
            return
        oldest = path.with_name(f"{path.stem}.{_LOG_BACKUPS}{path.suffix}")
        if oldest.exists():
            oldest.unlink()
        for index in range(_LOG_BACKUPS - 1, 0, -1):
            source = path.with_name(f"{path.stem}.{index}{path.suffix}")
            if source.exists():
                source.replace(
                    path.with_name(f"{path.stem}.{index + 1}{path.suffix}")
                )
        path.replace(path.with_name(f"{path.stem}.1{path.suffix}"))
    except OSError:
        pass


def log_event(event: str, **details: Any) -> None:
    """以 JSONL 写入可直接发给开发者分析的中文日志。"""
    path = diagnostics_directory() / "diagnostics.log"
    record = {
        "时间": datetime.now().astimezone().isoformat(timespec="milliseconds"),
        "事件": event,
        **details,
    }
    line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
    with _LOG_LOCK:
        _rotate_log(path)
        try:
            with path.open("a", encoding="utf-8") as stream:
                stream.write(line)
        except OSError:
            pass


def log_exception(event: str, error: BaseException, **details: Any) -> None:
    trace = traceback.format_exc()
    if trace.strip() == "NoneType: None":
        trace = ""
    log_event(
        event,
        错误类型=type(error).__name__,
        错误信息=str(error),
        调用栈=trace,
        **details,
    )


def recognition_details(recognitions: Any) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, item in enumerate(recognitions, start=1):
        result.append(
            {
                "位置": index,
                "牌": tile_name_to_chinese(item.tile),
                "内部牌名": item.tile,
                "置信度": round(float(item.confidence), 4),
                "检测框": list(item.box),
                "候选": [
                    {
                        "牌": tile_name_to_chinese(tile),
                        "内部牌名": tile,
                        "匹配分": round(float(score), 4),
                    }
                    for tile, score in item.alternatives[:5]
                ],
            }
        )
    return result


def save_diagnostic_frame(
    frame: Any,
    *,
    reason: str,
    recognitions: Any = (),
    minimum_interval: float = 8.0,
) -> Path | None:
    """限频保存用户框选区域；只保留最近若干组图片与说明。"""
    global _LAST_CAPTURE_AT
    now = time.monotonic()
    if now - _LAST_CAPTURE_AT < minimum_interval:
        return None
    _LAST_CAPTURE_AT = now
    try:
        import cv2  # type: ignore[import-not-found]

        annotated = frame.copy()
        for index, item in enumerate(recognitions, start=1):
            x, y, width, height = (int(value) for value in item.box)
            color = (40, 190, 40) if item.confidence >= 0.62 else (30, 80, 230)
            cv2.rectangle(
                annotated, (x, y), (x + width, y + height), color, 2
            )
            cv2.putText(
                annotated,
                str(index),
                (x + 3, max(16, y + 17)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
                cv2.LINE_AA,
            )
        success, encoded = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not success:
            return None
        directory = diagnostics_directory()
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        image_path = directory / f"capture-{stamp}.jpg"
        image_path.write_bytes(encoded.tobytes())
        metadata_path = image_path.with_suffix(".json")
        metadata_path.write_text(
            json.dumps(
                {
                    "原因": reason,
                    "识别结果": recognition_details(recognitions),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        _prune_capture_pairs(directory)
        return image_path
    except Exception as error:
        log_exception("保存诊断画面失败", error, 原因=reason)
        return None


def _prune_capture_pairs(directory: Path) -> None:
    images = sorted(
        directory.glob("capture-*.jpg"),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    for image_path in images[_MAX_CAPTURE_PAIRS:]:
        try:
            image_path.unlink()
            metadata_path = image_path.with_suffix(".json")
            if metadata_path.exists():
                metadata_path.unlink()
        except OSError:
            pass
