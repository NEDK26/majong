# 本程序仅用于立直麻将算法学习研究，禁止用于线上游戏对局辅助。

"""本机屏幕捕获与分析桥接层。

只通过 mss 读取用户选择的显示器区域；不读取游戏进程、不抓包、不注入，
也不控制鼠标键盘或向外部网络发送画面。
"""

from __future__ import annotations

import time
from typing import Any

from analyzer import AnalysisResult, calculate_shanten, core_analyze
from input_layer import hand_input
from ocr_input import (
    DEFAULT_MAHJONG_SOUL_TEMPLATE_DIR,
    DEFAULT_TEMPLATE_DIR,
    OCRResult,
    recognize_hand_frame,
)
from tile_utils import HandValidationError, tiles_from_34


class ScreenCaptureError(RuntimeError):
    """屏幕捕获或图像转换失败。"""


def _screen_modules() -> tuple[Any, Any, Any]:
    try:
        import cv2  # type: ignore[import-not-found]
        import mss  # type: ignore[import-not-found]
        import numpy as np  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ScreenCaptureError(
            "屏幕分析依赖未安装；请执行 pip install -r requirements.txt。"
        ) from exc
    return cv2, mss, np


def list_monitors() -> list[dict[str, int | str]]:
    """列出真实显示器；mss 索引 0 是所有显示器的联合区域，因此跳过。"""
    _, mss, _ = _screen_modules()
    try:
        with mss.mss() as capturer:
            return [
                {
                    "id": index,
                    "name": f"显示器 {index}",
                    "left": int(monitor["left"]),
                    "top": int(monitor["top"]),
                    "width": int(monitor["width"]),
                    "height": int(monitor["height"]),
                }
                for index, monitor in enumerate(capturer.monitors[1:], start=1)
            ]
    except Exception as exc:
        raise ScreenCaptureError(
            "无法读取显示器列表。请确认显示器已连接，并允许桌面应用捕获屏幕。"
        ) from exc


def clamp_region(
    region: dict[str, Any] | None, monitor: dict[str, int]
) -> dict[str, int]:
    """把相对显示器的框选区域限制在显示器范围内。"""
    if not region:
        return dict(monitor)

    relative_x = max(0, int(region.get("x", 0)))
    relative_y = max(0, int(region.get("y", 0)))
    width = max(1, int(region.get("width", monitor["width"])))
    height = max(1, int(region.get("height", monitor["height"])))
    relative_x = min(relative_x, monitor["width"] - 1)
    relative_y = min(relative_y, monitor["height"] - 1)
    width = min(width, monitor["width"] - relative_x)
    height = min(height, monitor["height"] - relative_y)
    return {
        "left": monitor["left"] + relative_x,
        "top": monitor["top"] + relative_y,
        "width": width,
        "height": height,
    }


def capture_screen(
    monitor_id: int, region: dict[str, Any] | None = None
) -> tuple[Any, dict[str, int]]:
    """返回 BGR 屏幕帧和实际捕获区域。"""
    cv2, mss, np = _screen_modules()
    try:
        with mss.mss() as capturer:
            if not 1 <= monitor_id < len(capturer.monitors):
                raise ScreenCaptureError(f"显示器 {monitor_id} 不存在。")
            monitor = {
                key: int(value) for key, value in capturer.monitors[monitor_id].items()
            }
            capture_region = clamp_region(region, monitor)
            shot = np.asarray(capturer.grab(capture_region))
    except ScreenCaptureError:
        raise
    except Exception as exc:
        raise ScreenCaptureError(
            "无法捕获屏幕。请允许桌面应用捕获屏幕，并将游戏切换为窗口化或无边框窗口。"
        ) from exc
    frame = cv2.cvtColor(shot, cv2.COLOR_BGRA2BGR)
    if frame.size == 0 or float(frame.mean()) < 1.0:
        raise ScreenCaptureError(
            "捕获画面为空。请把游戏切换为窗口化或无边框窗口，并允许屏幕捕获。"
        )
    return frame, capture_region


def _effective_payload(items: Any) -> list[dict[str, int | str]]:
    return [{"tile": item.tile, "remaining": item.remaining} for item in items]


def analysis_payload(ocr_result: OCRResult) -> dict[str, Any]:
    """把 OCR 结果转换为独立于界面框架的展示数据。"""
    hand_tiles = hand_input(ocr_result.tiles)
    shanten = calculate_shanten(hand_tiles)
    result: AnalysisResult = core_analyze(hand_tiles)
    payload: dict[str, Any] = {
        "tiles": ocr_result.tiles,
        "sortedTiles": tiles_from_34(hand_tiles),
        "confidences": [round(item.confidence, 4) for item in ocr_result.recognitions],
        "minimumConfidence": round(ocr_result.minimum_confidence, 4),
        "shanten": shanten,
        "mode": result.mode,
        "recommendations": [],
        "candidates": [],
    }

    if result.mode == "draw":
        payload["effectiveDraws"] = _effective_payload(result.effective_draws)
        payload["drawUkeire"] = result.draw_ukeire
        return payload
    if result.mode == "agari" or not result.candidates:
        return payload

    best = result.candidates[0]
    payload["recommendations"] = [
        candidate.discard
        for candidate in result.candidates
        if candidate.shanten == best.shanten and candidate.ukeire == best.ukeire
    ]
    payload["candidates"] = [
        {
            "discard": candidate.discard,
            "shanten": candidate.shanten,
            "ukeire": candidate.ukeire,
            "effectiveTiles": _effective_payload(candidate.effective_tiles),
        }
        for candidate in result.candidates
    ]
    return payload


def analyze_capture(
    monitor_id: int,
    region: dict[str, Any] | None,
    expected_count: int | None,
) -> dict[str, Any]:
    """捕获一次框选区域并返回 OCR + 牌理分析结果。"""
    frame, capture_region = capture_screen(monitor_id, region)
    template_dir = (
        DEFAULT_MAHJONG_SOUL_TEMPLATE_DIR
        if DEFAULT_MAHJONG_SOUL_TEMPLATE_DIR.is_dir()
        else DEFAULT_TEMPLATE_DIR
    )
    ocr_result = recognize_hand_frame(
        frame,
        expected_count=expected_count,
        template_dir=template_dir,
    )
    payload = analysis_payload(ocr_result)
    payload.update(
        {
            "capturedAt": time.strftime("%H:%M:%S"),
            "captureRegion": capture_region,
            "template": "mahjong_soul"
            if template_dir == DEFAULT_MAHJONG_SOUL_TEMPLATE_DIR
            else "default",
        }
    )
    return payload
