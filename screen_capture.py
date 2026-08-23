# 本程序仅用于立直麻将算法学习研究，禁止用于线上游戏对局辅助。

"""本机屏幕捕获与分析桥接层。

Windows 优先通过 DXGI Desktop Duplication 读取用户选择的画面区域，失败时
退回 MSS；不读取游戏进程、不抓包、不注入，也不控制鼠标键盘或上传画面。
"""

from __future__ import annotations

import os
import re
import time
from collections import Counter
from dataclasses import replace
from typing import Any

from analyzer import AnalysisResult, calculate_shanten, core_analyze
from input_layer import hand_input
from ocr_input import (
    DEFAULT_MAHJONG_SOUL_TEMPLATE_DIR,
    DEFAULT_TEMPLATE_DIR,
    OCRResult,
    TileRecognition,
    recognize_hand_frame,
)
from tile_utils import HandValidationError, TILE_NAMES, tiles_from_34


class ScreenCaptureError(RuntimeError):
    """屏幕捕获或图像转换失败。"""


CAPTURE_BACKENDS = ("auto", "dxgi", "mss")
_DXGI_CAMERAS: dict[tuple[int, int], Any] = {}


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


def _dxcam_module() -> Any:
    """加载 Windows Desktop Duplication 捕获组件。"""
    try:
        import dxcam  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ScreenCaptureError(
            "缺少 Windows DXGI 捕获组件，请重新下载最新版 EXE。"
        ) from exc
    return dxcam


def _is_windows() -> bool:
    return os.name == "nt"


def _selected_monitor(mss: Any, monitor_id: int) -> dict[str, int]:
    """读取显示器几何信息；MSS 枚举可靠，实际抓图可改走 DXGI。"""
    try:
        with mss.mss() as capturer:
            if not 1 <= monitor_id < len(capturer.monitors):
                raise ScreenCaptureError(f"显示器 {monitor_id} 不存在。")
            return {
                key: int(value) for key, value in capturer.monitors[monitor_id].items()
            }
    except ScreenCaptureError:
        raise
    except Exception as exc:
        raise ScreenCaptureError(
            "无法读取显示器信息。请重新连接显示器后再试。"
        ) from exc


def _dxgi_output_for_monitor(
    dxcam: Any, monitor_id: int, monitor: dict[str, int]
) -> tuple[int, int]:
    """按分辨率和显示器顺序，把 MSS 显示器映射到 DXGI 输出。"""
    outputs: list[tuple[int, int, int, int]] = []
    try:
        for device, output, width, height in re.findall(
            r"Device\[(\d+)\]\s+Output\[(\d+)\]:\s+"
            r"Res:\((\d+),\s*(\d+)\)",
            dxcam.output_info(),
        ):
            outputs.append((int(device), int(output), int(width), int(height)))
    except Exception:
        outputs = []

    expected = (monitor["width"], monitor["height"])
    same_size = [
        item
        for item in outputs
        if (item[2], item[3]) == expected or (item[3], item[2]) == expected
    ]
    if same_size:
        # 多台同分辨率显示器时，尽量保持 MSS 与 DXGI 的顺序一致。
        return same_size[min(monitor_id - 1, len(same_size) - 1)][:2]
    if 0 <= monitor_id - 1 < len(outputs):
        return outputs[monitor_id - 1][:2]
    return 0, max(0, monitor_id - 1)


def _frame_is_usable(frame: Any) -> bool:
    if frame is None or getattr(frame, "size", 0) == 0:
        return False
    # 纯黑帧通常表示游戏使用了 MSS/GDI 无法读取的 Direct3D 交换链。
    return not (float(frame.mean()) < 0.5 and float(frame.std()) < 0.5)


def _capture_with_dxgi(
    monitor_id: int,
    monitor: dict[str, int],
    capture_region: dict[str, int],
    np: Any,
) -> Any:
    """通过 Windows Desktop Duplication API 抓取 Direct3D 游戏画面。"""
    if not _is_windows():
        raise ScreenCaptureError("DXGI 捕获只支持 Windows 10/11。")

    dxcam = _dxcam_module()
    device_idx, output_idx = _dxgi_output_for_monitor(dxcam, monitor_id, monitor)
    left = capture_region["left"] - monitor["left"]
    top = capture_region["top"] - monitor["top"]
    region = (
        left,
        top,
        left + capture_region["width"],
        top + capture_region["height"],
    )
    camera_key = (device_idx, output_idx)
    try:
        camera = _DXGI_CAMERAS.get(camera_key)
        if camera is None:
            camera = dxcam.create(
                device_idx=device_idx,
                output_idx=output_idx,
                output_color="BGR",
                backend="dxgi",
                processor_backend="cv2",
            )
            _DXGI_CAMERAS[camera_key] = camera
        frame = None
        for _ in range(3):
            frame = camera.grab(region=region, new_frame_only=False)
            if frame is not None:
                break
            time.sleep(0.01)
    except Exception as exc:
        camera = _DXGI_CAMERAS.pop(camera_key, None)
        if camera is not None:
            try:
                camera.release()
            except Exception:
                pass
        raise ScreenCaptureError(
            "DXGI 捕获失败。请把游戏和牌理镜设为相同权限级别，并更新显卡驱动。"
        ) from exc

    if frame is None:
        raise ScreenCaptureError(
            "DXGI 没有返回画面。请将游戏切换为窗口化或无边框窗口后重试。"
        )
    frame = np.asarray(frame)
    if frame.ndim != 3 or frame.shape[2] < 3:
        raise ScreenCaptureError("DXGI 返回了无法识别的图像格式。")
    return np.ascontiguousarray(frame[:, :, :3])


def release_capture_resources() -> None:
    """释放复用的 DXGI 捕获器；应尽量在创建它的工作线程中调用。"""
    for camera in list(_DXGI_CAMERAS.values()):
        try:
            camera.release()
        except Exception:
            pass
    _DXGI_CAMERAS.clear()


def _capture_with_mss(
    capture_region: dict[str, int], cv2: Any, mss: Any, np: Any
) -> Any:
    """通过传统桌面位图方式抓取普通窗口。"""
    try:
        with mss.mss() as capturer:
            shot = np.asarray(capturer.grab(capture_region))
    except Exception as exc:
        raise ScreenCaptureError(
            "MSS 捕获失败。请把游戏切换为窗口化或无边框窗口。"
        ) from exc
    return cv2.cvtColor(shot, cv2.COLOR_BGRA2BGR)


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
    monitor_id: int,
    region: dict[str, Any] | None = None,
    backend: str = "auto",
    monitor_geometry: dict[str, Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """返回 BGR 屏幕帧和实际捕获区域。

    Windows 自动模式优先使用 DXGI Desktop Duplication，以支持 Direct3D 游戏；
    DXGI 不可用时退回 MSS。也可从界面强制选择其中一种方式。
    """
    if backend not in CAPTURE_BACKENDS:
        raise ScreenCaptureError(f"未知捕获方式：{backend}")
    cv2, mss, np = _screen_modules()
    if monitor_geometry is None:
        monitor = _selected_monitor(mss, monitor_id)
    else:
        try:
            monitor = {
                key: int(monitor_geometry[key])
                for key in ("left", "top", "width", "height")
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise ScreenCaptureError("已保存的显示器坐标无效，请重新选择显示器。") from exc
    capture_region: dict[str, Any] = clamp_region(region, monitor)
    errors: list[str] = []

    methods = [backend]
    if backend == "auto":
        methods = ["dxgi", "mss"] if _is_windows() else ["mss"]

    for method in methods:
        try:
            if method == "dxgi":
                frame = _capture_with_dxgi(
                    monitor_id, monitor, capture_region, np
                )
                label = "DXGI"
            else:
                frame = _capture_with_mss(capture_region, cv2, mss, np)
                label = "MSS"
            if not _frame_is_usable(frame):
                raise ScreenCaptureError(f"{label} 捕获到了纯黑画面。")
            capture_region["backend"] = label
            return frame, capture_region
        except ScreenCaptureError as exc:
            errors.append(str(exc))

    detail = "；".join(errors)
    raise ScreenCaptureError(
        "无法捕获游戏画面。请使用窗口化或无边框窗口，并确保游戏与牌理镜"
        f"以相同权限运行。详细信息：{detail}"
    )


def _effective_payload(items: Any) -> list[dict[str, int | str]]:
    return [{"tile": item.tile, "remaining": item.remaining} for item in items]


def _legalize_recognized_tiles(ocr_result: OCRResult) -> tuple[OCRResult, int]:
    """用次优候选修正单种牌超过四张的明显 OCR 冲突。"""
    recognitions = list(ocr_result.recognitions)
    counts = Counter(item.tile for item in recognitions)
    correction_count = 0

    for duplicated_tile, count in list(counts.items()):
        excess = count - 4
        if excess <= 0:
            continue
        indexes = sorted(
            (
                index
                for index, item in enumerate(recognitions)
                if item.tile == duplicated_tile
            ),
            key=lambda index: (
                recognitions[index].match_score,
                recognitions[index].confidence,
            ),
        )
        for index in indexes[:excess]:
            current = recognitions[index]
            replacement = next(
                (
                    (tile, score)
                    for tile, score in current.alternatives
                    if counts[tile] < 4
                ),
                None,
            )
            if replacement is None:
                replacement = next(
                    ((tile, 0.0) for tile in TILE_NAMES if counts[tile] < 4),
                    None,
                )
            if replacement is None:
                continue
            tile, score = replacement
            counts[current.tile] -= 1
            counts[tile] += 1
            remaining_alternatives = tuple(
                item for item in current.alternatives if item[0] != tile
            )
            recognitions[index] = replace(
                current,
                tile=tile,
                confidence=min(current.confidence, max(0.0, float(score))),
                alternatives=((current.tile, current.match_score),)
                + remaining_alternatives,
                match_score=float(score),
            )
            correction_count += 1

    return replace(ocr_result, recognitions=tuple(recognitions)), correction_count


def analysis_payload(ocr_result: OCRResult) -> dict[str, Any]:
    """把 OCR 结果转换为独立于界面框架的展示数据。"""
    ocr_result, correction_count = _legalize_recognized_tiles(ocr_result)
    hand_tiles = hand_input(ocr_result.tiles)
    shanten = calculate_shanten(hand_tiles)
    result: AnalysisResult = core_analyze(hand_tiles)
    payload: dict[str, Any] = {
        "tiles": ocr_result.tiles,
        "sortedTiles": tiles_from_34(hand_tiles),
        "confidences": [round(item.confidence, 4) for item in ocr_result.recognitions],
        "minimumConfidence": round(ocr_result.minimum_confidence, 4),
        "correctedTileCount": correction_count,
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
    backend: str = "auto",
    monitor_geometry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """捕获一次框选区域并返回 OCR + 牌理分析结果。"""
    frame, capture_region = capture_screen(
        monitor_id,
        region,
        backend,
        monitor_geometry=monitor_geometry,
    )
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
