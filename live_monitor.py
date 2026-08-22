# 本程序仅用于立直麻将算法学习研究，禁止用于线上游戏对局辅助。

"""Windows 本地实时研究面板。

仅通过 mss 捕获用户选择的显示器区域，服务只监听 127.0.0.1。模块不读取
游戏进程、不抓包、不注入、不控制鼠标键盘，也不向外部网络发送画面。
"""

from __future__ import annotations

import json
import sys
import threading
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from analyzer import AnalysisResult, calculate_shanten, core_analyze
from input_layer import hand_input
from ocr_input import (
    DEFAULT_MAHJONG_SOUL_TEMPLATE_DIR,
    DEFAULT_TEMPLATE_DIR,
    OCRResult,
    recognize_hand_frame,
)
from tile_utils import HandValidationError, tiles_from_34


WEB_DIR = Path(__file__).resolve().parent / "monitor_web"
HOST = "127.0.0.1"


class LiveMonitorError(RuntimeError):
    """屏幕捕获或本地服务错误。"""


def _screen_modules() -> tuple[Any, Any, Any]:
    try:
        import cv2  # type: ignore[import-not-found]
        import mss  # type: ignore[import-not-found]
        import numpy as np  # type: ignore[import-not-found]
    except ImportError as exc:
        raise LiveMonitorError(
            "实时监看依赖未安装；请执行 pip install -r requirements.txt。"
        ) from exc
    return cv2, mss, np


def list_monitors() -> list[dict[str, int | str]]:
    """列出真实显示器；mss 的索引 0 是所有显示器的联合区域，因此跳过。"""
    _, mss, _ = _screen_modules()
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


def _clamp_region(
    region: dict[str, Any] | None, monitor: dict[str, int]
) -> dict[str, int]:
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
    with mss.mss() as capturer:
        if not 1 <= monitor_id < len(capturer.monitors):
            raise LiveMonitorError(f"显示器 {monitor_id} 不存在。")
        monitor = {key: int(value) for key, value in capturer.monitors[monitor_id].items()}
        capture_region = _clamp_region(region, monitor)
        shot = np.asarray(capturer.grab(capture_region))
    frame = cv2.cvtColor(shot, cv2.COLOR_BGRA2BGR)
    if frame.size == 0 or float(frame.mean()) < 1.0:
        raise LiveMonitorError(
            "捕获画面为空。请确认窗口可见，并允许 Python/终端进行屏幕捕获。"
        )
    return frame, capture_region


def _effective_payload(items: Any) -> list[dict[str, int | str]]:
    return [{"tile": item.tile, "remaining": item.remaining} for item in items]


def _analysis_payload(ocr_result: OCRResult) -> dict[str, Any]:
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
    payload = _analysis_payload(ocr_result)
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


class MonitorRequestHandler(BaseHTTPRequestHandler):
    server_version = "MahjongStudyMonitor/1.0"

    def log_message(self, format_string: str, *args: Any) -> None:
        # 只保留错误日志，避免实时轮询刷满终端。
        if sys.stderr is not None and args and str(args[1]).startswith(("4", "5")):
            super().log_message(format_string, *args)

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def _send_static(self, filename: str, content_type: str) -> None:
        path = WEB_DIR / filename
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        data = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/":
                self._send_static("index.html", "text/html; charset=utf-8")
                return
            if parsed.path == "/styles.css":
                self._send_static("styles.css", "text/css; charset=utf-8")
                return
            if parsed.path == "/app.js":
                self._send_static("app.js", "text/javascript; charset=utf-8")
                return
            if parsed.path == "/api/monitors":
                self._send_json({"monitors": list_monitors()})
                return
            if parsed.path == "/api/frame":
                query = parse_qs(parsed.query)
                monitor_id = int(query.get("monitor", ["1"])[0])
                frame, _ = capture_screen(monitor_id)
                cv2, _, _ = _screen_modules()
                success, encoded = cv2.imencode(
                    ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82]
                )
                if not success:
                    raise LiveMonitorError("无法编码屏幕预览。")
                data = encoded.tobytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)
                return
            self.send_error(HTTPStatus.NOT_FOUND)
        except Exception as exc:  # 浏览器需要可读错误，而不是断开连接
            self._send_json({"error": str(exc)}, HTTPStatus.UNPROCESSABLE_ENTITY)

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        path = urlparse(self.path).path
        if path == "/api/shutdown":
            self._send_json({"ok": True})
            threading.Thread(target=self.server.shutdown, daemon=True).start()
            return
        if path != "/api/analyze":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(length) or b"{}")
            expected_count = request.get("expectedCount")
            if expected_count in ("", None, 0):
                expected_count = None
            else:
                expected_count = int(expected_count)
            payload = analyze_capture(
                monitor_id=int(request.get("monitor", 1)),
                region=request.get("region"),
                expected_count=expected_count,
            )
            self._send_json(payload)
        except (LiveMonitorError, HandValidationError, ValueError) as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.UNPROCESSABLE_ENTITY)
        except Exception as exc:
            self._send_json(
                {"error": f"实时分析失败：{exc}"},
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )


def run_live_monitor(port: int = 8765, open_browser: bool = True) -> None:
    """启动只监听本机回环地址的研究面板。"""
    if not 1024 <= port <= 65535:
        raise LiveMonitorError("端口必须在 1024～65535 之间。")
    if not WEB_DIR.is_dir():
        raise LiveMonitorError(f"监看界面文件不存在：{WEB_DIR}")

    # 启动前只检查依赖；显示器/权限问题交给界面提示，保证面板仍能打开。
    _screen_modules()

    server = ThreadingHTTPServer((HOST, port), MonitorRequestHandler)
    url = f"http://{HOST}:{port}/"
    if sys.stdout is not None:
        print(f"本地牌理研究面板已启动：{url}")
        print("服务仅监听 127.0.0.1；按 Ctrl+C 停止。")
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        if sys.stdout is not None:
            print("\n实时研究面板已停止。")
    finally:
        server.server_close()
