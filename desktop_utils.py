# 本程序仅用于立直麻将算法学习研究，禁止用于线上游戏对局辅助。

"""桌面展示层的无状态辅助函数，便于脱离 GUI 环境测试。"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class AnalysisStabilizer:
    """只在张数/状态连续出现后接受变化，避免单帧 OCR 抖动覆盖界面。"""

    required_hits: int = 2
    accepted_signature: tuple[int, str] | None = None
    pending_signature: tuple[int, str] | None = None
    pending_hits: int = 0

    def reset(self) -> None:
        self.accepted_signature = None
        self.pending_signature = None
        self.pending_hits = 0

    def should_accept(self, signature: tuple[int, str]) -> bool:
        if signature == self.accepted_signature:
            self.pending_signature = None
            self.pending_hits = 0
            return True
        if signature != self.pending_signature:
            self.pending_signature = signature
            self.pending_hits = 1
            if self.required_hits <= 1:
                self.accepted_signature = signature
                self.pending_signature = None
                self.pending_hits = 0
                return True
            return False
        self.pending_hits += 1
        if self.pending_hits < self.required_hits:
            return False
        self.accepted_signature = signature
        self.pending_signature = None
        self.pending_hits = 0
        return True


def fit_preview_size(
    source_width: int, source_height: int, target_width: int, target_height: int
) -> tuple[int, int]:
    """等比缩放显示器预览，不放大源图。"""
    scale = min(
        1.0,
        max(1, target_width) / max(1, source_width),
        max(1, target_height) / max(1, source_height),
    )
    return max(1, round(source_width * scale)), max(1, round(source_height * scale))


def shanten_label(value: int) -> str:
    if value == -1:
        return "和牌"
    if value == 0:
        return "听牌"
    return f"{value} 向听"


def friendly_error_message(error: BaseException) -> str:
    """把底层库异常转换为用户可操作的中文提示。"""
    message = str(error).strip()
    # 项目自己的异常已经提供中文说明，保留其中的具体处理建议。
    if re.search(r"[\u4e00-\u9fff]", message):
        return message

    lowered = message.lower()
    if isinstance(error, (ModuleNotFoundError, ImportError)):
        return "程序组件不完整，请重新下载最新版 EXE。"
    if isinstance(error, PermissionError) or any(
        keyword in lowered for keyword in ("access denied", "permission denied")
    ):
        return "没有屏幕捕获权限。请在 Windows 隐私设置中允许桌面应用捕获屏幕。"
    if isinstance(error, FileNotFoundError) or "no such file" in lowered:
        return "选择的图片或程序资源不存在，请重新选择文件。"
    if any(keyword in lowered for keyword in ("monitor", "display", "screen")):
        return "无法读取显示器。请确认显示器已连接，然后重新打开牌理镜。"
    if any(keyword in lowered for keyword in ("bitmap", "image", "jpeg", "png")):
        return "无法读取图像。请使用清晰的 PNG 或 JPG 图片后重试。"
    return "操作没有完成。请重新选择显示器和手牌区域后再试。"
