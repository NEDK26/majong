# 本程序仅用于立直麻将算法学习研究，禁止用于线上游戏对局辅助。

"""桌面展示层的纯辅助函数和可重置识别状态，便于脱离 GUI 环境测试。"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from ocr_input import OCRResult, TileRecognition


@dataclass
class OCRTemporalConsensus:
    """逐牌融合连续帧；相同手牌的稳定重复可消除单帧小分差。"""

    required_frames: int = 2
    maximum_frames: int = 3
    _history: list[OCRResult] = field(default_factory=list)

    @property
    def history_size(self) -> int:
        return len(self._history)

    def reset(self) -> None:
        self._history.clear()

    @staticmethod
    def _candidates(item: TileRecognition) -> set[str]:
        return {item.tile, *(tile for tile, _ in item.alternatives[:3])}

    def _is_same_hand(self, previous: OCRResult, current: OCRResult) -> bool:
        if len(previous.recognitions) != len(current.recognitions):
            return False
        if not current.recognitions:
            return False

        compatible_positions = 0
        for first, second in zip(
            previous.recognitions, current.recognitions, strict=True
        ):
            first_center = first.box[0] + first.box[2] / 2
            second_center = second.box[0] + second.box[2] / 2
            tolerance = max(first.box[2], second.box[2], 1) * 0.42
            if abs(first_center - second_center) > tolerance:
                return False
            if self._candidates(first) & self._candidates(second):
                compatible_positions += 1
        return compatible_positions >= max(
            1, round(len(current.recognitions) * 0.70)
        )

    def update(self, current: OCRResult) -> OCRResult:
        if self._history and not self._is_same_hand(self._history[-1], current):
            self.reset()
        self._history.append(current)
        self._history = self._history[-self.maximum_frames :]
        if len(self._history) < self.required_frames:
            return current
        return self._aggregate()

    def _aggregate(self) -> OCRResult:
        recognitions: list[TileRecognition] = []
        frame_count = len(self._history)
        for position in range(len(self._history[0].recognitions)):
            items = [frame.recognitions[position] for frame in self._history]
            scores: dict[str, float] = defaultdict(float)
            primary_votes = Counter(item.tile for item in items)
            for item in items:
                primary_score = item.match_score or item.confidence
                scores[item.tile] += float(primary_score)
                for tile, score in item.alternatives:
                    scores[tile] += float(score)

            ranking = sorted(
                (
                    (tile, score / frame_count)
                    for tile, score in scores.items()
                ),
                key=lambda pair: (pair[1], primary_votes[pair[0]]),
                reverse=True,
            )
            best_tile, average_score = ranking[0]
            unanimous = primary_votes[best_tile] == frame_count
            stability_bonus = 0.04 if unanimous else 0.0
            best_score = min(1.0, average_score + stability_bonus)
            alternatives = tuple(ranking[1:9])
            second_score = alternatives[0][1] if alternatives else 0.0
            margin = max(0.0, best_score - second_score)
            confidence = min(
                1.0, best_score * 0.86 + min(0.14, margin * 1.8)
            )
            averaged_box = tuple(
                round(sum(item.box[index] for item in items) / frame_count)
                for index in range(4)
            )
            recognitions.append(
                TileRecognition(
                    tile=best_tile,
                    confidence=confidence,
                    alternatives=alternatives,
                    box=averaged_box,
                    match_score=best_score,
                )
            )
        return OCRResult(
            image_path=self._history[-1].image_path,
            recognitions=tuple(recognitions),
        )


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
