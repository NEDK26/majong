# 本程序仅用于立直麻将算法学习研究，禁止用于线上游戏对局辅助。

from __future__ import annotations

import json
from collections.abc import Sequence

from tile_utils import HandValidationError, tiles_to_34


def _parse_manual_text(raw: str) -> list[str]:
    """支持 JSON 列表，以及空格或逗号分隔的便捷输入。"""
    raw = raw.strip()
    if not raw:
        raise HandValidationError("手牌输入不能为空。")

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = raw.replace(",", " ").split()

    if not isinstance(parsed, list) or not all(isinstance(tile, str) for tile in parsed):
        raise HandValidationError('请输入字符串列表，例如 ["1m", "2m", "3m", ...]。')
    return parsed


def hand_input(source_tiles: Sequence[str] | None = None) -> list[int]:
    """输入层：返回核心算法使用的 34 长度手牌数组。

    当前 ``source_tiles`` 为空时从命令行读取。OCR 模块把识别得到的标准化
    字符串列表作为 ``source_tiles`` 传入，不需要改动分析核心。
    """
    if source_tiles is None:
        raw = input(
            "请输入暗牌（闭门通常为 13/14 张；副露后也支持 10/11、7/8、4/5 张）\n"
            '（例如 ["1m","2m","3m",...]，也可用空格分隔）：\n> '
        )
        tile_names = _parse_manual_text(raw)
    else:
        if isinstance(source_tiles, (str, bytes)):
            raise HandValidationError("OCR/外部输入必须提供字符串列表。")
        tile_names = list(source_tiles)

    return tiles_to_34(tile_names)
