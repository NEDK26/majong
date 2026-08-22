# 本程序仅用于立直麻将算法学习研究，禁止用于线上游戏对局辅助。

from __future__ import annotations

from collections.abc import Sequence


SUITS = ("m", "p", "s", "z")
VALID_HAND_TOTALS = frozenset({1, 2, 4, 5, 7, 8, 10, 11, 13, 14})
TILE_NAMES: tuple[str, ...] = tuple(
    [f"{number}m" for number in range(1, 10)]
    + [f"{number}p" for number in range(1, 10)]
    + [f"{number}s" for number in range(1, 10)]
    + [f"{number}z" for number in range(1, 8)]
)
TILE_TO_INDEX = {tile: index for index, tile in enumerate(TILE_NAMES)}
CHINESE_NUMBERS = ("一", "二", "三", "四", "五", "六", "七", "八", "九")
HONOR_NAMES = ("东", "南", "西", "北", "白", "发", "中")


class HandValidationError(ValueError):
    """手牌输入不符合 34 种牌格式时抛出。"""


def tile_name_to_index(tile: str) -> int:
    """把 ``1m`` 形式的牌名转换为 0～33 的索引。"""
    if not isinstance(tile, str):
        raise HandValidationError(f"牌名必须是字符串，收到：{tile!r}")

    normalized = tile.strip().lower()
    if normalized not in TILE_TO_INDEX:
        raise HandValidationError(
            f"无效牌名 {tile!r}；仅支持 1m-9m、1p-9p、1s-9s、1z-7z。"
        )
    return TILE_TO_INDEX[normalized]


def tile_index_to_name(index: int) -> str:
    """把 0～33 的索引转换为 ``1m`` 形式的牌名。"""
    if not 0 <= index < len(TILE_NAMES):
        raise HandValidationError(f"牌索引必须在 0～33，收到：{index}")
    return TILE_NAMES[index]


def tile_name_to_chinese(tile: str) -> str:
    """把内部牌名转换成适合界面展示的中文牌名。"""
    index = tile_name_to_index(tile)
    if index < 9:
        return f"{CHINESE_NUMBERS[index]}万"
    if index < 18:
        return f"{CHINESE_NUMBERS[index - 9]}筒"
    if index < 27:
        return f"{CHINESE_NUMBERS[index - 18]}索"
    return HONOR_NAMES[index - 27]


def tiles_to_34(tiles: Sequence[str]) -> list[int]:
    """把标准化牌名列表转换为 ``mahjong`` 使用的 34 长度计数数组。"""
    if isinstance(tiles, (str, bytes)):
        raise HandValidationError("手牌必须是字符串列表，不能是单个字符串。")

    counts = [0] * 34
    for tile in tiles:
        index = tile_name_to_index(tile)
        counts[index] += 1
        if counts[index] > 4:
            raise HandValidationError(f"{tile_index_to_name(index)} 超过了 4 枚。")

    validate_34_array(counts)
    return counts


def tiles_from_34(tiles_34: Sequence[int]) -> list[str]:
    """把 34 长度计数数组展开为已按牌序排列的牌名列表。"""
    validate_34_array(tiles_34)
    return [
        tile_index_to_name(index)
        for index, count in enumerate(tiles_34)
        for _ in range(count)
    ]


def validate_34_array(tiles_34: Sequence[int], allowed_totals: set[int] | None = None) -> None:
    """校验内部 34 数组；同时支持闭门和已有副露的合法张数。"""
    if isinstance(tiles_34, (str, bytes)) or len(tiles_34) != 34:
        raise HandValidationError("内部手牌必须是长度为 34 的计数数组。")

    for index, count in enumerate(tiles_34):
        if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= 4:
            raise HandValidationError(
                f"{tile_index_to_name(index)} 的数量必须是 0～4 的整数，收到：{count!r}"
            )

    totals = allowed_totals if allowed_totals is not None else VALID_HAND_TOTALS
    total = sum(tiles_34)
    if total not in totals:
        expected = "、".join(str(value) for value in sorted(totals))
        raise HandValidationError(f"手牌总数必须是 {expected} 张，收到：{total} 张。")
