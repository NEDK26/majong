# 本程序仅用于立直麻将算法学习研究，禁止用于线上游戏对局辅助。

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from mahjong.shanten import Shanten

from tile_utils import VALID_HAND_TOTALS, tile_index_to_name, validate_34_array


@dataclass(frozen=True)
class EffectiveTile:
    """一种能让向听数下降的牌，以及牌山中的理论剩余数。"""

    tile: str
    remaining: int


@dataclass(frozen=True)
class DiscardCandidate:
    """一种舍牌选择的完整分析结果。"""

    discard: str
    shanten: int
    effective_tiles: tuple[EffectiveTile, ...]
    ukeire: int


@dataclass(frozen=True)
class AnalysisResult:
    """核心分析的结构化输出，供控制台或未来其他展示层使用。"""

    original_hand: tuple[int, ...]
    mode: Literal["draw", "discard", "agari"]
    candidates: tuple[DiscardCandidate, ...] = ()
    effective_draws: tuple[EffectiveTile, ...] = ()
    draw_ukeire: int = 0


_SHANTEN = Shanten()


def calculate_shanten(hand_tiles: list[int] | tuple[int, ...]) -> int:
    """通过开源 ``mahjong`` 库计算普通形、七对子、国士的最小向听数。"""
    validate_34_array(hand_tiles)
    return _SHANTEN.calculate_shanten(list(hand_tiles))


def _effective_tiles(
    hand_before_draw: list[int], shanten: int
) -> tuple[tuple[EffectiveTile, ...], int]:
    """枚举 34 种摸牌，找出能严格降低向听数的牌。"""
    draw_state_totals = {total for total in VALID_HAND_TOTALS if total % 3 == 1}
    validate_34_array(hand_before_draw, allowed_totals=draw_state_totals)
    effective: list[EffectiveTile] = []

    for draw_index in range(34):
        # 自己手中已有四枚时，不可能再摸到同种牌。
        if hand_before_draw[draw_index] >= 4:
            continue

        drawn_hand = hand_before_draw.copy()
        drawn_hand[draw_index] += 1
        if calculate_shanten(drawn_hand) < shanten:
            # 按需求只扣除当前手牌，不扣除任何场上可见牌。
            remaining = 4 - hand_before_draw[draw_index]
            effective.append(
                EffectiveTile(tile=tile_index_to_name(draw_index), remaining=remaining)
            )

    return tuple(effective), sum(item.remaining for item in effective)


def core_analyze(hand_tiles: list[int]) -> AnalysisResult:
    """纯计算手牌分析。

    - 3n+2 张（14/11/8/5/2）：逐种模拟舍牌；缺少的面子视为已经副露。
    - 3n+1 张（13/10/7/4/1）：返回当前有效摸牌。

    理论枚数采用 ``4 - 手牌中已有枚数``，暂不扣除牌河、副露等场上信息。
    """
    validate_34_array(hand_tiles)
    original_hand = tuple(hand_tiles)
    total = sum(hand_tiles)
    original_shanten = calculate_shanten(hand_tiles)

    if total % 3 == 1:
        effective_draws, draw_ukeire = _effective_tiles(hand_tiles.copy(), original_shanten)
        return AnalysisResult(
            original_hand=original_hand,
            mode="draw",
            effective_draws=effective_draws,
            draw_ukeire=draw_ukeire,
        )

    # 已经和牌时不再建议舍牌。
    if original_shanten == Shanten.AGARI_STATE:
        return AnalysisResult(original_hand=original_hand, mode="agari")

    candidates: list[DiscardCandidate] = []
    for discard_index, count in enumerate(hand_tiles):
        if count == 0:
            continue

        hand_after_discard = hand_tiles.copy()
        hand_after_discard[discard_index] -= 1
        shanten_after_discard = calculate_shanten(hand_after_discard)
        effective_tiles, ukeire = _effective_tiles(
            hand_after_discard, shanten_after_discard
        )
        candidates.append(
            DiscardCandidate(
                discard=tile_index_to_name(discard_index),
                shanten=shanten_after_discard,
                effective_tiles=effective_tiles,
                ukeire=ukeire,
            )
        )

    # 只使用用户指定的两个策略优先级。完全同分时保留自然牌序，便于阅读。
    candidates.sort(key=lambda item: (item.shanten, -item.ukeire))
    return AnalysisResult(
        original_hand=original_hand,
        mode="discard",
        candidates=tuple(candidates),
    )
