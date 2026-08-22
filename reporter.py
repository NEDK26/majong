# 本程序仅用于立直麻将算法学习研究，禁止用于线上游戏对局辅助。

from __future__ import annotations

import json
import sys

from analyzer import AnalysisResult, DiscardCandidate, EffectiveTile
from tile_utils import tiles_from_34


def _shanten_text(shanten: int) -> str:
    if shanten == -1:
        return "-1（已和牌）"
    if shanten == 0:
        return "0（听牌）"
    return f"{shanten}（{shanten} 向听）"


def _effective_text(effective_tiles: tuple[EffectiveTile, ...]) -> str:
    if not effective_tiles:
        return "无"
    return "、".join(f"{item.tile}×{item.remaining}" for item in effective_tiles)


def _highlight(text: str) -> str:
    """交互式终端使用绿色粗体；重定向输出仍保留星号标识。"""
    if sys.stdout.isatty():
        return f"\033[1;32m{text}\033[0m"
    return text


def _is_best(candidate: DiscardCandidate, best: DiscardCandidate) -> bool:
    return candidate.shanten == best.shanten and candidate.ukeire == best.ukeire


def report_printer(result: AnalysisResult, original_shanten: int) -> None:
    """把核心结果格式化输出到控制台。"""
    hand_names = tiles_from_34(result.original_hand)
    print("\n" + "=" * 68)
    print("立直麻将手牌分析（仅供算法学习研究）")
    print("当前手牌：", json.dumps(hand_names, ensure_ascii=False))
    print("当前向听数：", _shanten_text(original_shanten))

    if result.mode == "agari":
        print(_highlight("★ 当前 14 张手牌已经和牌，无需舍牌。"))
        print("=" * 68)
        return

    if result.mode == "draw":
        current_count = sum(result.original_hand)
        print(
            f"状态：{current_count} 张待摸牌；当前没有合法候选舍牌。"
            "张数少于 13 时，缺少的面子按已副露处理。"
        )
        print("有效摸牌：", _effective_text(result.effective_draws))
        print(f"理论有效枚数：{result.draw_ukeire}")
        print(f"摸牌后请把第 {current_count + 1} 张加入暗牌，再进行舍牌分析。")
        print("=" * 68)
        return

    if not result.candidates:
        print("没有可用的候选舍牌。")
        print("=" * 68)
        return

    best = result.candidates[0]
    best_discards = [
        candidate.discard
        for candidate in result.candidates
        if _is_best(candidate, best)
    ]
    recommendation = "、".join(best_discards)
    if len(best_discards) > 1:
        recommendation += "（并列最优）"
    print(_highlight(f"★ 推荐舍牌：{recommendation}"))
    print("\n候选舍牌（向听数升序，其次理论有效枚数降序）：")

    for rank, candidate in enumerate(result.candidates, start=1):
        marker = "★" if _is_best(candidate, best) else " "
        line = (
            f"{marker} {rank:>2}. 打 {candidate.discard:<2} | "
            f"打后向听 {_shanten_text(candidate.shanten):<10} | "
            f"有效进张：{_effective_text(candidate.effective_tiles)} | "
            f"总枚数：{candidate.ukeire}"
        )
        print(_highlight(line) if marker == "★" else line)

    print("\n注：理论枚数仅扣除当前手牌，不扣除牌河、副露等场上可见牌。")
    print("=" * 68)
