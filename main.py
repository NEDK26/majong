# 本程序仅用于立直麻将算法学习研究，禁止用于线上游戏对局辅助。

from __future__ import annotations

import argparse

from analyzer import calculate_shanten, core_analyze
from input_layer import hand_input
from reporter import report_printer
from tile_utils import HandValidationError, tiles_from_34


DEMO_HANDS: tuple[tuple[str, list[str]], ...] = (
    (
        "需求中的 13 张示例（展示当前有效摸牌）",
        ["1m", "2m", "3m", "4p", "5p", "6s", "7s", "8s", "9s", "1s", "2s", "3s", "5m"],
    ),
    (
        "14 张四面子形（展示候选舍牌）",
        ["1m", "2m", "3m", "1p", "2p", "3p", "1s", "2s", "3s", "7s", "8s", "9s", "1z", "9m"],
    ),
    (
        "14 张七对子方向",
        ["1m", "1m", "2m", "2m", "3p", "3p", "4p", "4p", "5s", "5s", "6s", "6s", "7z", "9m"],
    ),
)


def analyze_and_print(hand_tiles: list[int]) -> None:
    """调用纯计算层，再把结果交给报告层。"""
    original_shanten = calculate_shanten(hand_tiles)
    result = core_analyze(hand_tiles)
    report_printer(result, original_shanten)


def run_demos() -> None:
    for title, tile_names in DEMO_HANDS:
        print(f"\n【测试用例】{title}")
        analyze_and_print(hand_input(tile_names))


def run_interactive() -> None:
    hand_tiles = hand_input()
    analyze_and_print(hand_tiles)

    # 13 张不能直接舍牌；允许用户补录摸到的第 14 张后立即继续分析。
    if sum(hand_tiles) == 13:
        drawn = input("\n可选：输入刚摸到的第 14 张牌（直接回车结束）：\n> ").strip()
        if drawn:
            full_hand = tiles_from_34(hand_tiles) + [drawn]
            analyze_and_print(hand_input(full_hand))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="仅供算法学习研究的立直麻将向听与理论进张分析工具。"
    )
    parser.add_argument(
        "--hand",
        nargs="+",
        metavar="TILE",
        help="直接传入 13 或 14 张牌，例如：--hand 1m 2m 3m ...",
    )
    parser.add_argument("--demo", action="store_true", help="运行内置的 3 组测试手牌")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        if args.demo:
            run_demos()
        elif args.hand:
            analyze_and_print(hand_input(args.hand))
        else:
            run_interactive()
    except (HandValidationError, ValueError) as exc:
        raise SystemExit(f"输入错误：{exc}") from exc


if __name__ == "__main__":
    main()
