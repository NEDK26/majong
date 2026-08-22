# 本程序仅用于立直麻将算法学习研究，禁止用于线上游戏对局辅助。

from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys

from analyzer import calculate_shanten, core_analyze
from desktop_utils import friendly_error_message
from input_layer import hand_input
from ocr_input import (
    DEFAULT_MAHJONG_SOUL_TEMPLATE_DIR,
    DEFAULT_TEMPLATE_DIR,
    OCRResult,
    build_mahjong_soul_templates,
    recognize_hand_image,
    save_debug_image,
)
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


def is_frozen_app() -> bool:
    """是否正在 PyInstaller 打包后的程序中运行。"""
    return bool(getattr(sys, "frozen", False))


def native_message(title: str, message: str, *, error: bool = False) -> None:
    """EXE 无控制台时用 Windows 对话框反馈校准结果和启动错误。"""
    if is_frozen_app() and os.name == "nt":
        icon = 0x10 if error else 0x40
        ctypes.windll.user32.MessageBoxW(None, message, title, icon)
        return
    print(f"{title}：{message}")


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

    # 3n+1 张是待摸牌状态；允许补录下一张后立即继续分析。
    if sum(hand_tiles) % 3 == 1:
        next_count = sum(hand_tiles) + 1
        drawn = input(
            f"\n可选：输入刚摸到的第 {next_count} 张暗牌（直接回车结束）：\n> "
        ).strip()
        if drawn:
            full_hand = tiles_from_34(hand_tiles) + [drawn]
            analyze_and_print(hand_input(full_hand))


def print_ocr_result(result: OCRResult) -> None:
    """在进入算法层前展示 OCR 原始结果，便于人工核对。"""
    print("\nOCR 识别结果：", json.dumps(result.tiles, ensure_ascii=False))
    details = "、".join(
        f"{index}:{item.tile}({item.confidence:.2f})"
        for index, item in enumerate(result.recognitions, start=1)
    )
    print("单牌置信度：", details)


def run_ocr(
    image_path: str,
    expected_count: int | None,
    template_dir: str | None,
    minimum_confidence: float,
    allow_low_confidence: bool,
    debug_path: str | None,
) -> None:
    """执行本地 OCR，并把标准化牌名交给原有输入层。"""
    options = {"expected_count": expected_count}
    if template_dir:
        options["template_dir"] = template_dir
    elif DEFAULT_MAHJONG_SOUL_TEMPLATE_DIR.is_dir():
        options["template_dir"] = DEFAULT_MAHJONG_SOUL_TEMPLATE_DIR
        print(f"使用本机雀魂模板：{DEFAULT_MAHJONG_SOUL_TEMPLATE_DIR}")
    result = recognize_hand_image(image_path, **options)
    print_ocr_result(result)

    if debug_path:
        saved_path = save_debug_image(result, debug_path)
        print(f"OCR 调试图已保存：{saved_path}")

    low_confidence = [
        (index, item)
        for index, item in enumerate(result.recognitions, start=1)
        if item.confidence < minimum_confidence
    ]
    if low_confidence and not allow_low_confidence:
        positions = "、".join(
            f"第 {index} 张 {item.tile}({item.confidence:.2f})"
            for index, item in low_confidence
        )
        raise HandValidationError(
            f"OCR 置信度不足：{positions}。请裁紧手牌区域、提供对应皮肤模板，"
            "或人工核对后使用 --ocr-allow-low-confidence。"
        )

    analyze_and_print(hand_input(result.tiles))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="仅供算法学习研究的立直麻将向听与理论进张分析工具。"
    )
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument(
        "--hand",
        nargs="+",
        metavar="TILE",
        help="直接传入暗牌；支持闭门及副露后的合法张数",
    )
    sources.add_argument(
        "--ocr",
        metavar="IMAGE",
        help="从本地静态图片识别一行手牌；不截图、不读取游戏进程",
    )
    sources.add_argument(
        "--ocr-calibrate-mahjong-soul",
        metavar="REFERENCE_IMAGE",
        help="从雀魂 34 种牌说明图生成本机专用模板",
    )
    sources.add_argument(
        "--live",
        action="store_true",
        help="启动原生桌面研究客户端",
    )
    parser.add_argument(
        "--ocr-count",
        type=int,
        choices=(1, 2, 4, 5, 7, 8, 10, 11, 13, 14),
        help="明确暗牌张数；副露两组且待舍牌时为 8",
    )
    parser.add_argument(
        "--ocr-templates",
        metavar="DIR",
        help="自定义牌面模板目录，文件名规则见 README",
    )
    parser.add_argument(
        "--ocr-min-confidence",
        type=float,
        default=0.62,
        metavar="FLOAT",
        help="最低可接受置信度，默认 0.62",
    )
    parser.add_argument(
        "--ocr-allow-low-confidence",
        action="store_true",
        help="人工核对后允许低置信度结果继续进入分析层",
    )
    parser.add_argument(
        "--ocr-debug",
        metavar="IMAGE",
        help="保存带检测框和置信度的调试图片",
    )
    sources.add_argument("--demo", action="store_true", help="运行内置的 3 组测试手牌")
    parser.add_argument("--smoke-test", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "drop_image",
        nargs="?",
        help=argparse.SUPPRESS,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        explicit_source = any(
            (
                args.hand,
                args.ocr,
                args.ocr_calibrate_mahjong_soul,
                args.live,
                args.demo,
                args.smoke_test,
            )
        )
        if args.drop_image and explicit_source:
            raise HandValidationError("拖入图片不能与其他输入参数同时使用。")
        if not 0.0 <= args.ocr_min_confidence <= 1.0:
            raise HandValidationError("--ocr-min-confidence 必须在 0～1 之间。")
        calibration_image = args.ocr_calibrate_mahjong_soul or args.drop_image
        if args.smoke_test:
            from ocr_input import _cv_modules, _template_descriptors
            from screen_capture import _dxcam_module, _screen_modules

            cv2, np = _cv_modules()
            _screen_modules()
            _template_descriptors(DEFAULT_TEMPLATE_DIR, cv2, np)
            sample = np.zeros((12, 12, 3), dtype=np.uint8)
            cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)

            # Windows 构建必须同时验证 Tk 桌面组件；非 Windows 开发机可能没有 Tk。
            if os.name == "nt":
                from desktop_app import desktop_ui_smoke_test

                _dxcam_module()
                desktop_ui_smoke_test()
            if not DEFAULT_TEMPLATE_DIR.is_dir():
                raise RuntimeError("内置牌面模板缺失。")

        elif calibration_image:
            template_path = build_mahjong_soul_templates(
                calibration_image
            )
            if is_frozen_app():
                native_message(
                    "牌理镜 · 校准完成",
                    f"34 种牌模板已保存到：\n{template_path}\n\n现在可以双击 EXE 开始测试。",
                )
            else:
                print(f"雀魂 OCR 模板已生成：{template_path}")
                print("之后使用 --ocr 时会自动优先加载该模板。")
        elif args.live:
            from desktop_app import run_desktop_app

            run_desktop_app()
        elif args.demo:
            run_demos()
        elif args.ocr:
            run_ocr(
                image_path=args.ocr,
                expected_count=args.ocr_count,
                template_dir=args.ocr_templates,
                minimum_confidence=args.ocr_min_confidence,
                allow_low_confidence=args.ocr_allow_low_confidence,
                debug_path=args.ocr_debug,
            )
        elif args.hand:
            analyze_and_print(hand_input(args.hand))
        elif is_frozen_app():
            from desktop_app import run_desktop_app

            run_desktop_app()
        else:
            run_interactive()
    except (HandValidationError, ValueError, RuntimeError) as exc:
        if args.smoke_test:
            raise SystemExit(1) from exc
        if is_frozen_app():
            native_message(
                "牌理镜 · 启动失败", friendly_error_message(exc), error=True
            )
            return
        raise SystemExit(f"错误：{exc}") from exc
    except Exception as exc:
        if args.smoke_test:
            raise SystemExit(1) from exc
        if is_frozen_app():
            native_message(
                "牌理镜 · 启动失败", friendly_error_message(exc), error=True
            )
            return
        raise


if __name__ == "__main__":
    main()
