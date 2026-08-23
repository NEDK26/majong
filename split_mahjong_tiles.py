#!/usr/bin/env python3
# 本程序仅用于立直麻将算法学习研究，禁止用于线上游戏对局辅助。

"""从麻将牌总览图中裁出 34 种牌，并使用中文牌名保存。

支持的参考图布局：

* 第一行：一万～九万、东、南、西、北
* 第二行：一筒～九筒、白、发、中
* 第三行：一索～九索

脚本只读取本地图片并写入裁剪结果，不会上传图片。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ocr_input import (
    OCRError,
    _composite_alpha,
    _cv_modules,
    _detected_boxes,
    _read_image,
    _reference_rows,
    _write_image,
)
from tile_utils import tile_name_to_chinese


TILES_BY_ROW: tuple[tuple[str, ...], ...] = (
    (
        *(f"{number}m" for number in range(1, 10)),
        "1z",
        "2z",
        "3z",
        "4z",
    ),
    (
        *(f"{number}p" for number in range(1, 10)),
        "5z",
        "6z",
        "7z",
    ),
    tuple(f"{number}s" for number in range(1, 10)),
)


def _expanded_box(
    box: tuple[int, int, int, int],
    padding: int,
    top_padding: int | None,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    x, y, width, height = box
    # 轮廓检测锁定的是浅色正面；雀魂牌顶还有约 11% 牌高的橙色立体边。
    # 默认把它一并保留，也允许用 --top-padding 0 只导出正面。
    top_extra = round(height * 0.11) if top_padding is None else top_padding
    left = max(0, x - padding)
    top = max(0, y - padding - top_extra)
    right = min(image_width, x + width + padding)
    bottom = min(image_height, y + height + padding)
    return left, top, right - left, bottom - top


def _save_preview(
    image: Any,
    records: list[dict[str, object]],
    preview_path: Path,
    cv2: Any,
) -> None:
    preview = image.copy()
    scale = max(0.7, min(image.shape[:2]) / 900.0)
    thickness = max(2, round(scale * 2))
    for record in records:
        box = record["box"]
        if not isinstance(box, list) or len(box) != 4:
            raise OCRError("内部错误：检测框格式无效。")
        x, y, width, height = (int(value) for value in box)
        index = int(record["index"])
        cv2.rectangle(
            preview,
            (x, y),
            (x + width - 1, y + height - 1),
            (0, 210, 255),
            thickness,
        )
        cv2.putText(
            preview,
            str(index),
            (x + 4, max(18, y - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (0, 210, 255),
            thickness,
            cv2.LINE_AA,
        )
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    if not _write_image(preview_path, preview, cv2):
        raise OCRError(f"无法写入检测预览图：{preview_path}")


def split_mahjong_tiles(
    source_image: str | Path,
    output_dir: str | Path,
    *,
    padding: int = 0,
    top_padding: int | None = None,
    overwrite: bool = False,
    preview_path: str | Path | None = None,
) -> list[dict[str, object]]:
    """检测并裁出 34 张麻将牌，返回牌名、文件名和坐标映射。"""
    if padding < 0:
        raise OCRError("padding 不能小于 0。")
    if top_padding is not None and top_padding < 0:
        raise OCRError("top_padding 不能小于 0。")

    cv2, np = _cv_modules()
    source = Path(source_image).expanduser().resolve()
    if not source.is_file():
        raise OCRError(f"参考图不存在：{source}")

    image = _read_image(source, cv2.IMREAD_UNCHANGED, cv2, np)
    if image is None or getattr(image, "size", 0) == 0:
        raise OCRError(f"无法读取参考图：{source}")
    image = _composite_alpha(image, cv2, np)

    rows = _reference_rows(_detected_boxes(image, cv2, np), np)
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    names = [tile for row in TILES_BY_ROW for tile in row]
    targets = [
        destination / f"{tile_name_to_chinese(tile)}.png" for tile in names
    ]
    existing = [path for path in targets if path.exists()]
    if existing and not overwrite:
        sample = "、".join(path.name for path in existing[:5])
        suffix = "……" if len(existing) > 5 else ""
        raise OCRError(
            f"输出目录中已有同名图片：{sample}{suffix}。"
            "如需覆盖，请添加 --force。"
        )

    records: list[dict[str, object]] = []
    image_height, image_width = image.shape[:2]
    index = 0
    for row, tile_names in zip(rows, TILES_BY_ROW):
        for box, tile in zip(row, tile_names):
            index += 1
            crop_box = _expanded_box(
                box,
                padding,
                top_padding,
                image_width=image_width,
                image_height=image_height,
            )
            x, y, width, height = crop_box
            crop = image[y : y + height, x : x + width]
            chinese_name = tile_name_to_chinese(tile)
            target = destination / f"{chinese_name}.png"
            if not _write_image(target, crop, cv2):
                raise OCRError(f"无法写入单牌图片：{target}")
            records.append(
                {
                    "index": index,
                    "internal_name": tile,
                    "chinese_name": chinese_name,
                    "filename": target.name,
                    "box": [x, y, width, height],
                }
            )

    if len(records) != 34:
        raise OCRError(f"内部错误：预期输出 34 张，实际输出 {len(records)} 张。")

    manifest_path = destination / "牌名映射.json"
    manifest_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if preview_path is not None:
        _save_preview(image, records, Path(preview_path).expanduser().resolve(), cv2)

    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="从三行麻将牌总览图中裁出 34 张中文命名的 PNG。"
    )
    parser.add_argument("image", help="麻将牌总览图路径")
    parser.add_argument("output", help="34 张 PNG 的输出目录")
    parser.add_argument(
        "--padding",
        type=int,
        default=0,
        help="每张牌四周额外保留的像素，默认 0",
    )
    parser.add_argument(
        "--top-padding",
        type=int,
        help="牌面上方保留的像素；默认自动保留立体牌边，设为 0 可只取正面",
    )
    parser.add_argument(
        "--preview",
        metavar="IMAGE",
        help="可选：保存带检测框和 1～34 编号的预览图",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="覆盖输出目录中已有的同名牌图",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        records = split_mahjong_tiles(
            args.image,
            args.output,
            padding=args.padding,
            top_padding=args.top_padding,
            overwrite=args.force,
            preview_path=args.preview,
        )
    except OCRError as exc:
        raise SystemExit(f"拆分失败：{exc}") from exc

    destination = Path(args.output).expanduser().resolve()
    print(f"拆分完成：{len(records)} 张图片")
    print(f"输出目录：{destination}")
    print(f"牌名映射：{destination / '牌名映射.json'}")
    if args.preview:
        print(f"检测预览：{Path(args.preview).expanduser().resolve()}")


if __name__ == "__main__":
    main()
