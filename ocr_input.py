# 本程序仅用于立直麻将算法学习研究，禁止用于线上游戏对局辅助。

"""本地麻将牌图片识别。

该模块只读取用户明确提供的静态图片，不截图、不读取游戏进程、不抓包，
也不执行鼠标键盘操作。识别方法为 OpenCV 轮廓分割 + 本地模板匹配。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tile_utils import HandValidationError


DEFAULT_TEMPLATE_DIR = Path(__file__).resolve().parent / "assets" / "ocr_templates"
DEFAULT_MAHJONG_SOUL_TEMPLATE_DIR = (
    Path(__file__).resolve().parent / "local_ocr_templates" / "mahjong_soul"
)
SUPPORTED_HAND_SIZES = (1, 2, 4, 5, 7, 8, 10, 11, 13, 14)
AUTO_DETECT_HAND_SIZES = (7, 8, 10, 11, 13, 14)


class OCRError(HandValidationError):
    """图片无法读取、分割或可靠识别时抛出。"""


@dataclass(frozen=True)
class TileRecognition:
    """单张牌的 OCR 结果。"""

    tile: str
    confidence: float
    alternatives: tuple[tuple[str, float], ...]
    box: tuple[int, int, int, int]


@dataclass(frozen=True)
class OCRResult:
    """整手牌的 OCR 结构化结果。"""

    image_path: Path
    recognitions: tuple[TileRecognition, ...]

    @property
    def tiles(self) -> list[str]:
        return [item.tile for item in self.recognitions]

    @property
    def minimum_confidence(self) -> float:
        if not self.recognitions:
            return 0.0
        return min(item.confidence for item in self.recognitions)


_TEMPLATE_FILES: dict[str, str] = {
    **{f"Man{number}.png": f"{number}m" for number in range(1, 10)},
    **{f"Pin{number}.png": f"{number}p" for number in range(1, 10)},
    **{f"Sou{number}.png": f"{number}s" for number in range(1, 10)},
    "Ton.png": "1z",
    "Nan.png": "2z",
    "Shaa.png": "3z",
    "Pei.png": "4z",
    "Haku.png": "5z",
    "Hatsu.png": "6z",
    "Chun.png": "7z",
    # 赤五在分析层按普通 5 处理。
    "Man5-Dora.png": "5m",
    "Pin5-Dora.png": "5p",
    "Sou5-Dora.png": "5s",
}


def _cv_modules() -> tuple[Any, Any]:
    """延迟导入，让纯手动输入在未安装 OCR 依赖时仍能给出清晰错误。"""
    try:
        import cv2  # type: ignore[import-not-found]
        import numpy as np  # type: ignore[import-not-found]
    except ImportError as exc:
        raise OCRError(
            "OCR 依赖未安装；请执行 pip install -r requirements.txt。"
        ) from exc
    return cv2, np


def _composite_alpha(image: Any, cv2: Any, np: Any) -> Any:
    if image.ndim != 3:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] != 4:
        return image[:, :, :3]

    alpha = image[:, :, 3:4].astype(np.float32) / 255.0
    foreground = image[:, :, :3].astype(np.float32)
    white = np.full_like(foreground, 255.0)
    return (foreground * alpha + white * (1.0 - alpha)).astype(np.uint8)


def _ink_descriptor(image: Any, cv2: Any, np: Any) -> tuple[Any, Any, float]:
    """把牌面转换为对尺寸、留白和轻微颜色变化较稳健的描述子。"""
    image = _composite_alpha(image, cv2, np)
    if image.size == 0:
        raise OCRError("检测到了空的牌面区域。")

    normalized = cv2.resize(image, (96, 128), interpolation=cv2.INTER_AREA)
    # 去掉牌框和阴影，只比较中央牌面图案。
    inner = normalized[8:120, 8:88]
    lab = cv2.cvtColor(inner, cv2.COLOR_BGR2LAB).astype(np.float32)
    lightness = lab[:, :, 0]
    bright_pixels = lightness >= np.percentile(lightness, 62)
    background = np.median(lab[bright_pixels], axis=0)
    color_distance = np.linalg.norm(lab - background, axis=2)

    # 深色笔画或与牌面底色有明显色差的像素都视为图案。
    mask = ((color_distance > 24.0) | (lightness < background[0] - 28.0)).astype(
        np.uint8
    )
    mask[:3, :] = 0
    mask[-3:, :] = 0
    mask[:, :3] = 0
    mask[:, -3:] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    components, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    cleaned = np.zeros_like(mask)
    for component in range(1, components):
        if stats[component, cv2.CC_STAT_AREA] >= 5:
            cleaned[labels == component] = 1
    mask = cleaned

    foreground_ratio = float(mask.mean())
    if foreground_ratio < 0.003:
        # 白板本身没有图案。
        return np.zeros((88, 64), np.uint8), np.zeros(5, np.float32), foreground_ratio

    ys, xs = np.where(mask > 0)
    x1, x2 = max(0, int(xs.min()) - 2), min(mask.shape[1], int(xs.max()) + 3)
    y1, y2 = max(0, int(ys.min()) - 2), min(mask.shape[0], int(ys.max()) + 3)
    glyph_mask = mask[y1:y2, x1:x2] * 255

    glyph = np.zeros((88, 64), np.uint8)
    scale = min(58 / glyph_mask.shape[1], 82 / glyph_mask.shape[0])
    width = max(1, round(glyph_mask.shape[1] * scale))
    height = max(1, round(glyph_mask.shape[0] * scale))
    resized = cv2.resize(glyph_mask, (width, height), interpolation=cv2.INTER_AREA)
    left = (glyph.shape[1] - width) // 2
    top = (glyph.shape[0] - height) // 2
    glyph[top : top + height, left : left + width] = resized

    hsv = cv2.cvtColor(inner, cv2.COLOR_BGR2HSV)
    pixels = hsv[mask > 0]
    hue, saturation, value = pixels[:, 0], pixels[:, 1], pixels[:, 2]
    colorful = saturation >= 55
    categories = np.array(
        [
            ((hue <= 12) | (hue >= 168)) & colorful,  # 红
            (hue >= 34) & (hue <= 90) & colorful,  # 绿
            (hue >= 91) & (hue <= 145) & colorful,  # 蓝
            (~colorful) & (value < 175),  # 黑/灰
            np.ones_like(hue, dtype=bool),  # 总前景，用于归一化
        ]
    )
    color_vector = categories[:4].sum(axis=1).astype(np.float32)
    color_vector /= max(1.0, float(categories[4].sum()))
    color_vector = np.append(color_vector, foreground_ratio).astype(np.float32)
    return glyph, color_vector, foreground_ratio


def _template_descriptors(template_dir: Path, cv2: Any, np: Any) -> list[tuple[str, Any, Any, float]]:
    if not template_dir.is_dir():
        raise OCRError(f"OCR 模板目录不存在：{template_dir}")

    descriptors: list[tuple[str, Any, Any, float]] = []
    missing: list[str] = []
    for filename, tile in _TEMPLATE_FILES.items():
        path = template_dir / filename
        if not path.is_file():
            # 赤五模板是可选的，基础 34 种牌必须齐全。
            if "-Dora" not in filename:
                missing.append(filename)
            continue
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise OCRError(f"无法读取 OCR 模板：{path}")
        glyph, colors, ratio = _ink_descriptor(image, cv2, np)
        descriptors.append((tile, glyph, colors, ratio))

    if missing:
        raise OCRError("OCR 模板缺失：" + "、".join(missing))
    return descriptors


def _iou(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    overlap = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - overlap
    return overlap / union if union else 0.0


def _covered_by(
    smaller: tuple[int, int, int, int], larger: tuple[int, int, int, int]
) -> float:
    """返回 smaller 被 larger 覆盖的面积比例，用于去除牌内图案轮廓。"""
    ax, ay, aw, ah = smaller
    bx, by, bw, bh = larger
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    overlap = max(0, x2 - x1) * max(0, y2 - y1)
    return overlap / max(1, aw * ah)


def _detected_boxes(image: Any, cv2: Any, np: Any) -> list[tuple[int, int, int, int]]:
    """用牌面中性亮色区域和边缘轮廓寻找直立牌。"""
    height, width = image.shape[:2]
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    light, channel_a, channel_b = cv2.split(lab)
    neutral_light = (
        (light > 145)
        & (np.abs(channel_a.astype(np.int16) - 128) < 22)
        & (np.abs(channel_b.astype(np.int16) - 128) < 28)
    ).astype(np.uint8) * 255

    kernel_size = max(3, round(min(height, width) * 0.004))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    neutral_light = cv2.morphologyEx(neutral_light, cv2.MORPH_CLOSE, kernel)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 45, 140)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    boxes: list[tuple[int, int, int, int]] = []
    for mask in (neutral_light, edges):
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, box_width, box_height = cv2.boundingRect(contour)
            aspect = box_width / box_height if box_height else 0.0
            if box_height < max(28, height * 0.055):
                continue
            if box_height > height * 0.98 or not 0.46 <= aspect <= 0.92:
                continue
            if box_width * box_height < width * height * 0.0006:
                continue
            boxes.append((x, y, box_width, box_height))

    # 同一张牌通常会产生多个近似轮廓，只保留较大的一个。
    kept: list[tuple[int, int, int, int]] = []
    for box in sorted(boxes, key=lambda item: item[2] * item[3], reverse=True):
        if all(
            _iou(box, existing) < 0.72 and _covered_by(box, existing) < 0.82
            for existing in kept
        ):
            kept.append(box)
    return kept


def _reference_rows(
    boxes: list[tuple[int, int, int, int]], np: Any
) -> list[list[tuple[int, int, int, int]]]:
    """从雀魂麻将牌说明图中找出长度为 13、12、9 的三行牌。"""
    best: tuple[float, list[list[tuple[int, int, int, int]]]] | None = None
    for seed in boxes:
        similar = [
            box
            for box in boxes
            if 0.88 <= box[2] / seed[2] <= 1.12
            and 0.88 <= box[3] / seed[3] <= 1.12
        ]
        if len(similar) < 34:
            continue

        similar.sort(key=lambda item: item[1] + item[3] / 2)
        rows: list[list[tuple[int, int, int, int]]] = []
        for box in similar:
            center_y = box[1] + box[3] / 2
            target = next(
                (
                    row
                    for row in rows
                    if abs(
                        center_y
                        - sum(item[1] + item[3] / 2 for item in row) / len(row)
                    )
                    <= seed[3] * 0.35
                ),
                None,
            )
            if target is None:
                rows.append([box])
            else:
                target.append(box)

        rows = [sorted(row, key=lambda item: item[0]) for row in rows]
        rows.sort(key=lambda row: sum(item[1] for item in row) / len(row))
        for start in range(max(1, len(rows) - 2)):
            candidate = rows[start : start + 3]
            if [len(row) for row in candidate] != [13, 12, 9]:
                continue
            score = sum(_layout_regularity(row, np) for row in candidate) / 3.0
            if best is None or score > best[0]:
                best = score, candidate

    if best is None:
        raise OCRError(
            "无法从参考图中定位 34 种牌。需要类似雀魂“麻将牌”说明页的三行布局："
            "第一行 9 万+4 风，第二行 9 筒+3 元，第三行 9 索。"
        )
    return best[1]


def build_mahjong_soul_templates(
    reference_image: str | Path,
    output_dir: str | Path = DEFAULT_MAHJONG_SOUL_TEMPLATE_DIR,
) -> Path:
    """从雀魂的 34 种牌说明图生成本机专用模板。

    参考图和裁出的模板默认保存在被 Git 忽略的本地目录，不会进入公开仓库。
    """
    cv2, np = _cv_modules()
    source = Path(reference_image).expanduser().resolve()
    if not source.is_file():
        raise OCRError(f"OCR 参考图不存在：{source}")
    image = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise OCRError(f"无法读取 OCR 参考图：{source}")
    image = _composite_alpha(image, cv2, np)
    rows = _reference_rows(_detected_boxes(image, cv2, np), np)

    filenames = [
        [*[f"Man{number}.png" for number in range(1, 10)], "Ton.png", "Nan.png", "Shaa.png", "Pei.png"],
        [*[f"Pin{number}.png" for number in range(1, 10)], "Haku.png", "Hatsu.png", "Chun.png"],
        [f"Sou{number}.png" for number in range(1, 10)],
    ]
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    for row, names in zip(rows, filenames, strict=True):
        for box, filename in zip(row, names, strict=True):
            x, y, width, height = box
            crop = image[y : y + height, x : x + width]
            if not cv2.imwrite(str(destination / filename), crop):
                raise OCRError(f"无法写入 OCR 模板：{destination / filename}")
    return destination


def _layout_regularity(boxes: list[tuple[int, int, int, int]], np: Any) -> float:
    widths = np.array([box[2] for box in boxes], dtype=np.float32)
    heights = np.array([box[3] for box in boxes], dtype=np.float32)
    centers_y = np.array([box[1] + box[3] / 2 for box in boxes], dtype=np.float32)
    lefts = np.array([box[0] for box in boxes], dtype=np.float32)
    steps = np.diff(lefts)

    penalties = [
        float(widths.std() / max(1.0, widths.mean())),
        float(heights.std() / max(1.0, heights.mean())),
        float(centers_y.std() / max(1.0, heights.mean())),
    ]
    if len(steps):
        penalties.append(float(steps.std() / max(1.0, steps.mean())))
    return max(0.0, 1.0 - sum(penalties) / len(penalties) * 2.5)


def _contour_layouts(
    image: Any, expected_count: int | None, cv2: Any, np: Any
) -> list[tuple[list[tuple[int, int, int, int]], float]]:
    boxes = _detected_boxes(image, cv2, np)
    layouts: list[tuple[list[tuple[int, int, int, int]], float]] = []
    counts = (expected_count,) if expected_count else AUTO_DETECT_HAND_SIZES

    for count in counts:
        for seed in boxes:
            seed_center = seed[1] + seed[3] / 2
            row = [
                box
                for box in boxes
                if 0.60 <= box[3] / seed[3] <= 1.55
                and abs((box[1] + box[3] / 2) - seed_center) <= max(box[3], seed[3]) * 0.38
            ]
            row.sort(key=lambda item: item[0])
            if len(row) < count:
                continue
            for start in range(len(row) - count + 1):
                window = row[start : start + count]
                regularity = _layout_regularity(window, np)
                if regularity >= 0.30:
                    layouts.append((window, regularity))

    # 去掉完全相同或高度重叠的重复布局。
    unique: list[tuple[list[tuple[int, int, int, int]], float]] = []
    signatures: set[tuple[tuple[int, int, int, int], ...]] = set()
    for boxes_in_layout, score in sorted(layouts, key=lambda item: item[1], reverse=True):
        signature = tuple(boxes_in_layout)
        if signature not in signatures:
            signatures.add(signature)
            unique.append((boxes_in_layout, score))
    return unique[:12]


def _inferred_grid_layouts(
    image: Any, expected_count: int | None, cv2: Any, np: Any
) -> list[tuple[list[tuple[int, int, int, int]], float]]:
    """根据同一行的部分完整外框补齐漏检牌，适合雀魂底部暗牌。"""
    detected = _detected_boxes(image, cv2, np)
    counts = (expected_count,) if expected_count else tuple(reversed(AUTO_DETECT_HAND_SIZES))
    layouts: list[tuple[list[tuple[int, int, int, int]], float]] = []

    for seed in detected:
        anchors = [
            box
            for box in detected
            if 0.78 <= box[2] / seed[2] <= 1.25
            and 0.78 <= box[3] / seed[3] <= 1.25
            and abs((box[1] + box[3] / 2) - (seed[1] + seed[3] / 2))
            <= seed[3] * 0.30
        ]
        anchors.sort(key=lambda item: item[0])
        if len(anchors) < 3:
            continue

        differences = np.diff([box[0] for box in anchors]).astype(np.float32)
        step_candidates: list[float] = []
        median_width = float(np.median([box[2] for box in anchors]))
        for difference in differences:
            multiples = max(1, round(float(difference) / max(1.0, median_width)))
            if multiples <= 3:
                step_candidates.append(float(difference) / multiples)
        if not step_candidates:
            continue
        step = float(np.median(step_candidates))
        if not median_width * 0.82 <= step <= median_width * 1.45:
            continue

        for count in counts:
            for first_anchor in anchors:
                predicted: list[tuple[int, int, int, int]] = []
                used: set[int] = set()
                for index in range(count):
                    target_x = first_anchor[0] + index * step
                    nearest_index = min(
                        range(len(anchors)),
                        key=lambda anchor_index: abs(anchors[anchor_index][0] - target_x),
                    )
                    nearest = anchors[nearest_index]
                    if (
                        nearest_index not in used
                        and abs(nearest[0] - target_x) <= step * 0.36
                    ):
                        predicted.append(nearest)
                        used.add(nearest_index)
                    else:
                        predicted.append(
                            (
                                round(target_x),
                                round(np.median([box[1] for box in anchors])),
                                round(median_width),
                                round(np.median([box[3] for box in anchors])),
                            )
                        )

                if len(used) < max(3, count // 2):
                    continue
                x2 = predicted[-1][0] + predicted[-1][2]
                if predicted[0][0] < 0 or x2 > image.shape[1]:
                    continue
                anchor_coverage = len(used) / count
                regularity = _layout_regularity(predicted, np)
                layouts.append((predicted, anchor_coverage * 0.7 + regularity * 0.3))

    return sorted(layouts, key=lambda item: item[1], reverse=True)[:16]


def _uniform_layout(
    image: Any, count: int
) -> tuple[list[tuple[int, int, int, int]], float] | None:
    """为已经紧密裁成一行的图片提供等分后备方案。"""
    height, width = image.shape[:2]
    cell_width = width / count
    if width / max(1, height) < count * 0.42:
        return None

    boxes: list[tuple[int, int, int, int]] = []
    for index in range(count):
        left = round(index * cell_width + cell_width * 0.06)
        right = round((index + 1) * cell_width - cell_width * 0.06)
        boxes.append((left, 0, max(1, right - left), height))
    return boxes, 0.45


def _similarity(
    query_glyph: Any,
    query_colors: Any,
    query_ratio: float,
    template_glyph: Any,
    template_colors: Any,
    template_ratio: float,
    cv2: Any,
    np: Any,
) -> float:
    # 空图案只应匹配白板。
    if query_ratio < 0.003 or template_ratio < 0.003:
        return 0.98 if query_ratio < 0.003 and template_ratio < 0.003 else 0.05

    correlation = float(
        cv2.matchTemplate(query_glyph, template_glyph, cv2.TM_CCOEFF_NORMED)[0, 0]
    )
    correlation = max(0.0, min(1.0, (correlation + 1.0) / 2.0))

    first = cv2.dilate((query_glyph > 80).astype(np.uint8), np.ones((3, 3), np.uint8))
    second = cv2.dilate((template_glyph > 80).astype(np.uint8), np.ones((3, 3), np.uint8))
    overlap = float(np.logical_and(first, second).sum())
    dice = 2.0 * overlap / max(1.0, float(first.sum() + second.sum()))

    color_distance = float(np.abs(query_colors[:4] - template_colors[:4]).sum())
    color_similarity = max(0.0, 1.0 - color_distance / 1.8)
    return correlation * 0.58 + dice * 0.30 + color_similarity * 0.12


def _classify_tile(
    crop: Any,
    box: tuple[int, int, int, int],
    templates: list[tuple[str, Any, Any, float]],
    cv2: Any,
    np: Any,
) -> TileRecognition:
    query_glyph, query_colors, query_ratio = _ink_descriptor(crop, cv2, np)
    best_by_tile: dict[str, float] = {}
    for tile, glyph, colors, ratio in templates:
        score = _similarity(
            query_glyph,
            query_colors,
            query_ratio,
            glyph,
            colors,
            ratio,
            cv2,
            np,
        )
        best_by_tile[tile] = max(score, best_by_tile.get(tile, 0.0))

    ranking = sorted(best_by_tile.items(), key=lambda item: item[1], reverse=True)
    best_tile, best_score = ranking[0]
    second_score = ranking[1][1]
    # 既考虑绝对匹配分，也考虑第一名与第二名的区分度。
    margin = max(0.0, best_score - second_score)
    confidence = min(1.0, best_score * 0.86 + min(0.14, margin * 1.8))
    return TileRecognition(
        tile=best_tile,
        confidence=confidence,
        alternatives=tuple(ranking[1:4]),
        box=box,
    )


def recognize_hand_image(
    image_path: str | Path,
    *,
    expected_count: int | None = None,
    template_dir: str | Path = DEFAULT_TEMPLATE_DIR,
) -> OCRResult:
    """识别静态图片中的一行 13/14 张正立牌。

    图片最好只保留底部手牌区域。若自动分割不稳定，可通过 ``expected_count``
    明确指定 13 或 14，并把图片裁成紧密的一行。
    """
    if expected_count is not None and expected_count not in SUPPORTED_HAND_SIZES:
        raise OCRError("expected_count 不是闭门或副露后的合法暗牌张数。")

    cv2, np = _cv_modules()
    path = Path(image_path).expanduser().resolve()
    if not path.is_file():
        raise OCRError(f"OCR 图片不存在：{path}")
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise OCRError(f"无法读取 OCR 图片：{path}")
    image = _composite_alpha(image, cv2, np)
    templates = _template_descriptors(Path(template_dir).expanduser().resolve(), cv2, np)

    layouts = _contour_layouts(image, expected_count, cv2, np)
    layouts.extend(_inferred_grid_layouts(image, expected_count, cv2, np))
    # 等分整图只适用于用户已明确张数且图片已经紧密裁成一行的情况。
    if expected_count is not None:
        fallback = _uniform_layout(image, expected_count)
        if fallback:
            layouts.append(fallback)

    if not layouts:
        raise OCRError(
            "未检测到合法张数的一行直立牌。请先把图片裁到只包含自己的横向暗牌，"
            "并使用 --ocr-count 明确暗牌张数。"
        )

    best_result: tuple[float, list[TileRecognition]] | None = None
    for boxes, regularity in layouts:
        recognitions: list[TileRecognition] = []
        for box in boxes:
            x, y, width, height = box
            crop = image[y : y + height, x : x + width]
            recognitions.append(_classify_tile(crop, box, templates, cv2, np))

        average_confidence = sum(item.confidence for item in recognitions) / len(
            recognitions
        )
        bottomness = sum(y + height / 2 for _, y, _, height in boxes) / (
            len(boxes) * image.shape[0]
        )
        count_preference = min(1.0, len(boxes) / 14.0)
        layout_score = (
            average_confidence * 0.68
            + regularity * 0.12
            + bottomness * 0.14
            + count_preference * 0.06
        )
        if best_result is None or layout_score > best_result[0]:
            best_result = layout_score, recognitions

    assert best_result is not None
    return OCRResult(image_path=path, recognitions=tuple(best_result[1]))


def save_debug_image(result: OCRResult, output_path: str | Path) -> Path:
    """保存带检测框、牌名和置信度的调试图。"""
    cv2, _ = _cv_modules()
    image = cv2.imread(str(result.image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise OCRError(f"无法重新读取 OCR 图片：{result.image_path}")

    for item in result.recognitions:
        x, y, width, height = item.box
        color = (40, 190, 40) if item.confidence >= 0.62 else (30, 80, 230)
        cv2.rectangle(image, (x, y), (x + width, y + height), color, 2)
        cv2.putText(
            image,
            f"{item.tile} {item.confidence:.2f}",
            (x, max(18, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OCRError(f"无法写入 OCR 调试图：{path}")
    return path
