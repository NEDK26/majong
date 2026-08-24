# 本程序仅用于立直麻将算法学习研究，禁止用于线上游戏对局辅助。

"""本地麻将牌图片识别。

该模块只读取用户明确提供的静态图片，不截图、不读取游戏进程、不抓包，
也不执行鼠标键盘操作。识别流程为牌面投影分槽、家族模板匹配和离线线性分类。
"""

from __future__ import annotations

import hashlib
import io
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from tile_utils import HandValidationError, TILE_NAMES, tile_name_to_chinese


DEFAULT_TEMPLATE_DIR = Path(__file__).resolve().parent / "assets" / "ocr_templates"
DEFAULT_FAMILY_CLASSIFIER_PATH = (
    Path(__file__).resolve().parent
    / "assets"
    / "ocr_models"
    / "mahjong_soul_family_classifier.npz"
)


def _mahjong_soul_template_dir() -> Path:
    """返回可持久化的雀魂模板目录。

    PyInstaller 单文件程序每次会解压到临时目录，不能把校准结果写在
    ``__file__`` 旁边。源码运行仍保留项目内目录，便于开发和调试。
    """
    override = os.environ.get("MAHJONG_STUDY_DATA_DIR")
    if override:
        return Path(override).expanduser().resolve() / "mahjong_soul"
    if getattr(sys, "frozen", False):
        base = Path(
            os.environ.get("LOCALAPPDATA")
            or os.environ.get("APPDATA")
            or Path.home()
        )
        return base / "MahjongStudyAnalyzer" / "ocr_templates" / "mahjong_soul"
    return Path(__file__).resolve().parent / "local_ocr_templates" / "mahjong_soul"


DEFAULT_MAHJONG_SOUL_TEMPLATE_DIR = _mahjong_soul_template_dir()
SUPPORTED_HAND_SIZES = (1, 2, 4, 5, 7, 8, 10, 11, 13, 14)
AUTO_DETECT_HAND_SIZES = (7, 8, 10, 11, 13, 14)
MINIMUM_RECOGNITION_CONFIDENCE = 0.62
MINIMUM_CLASSIFICATION_MARGIN = 0.03
_MINIMUM_TILE_SURFACE_COVERAGE = 0.20
_WAITING_HAND_SIZES = {1, 4, 7, 10, 13}
_MODEL_TEMPLATE_DECISIVE_MARGIN = 0.08
_MODEL_STYLE_ADVANTAGE = 0.02
_MODEL_OVERRIDE_MARGIN = 0.04
_MODEL_AGREEMENT_MARGIN = 0.03


class OCRError(HandValidationError):
    """图片无法读取、分割或可靠识别时抛出。"""


@dataclass(frozen=True)
class TileRecognition:
    """单张牌的 OCR 结果。"""

    tile: str
    confidence: float
    alternatives: tuple[tuple[str, float], ...]
    box: tuple[int, int, int, int]
    match_score: float = 0.0

    @property
    def classification_margin(self) -> float:
        """第一名与第二名的匹配分差；无候选的测试/外部结果视为已确认。"""
        if self.match_score <= 0.0 or not self.alternatives:
            return 1.0
        return max(0.0, self.match_score - float(self.alternatives[0][1]))

    @property
    def is_reliable(self) -> bool:
        return (
            self.confidence >= MINIMUM_RECOGNITION_CONFIDENCE
            and self.classification_margin >= MINIMUM_CLASSIFICATION_MARGIN
        )


@dataclass(frozen=True)
class OCRResult:
    """整手牌的 OCR 结构化结果。"""

    image_path: Path | None
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
TemplateDescriptor = tuple[str, Any, Any, float, str, Any]
_TEMPLATE_CACHE: dict[str, list[TemplateDescriptor]] = {}
_FAMILY_CLASSIFIER_CACHE: dict[str, dict[str, Any] | None] = {}
_DESCRIPTOR_CACHE_VERSION = "ink-v3-annotation-filter-family-feature-v1"
_FAMILY_CLASSIFIER_FEATURE_VERSION = "glyph-pixels-24x32-color-v1"
_CANONICAL_STEM_BY_TILE = {
    tile: Path(filename).stem
    for filename, tile in _TEMPLATE_FILES.items()
    if "-Dora" not in filename
}
_TILE_BY_CHINESE_NAME = {tile_name_to_chinese(tile): tile for tile in TILE_NAMES}
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


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


def _read_image(path: Path, flags: int, cv2: Any, np: Any) -> Any:
    """通过字节读取图片，绕过 Windows OpenCV 对中文路径的限制。"""
    try:
        data = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(data, flags)


def _write_image(path: Path, image: Any, cv2: Any) -> bool:
    """通过 pathlib 写入编码结果，可靠支持 Windows 中文路径。"""
    extension = path.suffix.lower() or ".png"
    success, encoded = cv2.imencode(extension, image)
    if not success:
        return False
    try:
        path.write_bytes(encoded.tobytes())
    except OSError:
        return False
    return True


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
    # 训练/诊断素材常在牌角绘制彩色类别或序号。只删除角落里的高饱和色像素，
    # 不整块挖掉牌角；后者会误伤九万、索子等真实牌面笔画。
    corner_height = max(6, round(mask.shape[0] * 0.28))
    corner_width = max(6, round(mask.shape[1] * 0.38))
    hsv = cv2.cvtColor(inner, cv2.COLOR_BGR2HSV)
    annotation_color = (hsv[:, :, 1] >= 90) & (hsv[:, :, 2] >= 90)
    annotation_band = np.zeros_like(mask, dtype=bool)
    annotation_band[:corner_height, :corner_width] = True
    annotation_band[:corner_height, -corner_width:] = True
    mask[annotation_band & annotation_color] = 0
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


def _classification_feature(
    glyph: Any, color_vector: Any, cv2: Any, np: Any, *, normalize: bool = False
) -> Any:
    """生成紧凑的牌面分类特征；训练脚本与运行时必须共用这一实现。"""
    pixels = cv2.resize(glyph, (24, 32), interpolation=cv2.INTER_AREA)
    feature = np.concatenate(
        [pixels.astype(np.float32).reshape(-1) / 255.0, color_vector * 3.0]
    ).astype(np.float32)
    if normalize:
        norm = float(np.linalg.norm(feature))
        if norm > 1e-9:
            feature /= norm
    return feature


def _family_classifier(path: Path, np: Any) -> dict[str, Any] | None:
    """加载离线、按花色分层的轻量线性分类器。"""
    cache_key = str(path.resolve())
    if cache_key in _FAMILY_CLASSIFIER_CACHE:
        return _FAMILY_CLASSIFIER_CACHE[cache_key]
    if not path.is_file():
        _FAMILY_CLASSIFIER_CACHE[cache_key] = None
        return None

    try:
        with np.load(path, allow_pickle=False) as data:
            if str(data["feature_version"].item()) != _FAMILY_CLASSIFIER_FEATURE_VERSION:
                _FAMILY_CLASSIFIER_CACHE[cache_key] = None
                return None
            model: dict[str, Any] = {}
            for suit in "mpsz":
                model[suit] = {
                    "labels": tuple(str(value) for value in data[f"{suit}_labels"]),
                    "weights": data[f"{suit}_weights"].copy(),
                    "biases": data[f"{suit}_biases"].copy(),
                    "positive": data[f"{suit}_positive"].copy(),
                    "negative": data[f"{suit}_negative"].copy(),
                    "style_prototypes": data[f"{suit}_style_prototypes"].copy(),
                }
    except (OSError, KeyError, ValueError, EOFError):
        _FAMILY_CLASSIFIER_CACHE[cache_key] = None
        return None
    _FAMILY_CLASSIFIER_CACHE[cache_key] = model
    return model


def _family_model_ranking(
    feature: Any, suit: str, model: dict[str, Any], np: Any
) -> tuple[list[str], float]:
    """用一对一线性判别器投票，并返回该画面属于训练风格的相似度。"""
    family = model[suit]
    decision_values = feature @ family["weights"].T + family["biases"]
    votes = np.zeros(len(family["labels"]), dtype=np.int16)
    strengths = np.zeros(len(family["labels"]), dtype=np.float32)
    for index, decision in enumerate(decision_values):
        winner = (
            int(family["positive"][index])
            if decision >= 0.0
            else int(family["negative"][index])
        )
        votes[winner] += 1
        strengths[winner] += min(4.0, abs(float(decision)))
    order = sorted(
        range(len(family["labels"])),
        key=lambda index: (int(votes[index]), float(strengths[index])),
        reverse=True,
    )

    normalized = feature.copy()
    norm = float(np.linalg.norm(normalized))
    if norm > 1e-9:
        normalized /= norm
    style_score = float((family["style_prototypes"] @ normalized).max())
    return [family["labels"][index] for index in order], style_score


def _template_fingerprint(template_dir: Path) -> str:
    digest = hashlib.sha256(_DESCRIPTOR_CACHE_VERSION.encode("ascii"))
    for path in sorted(template_dir.glob("*")):
        if not path.is_file() or path.suffix.lower() not in _IMAGE_EXTENSIONS:
            continue
        stat = path.stat()
        digest.update(path.name.encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
    return digest.hexdigest()


def _load_descriptor_cache(
    template_dir: Path, fingerprint: str, np: Any
) -> list[TemplateDescriptor] | None:
    try:
        raw = (template_dir / ".ocr_features.npz").read_bytes()
        with np.load(io.BytesIO(raw), allow_pickle=False) as data:
            if str(data["fingerprint"].item()) != fingerprint:
                return None
            return [
                (
                    str(tile),
                    glyph,
                    color,
                    float(ratio),
                    str(orientation),
                    model_feature,
                )
                for tile, glyph, color, ratio, orientation, model_feature in zip(
                    data["tiles"].tolist(),
                    data["glyphs"],
                    data["colors"],
                    data["ratios"],
                    data["orientations"].tolist(),
                    data["model_features"],
                    strict=True,
                )
            ]
    except (OSError, KeyError, ValueError, EOFError):
        return None


def _save_descriptor_cache(
    template_dir: Path,
    fingerprint: str,
    descriptors: list[TemplateDescriptor],
    np: Any,
) -> None:
    if not descriptors:
        return
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        fingerprint=np.array(fingerprint),
        tiles=np.array([item[0] for item in descriptors]),
        glyphs=np.stack([item[1] for item in descriptors]),
        colors=np.stack([item[2] for item in descriptors]),
        ratios=np.array([item[3] for item in descriptors], dtype=np.float32),
        orientations=np.array([item[4] for item in descriptors]),
        model_features=np.stack([item[5] for item in descriptors]),
    )
    try:
        (template_dir / ".ocr_features.npz").write_bytes(buffer.getvalue())
    except OSError:
        pass


def _template_descriptors(
    template_dir: Path,
    cv2: Any,
    np: Any,
    progress: Callable[[str, int, int], None] | None = None,
) -> list[TemplateDescriptor]:
    if not template_dir.is_dir():
        raise OCRError(f"OCR 模板目录不存在：{template_dir}")

    cache_key = str(template_dir.resolve())
    if cache_key in _TEMPLATE_CACHE:
        return _TEMPLATE_CACHE[cache_key]

    fingerprint = _template_fingerprint(template_dir)
    cached = _load_descriptor_cache(template_dir, fingerprint, np)
    if cached is not None:
        _TEMPLATE_CACHE[cache_key] = cached
        return cached

    descriptors: list[TemplateDescriptor] = []
    missing: list[str] = []
    prepared_tiles: set[str] = set()
    seen_samples_by_tile: dict[str, set[str]] = {}
    for filename, tile in _TEMPLATE_FILES.items():
        path = template_dir / filename
        if not path.is_file():
            if "-Dora" not in filename:
                missing.append(filename)
            continue
        sample_paths = [path]
        sample_paths.extend(
            candidate
            for candidate in sorted(template_dir.glob(f"{path.stem}_样本*"))
            if candidate.suffix.lower() in _IMAGE_EXTENSIONS
        )
        for sample_path in sample_paths:
            image = _read_image(sample_path, cv2.IMREAD_UNCHANGED, cv2, np)
            if image is None:
                raise OCRError(f"无法读取 OCR 模板：{sample_path}")
            sample_digest = hashlib.sha256()
            sample_digest.update(str(image.shape).encode("ascii"))
            sample_digest.update(str(image.dtype).encode("ascii"))
            sample_digest.update(image.tobytes())
            digest = sample_digest.hexdigest()
            seen_samples = seen_samples_by_tile.setdefault(tile, set())
            if digest in seen_samples:
                continue
            seen_samples.add(digest)
            variants = (
                image,
                cv2.rotate(image, cv2.ROTATE_180),
                cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE),
                cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE),
            )
            for variant in variants:
                glyph, colors, ratio = _ink_descriptor(variant, cv2, np)
                orientation = (
                    "竖向" if variant.shape[0] >= variant.shape[1] else "横向"
                )
                model_feature = _classification_feature(
                    glyph, colors, cv2, np, normalize=True
                )
                descriptors.append(
                    (tile, glyph, colors, ratio, orientation, model_feature)
                )

        if "-Dora" not in filename and tile not in prepared_tiles:
            prepared_tiles.add(tile)
            if progress is not None:
                progress(tile, len(prepared_tiles), 34)

    if missing:
        raise OCRError("OCR 模板缺失：" + "、".join(missing))
    _TEMPLATE_CACHE[cache_key] = descriptors
    _save_descriptor_cache(template_dir, fingerprint, descriptors, np)
    return descriptors


def prepare_ocr_templates(
    template_dir: str | Path = DEFAULT_MAHJONG_SOUL_TEMPLATE_DIR,
    progress: Callable[[str, int, int], None] | None = None,
) -> tuple[int, int]:
    """初始化并持久缓存34种牌的四方向特征。"""
    cv2, np = _cv_modules()
    descriptors = _template_descriptors(
        Path(template_dir).expanduser().resolve(), cv2, np, progress
    )
    return len({item[0] for item in descriptors}), len(descriptors)


def _tile_from_sample_filename(path: Path) -> str | None:
    """支持“八筒.png / 八筒_2.png”和内部标准文件名。"""
    stem = path.stem
    for chinese_name, tile in _TILE_BY_CHINESE_NAME.items():
        if stem == chinese_name or stem.startswith(f"{chinese_name}_"):
            return tile
    normalized = stem.split("_", 1)[0]
    for filename, tile in _TEMPLATE_FILES.items():
        if normalized.lower() == Path(filename).stem.lower():
            return tile
    return None


def import_labeled_template_folder(
    source_dir: str | Path,
    output_dir: str | Path = DEFAULT_MAHJONG_SOUL_TEMPLATE_DIR,
) -> tuple[Path, int, int]:
    """导入用户按中文牌名标注的实际游戏单牌样本。"""
    cv2, np = _cv_modules()
    source = Path(source_dir).expanduser().resolve()
    if not source.is_dir():
        raise OCRError(f"逐张样本文件夹不存在：{source}")
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    imported = 0
    covered_tiles: set[str] = set()
    counters: dict[str, int] = {}
    for sample_path in sorted(source.rglob("*")):
        if not sample_path.is_file() or sample_path.suffix.lower() not in _IMAGE_EXTENSIONS:
            continue
        tile = _tile_from_sample_filename(sample_path)
        if tile is None:
            continue
        image = _read_image(sample_path, cv2.IMREAD_UNCHANGED, cv2, np)
        if image is None or getattr(image, "size", 0) == 0:
            continue
        stem = _CANONICAL_STEM_BY_TILE[tile]
        base_target = destination / f"{stem}.png"
        existing_paths = [base_target]
        existing_paths.extend(
            candidate
            for candidate in sorted(destination.glob(f"{stem}_样本*"))
            if candidate.suffix.lower() in _IMAGE_EXTENSIONS
        )
        duplicate = False
        for existing_path in existing_paths:
            if not existing_path.is_file():
                continue
            existing_image = _read_image(
                existing_path, cv2.IMREAD_UNCHANGED, cv2, np
            )
            if (
                existing_image is not None
                and existing_image.shape == image.shape
                and np.array_equal(existing_image, image)
            ):
                duplicate = True
                break
        covered_tiles.add(tile)
        if duplicate:
            continue
        if not base_target.exists():
            target = base_target
        else:
            counters[stem] = counters.get(stem, 0) + 1
            sample_number = counters[stem]
            target = destination / f"{stem}_样本{sample_number:03d}.png"
            while target.exists():
                sample_number += 1
                target = destination / f"{stem}_样本{sample_number:03d}.png"
            counters[stem] = sample_number
        if not _write_image(target, image, cv2):
            raise OCRError(f"无法保存逐张样本：{target}")
        imported += 1

    if not covered_tiles:
        raise OCRError(
            "没有找到已标注样本。请使用“八筒.png、六索_2.png、北.png”"
            "这样的中文文件名。"
        )
    _TEMPLATE_CACHE.pop(str(destination), None)
    return destination, imported, len(covered_tiles)


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
            is_vertical = 0.46 <= aspect <= 0.92
            is_horizontal = 1.08 <= aspect <= 2.20
            if box_height > height * 0.98 or not (is_vertical or is_horizontal):
                continue
            minimum_height = (
                max(28, height * 0.055)
                if is_vertical
                else max(18, height * 0.035)
            )
            if box_height < minimum_height:
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
    image = _read_image(source, cv2.IMREAD_UNCHANGED, cv2, np)
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
            if not _write_image(destination / filename, crop, cv2):
                raise OCRError(f"无法写入 OCR 模板：{destination / filename}")
    _TEMPLATE_CACHE.pop(str(destination), None)
    return destination


def _layout_regularity(boxes: list[tuple[int, int, int, int]], np: Any) -> float:
    widths = np.array([box[2] for box in boxes], dtype=np.float32)
    heights = np.array([box[3] for box in boxes], dtype=np.float32)
    centers_y = np.array([box[1] + box[3] / 2 for box in boxes], dtype=np.float32)
    lefts = np.array([box[0] for box in boxes], dtype=np.float32)
    steps = np.diff(lefts)
    # 雀魂把刚摸入的最右一张与原手牌留出小间隔；这仍属于同一行，不能因为
    # 唯一的末尾间隔而把完整 14 张候选降到短子窗口之后。
    if len(steps) >= 2:
        median_step = float(np.median(steps[:-1]))
        if median_step * 1.12 <= float(steps[-1]) <= median_step * 1.65:
            steps = steps[:-1]

    penalties = [
        float(widths.std() / max(1.0, widths.mean())),
        float(heights.std() / max(1.0, heights.mean())),
        float(centers_y.std() / max(1.0, heights.mean())),
    ]
    if len(steps):
        penalties.append(float(steps.std() / max(1.0, steps.mean())))
    return max(0.0, 1.0 - sum(penalties) / len(penalties) * 2.5)


def _tile_surface_coverage(
    image: Any, box: tuple[int, int, int, int]
) -> float:
    """估算候选框内真实亮色牌面的占比，排除深色桌布等空白区域。"""
    x, y, width, height = box
    image_height, image_width = image.shape[:2]
    left = max(0, int(x))
    top = max(0, int(y))
    right = min(image_width, int(x + width))
    bottom = min(image_height, int(y + height))
    if right <= left or bottom <= top:
        return 0.0
    crop = image[top:bottom, left:right]
    brightest = crop.max(axis=2)
    darkest = crop.min(axis=2)
    light_neutral = (brightest > 115) & ((brightest - darkest) < 70)
    return float(light_neutral.mean())


def _neutral_tile_mask(image: Any, cv2: Any, np: Any) -> Any:
    """提取牌面本身，不把牌框轮廓是否粘连当作分牌依据。"""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    light, channel_a, channel_b = cv2.split(lab)
    return (
        (light > 145)
        & (np.abs(channel_a.astype(np.int16) - 128) < 22)
        & (np.abs(channel_b.astype(np.int16) - 128) < 28)
    )


def _projection_tile_boxes(
    image: Any, cv2: Any, np: Any
) -> list[tuple[int, int, int, int]]:
    """按白色牌面的水平投影直接切出每张牌。

    雀魂相邻牌的外框经常在二值化后粘成一个大轮廓，但两张牌之间仍有稳定的
    深色竖缝。水平投影保留这条缝，因此它才是牌数判断的主信号；轮廓只用于
    参考图校准，不再参与实战手牌张数竞争。
    """
    height, width = image.shape[:2]
    mask = _neutral_tile_mask(image, cv2, np)
    contour_boxes = _detected_boxes(image, cv2, np)
    vertical_contours = [
        box
        for box in contour_boxes
        if 0.45 <= box[2] / max(1, box[3]) <= 0.95
    ]
    if not vertical_contours:
        row_top, row_bottom = 0, height
        reference_width = max(12.0, height * 0.55)
    else:
        tallest = max(box[3] for box in vertical_contours)
        row_contours = [
            box for box in vertical_contours if box[3] >= tallest * 0.76
        ]
        row_top = min(box[1] for box in row_contours)
        row_bottom = max(box[1] + box[3] for box in row_contours)
        reference_width = float(np.median([box[2] for box in row_contours]))

    row_mask = mask[row_top:row_bottom]
    minimum_column_pixels = max(4, round((row_bottom - row_top) * 0.10))
    active_columns = row_mask.sum(axis=0) >= minimum_column_pixels
    minimum_run_width = max(6, round(reference_width * 0.16))

    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, active in enumerate(active_columns.tolist() + [False]):
        if active and start is None:
            start = index
        elif not active and start is not None:
            if index - start >= minimum_run_width:
                runs.append((start, index))
            start = None

    boxes: list[tuple[int, int, int, int]] = []
    for left, right in runs:
        column_mask = row_mask[:, left:right]
        minimum_row_pixels = max(3, round((right - left) * 0.10))
        active_rows = np.where(column_mask.sum(axis=1) >= minimum_row_pixels)[0]
        if not len(active_rows):
            continue
        top = row_top + int(active_rows.min())
        bottom = row_top + int(active_rows.max()) + 1
        box_width = right - left
        box_height = bottom - top
        aspect = box_width / max(1, box_height)
        if box_height < max(24, height * 0.05) or not 0.32 <= aspect <= 2.30:
            continue
        # 白色投影从牌面内侧开始；向外补一像素可保留完整边缘且不会吃到邻牌。
        expanded_left = max(0, left - 1)
        expanded_right = min(width, right + 1)
        projected = (expanded_left, top, expanded_right - expanded_left, box_height)
        matching_contour = next(
            (
                contour
                for contour in vertical_contours
                if abs(contour[0] - projected[0]) <= max(3, reference_width * 0.08)
                and 0.72 <= contour[2] / max(1, projected[2]) <= 1.28
            ),
            None,
        )
        boxes.append(matching_contour or projected)
    return boxes


def _concealed_hand_layout(
    image: Any,
    expected_count: int | None,
    cv2: Any,
    np: Any,
) -> list[tuple[int, int, int, int]] | None:
    """按麻将桌面语义确定暗牌：左起连续牌列，加可选的末尾摸入牌。

    这里不枚举 7/8/10/11/13/14 后再按识别分数竞猜。暗牌必须从最左一张
    开始；普通相邻间距组成基础手牌，只有在 3n+1 张之后允许一次较大的间隔，
    间隔后的单张就是摸入牌并立即结束暗牌区域。右侧更小、横置或更远的牌属于
    副露，不会进入候选舍牌。
    """
    projected = _projection_tile_boxes(image, cv2, np)
    if not projected:
        return None

    # 暗牌是画面里尺寸最大的同一行直立牌；右侧副露通常更小或横置。
    vertical = [box for box in projected if box[2] / max(1, box[3]) <= 0.95]
    if not vertical:
        return None
    tallest = max(box[3] for box in vertical)
    size_candidates = [box for box in vertical if box[3] >= tallest * 0.78]
    widths = np.array([box[2] for box in size_candidates], dtype=np.float32)
    # 索子图案会把白色牌面投影切成多个窄片；用较宽分位估计真实牌宽，
    # 再以完整牌面作为网格锚点，窄片只作为槽位存在性的证据。
    width_floor = float(np.percentile(widths, 65))
    anchors = [box for box in size_candidates if box[2] >= width_floor * 0.82]
    anchors.sort(key=lambda item: item[0])
    if not anchors:
        return None

    median_width = float(np.median([box[2] for box in anchors]))
    median_height = float(np.median([box[3] for box in anchors]))
    median_y = float(np.median([box[1] for box in anchors]))

    step_candidates: list[float] = []
    for difference in np.diff([box[0] for box in anchors]).astype(np.float32):
        multiples = max(1, round(float(difference) / max(1.0, median_width)))
        if multiples <= 8:
            candidate = float(difference) / multiples
            if median_width * 0.88 <= candidate <= median_width * 1.35:
                step_candidates.append(candidate)
    if not step_candidates:
        return None
    regular_step = float(np.median(step_candidates))

    origin = float(anchors[0][0])
    template_box = (
        round(origin),
        round(median_y),
        round(median_width),
        round(median_height),
    )
    while origin - regular_step >= 0:
        previous = (
            round(origin - regular_step),
            template_box[1],
            template_box[2],
            template_box[3],
        )
        if _tile_surface_coverage(image, previous) < _MINIMUM_TILE_SURFACE_COVERAGE:
            break
        origin -= regular_step

    selected: list[tuple[int, int, int, int]] = []
    used_anchors: set[int] = set()
    for index in range(14):
        target_x = origin + index * regular_step
        nearest_index = min(
            range(len(anchors)),
            key=lambda anchor_index: abs(anchors[anchor_index][0] - target_x),
        )
        nearest = anchors[nearest_index]
        distance = abs(nearest[0] - target_x)
        maximum_distance = regular_step * 0.24
        # 3n+1 张之后允许末张以摸牌间隔出现；其余槽位必须落在普通网格上。
        if len(selected) in _WAITING_HAND_SIZES:
            maximum_distance = regular_step * 0.38

        if nearest_index not in used_anchors and distance <= maximum_distance:
            box = nearest
            used_anchors.add(nearest_index)
        else:
            box = (
                round(target_x),
                round(median_y),
                round(median_width),
                round(median_height),
            )
            if _tile_surface_coverage(image, box) < _MINIMUM_TILE_SURFACE_COVERAGE:
                break

        if selected:
            actual_step = box[0] - selected[-1][0]
            if actual_step > regular_step * 1.18:
                if len(selected) not in _WAITING_HAND_SIZES:
                    break
                selected.append(box)
                break
        selected.append(box)

    if expected_count is not None:
        if len(selected) >= expected_count:
            return selected[:expected_count]
        return None

    if len(selected) not in AUTO_DETECT_HAND_SIZES:
        return None
    return selected


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
    templates: list[TemplateDescriptor],
    cv2: Any,
    np: Any,
) -> TileRecognition:
    query_glyph, query_colors, query_ratio = _ink_descriptor(crop, cv2, np)
    query_orientation = "竖向" if crop.shape[0] >= crop.shape[1] else "横向"
    best_by_tile: dict[str, float] = {}
    for tile, glyph, colors, ratio, orientation, _ in templates:
        if orientation != query_orientation:
            continue
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
    selected_ranking = ranking

    # 先由全局模板确定万/筒/索/字牌家族，再只在同一家族内辨别数字或字牌。
    # 这避免所有万子共有的“萬”吞掉上方数字特征，也不会让 5p/5s 跨花色乱跳。
    # 只有当前模板在同家族内含糊、且画面与训练风格更接近时，模型才覆盖模板。
    classifier = _family_classifier(DEFAULT_FAMILY_CLASSIFIER_PATH, np)
    if classifier is not None:
        suit = best_tile[-1]
        family_ranking = [item for item in ranking if item[0].endswith(suit)]
        if len(family_ranking) >= 2:
            family_margin = family_ranking[0][1] - family_ranking[1][1]
            query_feature = _classification_feature(
                query_glyph, query_colors, cv2, np
            )
            model_ranking, model_style_score = _family_model_ranking(
                query_feature, suit, classifier, np
            )
            model_tile = model_ranking[0]
            if family_margin < _MODEL_TEMPLATE_DECISIVE_MARGIN:
                if model_tile == best_tile:
                    boosted_score = min(
                        1.0, best_score + _MODEL_AGREEMENT_MARGIN
                    )
                    selected_ranking = [(best_tile, boosted_score), *ranking[1:]]
                else:
                    normalized_query = query_feature / max(
                        1e-9, float(np.linalg.norm(query_feature))
                    )
                    active_style_score = max(
                        float(template_feature @ normalized_query)
                        for tile, _, _, _, orientation, template_feature in templates
                        if tile.endswith(suit) and orientation == query_orientation
                    )
                    if (
                        model_style_score
                        >= active_style_score + _MODEL_STYLE_ADVANTAGE
                    ):
                        boosted_score = min(
                            1.0, best_score + _MODEL_OVERRIDE_MARGIN
                        )
                        selected_ranking = [
                            (model_tile, boosted_score),
                            *[item for item in ranking if item[0] != model_tile],
                        ]
                best_tile, best_score = selected_ranking[0]

    second_score = selected_ranking[1][1]
    # 既考虑绝对匹配分，也考虑第一名与第二名的区分度。
    margin = max(0.0, best_score - second_score)
    confidence = min(1.0, best_score * 0.86 + min(0.14, margin * 1.8))
    return TileRecognition(
        tile=best_tile,
        confidence=confidence,
        alternatives=tuple(selected_ranking[1:9]),
        box=box,
        match_score=best_score,
    )


def _recognize_hand_array(
    image: Any,
    *,
    expected_count: int | None = None,
    template_dir: str | Path = DEFAULT_TEMPLATE_DIR,
    image_path: Path | None = None,
) -> OCRResult:
    if expected_count is not None and expected_count not in SUPPORTED_HAND_SIZES:
        raise OCRError("expected_count 不是闭门或副露后的合法暗牌张数。")

    cv2, np = _cv_modules()
    image = _composite_alpha(image, cv2, np)
    templates = _template_descriptors(Path(template_dir).expanduser().resolve(), cv2, np)

    boxes = _concealed_hand_layout(image, expected_count, cv2, np)
    # 横置单牌或已经紧密裁切、没有可用白色投影的静态图片，仅在用户明确
    # 指定张数时使用整图等分；实时自动模式绝不靠等分猜张数。
    if boxes is None and expected_count is not None:
        fallback = _uniform_layout(image, expected_count)
        boxes = fallback[0] if fallback else None

    if not boxes:
        raise OCRError(
            "未检测到合法张数的一行直立牌。请先把图片裁到只包含自己的横向暗牌，"
            "并使用 --ocr-count 明确暗牌张数。"
        )

    recognitions: list[TileRecognition] = []
    for box in boxes:
        x, y, width, height = box
        crop = image[y : y + height, x : x + width]
        recognitions.append(_classify_tile(crop, box, templates, cv2, np))
    return OCRResult(image_path=image_path, recognitions=tuple(recognitions))


def recognize_hand_frame(
    image: Any,
    *,
    expected_count: int | None = None,
    template_dir: str | Path = DEFAULT_TEMPLATE_DIR,
) -> OCRResult:
    """直接识别 OpenCV/NumPy 图像，用于本地实时屏幕捕获。"""
    return _recognize_hand_array(
        image,
        expected_count=expected_count,
        template_dir=template_dir,
    )


def recognize_hand_image(
    image_path: str | Path,
    *,
    expected_count: int | None = None,
    template_dir: str | Path = DEFAULT_TEMPLATE_DIR,
) -> OCRResult:
    """识别静态图片中的一行正立牌。

    图片最好只保留底部手牌区域。若自动分割不稳定，可通过 ``expected_count``
    明确指定暗牌张数，并把图片裁成紧密的一行。
    """
    cv2, np = _cv_modules()
    path = Path(image_path).expanduser().resolve()
    if not path.is_file():
        raise OCRError(f"OCR 图片不存在：{path}")
    image = _read_image(path, cv2.IMREAD_UNCHANGED, cv2, np)
    if image is None:
        raise OCRError(f"无法读取 OCR 图片：{path}")
    return _recognize_hand_array(
        image,
        expected_count=expected_count,
        template_dir=template_dir,
        image_path=path,
    )


def save_debug_image(result: OCRResult, output_path: str | Path) -> Path:
    """保存带检测框、牌名和置信度的调试图。"""
    if result.image_path is None:
        raise OCRError("实时帧没有源文件路径，无法使用 save_debug_image。")
    cv2, np = _cv_modules()
    image = _read_image(result.image_path, cv2.IMREAD_COLOR, cv2, np)
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
    if not _write_image(path, image, cv2):
        raise OCRError(f"无法写入 OCR 调试图：{path}")
    return path
