# 本程序仅用于立直麻将算法学习研究，禁止用于线上游戏对局辅助。

"""从 pjura/mahjong_souls_tiles 训练轻量、离线的分层 OCR 分类器。

数据集图片右上角带有类别角标。运行时共用的 ``_ink_descriptor`` 会先删除
角落高饱和色角标，避免模型直接读取答案。输出只包含线性边界和归一化风格
原型，不复制原始图片，Windows EXE 无需 PyTorch 或网络连接。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ocr_input import (
    _FAMILY_CLASSIFIER_FEATURE_VERSION,
    _classification_feature,
    _cv_modules,
    _ink_descriptor,
    _read_image,
)


_HONOR_LABELS = {
    "ew": "1z",
    "sw": "2z",
    "ww": "3z",
    "nw": "4z",
    "wd": "5z",
    "gd": "6z",
    "rd": "7z",
}


def _tile_from_dataset_folder(name: str) -> str | None:
    if name in _HONOR_LABELS:
        return _HONOR_LABELS[name]
    if len(name) == 2 and name[0] in "123456789":
        suit = {"n": "m", "p": "p", "b": "s"}.get(name[1])
        if suit is not None:
            return f"{name[0]}{suit}"
    return None


def _sample_rows(dataset_root: Path, cv2: Any, np: Any) -> list[tuple[str, Any]]:
    rows: list[tuple[str, Any]] = []
    for image_path in sorted(dataset_root.rglob("*.png")):
        tile = _tile_from_dataset_folder(image_path.parent.name)
        if tile is None:
            continue
        image = _read_image(image_path, cv2.IMREAD_UNCHANGED, cv2, np)
        if image is None or getattr(image, "size", 0) == 0:
            continue
        glyph, colors, _ = _ink_descriptor(image, cv2, np)
        feature = _classification_feature(glyph, colors, cv2, np)
        rows.append((tile, feature))
    return rows


def _train_family(
    rows: list[tuple[str, Any]], cv2: Any, np: Any
) -> dict[str, Any]:
    labels = sorted({tile for tile, _ in rows})
    label_index = {tile: index for index, tile in enumerate(labels)}
    samples = np.stack([feature for _, feature in rows]).astype(np.float32)
    targets = np.array([label_index[tile] for tile, _ in rows], dtype=np.int32)

    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setC(0.1)
    svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 2000, 1e-6))
    if not svm.train(samples, cv2.ml.ROW_SAMPLE, targets):
        raise RuntimeError("OpenCV SVM 训练失败。")

    support_vectors = svm.getSupportVectors()
    weights: list[Any] = []
    biases: list[float] = []
    positive: list[int] = []
    negative: list[int] = []
    pair_index = 0
    for first in range(len(labels)):
        for second in range(first + 1, len(labels)):
            rho, alpha, support_indices = svm.getDecisionFunction(pair_index)
            indices = support_indices.reshape(-1).astype(np.int32)
            weight = (alpha.reshape(-1) @ support_vectors[indices]).astype(np.float32)
            decision = samples @ weight - rho
            first_mean = float(decision[targets == first].mean())
            second_mean = float(decision[targets == second].mean())
            winner, loser = (
                (first, second) if first_mean >= second_mean else (second, first)
            )
            weights.append(weight)
            biases.append(float(-rho))
            positive.append(winner)
            negative.append(loser)
            pair_index += 1

    normalized = samples.copy()
    norms = np.linalg.norm(normalized, axis=1, keepdims=True)
    normalized /= np.maximum(norms, 1e-9)
    return {
        "labels": np.array(labels),
        "weights": np.stack(weights).astype(np.float32),
        "biases": np.array(biases, dtype=np.float32),
        "positive": np.array(positive, dtype=np.int16),
        "negative": np.array(negative, dtype=np.int16),
        "style_prototypes": normalized.astype(np.float32),
    }


def train(dataset_root: Path, output: Path) -> tuple[int, dict[str, int]]:
    cv2, np = _cv_modules()
    rows = _sample_rows(dataset_root, cv2, np)
    grouped = {
        suit: [(tile, feature) for tile, feature in rows if tile.endswith(suit)]
        for suit in "mpsz"
    }
    expected_classes = {"m": 9, "p": 9, "s": 9, "z": 7}
    for suit, family_rows in grouped.items():
        actual = len({tile for tile, _ in family_rows})
        if actual != expected_classes[suit]:
            raise RuntimeError(
                f"数据不完整：{suit} 家族只有 {actual}/{expected_classes[suit]} 类。"
            )

    payload: dict[str, Any] = {
        "feature_version": np.array(_FAMILY_CLASSIFIER_FEATURE_VERSION),
        "dataset": np.array("pjura/mahjong_souls_tiles"),
        "dataset_license": np.array("Apache-2.0"),
        "sample_count": np.array(len(rows), dtype=np.int32),
    }
    counts: dict[str, int] = {}
    for suit, family_rows in grouped.items():
        counts[suit] = len(family_rows)
        trained = _train_family(family_rows, cv2, np)
        for name, value in trained.items():
            payload[f"{suit}_{name}"] = value

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)
    return len(rows), counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    total, counts = train(args.dataset_root.resolve(), args.output.resolve())
    print(f"已写入 {args.output}：{total} 张，分家族 {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
