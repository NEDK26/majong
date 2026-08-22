# 本程序仅用于立直麻将算法学习研究，禁止用于线上游戏对局辅助。

from __future__ import annotations

import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from ocr_input import (
    DEFAULT_TEMPLATE_DIR,
    _mahjong_soul_template_dir,
    build_mahjong_soul_templates,
    import_labeled_template_folder,
    recognize_hand_image,
)


class OCRTests(unittest.TestCase):
    @staticmethod
    def _load_tile(cv2, np, filename: str, width: int, height: int):
        tile = cv2.imread(str(DEFAULT_TEMPLATE_DIR / filename), cv2.IMREAD_UNCHANGED)
        if tile is None:
            raise AssertionError(f"Missing OCR template: {filename}")
        alpha = tile[:, :, 3:4].astype(np.float32) / 255.0
        foreground = tile[:, :, :3].astype(np.float32)
        tile = (foreground * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)
        return cv2.resize(tile, (width, height), interpolation=cv2.INTER_AREA)

    def test_recognizes_synthetic_horizontal_hand(self) -> None:
        try:
            import cv2
            import numpy as np
        except ImportError as exc:  # pragma: no cover - 安装 requirements 后不会触发
            self.skipTest(f"OCR dependencies are not installed: {exc}")

        filenames = [
            "Man1.png",
            "Man2.png",
            "Man3.png",
            "Pin1.png",
            "Pin2.png",
            "Pin3.png",
            "Sou1.png",
            "Sou2.png",
            "Sou3.png",
            "Sou7.png",
            "Sou8.png",
            "Sou9.png",
            "Ton.png",
        ]
        expected = [
            "1m",
            "2m",
            "3m",
            "1p",
            "2p",
            "3p",
            "1s",
            "2s",
            "3s",
            "7s",
            "8s",
            "9s",
            "1z",
        ]

        tile_width, tile_height, gap = 90, 120, 8
        canvas = np.full(
            (tile_height + 28, len(filenames) * (tile_width + gap) + gap, 3),
            (122, 92, 55),
            dtype=np.uint8,
        )
        for index, filename in enumerate(filenames):
            tile = self._load_tile(cv2, np, filename, tile_width, tile_height)
            left = gap + index * (tile_width + gap)
            canvas[10 : 10 + tile_height, left : left + tile_width] = tile

        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "synthetic-hand.png"
            self.assertTrue(cv2.imwrite(str(image_path), canvas))
            result = recognize_hand_image(image_path, expected_count=13)

        self.assertEqual(result.tiles, expected)
        self.assertGreaterEqual(result.minimum_confidence, 0.62)

    def test_frozen_app_uses_persistent_windows_data_directory(self) -> None:
        with (
            patch.dict("os.environ", {"LOCALAPPDATA": "C:/Users/Test/AppData/Local"}, clear=False),
            patch("ocr_input.sys.frozen", True, create=True),
        ):
            result = _mahjong_soul_template_dir()

        self.assertEqual(
            result.as_posix(),
            "C:/Users/Test/AppData/Local/MahjongStudyAnalyzer/ocr_templates/mahjong_soul",
        )

    def test_builds_34_templates_from_reference_layout(self) -> None:
        try:
            import cv2
            import numpy as np
        except ImportError as exc:  # pragma: no cover
            self.skipTest(f"OCR dependencies are not installed: {exc}")

        rows = [
            [*[f"Man{number}.png" for number in range(1, 10)], "Ton.png", "Nan.png", "Shaa.png", "Pei.png"],
            [*[f"Pin{number}.png" for number in range(1, 10)], "Haku.png", "Hatsu.png", "Chun.png"],
            [f"Sou{number}.png" for number in range(1, 10)],
        ]
        tile_width, tile_height, gap = 78, 116, 8
        canvas = np.full((420, 1250, 3), (105, 110, 70), dtype=np.uint8)
        for row_index, filenames in enumerate(rows):
            top = 18 + row_index * 135
            left_start = 20 + row_index * 35
            for column, filename in enumerate(filenames):
                tile = self._load_tile(cv2, np, filename, tile_width, tile_height)
                left = left_start + column * (tile_width + gap)
                canvas[top : top + tile_height, left : left + tile_width] = tile

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "reference.png"
            output = Path(directory) / "templates"
            self.assertTrue(cv2.imwrite(str(source), canvas))
            result_dir = build_mahjong_soul_templates(source, output)

            generated = sorted(path.name for path in result_dir.glob("*.png"))
            self.assertEqual(len(generated), 34)
            self.assertIn("Man1.png", generated)
            self.assertIn("Chun.png", generated)
            self.assertIn("Sou9.png", generated)

    def test_imports_chinese_named_single_tile_samples(self) -> None:
        try:
            import cv2
            import numpy as np
        except ImportError as exc:  # pragma: no cover
            self.skipTest(f"OCR dependencies are not installed: {exc}")

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "samples"
            output = Path(directory) / "templates"
            source.mkdir()
            sample = self._load_tile(cv2, np, "Pin8.png", 60, 88)
            self.assertTrue(cv2.imwrite(str(source / "八筒.png"), sample))
            self.assertTrue(cv2.imwrite(str(source / "八筒_2.png"), sample))

            result_dir, sample_count, tile_count = import_labeled_template_folder(
                source, output
            )

            self.assertEqual(sample_count, 2)
            self.assertEqual(tile_count, 1)
            self.assertEqual(len(list(result_dir.glob("Pin8_样本*.png"))), 2)

    def test_auto_detects_eight_concealed_tiles_below_smaller_open_melds(self) -> None:
        try:
            import cv2
            import numpy as np
        except ImportError as exc:  # pragma: no cover
            self.skipTest(f"OCR dependencies are not installed: {exc}")

        concealed_files = [
            "Pin6.png",
            "Pin7.png",
            "Sou4.png",
            "Sou4.png",
            "Sou6.png",
            "Sou7.png",
            "Sou8.png",
            "Pei.png",
        ]
        canvas = np.full((468, 1000, 3), (80, 105, 100), dtype=np.uint8)
        for index, filename in enumerate(concealed_files):
            tile = self._load_tile(cv2, np, filename, 40, 59)
            left = 218 + index * 41
            canvas[398:457, left : left + 40] = tile
            cv2.rectangle(canvas, (left, 398), (left + 39, 456), (35, 45, 50), 1)

        # 右侧副露比暗牌小，不应被当作候选舍牌。
        exposed_files = [
            "Man1.png",
            "Man2.png",
            "Man3.png",
            "Pin4.png",
            "Pin5.png",
            "Pin6.png",
        ]
        for index, filename in enumerate(exposed_files):
            tile = self._load_tile(cv2, np, filename, 25, 38)
            left = 692 + index * 27
            canvas[405:443, left : left + 25] = tile
            cv2.rectangle(canvas, (left, 405), (left + 24, 442), (35, 45, 50), 1)

        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "open-hand.png"
            self.assertTrue(cv2.imwrite(str(image_path), canvas))
            result = recognize_hand_image(image_path)

        self.assertEqual(
            result.tiles,
            ["6p", "7p", "4s", "4s", "6s", "7s", "8s", "4z"],
        )


if __name__ == "__main__":
    unittest.main()
