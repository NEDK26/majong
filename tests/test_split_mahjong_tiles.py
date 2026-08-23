# 本程序仅用于立直麻将算法学习研究，禁止用于线上游戏对局辅助。

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ocr_input import DEFAULT_TEMPLATE_DIR
from split_mahjong_tiles import split_mahjong_tiles


class SplitMahjongTilesTests(unittest.TestCase):
    @staticmethod
    def _load_tile(cv2, np, filename: str, width: int, height: int):
        tile = cv2.imread(str(DEFAULT_TEMPLATE_DIR / filename), cv2.IMREAD_UNCHANGED)
        if tile is None:
            raise AssertionError(f"Missing OCR template: {filename}")
        alpha = tile[:, :, 3:4].astype(np.float32) / 255.0
        foreground = tile[:, :, :3].astype(np.float32)
        tile = (foreground * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)
        return cv2.resize(tile, (width, height), interpolation=cv2.INTER_AREA)

    def test_splits_34_tiles_with_chinese_filenames(self) -> None:
        try:
            import cv2
            import numpy as np
        except ImportError as exc:  # pragma: no cover
            self.skipTest(f"OCR dependencies are not installed: {exc}")

        rows = [
            [
                *[f"Man{number}.png" for number in range(1, 10)],
                "Ton.png",
                "Nan.png",
                "Shaa.png",
                "Pei.png",
            ],
            [
                *[f"Pin{number}.png" for number in range(1, 10)],
                "Haku.png",
                "Hatsu.png",
                "Chun.png",
            ],
            [f"Sou{number}.png" for number in range(1, 10)],
        ]
        tile_width, tile_height, gap = 78, 116, 8
        canvas = np.full((420, 1250, 3), (105, 110, 70), dtype=np.uint8)
        for row_index, filenames in enumerate(rows):
            top = 18 + row_index * 135
            left_start = 20 + row_index * 35
            for column, filename in enumerate(filenames):
                tile = self._load_tile(
                    cv2, np, filename, tile_width, tile_height
                )
                left = left_start + column * (tile_width + gap)
                canvas[top : top + tile_height, left : left + tile_width] = tile

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "reference.png"
            output = root / "tiles"
            preview = root / "preview.png"
            self.assertTrue(cv2.imwrite(str(source), canvas))

            records = split_mahjong_tiles(
                source, output, preview_path=preview
            )

            generated = sorted(path.name for path in output.glob("*.png"))
            manifest = json.loads((output / "牌名映射.json").read_text("utf-8"))
            preview_exists = preview.exists()

        self.assertEqual(len(records), 34)
        self.assertEqual(len(generated), 34)
        self.assertEqual(len(manifest), 34)
        self.assertIn("一万.png", generated)
        self.assertIn("九筒.png", generated)
        self.assertIn("九索.png", generated)
        self.assertIn("东.png", generated)
        self.assertIn("白.png", generated)
        self.assertIn("中.png", generated)
        self.assertTrue(preview_exists)


if __name__ == "__main__":
    unittest.main()
