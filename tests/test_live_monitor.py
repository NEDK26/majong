# 本程序仅用于立直麻将算法学习研究，禁止用于线上游戏对局辅助。

from __future__ import annotations

import unittest
from unittest.mock import patch

from live_monitor import HOST, _analysis_payload, analyze_capture
from ocr_input import OCRResult, TileRecognition


class LiveMonitorTests(unittest.TestCase):
    @staticmethod
    def _ocr_result() -> OCRResult:
        tiles = ["6p", "7p", "4s", "4s", "6s", "7s", "8s", "4z"]
        return OCRResult(
            image_path=None,
            recognitions=tuple(
                TileRecognition(
                    tile=tile,
                    confidence=0.95,
                    alternatives=(),
                    box=(index * 40, 0, 40, 60),
                )
                for index, tile in enumerate(tiles)
            ),
        )

    def test_service_is_loopback_only(self) -> None:
        self.assertEqual(HOST, "127.0.0.1")

    def test_analysis_payload_supports_open_hand(self) -> None:
        payload = _analysis_payload(self._ocr_result())

        self.assertEqual(payload["tiles"], ["6p", "7p", "4s", "4s", "6s", "7s", "8s", "4z"])
        self.assertEqual(payload["shanten"], 0)
        self.assertEqual(payload["recommendations"], ["4z"])

    def test_capture_pipeline_does_not_require_a_temporary_image(self) -> None:
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover
            self.skipTest(f"NumPy is not installed: {exc}")

        frame = np.zeros((120, 800, 3), dtype=np.uint8)
        with (
            patch(
                "live_monitor.capture_screen",
                return_value=(frame, {"left": 0, "top": 700, "width": 800, "height": 120}),
            ),
            patch("live_monitor.recognize_hand_frame", return_value=self._ocr_result()),
        ):
            payload = analyze_capture(1, {"x": 0, "y": 700, "width": 800, "height": 120}, 8)

        self.assertEqual(payload["recommendations"], ["4z"])
        self.assertEqual(payload["captureRegion"]["height"], 120)


if __name__ == "__main__":
    unittest.main()
