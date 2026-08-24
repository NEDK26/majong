# 本程序仅用于立直麻将算法学习研究，禁止用于线上游戏对局辅助。

from __future__ import annotations

import unittest
from unittest.mock import patch

from desktop_utils import (
    AnalysisStabilizer,
    OCRTemporalConsensus,
    fit_preview_size,
    friendly_error_message,
    shanten_label,
)
from ocr_input import OCRResult, TileRecognition
from screen_capture import (
    _DXGI_CAMERAS,
    _capture_with_dxgi,
    ScreenCaptureError,
    _dxgi_output_for_monitor,
    _legalize_recognized_tiles,
    analysis_payload,
    analyze_capture,
    capture_screen,
    clamp_region,
)


class ScreenCaptureTests(unittest.TestCase):
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

    def test_analysis_payload_supports_open_hand(self) -> None:
        payload = analysis_payload(self._ocr_result())

        self.assertEqual(payload["tiles"], ["6p", "7p", "4s", "4s", "6s", "7s", "8s", "4z"])
        self.assertEqual(payload["shanten"], 0)
        self.assertEqual(payload["recommendations"], ["4z"])

    def test_ambiguous_tile_does_not_produce_a_discard_recommendation(self) -> None:
        recognitions = list(self._ocr_result().recognitions)
        first = recognitions[0]
        recognitions[0] = TileRecognition(
            tile=first.tile,
            confidence=0.90,
            alternatives=(("7p", 0.89),),
            box=first.box,
            match_score=0.90,
        )

        payload = analysis_payload(
            OCRResult(image_path=None, recognitions=tuple(recognitions))
        )

        self.assertEqual(payload["mode"], "uncertain")
        self.assertEqual(payload["uncertainPositions"], [1])
        self.assertIsNone(payload["shanten"])
        self.assertEqual(payload["recommendations"], [])

    def test_duplicate_ocr_tiles_are_corrected_before_hand_validation(self) -> None:
        recognitions = []
        for index in range(5):
            recognitions.append(
                TileRecognition(
                    tile="8p",
                    confidence=0.90 if index < 4 else 0.55,
                    alternatives=(("7p", 0.80),),
                    box=(index * 40, 0, 40, 60),
                    match_score=0.90 if index < 4 else 0.56,
                )
            )
        result, correction_count = _legalize_recognized_tiles(
            OCRResult(image_path=None, recognitions=tuple(recognitions))
        )

        self.assertEqual(correction_count, 1)
        self.assertEqual(result.tiles.count("8p"), 4)
        self.assertEqual(result.tiles.count("7p"), 1)

    def test_capture_pipeline_does_not_require_a_temporary_image(self) -> None:
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover
            self.skipTest(f"NumPy is not installed: {exc}")

        frame = np.zeros((120, 800, 3), dtype=np.uint8)
        with (
            patch(
                "screen_capture.capture_screen",
                return_value=(frame, {"left": 0, "top": 700, "width": 800, "height": 120}),
            ),
            patch("screen_capture.recognize_hand_frame", return_value=self._ocr_result()),
        ):
            payload = analyze_capture(1, {"x": 0, "y": 700, "width": 800, "height": 120}, 8)

        self.assertEqual(payload["recommendations"], ["4z"])
        self.assertEqual(payload["captureRegion"]["height"], 120)

    def test_region_is_clamped_to_monitor(self) -> None:
        monitor = {"left": -1920, "top": 0, "width": 1920, "height": 1080}
        self.assertEqual(
            clamp_region({"x": 1800, "y": 1000, "width": 500, "height": 500}, monitor),
            {"left": -120, "top": 1000, "width": 120, "height": 80},
        )

    def test_dxgi_output_is_matched_by_monitor_resolution(self) -> None:
        class FakeDxcam:
            @staticmethod
            def output_info() -> str:
                return (
                    "Device[0] Output[0]: Res:(2560, 1440) Rot:0 Primary:True\n"
                    "Device[0] Output[1]: Res:(1920, 1080) Rot:0 Primary:False\n"
                )

        self.assertEqual(
            _dxgi_output_for_monitor(
                FakeDxcam(), 1, {"width": 1920, "height": 1080}
            ),
            (0, 1),
        )

    def test_windows_auto_capture_falls_back_from_dxgi_to_mss(self) -> None:
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover
            self.skipTest(f"NumPy is not installed: {exc}")

        frame = np.full((40, 80, 3), 90, dtype=np.uint8)
        fake_cv2 = object()
        fake_mss = object()
        with (
            patch("screen_capture._screen_modules", return_value=(fake_cv2, fake_mss, np)),
            patch(
                "screen_capture._selected_monitor",
                return_value={"left": 0, "top": 0, "width": 80, "height": 40},
            ),
            patch("screen_capture._is_windows", return_value=True),
            patch(
                "screen_capture._capture_with_dxgi",
                side_effect=ScreenCaptureError("DXGI 测试失败"),
            ),
            patch("screen_capture._capture_with_mss", return_value=frame),
        ):
            captured, region = capture_screen(1, backend="auto")

        self.assertIs(captured, frame)
        self.assertEqual(region["backend"], "MSS")

    def test_dxgi_camera_is_reused_between_frames(self) -> None:
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover
            self.skipTest(f"NumPy is not installed: {exc}")

        class FakeCamera:
            def grab(self, **_: object):
                return np.full((20, 40, 3), 80, dtype=np.uint8)

            def release(self) -> None:
                pass

        class FakeDxcam:
            create_count = 0

            @classmethod
            def create(cls, **_: object):
                cls.create_count += 1
                return FakeCamera()

        monitor = {"left": 0, "top": 0, "width": 100, "height": 100}
        region = {"left": 0, "top": 0, "width": 40, "height": 20}
        _DXGI_CAMERAS.clear()
        with (
            patch("screen_capture._is_windows", return_value=True),
            patch("screen_capture._dxcam_module", return_value=FakeDxcam),
            patch("screen_capture._dxgi_output_for_monitor", return_value=(0, 0)),
        ):
            _capture_with_dxgi(1, monitor, region, np)
            _capture_with_dxgi(1, monitor, region, np)

        self.assertEqual(FakeDxcam.create_count, 1)
        _DXGI_CAMERAS.clear()

    def test_capture_reuses_geometry_without_enumerating_monitor_again(self) -> None:
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover
            self.skipTest(f"NumPy is not installed: {exc}")

        frame = np.full((40, 80, 3), 100, dtype=np.uint8)
        geometry = {"left": -80, "top": 0, "width": 80, "height": 40}
        with (
            patch("screen_capture._screen_modules", return_value=(object(), object(), np)),
            patch("screen_capture._selected_monitor") as enumerate_monitor,
            patch("screen_capture._capture_with_mss", return_value=frame),
        ):
            captured, region = capture_screen(
                2,
                backend="mss",
                monitor_geometry=geometry,
            )

        enumerate_monitor.assert_not_called()
        self.assertIs(captured, frame)
        self.assertEqual(region["left"], -80)

    def test_desktop_preview_preserves_aspect_ratio(self) -> None:
        self.assertEqual(fit_preview_size(1920, 1080, 800, 600), (800, 450))
        self.assertEqual(fit_preview_size(640, 480, 1200, 900), (640, 480))
        self.assertEqual(shanten_label(0), "听牌")

    def test_analysis_stabilizer_requires_repeated_count_changes(self) -> None:
        stabilizer = AnalysisStabilizer()

        self.assertFalse(stabilizer.should_accept((14, "discard")))
        self.assertTrue(stabilizer.should_accept((14, "discard")))
        self.assertTrue(stabilizer.should_accept((14, "discard")))
        self.assertFalse(stabilizer.should_accept((7, "draw")))
        self.assertTrue(stabilizer.should_accept((14, "discard")))
        self.assertFalse(stabilizer.should_accept((7, "draw")))
        self.assertTrue(stabilizer.should_accept((7, "draw")))

    def test_temporal_consensus_confirms_repeated_ambiguous_tile(self) -> None:
        stabilizer = OCRTemporalConsensus()

        def frame(best: str, alternative: str) -> OCRResult:
            return OCRResult(
                image_path=None,
                recognitions=(
                    TileRecognition(
                        tile=best,
                        confidence=0.70,
                        alternatives=((alternative, 0.797),),
                        box=(0, 0, 40, 60),
                        match_score=0.80,
                    ),
                ),
            )

        first = stabilizer.update(frame("4m", "6m"))
        confirmed = stabilizer.update(frame("4m", "6m"))
        after_noise = stabilizer.update(frame("6m", "4m"))

        self.assertFalse(first.recognitions[0].is_reliable)
        self.assertTrue(confirmed.recognitions[0].is_reliable)
        self.assertEqual(confirmed.tiles, ["4m"])
        self.assertEqual(after_noise.tiles, ["4m"])

    def test_temporal_consensus_resets_when_hand_count_changes(self) -> None:
        stabilizer = OCRTemporalConsensus()
        stable = self._ocr_result()
        stabilizer.update(stable)
        stabilizer.update(stable)

        changed = OCRResult(
            image_path=None,
            recognitions=stable.recognitions + (
                TileRecognition("1m", 0.7, (("2m", 0.69),), (320, 0, 40, 60), 0.7),
            ),
        )

        result = stabilizer.update(changed)

        self.assertEqual(len(result.recognitions), 9)
        self.assertEqual(stabilizer.history_size, 1)

    def test_low_level_errors_are_translated_to_chinese(self) -> None:
        translated = friendly_error_message(PermissionError("Access denied"))
        self.assertIn("屏幕捕获权限", translated)
        self.assertNotIn("Access denied", translated)


if __name__ == "__main__":
    unittest.main()
