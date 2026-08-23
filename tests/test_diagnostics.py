# 本程序仅用于立直麻将算法学习研究，禁止用于线上游戏对局辅助。

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from diagnostics import (
    diagnostics_directory,
    log_event,
    log_exception,
    recognition_details,
)
from ocr_input import TileRecognition


class DiagnosticsTests(unittest.TestCase):
    def test_writes_chinese_json_log_to_local_data_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(
                "os.environ", {"MAHJONG_STUDY_DATA_DIR": directory}, clear=False
            ):
                log_event("识别测试", 识别张数=14, OCR耗时毫秒=123.4)
                path = diagnostics_directory() / "diagnostics.log"
                record = json.loads(path.read_text(encoding="utf-8").splitlines()[-1])

        self.assertEqual(record["事件"], "识别测试")
        self.assertEqual(record["识别张数"], 14)
        self.assertEqual(record["OCR耗时毫秒"], 123.4)

    def test_exception_log_contains_error_type_and_message(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(
                "os.environ", {"MAHJONG_STUDY_DATA_DIR": directory}, clear=False
            ):
                try:
                    raise RuntimeError("测试异常")
                except RuntimeError as error:
                    log_exception("分析失败", error)
                path = Path(directory) / "diagnostics" / "diagnostics.log"
                record = json.loads(path.read_text(encoding="utf-8").splitlines()[-1])

        self.assertEqual(record["事件"], "分析失败")
        self.assertEqual(record["错误类型"], "RuntimeError")
        self.assertEqual(record["错误信息"], "测试异常")
        self.assertIn("RuntimeError", record["调用栈"])

    def test_recognition_details_include_best_score_and_margin(self) -> None:
        details = recognition_details(
            [
                TileRecognition(
                    tile="2m",
                    confidence=0.71,
                    alternatives=(("9m", 0.74),),
                    box=(1, 2, 3, 4),
                    match_score=0.75,
                )
            ]
        )

        self.assertEqual(details[0]["匹配分"], 0.75)
        self.assertEqual(details[0]["区分度"], 0.01)


if __name__ == "__main__":
    unittest.main()
