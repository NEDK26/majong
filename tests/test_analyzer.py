# 本程序仅用于立直麻将算法学习研究，禁止用于线上游戏对局辅助。

import unittest

from analyzer import calculate_shanten, core_analyze
from input_layer import hand_input
from tile_utils import HandValidationError


class AnalyzerTests(unittest.TestCase):
    def test_thirteen_tile_ready_hand_reports_winning_draw(self) -> None:
        hand = hand_input(
            ["1m", "2m", "3m", "1p", "2p", "3p", "1s", "2s", "3s", "7s", "8s", "9s", "1z"]
        )
        result = core_analyze(hand)

        self.assertEqual(calculate_shanten(hand), 0)
        self.assertEqual(result.mode, "draw")
        self.assertEqual([(tile.tile, tile.remaining) for tile in result.effective_draws], [("1z", 3)])
        self.assertEqual(result.draw_ukeire, 3)

    def test_fourteen_tile_candidates_are_sorted_by_requested_strategy(self) -> None:
        hand = hand_input(
            ["1m", "2m", "3m", "1p", "2p", "3p", "1s", "2s", "3s", "7s", "8s", "9s", "1z", "9m"]
        )
        result = core_analyze(hand)

        self.assertEqual(result.mode, "discard")
        priorities = [(item.shanten, -item.ukeire) for item in result.candidates]
        self.assertEqual(priorities, sorted(priorities))
        self.assertEqual(result.candidates[0].shanten, 0)

    def test_duplicate_physical_tiles_create_one_discard_candidate(self) -> None:
        hand = hand_input(
            ["1m", "1m", "2m", "2m", "3p", "3p", "4p", "4p", "5s", "5s", "6s", "6s", "7z", "9m"]
        )
        result = core_analyze(hand)

        discards = [item.discard for item in result.candidates]
        self.assertEqual(len(discards), len(set(discards)))
        self.assertEqual(len(discards), 8)

    def test_rejects_more_than_four_identical_tiles(self) -> None:
        with self.assertRaises(HandValidationError):
            hand_input(["1m"] * 5 + ["2m"] * 4 + ["3m"] * 4)

    def test_open_hand_with_two_melds_accepts_eight_concealed_tiles(self) -> None:
        hand = hand_input(["1p", "2p", "3p", "2s", "3s", "4s", "1z", "9m"])
        result = core_analyze(hand)

        self.assertEqual(result.mode, "discard")
        self.assertTrue(result.candidates)


if __name__ == "__main__":
    unittest.main()
