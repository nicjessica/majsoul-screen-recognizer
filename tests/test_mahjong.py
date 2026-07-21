import unittest

from mahjong.analyzer import analyze_hand
from mahjong.shanten import calculate_shanten
from mahjong.tiles import tiles_to_counts


class MahjongAnalysisTests(unittest.TestCase):
    def test_completed_hand_is_minus_one_shanten(self):
        counts = tiles_to_counts(
            ["1m", "2m", "3m", "2p", "3p", "4p", "3s", "4s", "5s", "east", "east", "east", "red", "red"]
        )
        self.assertEqual(calculate_shanten(counts), -1)

    def test_tenpai_hand_is_zero_shanten(self):
        counts = tiles_to_counts(
            ["1m", "2m", "3m", "2p", "3p", "4p", "3s", "4s", "5s", "east", "east", "east", "red"]
        )
        self.assertEqual(calculate_shanten(counts), 0)

    def test_analyzer_returns_sorted_recommendations_for_14_tiles(self):
        analysis = analyze_hand(
            ["1m", "2m", "3m", "2p", "3p", "4p", "3s", "4s", "5s", "east", "east", "red", "red", "9m"]
        )
        self.assertGreaterEqual(len(analysis.recommendations), 1)
        first = analysis.recommendations[0]
        self.assertLessEqual(first.resulting_shanten, analysis.recommendations[-1].resulting_shanten)

    def test_screenshot_hand_exposes_chiitoitsu_breakdown_without_changing_sort(self):
        tiles = [
            "1m", "1m", "4s", "4s", "7s", "7s", "8s", "8s",
            "east", "east", "north", "white", "red", "9s",
        ]

        analysis = analyze_hand(tiles)

        self.assertEqual(analysis.shanten, 1)
        self.assertIsNotNone(analysis.breakdown)
        assert analysis.breakdown is not None
        self.assertEqual(analysis.breakdown.normal, 2)
        self.assertEqual(analysis.breakdown.chiitoitsu, 1)
        self.assertEqual(analysis.breakdown.kokushi, 6)
        self.assertEqual(
            [item.discard for item in analysis.recommendations[:4]],
            ["9s", "north", "white", "red"],
        )
        self.assertTrue(all(
            (item.resulting_shanten, item.ukeire_count) == (1, 9)
            for item in analysis.recommendations[:4]
        ))
        self.assertTrue(all(
            item.resulting_shanten == 2 for item in analysis.recommendations[4:]
        ))

    def test_invalid_tile_count(self):
        with self.assertRaises(ValueError):
            analyze_hand(["1m", "2m"])

    def test_one_open_meld_tenpai_and_effective_draw(self):
        tiles = ["1m", "2m", "3m", "2p", "3p", "4p", "3s", "4s", "5s", "red"]
        analysis = analyze_hand(tiles, open_meld_count=1)

        self.assertEqual(analysis.shanten, 0)
        self.assertEqual(analysis.recommendations[0].effective_tiles, ["red"])
        self.assertEqual(analysis.recommendations[0].ukeire_count, 3)

    def test_one_open_meld_recommendations_for_eleven_tiles(self):
        tiles = ["1m", "2m", "3m", "2p", "3p", "4p", "3s", "4s", "5s", "red", "9m"]
        analysis = analyze_hand(tiles, open_meld_count=1)

        self.assertEqual(analysis.shanten, 0)
        self.assertEqual(analysis.recommendations[0].discard, "9m")
        self.assertEqual(analysis.recommendations[0].effective_tiles, ["red"])

    def test_two_open_melds_support_seven_and_eight_tiles(self):
        before_draw = ["1m", "2m", "3m", "2p", "3p", "4p", "red"]
        analysis = analyze_hand(before_draw, open_meld_count=2)
        self.assertEqual(analysis.shanten, 0)
        self.assertEqual(analysis.recommendations[0].effective_tiles, ["red"])

        after_draw = analyze_hand([*before_draw, "9m"], open_meld_count=2)
        self.assertEqual(after_draw.recommendations[0].discard, "9m")

    def test_open_hand_uses_normal_shape_only(self):
        seven_pairs_tenpai = ["1m", "1m", "2m", "2m", "3p", "3p", "4p", "4p", "east", "east"]
        self.assertGreater(calculate_shanten(tiles_to_counts(seven_pairs_tenpai), open_meld_count=1), 0)

    def test_chi_pon_and_kan_each_count_as_one_fixed_meld(self):
        concealed_tiles = ["1m", "2m", "3m", "2p", "3p", "4p", "3s", "4s", "5s", "red"]

        # 吃、碰和杠的牌面结构不同，但在向听计算中都只占一个已完成面子。
        chi_meld_count = pon_meld_count = kan_meld_count = 1
        expected = analyze_hand(concealed_tiles, open_meld_count=chi_meld_count)

        self.assertEqual(expected.shanten, 0)
        self.assertEqual(
            analyze_hand(concealed_tiles, open_meld_count=pon_meld_count),
            expected,
        )
        self.assertEqual(
            analyze_hand(concealed_tiles, open_meld_count=kan_meld_count),
            expected,
        )

    def test_two_fixed_melds_including_a_kan_count_as_two(self):
        concealed_tiles = ["1m", "2m", "3m", "2p", "3p", "4p", "red"]

        # 一组普通副露加一组杠仍是两个完成面子，不按五组或七张可见牌计数。
        analysis = analyze_hand(concealed_tiles, open_meld_count=2)

        self.assertEqual(analysis.shanten, 0)
        self.assertEqual(analysis.recommendations[0].effective_tiles, ["red"])

    def test_kan_replacement_draw_allows_eleven_concealed_tiles(self):
        concealed_tiles_after_replacement_draw = [
            "1m", "2m", "3m", "2p", "3p", "4p", "3s", "4s", "5s", "red", "9m"
        ]

        analysis = analyze_hand(concealed_tiles_after_replacement_draw, open_meld_count=1)

        self.assertEqual(analysis.shanten, 0)
        self.assertEqual(analysis.recommendations[0].discard, "9m")
        self.assertEqual(analysis.recommendations[0].effective_tiles, ["red"])

    def test_visible_tiles_from_multiple_sources_reduce_ukeire(self):
        tiles = ["1m", "2m", "3m", "2p", "3p", "4p", "3s", "4s", "5s", "red"]

        analysis = analyze_hand(
            tiles,
            open_meld_count=1,
            visible_tiles=["red", "red"],
        )

        self.assertEqual(analysis.recommendations[0].effective_tiles, ["red"])
        self.assertEqual(analysis.recommendations[0].ukeire_count, 1)

    def test_visible_red_five_is_normalized_for_ukeire(self):
        tiles = ["1m", "2m", "3m", "2p", "3p", "4p", "3s", "4s", "5s", "5m"]

        analysis = analyze_hand(tiles, open_meld_count=1, visible_tiles=["5mr"])

        self.assertEqual(analysis.recommendations[0].effective_tiles, ["5m"])
        self.assertEqual(analysis.recommendations[0].ukeire_count, 2)

    def test_dora_indicators_are_plain_visible_tile_inputs(self):
        tiles = ["1m", "2m", "3m", "2p", "3p", "4p", "3s", "4s", "5s", "red"]
        indicator_sets = [
            ["red"],
            ["red", "south", "west"],
            ["red", "south", "west", "north", "white"],
        ]

        for indicators in indicator_sets:
            with self.subTest(indicator_count=len(indicators)):
                analysis = analyze_hand(
                    tiles,
                    open_meld_count=1,
                    visible_tiles=indicators,
                )
                self.assertEqual(analysis.recommendations[0].effective_tiles, ["red"])
                self.assertEqual(analysis.recommendations[0].ukeire_count, 2)

    def test_candidate_discard_is_counted_as_visible(self):
        completed_hand = [
            "1m", "2m", "3m", "2p", "3p", "4p", "3s", "4s", "5s",
            "east", "east", "east", "red", "red",
        ]

        analysis = analyze_hand(completed_hand)
        discard_red = next(item for item in analysis.recommendations if item.discard == "red")

        self.assertIn("red", discard_red.effective_tiles)
        self.assertEqual(discard_red.ukeire_count, 2)

    def test_concealed_and_visible_tiles_cannot_exceed_four(self):
        tiles = ["1m", "2m", "3m", "2p", "3p", "4p", "3s", "4s", "5s", "red"]

        with self.assertRaisesRegex(ValueError, "red"):
            analyze_hand(
                tiles,
                open_meld_count=1,
                visible_tiles=["red", "red", "red", "red"],
            )


if __name__ == "__main__":
    unittest.main()
