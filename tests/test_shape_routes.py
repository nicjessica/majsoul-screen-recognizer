import unittest

from mahjong.shape_routes import KnownMeld, ShapeRouteContext, match_shape_routes


def _by_id(report, route_id):
    return next(item for item in report.candidates if item.id == route_id)


class ShapeRouteTests(unittest.TestCase):
    def test_exact_base_shape_routes_choose_seven_pairs_for_screenshot_hand(self):
        report = match_shape_routes(ShapeRouteContext(concealed_tiles=(
            "1m", "1m", "4s", "4s", "7s", "7s", "8s", "8s",
            "east", "east", "north", "white", "red", "9s",
        )))

        self.assertEqual(_by_id(report, "normal").metric, 2)
        self.assertEqual(_by_id(report, "chiitoitsu").metric, 1)
        self.assertEqual(_by_id(report, "kokushi_musou").metric, 6)
        self.assertEqual(report.primary_candidate.id, "chiitoitsu")
        self.assertEqual(report.primary_candidate.metric_kind, "shanten")

    def test_any_fixed_meld_blocks_special_closed_shapes(self):
        report = match_shape_routes(ShapeRouteContext(
            concealed_tiles=("1m", "1m", "2m", "2m", "3p", "3p", "4p", "4p", "east", "east"),
            open_meld_count=1,
            melds=(KnownMeld("ankan", ("red", "red", "red", "red"), False),),
        ))

        self.assertEqual(_by_id(report, "chiitoitsu").status, "blocked")
        self.assertEqual(_by_id(report, "kokushi_musou").status, "blocked")

    def test_red_five_normalizes_for_structure_and_keeps_deterministic_result(self):
        first = match_shape_routes(ShapeRouteContext(concealed_tiles=(
            "2m", "3m", "4m", "5mr", "6m", "7m", "2m", "3m", "4m", "5m", "6m", "7m", "8m",
        )))
        second = match_shape_routes(ShapeRouteContext(concealed_tiles=tuple(reversed((
            "2m", "3m", "4m", "5mr", "6m", "7m", "2m", "3m", "4m", "5m", "6m", "7m", "8m",
        )))))

        self.assertEqual(first, second)
        self.assertEqual(_by_id(first, "tanyao").metric, 0)
        self.assertEqual(_by_id(first, "chinitsu:m").metric, 0)

    def test_suit_families_keep_only_best_suit_and_count_conflicts(self):
        report = match_shape_routes(ShapeRouteContext(concealed_tiles=(
            "1m", "2m", "3m", "4m", "5m", "6m", "east", "east", "red", "2p", "3p", "4p", "5p",
        )))

        honitsu = _by_id(report, "honitsu:m")
        chinitsu = _by_id(report, "chinitsu:m")
        self.assertEqual(honitsu.metric, 4)  # four known pin tiles conflict; honors are allowed
        self.assertEqual(chinitsu.metric, 7)  # pin tiles plus three honors conflict
        self.assertFalse(any(item.id.startswith("honitsu:") and item.id != "honitsu:m" for item in report.candidates))

    def test_tanyao_obeys_open_rule_and_unknown_melds_degrade_whole_hand_routes(self):
        open_hand = ShapeRouteContext(
            concealed_tiles=("2m", "3m", "4m", "2p", "3p", "4p", "3s", "4s", "5s", "6s"),
            open_meld_count=1,
            melds=(KnownMeld("chi", ("2m", "3m", "4m")),),
            open_tanyao=False,
        )
        blocked = match_shape_routes(open_hand)
        self.assertEqual(_by_id(blocked, "tanyao").status, "blocked")

        unknown = match_shape_routes(ShapeRouteContext(
            concealed_tiles=("2m", "3m", "4m", "2p", "3p", "4p", "3s", "4s", "5s", "6s"),
            open_meld_count=1,
            unknown_meld_count=1,
        ))
        self.assertEqual(_by_id(unknown, "tanyao").status, "insufficient_data")
        self.assertEqual(_by_id(unknown, "honitsu").status, "insufficient_data")
        self.assertEqual(_by_id(unknown, "honroutou").status, "insufficient_data")

    def test_yakuhai_uses_reliable_winds_and_keeps_double_wind_separate(self):
        report = match_shape_routes(ShapeRouteContext(
            concealed_tiles=("east", "east", "east", "white", "white", "1m", "2m", "3m", "2p", "3p", "4p", "3s", "4s"),
            seat_wind="east",
            round_wind="east",
        ))

        self.assertEqual(_by_id(report, "yakuhai:seat").metric, 0)
        self.assertEqual(_by_id(report, "yakuhai:round").metric, 0)
        self.assertEqual(_by_id(report, "yakuhai:white").metric, 1)

        no_wind = match_shape_routes(ShapeRouteContext(concealed_tiles=(
            "east", "east", "east", "white", "white", "1m", "2m", "3m", "2p", "3p", "4p", "3s", "4s",
        )))
        self.assertFalse(any(item.id == "yakuhai:seat" for item in no_wind.candidates))
        self.assertFalse(any(item.id == "yakuhai:round" for item in no_wind.candidates))

    def test_validates_unknown_tiles_copy_limits_and_meld_accounting(self):
        with self.assertRaisesRegex(ValueError, "未知牌名"):
            match_shape_routes(ShapeRouteContext(concealed_tiles=("bogus",) * 13))
        with self.assertRaisesRegex(ValueError, "超过4张"):
            match_shape_routes(ShapeRouteContext(concealed_tiles=("1m",) * 5 + ("2m",) * 8))
        with self.assertRaisesRegex(ValueError, "不能超过"):
            match_shape_routes(ShapeRouteContext(
                concealed_tiles=("1m",) * 10,
                open_meld_count=1,
                melds=(KnownMeld("pon", ("east",) * 3), KnownMeld("pon", ("south",) * 3)),
            ))


if __name__ == "__main__":
    unittest.main()
