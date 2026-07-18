import unittest

from mahjong.decision import (
    ActionCandidate,
    MeldState,
    RoundContext,
    evaluate_actions,
    generate_call_candidates,
    generate_state_candidates,
)


class DecisionTests(unittest.TestCase):
    def setUp(self):
        self.waiting_hand = [
            "1m", "2m", "3m", "2p", "3p", "4p", "3s", "4s", "5s",
            "east", "east", "red", "red",
        ]

    def test_no_candidates_generates_discard_damaten_and_riichi(self):
        hand = [*self.waiting_hand, "east"]

        candidates = generate_state_candidates(hand)

        east_actions = {item.kind for item in candidates if item.discard_tile == "east"}
        self.assertIn("discard", east_actions)
        self.assertIn("damaten", east_actions)
        self.assertIn("riichi", east_actions)

    def test_riichi_legality_checks_closed_points_and_remaining_tiles(self):
        hand = [*self.waiting_hand, "east"]
        candidate = ActionCandidate("riichi", discard_tile="east")

        legal = evaluate_actions(hand, candidates=[candidate])
        no_points = evaluate_actions(
            hand,
            candidates=[candidate],
            context=RoundContext(points=900),
        )
        too_late = evaluate_actions(
            hand,
            candidates=[candidate],
            context=RoundContext(points=25000, remaining_draws=3),
        )

        self.assertTrue(legal.evaluations[0].legal)
        self.assertEqual(legal.evaluations[0].legality, "unverified")
        self.assertEqual(legal.evaluations[0].value.guaranteed_yaku, ("riichi",))
        self.assertFalse(no_points.evaluations[0].legal)
        self.assertFalse(too_late.evaluations[0].legal)

    def test_open_meld_makes_riichi_illegal(self):
        hand = [
            "1m", "2m", "3m", "2p", "3p", "4p", "3s", "4s", "5s", "red", "red",
        ]
        report = evaluate_actions(
            hand,
            melds=[MeldState("pon", ("east", "east", "east"))],
            candidates=[ActionCandidate("riichi", discard_tile="red")],
        )

        self.assertFalse(report.evaluations[0].legal)
        self.assertIn("不能立直", report.evaluations[0].reasons[0])

    def test_chi_is_only_legal_from_left_and_requires_exact_sequence(self):
        hand = [
            "1m", "2m", "4m", "5m", "6m", "2p", "3p", "4p", "3s", "4s", "5s", "red", "red",
        ]
        valid = ActionCandidate("chi", "3m", ("1m", "2m"), source="left")
        wrong_source = ActionCandidate("chi", "3m", ("1m", "2m"), source="right")
        wrong_shape = ActionCandidate("chi", "3m", ("1m", "4m"), source="left")

        report = evaluate_actions(hand, candidates=[valid, wrong_source, wrong_shape])

        self.assertTrue(report.evaluations[0].legal)
        self.assertFalse(report.evaluations[1].legal)
        self.assertFalse(report.evaluations[2].legal)

    def test_pon_and_minkan_require_two_or_three_matching_tiles(self):
        hand = [
            "1m", "2m", "3m", "2p", "3p", "4p", "3s", "4s", "5s",
            "red", "red", "red", "east",
        ]
        pon = ActionCandidate("pon", "red", ("red", "red"), source="across")
        minkan = ActionCandidate("minkan", "red", ("red", "red", "red"), source="right")
        invalid = ActionCandidate("pon", "red", ("red", "east"), source="left")

        report = evaluate_actions(hand, candidates=[pon, minkan, invalid])

        self.assertTrue(report.evaluations[0].legal)
        self.assertTrue(report.evaluations[1].legal)
        self.assertFalse(report.evaluations[2].legal)
        self.assertEqual(report.evaluations[0].value.guaranteed_han, 1)

    def test_call_candidates_always_start_with_skip_and_obey_source(self):
        hand = ["1m", "2m", "4m", "5m", "3p", "3p", "3p"]

        left = generate_call_candidates(hand, "3m", "left")
        across = generate_call_candidates(hand, "3m", "across")

        self.assertEqual(left[0], ActionCandidate("skip"))
        self.assertEqual(across[0], ActionCandidate("skip"))
        self.assertEqual(
            {item.consumed_tiles for item in left if item.kind == "chi"},
            {("1m", "2m"), ("2m", "4m"), ("4m", "5m")},
        )
        self.assertFalse(any(item.kind == "chi" for item in across))

    def test_call_candidates_generate_pon_and_minkan_only_with_enough_copies(self):
        two = generate_call_candidates(["red", "red", "east"], "red", "right")
        three = generate_call_candidates(["red", "red", "red"], "red", "across")

        self.assertEqual([item.kind for item in two], ["skip", "pon"])
        self.assertEqual([item.kind for item in three], ["skip", "pon", "minkan"])

        call_hand = [
            "1m", "2m", "3m", "2p", "3p", "4p", "3s", "4s", "5s",
            "red", "red", "red", "east",
        ]
        report = evaluate_actions(call_hand, candidates=generate_call_candidates(
            call_hand, "red", "across"
        ))
        kan = next(item for item in report.evaluations if item.action.kind == "minkan")
        self.assertEqual(kan.relative_win_chance, "unknown")
        self.assertEqual(kan.recommendation, "consider")

    def test_call_candidates_preserve_red_five_consumption_without_duplicates(self):
        candidates = generate_call_candidates(
            ["3m", "4m", "5m", "5mr", "5m"], "5m", "left"
        )

        chi_consumed = [
            item.consumed_tiles for item in candidates if item.kind == "chi"
        ]
        pon_consumed = [
            item.consumed_tiles for item in candidates if item.kind == "pon"
        ]
        self.assertEqual(chi_consumed, [("3m", "4m")])
        self.assertEqual(set(pon_consumed), {("5m", "5m"), ("5m", "5mr")})
        self.assertEqual(len(pon_consumed), 2)

    def test_call_candidates_reject_invalid_event_data(self):
        with self.assertRaisesRegex(ValueError, "来源"):
            generate_call_candidates(["1m", "2m"], "3m", "self")  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "未知牌名"):
            generate_call_candidates(["1m", "2m"], "bogus", "left")

    def test_ankan_keeps_closed_hand_and_kakan_requires_existing_pon(self):
        ankan_hand = [
            "1m", "2m", "3m", "2p", "3p", "4p", "3s", "4s", "5s",
            "red", "red", "red", "red", "east",
        ]
        ankan = evaluate_actions(
            ankan_hand,
            candidates=[ActionCandidate("ankan", consumed_tiles=("red",) * 4)],
        )
        kakan_hand = [
            "1m", "2m", "3m", "2p", "3p", "4p", "3s", "4s", "5s", "red", "east",
        ]
        kakan = evaluate_actions(
            kakan_hand,
            melds=[MeldState("pon", ("red", "red", "red"))],
            candidates=[ActionCandidate("kakan", consumed_tiles=("red",))],
        )

        self.assertTrue(ankan.evaluations[0].legal)
        self.assertTrue(kakan.evaluations[0].legal)
        self.assertEqual(kakan.evaluations[0].value.guaranteed_han, 1)

    def test_value_estimate_counts_known_dora_but_does_not_invent_points(self):
        report = evaluate_actions(
            self.waiting_hand,
            candidates=[ActionCandidate("skip")],
            context=RoundContext(dora_indicators=("north", "9m")),
        )

        value = report.evaluations[0].value
        self.assertEqual(value.known_dora, 3)  # two east + one 1m
        self.assertIsNone(value.fu)
        self.assertIsNone(value.points_range)
        self.assertIn("fu and exact points", value.unknown)

    def test_red_five_value_is_a_conservative_post_action_floor(self):
        hand = [
            "1m", "2m", "3m", "2p", "3p", "4p", "3s", "4s", "5s",
            "east", "east", "red", "5mr",
        ]
        report = evaluate_actions(
            [*hand, "5m"],
            candidates=[ActionCandidate("discard", discard_tile="5m")],
        )

        self.assertEqual(report.evaluations[0].value.known_dora, 0)

    def test_win_chance_is_relative_label_with_explicit_high_uncertainty(self):
        report = evaluate_actions(self.waiting_hand, candidates=[ActionCandidate("skip")])
        evaluation = report.evaluations[0]

        self.assertIn(evaluation.relative_win_chance, {"higher", "similar", "lower", "unknown"})
        self.assertEqual(evaluation.win_chance_uncertainty, "high")
        self.assertFalse(hasattr(evaluation, "win_rate"))

    def test_kan_is_not_hard_ranked_against_normal_draw_shape(self):
        hand = [
            "1m", "2m", "3m", "2p", "3p", "4p", "3s", "4s", "5s",
            "red", "red", "red", "red", "east",
        ]
        report = evaluate_actions(
            hand,
            candidates=[ActionCandidate("ankan", consumed_tiles=("red",) * 4)],
        )

        self.assertEqual(report.evaluations[0].relative_win_chance, "unknown")
        self.assertEqual(report.evaluations[0].recommendation, "consider")

    def test_meld_structure_validation_is_strict(self):
        with self.assertRaisesRegex(ValueError, "chi"):
            evaluate_actions(
                self.waiting_hand[:10],
                melds=[MeldState("chi", ("1m", "2m", "4m"))],
            )
        with self.assertRaisesRegex(ValueError, "暗杠"):
            evaluate_actions(
                self.waiting_hand[:10],
                melds=[MeldState("ankan", ("east",) * 4, is_open=True)],
            )


if __name__ == "__main__":
    unittest.main()
