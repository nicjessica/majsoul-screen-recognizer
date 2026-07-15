import unittest

from recognizer.models import (
    MeldRecognition,
    MeldTileRecognition,
    PlayerMeldRecognition,
    RecognitionResult,
)
from recognizer.visible_tiles import collect_visible_tiles


def _result(**changes) -> RecognitionResult:
    values = dict(
        hand=["1m"],
        draw=None,
        dora_indicators=[],
        meld_tiles=[],
        confidence=0.9,
        matches=[],
    )
    values.update(changes)
    return RecognitionResult(**values)


def _meld(*names: str | None) -> MeldRecognition:
    return MeldRecognition(
        kind="unknown",
        tiles=[MeldTileRecognition(name, None) for name in names],
        confidence=None,
    )


class VisibleTilesTests(unittest.TestCase):
    def test_collects_dora_and_only_known_structured_slots_without_duplicates(self):
        result = _result(
            dora_indicators=["east"],
            meld_tiles=["legacy-self"],
            melds=[_meld("2p", None, "4p")],
            opponent_melds=[
                PlayerMeldRecognition(
                    "right", ["legacy-right"], [_meld(None, "5s")]
                )
            ],
        )

        self.assertEqual(collect_visible_tiles(result), ["east", "2p", "4p", "5s"])

    def test_uses_legacy_flattened_tiles_when_structured_melds_are_empty(self):
        result = _result(
            meld_tiles=["1p", "2p", "3p"],
            opponent_melds=[
                PlayerMeldRecognition("left", ["white", "white", "white"], [])
            ],
        )

        self.assertEqual(
            collect_visible_tiles(result),
            ["1p", "2p", "3p", "white", "white", "white"],
        )

    def test_opponents_are_collected_in_right_across_left_order(self):
        result = _result(opponent_melds=[
            PlayerMeldRecognition("left", ["3s"], []),
            PlayerMeldRecognition("across", ["2s"], []),
            PlayerMeldRecognition("right", ["1s"], []),
        ])

        self.assertEqual(collect_visible_tiles(result), ["1s", "2s", "3s"])


if __name__ == "__main__":
    unittest.main()
