import unittest

import numpy as np

from recognizer.config import MeldConfig, MeldTileSlotConfig, RelativeRegion, default_config
from recognizer.models import MeldTileRecognition, TileMatch
from recognizer.recognizer import TileRecognizer, _is_tile_back, infer_meld_kind


def _tile(name):
    return MeldTileRecognition(name, TileMatch(name, 0.95))


class MeldKindRecognitionTests(unittest.TestCase):
    def test_infers_chi_and_pon_from_recognized_faces(self):
        self.assertEqual(infer_meld_kind([_tile("3m"), _tile("4m"), _tile("5mr")]), "chi")
        self.assertEqual(infer_meld_kind([_tile("east"), _tile("east"), _tile("east")]), "pon")

    def test_four_equal_separate_faces_are_minkan(self):
        config = MeldConfig("unknown", [
            MeldTileSlotConfig(RelativeRegion(index * 0.2, 0, 0.18, 1))
            for index in range(4)
        ])
        self.assertEqual(infer_meld_kind([_tile("7p") for _ in range(4)], config), "minkan")

    def test_kakan_requires_explicit_stack_geometry(self):
        config = MeldConfig("unknown", [
            MeldTileSlotConfig(RelativeRegion(0, 0, 0.2, 1)),
            MeldTileSlotConfig(RelativeRegion(0.2, 0, 0.2, 1)),
            MeldTileSlotConfig(RelativeRegion(0.4, 0, 0.2, 1)),
            MeldTileSlotConfig(RelativeRegion(0.2, 0, 0.2, 0.5), stack_level=1),
        ])
        self.assertEqual(infer_meld_kind([_tile("5s") for _ in range(4)], config), "kakan")

    def test_two_detected_backs_make_ankan_without_exposing_tile_names(self):
        hidden = MeldTileRecognition(None, None)
        self.assertEqual(
            infer_meld_kind([hidden, _tile("red"), _tile("red"), hidden]),
            "ankan",
        )

    def test_green_back_detection_is_conservative(self):
        back = np.full((40, 30, 3), (45, 125, 75), dtype=np.uint8)
        face = np.full((40, 30, 3), 220, dtype=np.uint8)
        face[12:28, 10:20] = (35, 70, 155)
        self.assertTrue(_is_tile_back(back))
        self.assertFalse(_is_tile_back(face))

    def test_unstructured_equal_width_slots_split_by_detected_group_count(self):
        class Templates:
            def __init__(self):
                self.names = iter(("1m", "2m", "3m", "east", "east", "east"))

            def match_candidates(self, tile, limit=2):
                name = next(self.names)
                return [TileMatch(name, 0.95)]

        recognizer = TileRecognizer.__new__(TileRecognizer)
        recognizer.config = default_config()
        recognizer.templates = Templates()
        face = np.full((40, 30, 3), 220, dtype=np.uint8)
        face[12:28, 10:20] = (35, 70, 155)

        groups, _ = recognizer._recognize_melds(
            [face.copy() for _ in range(6)],
            configs=[],
            expected_group_count=2,
        )

        self.assertEqual([group.kind for group in groups], ["chi", "pon"])


if __name__ == "__main__":
    unittest.main()
