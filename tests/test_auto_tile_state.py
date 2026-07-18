import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from recognizer.config import RelativeRegion, default_config, load_config, save_config
from recognizer.models import RecognitionResult, TileMatch
from recognizer.recognizer import RecognitionError, TileRecognizer
from recognizer.stability import KeyRegionSnapshot, result_key


class _StateTemplates:
    def __init__(self, score_overrides=None):
        self.templates = {"shape": np.zeros((30, 20, 3), dtype=np.uint8)}
        self.score_overrides = score_overrides or {}

    def match_candidates(self, tile, limit=2):
        code = int(tile[0, 0, 0])
        score = self.score_overrides.get(code, 0.95 if code else 0.20)
        return [TileMatch(f"{max(code, 1)}m", score)][:limit]


def _region(x, y, width, height, frame_width=300, frame_height=60):
    return RelativeRegion(
        x / frame_width,
        y / frame_height,
        width / frame_width,
        height / frame_height,
    )


def _recognizer(hand_count, draw_present, *, manual_hand=10, manual_draw=0, manual_open=1):
    frame = np.zeros((60, 300, 3), dtype=np.uint8)
    for index in range(hand_count):
        frame[0:30, index * 20 : (index + 1) * 20] = 220
        # A real tile face is light, but it also contains a visible glyph/pip.
        # Keep the template lookup marker at [0, 0] and add a separate interior
        # feature so occupancy detection does not need template confidence.
        frame[9:21, index * 20 + 7 : index * 20 + 13] = (35, 70, 155)
        frame[0, index * 20, 0] = index + 1
    if draw_present:
        frame[0:30, 260:280] = 220
        frame[9:21, 267:273] = (35, 70, 155)
        frame[0, 260, 0] = 20

    config = default_config()
    config.layout.hand_region = _region(0, 0, 260, 30)
    config.layout.draw_region = _region(260, 0, 20, 30)
    config.layout.hand_tile_count = manual_hand
    config.layout.draw_tile_count = manual_draw
    config.layout.open_meld_count = manual_open
    config.layout.dora_tile_count = 0

    recognizer = TileRecognizer.__new__(TileRecognizer)
    recognizer.config = config
    recognizer.templates = _StateTemplates()
    return recognizer, frame


class AutoTileStateTests(unittest.TestCase):
    def test_detects_closed_hand_and_draw_despite_stale_manual_counts(self):
        recognizer, frame = _recognizer(13, True)

        result = recognizer.recognize(frame)

        self.assertEqual(len(result.hand), 13)
        self.assertEqual(result.draw, "20m")
        self.assertEqual(result.open_meld_count, 0)
        self.assertEqual(len(result.matches), 14)

    def test_detects_open_hand_and_empty_draw_region(self):
        recognizer, frame = _recognizer(
            10,
            False,
            manual_hand=13,
            manual_draw=1,
            manual_open=0,
        )

        result = recognizer.recognize(frame)

        self.assertEqual(len(result.hand), 10)
        self.assertIsNone(result.draw)
        self.assertEqual(result.open_meld_count, 1)

    def test_detects_all_deeper_open_hand_states_with_and_without_draw(self):
        for hand_count, expected_melds in ((7, 2), (4, 3), (1, 4)):
            for draw_present in (False, True):
                with self.subTest(hand_count=hand_count, draw_present=draw_present):
                    recognizer, frame = _recognizer(hand_count, draw_present)

                    result = recognizer.recognize(frame)

                    self.assertEqual(len(result.hand), hand_count)
                    self.assertEqual(result.draw is not None, draw_present)
                    self.assertEqual(result.open_meld_count, expected_melds)

    def test_keeps_contiguous_open_hand_waiting_to_discard_states_in_hand(self):
        for total_count, expected_melds in (
            (11, 1),
            (8, 2),
            (5, 3),
            (2, 4),
        ):
            with self.subTest(total_count=total_count):
                recognizer, frame = _recognizer(
                    total_count,
                    False,
                    manual_hand=total_count - 1,
                    manual_draw=0,
                    manual_open=expected_melds,
                )

                result = recognizer.recognize(frame)

                self.assertEqual(len(result.hand), total_count)
                self.assertIsNone(result.draw)
                self.assertEqual(result.open_meld_count, expected_melds)

    def test_rejects_contiguous_extra_tile_and_external_draw_collision(self):
        for total_count, base_count, expected_melds in (
            (11, 10, 1),
            (8, 7, 2),
            (5, 4, 3),
            (2, 1, 4),
        ):
            with self.subTest(total_count=total_count):
                recognizer, frame = _recognizer(
                    total_count,
                    True,
                    manual_hand=base_count,
                    manual_draw=0,
                    manual_open=expected_melds,
                )
                # A stale manual fallback must not be what happens to stop this
                # illegal total; make blank slots match successfully so the
                # contradictory layout itself has to block publication.
                recognizer.templates = _StateTemplates({0: 0.95})

                with self.assertRaises(RecognitionError) as context:
                    recognizer.recognize(frame)

                self.assertIn("同時有牌", str(context.exception))

    def test_ambiguous_empty_boundary_defers_to_manual_configuration(self):
        recognizer, frame = _recognizer(10, False)
        frame[3:8, 200:220] = 220

        self.assertIsNone(recognizer._try_detect_tile_state(frame))
        self.assertIn("占位介于", recognizer._auto_state_error)

    def test_low_template_scores_on_last_three_real_tiles_do_not_imply_a_meld(self):
        recognizer, frame = _recognizer(13, False)
        recognizer.templates = _StateTemplates({11: 0.60, 12: 0.60, 13: 0.60})

        with self.assertRaises(RecognitionError) as context:
            recognizer.recognize(frame)

        self.assertIn("第 11 张", str(context.exception))

    def test_high_template_score_for_blank_slots_does_not_create_tiles(self):
        recognizer, frame = _recognizer(10, False)
        recognizer.templates = _StateTemplates({0: 0.95})

        result = recognizer.recognize(frame)

        self.assertEqual(len(result.hand), 10)
        self.assertEqual(result.open_meld_count, 1)

    def test_white_placeholder_faces_after_open_hand_are_empty_slots(self):
        recognizer, frame = _recognizer(10, False)
        # Mahjong Soul leaves bright, low-saturation placeholder faces in the
        # three vacated positions.  Their light-face ratio is higher than that
        # of real tiles, but they contain no tile glyphs at all.
        frame[0:30, 200:260] = 220

        result = recognizer.recognize(frame)

        self.assertEqual(len(result.hand), 10)
        self.assertIsNone(result.draw)
        self.assertEqual(result.open_meld_count, 1)

    def test_open_hand_extra_tile_can_occupy_terminal_hand_slot(self):
        recognizer, frame = _recognizer(10, False)
        frame[0:30, 200:260] = 220
        frame[9:21, 247:253] = (35, 70, 155)
        frame[0, 240, 0] = 20

        result = recognizer.recognize(frame)

        self.assertEqual(len(result.hand), 10)
        self.assertEqual(result.draw, "20m")
        self.assertEqual(result.open_meld_count, 1)

    def test_explicit_disable_keeps_manual_counts_and_open_meld_count(self):
        recognizer, frame = _recognizer(13, True, manual_hand=10, manual_draw=0, manual_open=2)
        recognizer.config.recognition.auto_detect_tile_state = False

        result = recognizer.recognize(frame)

        self.assertEqual(len(result.hand), 10)
        self.assertIsNone(result.draw)
        self.assertEqual(result.open_meld_count, 2)

    def test_auto_snapshot_tracks_draw_even_when_manual_draw_count_is_zero(self):
        recognizer, frame = _recognizer(10, False)

        original = KeyRegionSnapshot.from_frame(
            frame,
            recognizer.config.layout,
            auto_detect_tile_state=True,
        )
        changed = frame.copy()
        changed[0:30, 260:280] = 255
        after_draw = KeyRegionSnapshot.from_frame(
            changed,
            recognizer.config.layout,
            auto_detect_tile_state=True,
        )

        self.assertEqual(original.regions.keys(), {"hand", "draw"})
        self.assertFalse(original.is_equivalent(after_draw))

    def test_result_key_includes_detected_open_meld_count(self):
        base = RecognitionResult([], None, [], [], 1.0, [], open_meld_count=0)
        opened = RecognitionResult([], None, [], [], 1.0, [], open_meld_count=1)

        self.assertNotEqual(result_key(base), result_key(opened))

    def test_old_config_enables_auto_detection_and_false_round_trips(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps({"recognition": {"threshold": 0.8}}), encoding="utf-8")
            self.assertTrue(load_config(path).recognition.auto_detect_tile_state)

            config = default_config()
            config.templates_dir = str(Path(tmp) / "templates")
            config.recognition.auto_detect_tile_state = False
            save_config(config, path)
            self.assertFalse(load_config(path).recognition.auto_detect_tile_state)


if __name__ == "__main__":
    unittest.main()
