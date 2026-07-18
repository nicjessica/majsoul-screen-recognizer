import unittest

from tests.test_auto_tile_state import _recognizer


def _add_inline_discard_tile(frame, base_count):
    frame[0:30, base_count * 20 : 260] = 220
    frame[9:21, base_count * 20 + 7 : base_count * 20 + 13] = (35, 70, 155)
    frame[0, base_count * 20, 0] = 20


class AutoDiscardStateTests(unittest.TestCase):
    def test_contiguous_11_8_5_2_states_keep_base_hand_and_extra_draw(self):
        for base_count, expected_melds in ((10, 1), (7, 2), (4, 3), (1, 4)):
            with self.subTest(base_count=base_count):
                recognizer, frame = _recognizer(base_count, False)
                _add_inline_discard_tile(frame, base_count)

                result = recognizer.recognize(frame)

                self.assertEqual(len(result.hand), base_count + 1)
                self.assertEqual(result.hand[-1], "20m")
                self.assertIsNone(result.draw)
                self.assertEqual(result.open_meld_count, expected_melds)

    def test_inline_and_external_draw_conflict_defers_to_manual_state(self):
        recognizer, frame = _recognizer(10, True)
        _add_inline_discard_tile(frame, 10)

        self.assertIsNone(recognizer._try_detect_tile_state(frame))
        self.assertIn("同时有牌", recognizer._auto_state_error)

    def test_two_inline_extra_tiles_are_treated_as_transition(self):
        recognizer, frame = _recognizer(10, False)
        _add_inline_discard_tile(frame, 10)
        frame[9:21, 227:233] = (35, 70, 155)
        frame[0, 220, 0] = 21

        self.assertIsNone(recognizer._try_detect_tile_state(frame))
        self.assertIn("不符合", recognizer._auto_state_error)


if __name__ == "__main__":
    unittest.main()
