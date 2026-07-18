import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from recognizer.config import PlayerScoreLayoutConfig, RelativeRegion, default_config
from recognizer.table_state import TableStateRecognizer
from recognizer.table_template_builder import build_table_state_templates_from_screenshot
from recognizer.models import PlayerScoreRecognition, RecognitionResult, TableStateRecognition
from recognizer.stability import result_key


def _pattern(seed: int, height: int = 24, width: int = 32) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    image[2:7, 3:12] = (seed * 31 % 255, 250, 20)
    return image


def _write_rgb(path: Path, image: np.ndarray) -> None:
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


class TableStateRecognizerTests(unittest.TestCase):
    def test_recognizes_round_wind_and_four_oriented_whole_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            round_image = _pattern(1)
            wind_image = _pattern(2)
            scores = {seat: _pattern(10 + index) for index, seat in enumerate(
                ("self", "right", "across", "left")
            )}
            _write_rgb(directory / "round_east_1.png", round_image)
            _write_rgb(directory / "round_south_2.png", _pattern(3))
            _write_rgb(directory / "wind_north.png", wind_image)
            _write_rgb(directory / "wind_east.png", _pattern(4))
            for index, image in enumerate(scores.values()):
                _write_rgb(directory / f"score_{25000 + index * 1000}.png", image)

            frame = np.zeros((96, 128, 3), dtype=np.uint8)
            frame[0:24, 0:32] = round_image
            frame[0:24, 32:64] = wind_image
            orientations = ("upright", "rotated_cw", "rotated_180", "rotated_ccw")
            regions = []
            for index, (seat, image, orientation) in enumerate(zip(scores, scores.values(), orientations)):
                displayed = {
                    "upright": image,
                    "rotated_cw": np.rot90(image, -1),
                    "rotated_180": np.rot90(image, 2),
                    "rotated_ccw": np.rot90(image, 1),
                }[orientation]
                y = 32
                x = index * 32
                h, w = displayed.shape[:2]
                frame[y:y + h, x:x + w] = displayed
                regions.append(PlayerScoreLayoutConfig(
                    seat, RelativeRegion(x / 128, y / 96, w / 128, h / 96), orientation
                ))

            config = default_config()
            config.table_templates_dir = tmp
            config.recognition.threshold = 0.9
            config.layout.table_state.round_region = RelativeRegion(0, 0, 0.25, 0.25)
            config.layout.table_state.self_wind_region = RelativeRegion(0.25, 0, 0.25, 0.25)
            config.layout.table_state.scores = regions
            result = TableStateRecognizer(config).recognize(frame)

            self.assertEqual((result.round.round_wind, result.round.hand_number), ("east", 1))
            self.assertEqual(result.self_wind.wind, "north")
            self.assertEqual([item.score for item in result.scores], [25000, 26000, 27000, 28000])
            self.assertTrue(all(item.error is None for item in result.scores))

    def test_empty_directory_and_low_confidence_degrade_independently(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = default_config()
            config.table_templates_dir = tmp
            config.layout.table_state.round_region = RelativeRegion(0, 0, 1, 1)
            config.layout.table_state.scores = [
                PlayerScoreLayoutConfig("self", RelativeRegion(0, 0, 1, 1))
            ]
            empty = TableStateRecognizer(config).recognize(_pattern(50))
            self.assertIsNone(empty.round.round_wind)
            self.assertIn("No round templates", empty.round.error)
            self.assertIsNone(empty.scores[0].score)

            _write_rgb(Path(tmp) / "round_west_3.png", _pattern(51))
            config.recognition.threshold = 0.99
            low = TableStateRecognizer(config).recognize(_pattern(52))
            self.assertIsNone(low.round.round_wind)
            self.assertIsNotNone(low.round.confidence)
            self.assertIn("below threshold", low.round.error)

    def test_unconfigured_fields_return_neutral_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = default_config()
            config.table_templates_dir = tmp
            result = TableStateRecognizer(config).recognize(np.zeros((10, 10, 3), dtype=np.uint8))
            self.assertIsNone(result.round.error)
            self.assertIsNone(result.self_wind.error)
            self.assertEqual(result.scores, [])

    def test_builds_user_labelled_templates_from_configured_regions(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            frame = np.zeros((40, 80, 3), dtype=np.uint8)
            frame[:20, :20] = _pattern(1, 20, 20)
            frame[:20, 20:40] = _pattern(2, 20, 20)
            frame[:20, 40:60] = _pattern(3, 20, 20)
            screenshot = directory / "source.jpg"
            from PIL import Image
            Image.fromarray(frame).save(screenshot)
            config = default_config()
            config.layout.table_state.round_region = RelativeRegion(0, 0, 0.25, 0.5)
            config.layout.table_state.self_wind_region = RelativeRegion(0.25, 0, 0.25, 0.5)
            config.layout.table_state.scores = [
                PlayerScoreLayoutConfig("self", RelativeRegion(0.5, 0, 0.25, 0.5))
            ]

            saved = build_table_state_templates_from_screenshot(
                screenshot,
                directory / "templates",
                config.layout,
                round_value=("east", 1),
                self_wind="north",
                scores={"self": 25000},
            )

            self.assertEqual(
                {path.name for path in saved},
                {"round_east_1.png", "wind_north.png", "score_25000.png"},
            )

    def test_stability_key_uses_values_but_ignores_table_confidence(self):
        base = RecognitionResult([], None, [], [], 1.0, [])
        first = RecognitionResult(
            **{**base.__dict__, "table_state": TableStateRecognition(
                scores=[PlayerScoreRecognition("self", 25000, 0.81)]
            )}
        )
        jitter = RecognitionResult(
            **{**base.__dict__, "table_state": TableStateRecognition(
                scores=[PlayerScoreRecognition("self", 25000, 0.96)]
            )}
        )
        changed = RecognitionResult(
            **{**base.__dict__, "table_state": TableStateRecognition(
                scores=[PlayerScoreRecognition("self", 24000, 0.96)]
            )}
        )

        self.assertEqual(result_key(first), result_key(jitter))
        self.assertNotEqual(result_key(first), result_key(changed))


if __name__ == "__main__":
    unittest.main()
