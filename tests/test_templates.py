import importlib.util
import tempfile
import unittest
from pathlib import Path


class TemplateMatchTests(unittest.TestCase):
    @unittest.skipIf(importlib.util.find_spec("cv2") is None, "opencv-python is not installed")
    def test_match_score_tolerates_small_crop_offset(self):
        import cv2
        import numpy as np

        template = np.zeros((120, 80, 3), dtype=np.uint8)
        template[:, :] = (20, 30, 40)
        cv2.rectangle(template, (8, 6), (72, 114), (230, 230, 230), -1)
        cv2.circle(template, (40, 60), 18, (30, 80, 210), -1)

        tile = np.zeros((128, 88, 3), dtype=np.uint8)
        tile[:, :] = (20, 30, 40)
        tile[5:125, 6:86] = cv2.resize(template, (80, 120))

        from recognizer.templates import match_score

        self.assertGreater(match_score(tile, template), 0.75)

    @unittest.skipIf(importlib.util.find_spec("cv2") is None, "opencv-python is not installed")
    def test_match_score_tolerates_extra_background_on_one_template_edge(self):
        import cv2
        import numpy as np

        tile = np.full((120, 80, 3), (225, 225, 225), dtype=np.uint8)
        cv2.circle(tile, (24, 36), 13, (35, 70, 210), 4)
        cv2.circle(tile, (56, 76), 16, (210, 60, 35), 5)
        cv2.line(tile, (10, 104), (70, 96), (20, 130, 60), 4)
        template = np.full((134, 80, 3), (25, 40, 45), dtype=np.uint8)
        template[:120] = tile

        from recognizer.templates import match_score

        self.assertGreater(match_score(tile, template), 0.78)

    @unittest.skipIf(importlib.util.find_spec("cv2") is None, "opencv-python is not installed")
    def test_edge_tolerance_does_not_match_an_obviously_different_pattern(self):
        import cv2
        import numpy as np

        tile = np.full((120, 80, 3), (225, 225, 225), dtype=np.uint8)
        cv2.circle(tile, (40, 60), 24, (35, 70, 210), 6)
        template = np.full((134, 80, 3), (225, 225, 225), dtype=np.uint8)
        cv2.rectangle(template, (12, 18), (68, 102), (210, 60, 35), 6)
        template[120:] = (25, 40, 45)

        from recognizer.templates import match_score

        self.assertLess(match_score(tile, template), 0.78)

    @unittest.skipIf(importlib.util.find_spec("cv2") is None, "opencv-python is not installed")
    def test_edge_tolerance_rejects_a_constant_trimmed_core(self):
        import cv2
        import numpy as np

        tile = np.full((120, 80, 3), (225, 225, 225), dtype=np.uint8)
        cv2.circle(tile, (40, 60), 24, (35, 70, 210), 6)
        template = np.full((134, 80, 3), (221, 222, 223), dtype=np.uint8)
        template[:14] = (25, 80, 40)

        from recognizer.templates import match_score

        self.assertLess(match_score(tile, template), 0.78)

    @unittest.skipIf(importlib.util.find_spec("cv2") is None, "opencv-python is not installed")
    def test_match_candidates_are_sorted_and_match_returns_first(self):
        import cv2
        import numpy as np

        from recognizer.templates import TemplateLibrary

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            exact = np.zeros((20, 20, 3), dtype=np.uint8)
            exact[3:17, 5:15] = (40, 180, 230)
            different = np.zeros((20, 20, 3), dtype=np.uint8)
            different[2:8, 2:8] = (230, 40, 60)
            cv2.imwrite(str(directory / "exact.png"), exact)
            cv2.imwrite(str(directory / "different.png"), different)
            library = TemplateLibrary(directory)
            tile_rgb = cv2.cvtColor(exact, cv2.COLOR_BGR2RGB)

            candidates = library.match_candidates(tile_rgb, limit=2)

            self.assertEqual(len(candidates), 2)
            self.assertGreaterEqual(candidates[0].score, candidates[1].score)
            self.assertEqual(candidates[0].name, "exact")
            self.assertEqual(library.match(tile_rgb), candidates[0])
            self.assertEqual(library.match_candidates(tile_rgb, limit=0), [])

    @unittest.skipIf(importlib.util.find_spec("cv2") is None, "opencv-python is not installed")
    def test_library_batch_cache_preserves_public_match_scores_and_ranking(self):
        import cv2
        import numpy as np

        from recognizer.templates import TemplateLibrary, match_score

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            first = np.full((120, 80, 3), (225, 225, 225), dtype=np.uint8)
            cv2.circle(first, (28, 42), 16, (40, 110, 230), -1)
            second = np.full((134, 80, 3), (225, 225, 225), dtype=np.uint8)
            second[:120] = first
            third = np.full((110, 90, 3), (225, 225, 225), dtype=np.uint8)
            cv2.rectangle(third, (20, 25), (68, 82), (230, 60, 40), -1)
            cv2.imwrite(str(directory / "first.png"), first)
            cv2.imwrite(str(directory / "second.png"), second)
            cv2.imwrite(str(directory / "third.png"), third)
            library = TemplateLibrary(directory)
            tile_rgb = cv2.cvtColor(first, cv2.COLOR_BGR2RGB)
            tile_bgr = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2BGR)

            candidates = library.match_candidates(tile_rgb, limit=3)
            expected = sorted(
                (
                    (name, match_score(tile_bgr, template))
                    for name, template in library.templates.items()
                ),
                key=lambda item: (-item[1], item[0]),
            )

            self.assertEqual([item.name for item in candidates], [name for name, _ in expected])
            for candidate, (_, score) in zip(candidates, expected, strict=True):
                self.assertAlmostEqual(candidate.score, score, places=7)


if __name__ == "__main__":
    unittest.main()
