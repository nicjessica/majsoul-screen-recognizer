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


if __name__ == "__main__":
    unittest.main()
