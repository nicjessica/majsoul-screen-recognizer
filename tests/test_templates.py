import importlib.util
import unittest


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


if __name__ == "__main__":
    unittest.main()
