import unittest

import numpy as np

from recognizer.config import MeldConfig, MeldTileSlotConfig, RelativeRegion
from recognizer.recognizer import crop_meld_slots


def _slot(x, y, width, height, orientation="upright", stack_level=0):
    return MeldTileSlotConfig(
        region=RelativeRegion(x=x, y=y, width=width, height=height),
        orientation=orientation,
        stack_level=stack_level,
    )


class MeldStructureCropTests(unittest.TestCase):
    def setUp(self):
        rows, columns = np.indices((10, 20))
        self.region = np.stack(
            (rows, columns, rows * 20 + columns), axis=2
        ).astype(np.int16)

    def test_group_slots_preserve_gaps_and_independent_widths(self):
        melds = [
            MeldConfig(kind="chi", tiles=[_slot(0.00, 0.0, 0.20, 1.0)]),
            MeldConfig(
                kind="pon",
                tiles=[
                    _slot(0.35, 0.0, 0.15, 1.0),
                    _slot(0.70, 0.0, 0.30, 1.0),
                ],
            ),
        ]

        groups = crop_meld_slots(self.region, melds)

        self.assertEqual([len(group) for group in groups], [1, 2])
        np.testing.assert_array_equal(groups[0][0], self.region[:, 0:4])
        np.testing.assert_array_equal(groups[1][0], self.region[:, 7:10])
        np.testing.assert_array_equal(groups[1][1], self.region[:, 14:20])

    def test_clockwise_slot_is_rotated_counterclockwise_to_upright(self):
        melds = [
            MeldConfig(
                kind="pon",
                tiles=[_slot(0.10, 0.20, 0.30, 0.40, "rotated_cw")],
            )
        ]
        source = self.region[2:6, 2:8]

        tile = crop_meld_slots(self.region, melds)[0][0]

        np.testing.assert_array_equal(tile, np.rot90(source, 1))
        self.assertTrue(tile.flags.c_contiguous)

    def test_counterclockwise_slot_is_rotated_clockwise_to_upright(self):
        melds = [
            MeldConfig(
                kind="pon",
                tiles=[_slot(0.10, 0.20, 0.30, 0.40, "rotated_ccw")],
            )
        ]
        source = self.region[2:6, 2:8]

        tile = crop_meld_slots(self.region, melds)[0][0]

        np.testing.assert_array_equal(tile, np.rot90(source, -1))
        self.assertTrue(tile.flags.c_contiguous)

    def test_overlapping_slots_are_cropped_independently(self):
        melds = [
            MeldConfig(
                kind="kakan",
                tiles=[
                    _slot(0.10, 0.10, 0.50, 0.80, stack_level=0),
                    _slot(0.40, 0.20, 0.50, 0.60, stack_level=1),
                ],
            )
        ]

        tiles = crop_meld_slots(self.region, melds)[0]

        np.testing.assert_array_equal(tiles[0], self.region[1:9, 2:12])
        np.testing.assert_array_equal(tiles[1], self.region[2:8, 8:18])
        np.testing.assert_array_equal(tiles[0][1:7, 6:10], tiles[1][:, :4])


if __name__ == "__main__":
    unittest.main()
