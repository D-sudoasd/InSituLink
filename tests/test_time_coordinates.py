"""Clock coordinate changes preserve the public offset convention."""

import unittest

from insitulink.time_coordinates import (
    clock_offset_local_s, clock_offset_public_s, time_relative_s,
)


class TimeCoordinatesTest(unittest.TestCase):
    def test_decimal_epoch_subtraction_retains_nanoseconds(self):
        self.assertEqual(time_relative_s("1767225600.0000002", "1767225600.0000001"), 1e-7)

    def test_nonzero_relative_origins_preserve_shared_and_offset_clocks(self):
        # Source public time = frame public time + dt, regardless of first row.
        for dt in (0, 2.5, -3):
            local = clock_offset_local_s(dt, "10", "18")
            self.assertEqual(local, dt - 8)
            self.assertEqual(clock_offset_public_s(local, "10", "18"), dt)
            self.assertEqual(3 + local, (10 + 3 + dt) - 18)

    def test_fractional_epoch_origin_difference_is_not_rounded_first(self):
        offset = clock_offset_local_s("0.0000003", "1767225600.0000001", "1767225600.0000002")
        self.assertEqual(offset, 2e-7)
        self.assertEqual(clock_offset_public_s(offset, "1767225600.0000001", "1767225600.0000002"), 3e-7)


if __name__ == "__main__":
    unittest.main()
