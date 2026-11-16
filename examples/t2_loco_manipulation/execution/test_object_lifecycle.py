import unittest
import numpy as np
from object_lifecycle import hold_lifecycle


class ObjectLifecycle(unittest.TestCase):
    def test_withdrawal_after_release_is_not_slip(self):
        t = np.arange(10.0)
        g = [0, 255, 255, 255, 255, 0, 0, 0, 0, 0]
        r = np.zeros((10, 3))
        r[5:, 0] = np.arange(5)
        c, o, mask, drift = hold_lifecycle(t, g, r)
        self.assertEqual(c.tolist(), [1])
        self.assertEqual(o.tolist(), [5])
        self.assertEqual(drift, 0.0)

    def test_intervening_drop_is_measured_until_last_release(self):
        t = np.arange(10.0)
        g = [0, 255, 255, 0, 0, 255, 255, 255, 0, 0]
        r = np.zeros((10, 3))
        r[3:7, 0] = 0.7
        c, o, mask, drift = hold_lifecycle(t, g, r)
        self.assertEqual(c.tolist(), [1, 5])
        self.assertEqual(o.tolist(), [3, 8])
        self.assertAlmostEqual(drift, 0.7)


if __name__ == "__main__":
    unittest.main()
