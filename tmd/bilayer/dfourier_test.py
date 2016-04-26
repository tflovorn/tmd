import unittest
from tmd.bilayer.dfourier import trapezoid_d_regions

class TestRegions(unittest.TestCase):
    def test_regions(self):
        ds = [[0, 0], [0, 0.5], [0, 1], [1, 0], [1, 0.5], [1, 1]]
        regions = trapezoid_d_regions(ds)

        self.assertEqual(len(regions), 2)
        self.assertEqual(regions[0], [0, 1, 3, 4])
        self.assertEqual(regions[1], [1, 2, 4, 5])

if __name__ == "__main__":
    unittest.main()
