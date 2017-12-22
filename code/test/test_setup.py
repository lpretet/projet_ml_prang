
import lib.custom_rnns
import unittest

class BasicTestSuite(unittest.TestCase):
    """Basic test cases."""

    def test_absolute_truth_and_meaning(self):
        # self.assertEqual(1,2, "test_setup is alive")
        self.assertEqual(1,1)
        

if __name__ == '__main__':
	unittest.main()