import unittest
from params import *
import data
import abc_class
import global_var as g

class TestABC(unittest.TestCase):
    def test_one_parameter_abc(self):

        LES_data = np.load(loadfile_LES)
        TEST_data = np.load(loadfile_TEST)
        g.LES = data.Data(LES_data, LES_delta)
        g.TEST = data.Data(TEST_data, TEST_delta)
        abc = abc_class.ABC(1000, 64)
        abc.main_loop()
        C = abc.calc_final_C()[0]
        self.assertAlmostEqual(C, 0.215, delta=0.001)

if __name__ == '__main__':
    unittest.main()
