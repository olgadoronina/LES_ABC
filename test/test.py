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
        abc = abc_class.ABC(10000, 64)
        abc.main_loop()
        abc.calc_final_C()
        self.assertAlmostEqual(abc.C_final_joint[0], 0.2225, delta=0.001)

if __name__ == '__main__':
    unittest.main()
