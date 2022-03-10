import unittest, logging, os
import numpy as np
from hypothesis import given, strategies as st

import traffic_model as tm


logging.getLogger('traffic_model').setLevel(logging.CRITICAL)


class TestTrafficModel(unittest.TestCase):
    @given(st.text())
    def test_constructor_ic(self, s):
        if s not in tm.TrafficModel._all_initial_conditions.keys():
            m = tm.TrafficModel(ic=s, visual=False)
            self.assertLogs(logger='traffic_model', level='ERROR')

    @given(st.floats(0.1, 0.9))
    def test_uniform_initial_conditions_lot_values(self, x):
        m = tm.TrafficModel(ic='uniform', ic_avg=x, visual=False)
        m.run(50)
        np.testing.assert_approx_equal(np.max(m.u), np.min(m.u))

    def test_uniform_initial_conditions_one_value_long_time(self):
        m = tm.TrafficModel(ic='uniform', visual=False)
        m.run(500)
        np.testing.assert_approx_equal(np.max(m.u), np.min(m.u))


class TestTrafficModelQualitative(unittest.TestCase):
    """Run standard simulations and manually check output plots."""
    def setUp(self):
        self.outpath = "./output_tests"
        if not os.path.exists(self.outpath):
            os.mkdir(self.outpath)
        with open("./output_tests/output_info.txt", 'w') as f:
            f.write("Please inspect the created images for consistency:\n"
                    "sin - with sinusoidal initial conditions expect\n"
                    "      the road to go towards equilibrium.\n"
                    "normal - with normal distribution initially\n"
                    "      expect the distribution to spread out.")

    def test_sin_initial_conditions(self):
        m = tm.TrafficModel(ic='sin')
        m.run(time=50)  # run for 50 minutes
        m.fig.savefig(self.outpath + "/sin.png")

    def test_normal_initial_conditions(self):
        m = tm.TrafficModel(ic='normal')
        m.run(time=50)  # run for 50 minutes
        m.fig.savefig(self.outpath + "/normal.png")


if __name__ == '__main__':
    unittest.main()
