import unittest, logging
from hypothesis import given, strategies as st
import traffic_model as tm


logging.getLogger('traffic_model').setLevel(logging.CRITICAL)

class TestTrafficModel(unittest.TestCase):
    def test_constructor_params(self):
        self.assertRaises(AssertionError, tm.TrafficModel,
                          params={'N': -900})

    @given(st.text())
    def test_constructor_ic(self, s):
        if s not in tm.TrafficModel._all_initial_conditions.keys():
            m = tm.TrafficModel(ic=s, visual=False)
            self.assertLogs(logger='traffic_model', level='ERROR')


if __name__ == '__main__':
    unittest.main()
