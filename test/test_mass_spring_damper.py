import numpy as np

import unittest

from world import World
from mass_spring_damper import MassSpringDamper

from test.test_pid_controller import InputHarness # input.y = 5

class TestMassSpringDamper(unittest.TestCase):

    def setUp(self):
        self.k = 4.0
        self.m = 2.0
        self.b = 6.0

        self.world = World("test")
        self.input = InputHarness(self.world, "input", dt=0.1)
        self.msd = MassSpringDamper(self.world, "msd", dt=0.1,
                                    m=self.m, k=self.k, b=self.b)

        self.msd.add_input('force', 'input.y')

        self.input.update(0.1)

    def tearDown(self) -> None:
        self.world.finish_logging()

    def test_compute_inputs(self):
        first_x = self.msd.x
        self.msd.compute_inputs()
        np.testing.assert_equal(first_x, self.msd.x) # x should not change, it's not an input
        self.assertEqual(self.msd.u, 5.0) # u should get set to input.y (which is 5)

    @unittest.skip("needs dynamics tests")
    def test_dynamics(self):
        # ... TODO: finish this test
        pass

    @unittest.skip("needs compute_outputs test")
    def test_compute_outputs(self):
        # ... TODO: finish this test
        pass