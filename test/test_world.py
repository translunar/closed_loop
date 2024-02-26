import unittest

from world import World
from mass_spring_damper import MassSpringDamper

from test.test_pid_controller import InputHarness

class TestWorld(unittest.TestCase):
    def setUp(self):
        self.world = World("test")
        self.input = InputHarness(self.world, "input", dt=0.1)
        self.msd1 = MassSpringDamper(self.world, "msd", dt=0.1, m=2.0, k=4.0, b=6.0)

    def tearDown(self) -> None:
        return self.world.finish_logging()
        
    def test_add_model(self):
        self.assertIn("input", self.world.models)
        self.assertIn("msd", self.world.models)

        # All models should be in the order list
        self.assertEqual(self.world.order, ["input", "msd"])

        with self.assertRaises(KeyError):
            msd2 = MassSpringDamper(self.world, "msd", dt=0.1, m=4.0, k=4.0, b=6.0)

    def test_add_input(self):
        # this really belongs on the model base test, but it's a pure base class,
        # and I'm in a hurry, so I'm putting it here. In a real project, I'd have
        # a separate TestModel class.
        with self.assertRaises(ValueError):
            self.msd1.add_input('force', 'invalid.input.y') # too many dots

        with self.assertRaises(KeyError):
            self.msd1.add_input('force', 'output.y') # output doesn't exist