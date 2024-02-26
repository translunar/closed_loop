import unittest

from world import World
from discrete_model import DiscreteModel
from pid_controller import PIDController

class InputHarness(DiscreteModel):
    """
    A simple model that provides a constant input to the PID controller.
    """
    def compute_inputs(self):
        self.x = 5.0
        self.u = 0.0

class TestPIDController(unittest.TestCase):

    def setUp(self):
        self.world = World("test")
        self.input = InputHarness(self.world, "input", dt=0.1)
        self.pid = PIDController(self.world, "pid", dt=0.1, setpoint=0.1)

        self.pid.add_input('process', 'input.y')

        self.input.update(0.1) # make sure the input is ready before we start

    def tearDown(self) -> None:
        self.world.finish_logging()

    def test_compute_inputs(self):
        """Should set the input x"""
        self.assertEqual(self.pid.x, 0.0)
        self.pid.compute_inputs()
        self.assertEqual(self.pid.x, 5.0)

    def test_compute_outputs(self):
        """Should set the output y on the second run-through"""
        self.pid.compute_outputs()
        self.assertEqual(self.pid.y, 0.0)
        self.pid.compute_outputs()
        self.assertNotEqual(self.pid.y, 0.0) # because the setpoint is 0.1, it should change

    def test_update(self):
        """update should call compute_inputs and compute_outputs, and update t"""
        self.pid.update(0.1)
        self.assertEqual(self.pid.t, 0.1)
        self.assertEqual(self.pid.y, 0.0)

        self.pid.update(0.2)
        self.assertEqual(self.pid.t, 0.2)
        self.assertNotEqual(self.pid.y, 0.0)

    def test_pid_terms(self):
        """The PID controller terms should update"""
        self.pid.update(0.1)
        self.assertEqual(self.pid.ep, 0.0)
        self.assertEqual(self.pid.e, -4.9)
        self.assertEqual(self.pid.E, 0.0)
        self.assertEqual(self.pid.de, 0.0)

        self.input.y = 4.0
        self.pid.update(0.2)
        self.assertEqual(self.pid.ep, -4.9)
        self.assertEqual(self.pid.e, -3.9)
        self.assertEqual(self.pid.E, -0.39)
        self.assertAlmostEqual(self.pid.de, 10.0)