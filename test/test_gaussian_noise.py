import unittest

from world import World
from gaussian_noise import GaussianNoise

from test.test_pid_controller import InputHarness

class TestGaussianNoise(unittest.TestCase):

    def setUp(self):
        self.world = World("test")
        self.input = InputHarness(self.world, "input", dt=0.1)
        self.noise = GaussianNoise(self.world, "noise_model", 1.0, 0.1)

        self.noise.add_input('process', 'input.y')

        self.input.update(0.1)

    def tearDown(self) -> None:
        self.world.finish_logging()

    def test_compute_inputs_and_outputs(self):
        self.noise.compute_inputs()
        self.assertEqual(self.noise.x, 5.0)
        self.noise.compute_outputs()
        self.assertNotEqual(self.noise.y, 5.0)

if __name__ == '__main__':
    unittest.main()
