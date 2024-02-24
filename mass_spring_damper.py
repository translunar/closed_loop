import numpy as np

from dynamic_model import DynamicModel

class MassSpringDamper(DynamicModel):
    def __init__(
            self,
            world,
            x = np.zeros(3),
            m: float = 1.0,
            k: float = 1.0,
            b: float = 1.0,
            dt: float = 0.01):
        super().__init__(world, dt=dt)

        # set the model parameters
        self.m = m
        self.k = k
        self.b = b

        # initialize the state
        self.x = x

    def dynamics(self, t, x):
        """
        The dynamics of the model. For some time t and state vector x, returns the
        derivative of the state, xdot.

        Note that in this particular model, it's independent of t. We still
        need to include it in the signature because the solver expects it.
        """
        xddot = (x[2] - self.k * x[0] - self.b * x[1]) / self.m
        return np.array([x[1], xddot, 0.0])

    def update(self, t):
        force = self.get_input('force')

        # Update any parts of the state with the inputs before we integrate.
        # Don't update the time in a dynamic state (that happens when we integrate).
        self.x[2] = force