import numpy as np
from scipy.integrate import solve_ivp

from model import Model

class DynamicModel(Model):
    """
    A model that can be integrated over time.
    """
    def __init__(
            self,
            world,
            x = np.zeros(1),
            dt: float = 0.1,
    ):
        super().__init__(world, dt)
        self.x = x

    @classmethod
    def dynamics(self, t, x):
        """
        The dynamics of the model. For some time t and state vector x, returns the
        derivative of the state, xdot.
        """
        raise NotImplementedError("expected dynamic model to have dynamics method defined")

    def integrate(self, t):
        """
        Integrate the model to the given time.
        """
        self.update(t)

        sol = solve_ivp(
            self.dynamics,
            [self.t, t],
            self.x
        )
        # Save previous state and time
        self.tp = self.t
        self.xp = np.array(self.x)

        # Save new state and time
        self.t = t
        self.x = sol.y[:,-1]

