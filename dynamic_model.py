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