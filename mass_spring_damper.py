import numpy as np

from dynamic_model import DynamicModel

class MassSpringDamper(DynamicModel):
    """
    Model of a mass-spring-damper system.

    It does not know how close it is to the fixed surface, so does not protect against
    collisions. It does not include any units, so you can use whatever units you like.
    """
    def __init__(
            self,
            world,
            x = np.zeros(2),
            m: float = 1.0,
            k: float = 1.0,
            b: float = 1.0,
            dt: float = 0.01
    ):
        """
        Args:
            world: the registry of models
            x: the initial state of the model
            m: the mass of the object
            k: the spring constant
            b: the damping constant
            dt: the time step for integration and updating
        """
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
        xddot = (self.force - self.k * x[0] - self.b * x[1]) / self.m
        return np.array([x[1], xddot])

    def update(self, t):
        self.force = self.get_input('force') # This comes from the PID controller (and the connection
                                        # is defined in the world setup)

        # Update any parts of the state with the inputs before we integrate.
        # Don't update the time in a dynamic state (that happens when we integrate).
        