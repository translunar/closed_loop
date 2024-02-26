import numpy as np
from scipy.integrate import solve_ivp

from model import Model

class DynamicModel(Model):
    """
    A model that can be integrated over time. Generally this sets us up to use state-space
    representation, but it's not strictly necessary.

    For any model we derive from DynamicsModel, we define three methods:

    * compute_inputs: setup control inputs for the model (e.g. self.u = self.get_input('force'))
    * dynamics: compute the derivative of the state vector, often in state-space form
    * compute_outputs: setup all outputs for the model (e.g. self.y = C.dot(self.x) + D.dot(self.u))

    """
    def __init__(
            self,
            world,
            name: str,
            x = np.zeros(1),
            u = np.zeros(1),
            dt: float = 0.1,
    ):
        super().__init__(world, name, dt)

        # initialize the state
        self.x = x

        # initialize the control input
        self.u = u

    def dynamics(self, t, x):
        """
        The dynamics of the model. For some time t and state vector x, returns the
        derivative of the state, xdot.
        """
        raise NotImplementedError("expected dynamic model to have dynamics method defined")

    def compute_outputs(self):
        """
        Post-integration step of update().

        The output of the model. Once this method is run, self.y should be defined.
        """
        C = np.ones(1)
        D = np.zeros(1)
        self.y = C.dot(self.x) + D.dot(self.u)
        
    def integrate(self, t):
        """
        Integrate the model to the given time. At the end of this function call,
        self.x should be updated to the state at time t.
        """

        # We should maybe consider putting x and u into a single state vector just
        # for the integration step, but this isn't a current necessity (since u is
        # a discrete rather than continuous input in our system), so consider it
        # forward work.

        sol = solve_ivp(
            self.dynamics,
            [self.t, t],
            self.x,
        )
        # Save previous state and time
        self.tp = self.t
        self.xp = np.array(self.x)

        # Save new state and time
        self.t = t

        # Now pull the state back out of the solution and update the model's state.
        self.x = sol.y[:,-1]

    def update(self, t):
        self.compute_inputs()

        # Update any parts of the state with the inputs before we integrate.
        # Don't update the time in a dynamic state (that happens when we integrate).
        
        self.integrate(t)

        self.compute_outputs()