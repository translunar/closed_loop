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
            name: str,
            x = np.zeros(2),
            u = 0.0,
            m: float = 1.0,
            k: float = 1.0,
            b: float = 1.0,
            dt: float = 0.01
    ):
        """
        Args:
            world: the registry of models
            name: the name of the model
            x: the initial state of the model
            u: the initial control input
            m: the mass of the object
            k: the spring constant
            b: the damping constant
            dt: the time step for integration and updating
        """
        super().__init__(world, name, x, u, dt)

        # set the model parameters
        self.m = m
        self.k = k
        self.b = b

    def compute_inputs(self):
        """
        Pre-integration step of update().
        """
        # This comes from the PID controller (and the connection
        # is defined in the world setup)
        self.u = self.get_input('force')
        print(f"msd: compute_inputs set u to {self.u}")

    def dynamics(self, t, x):
        """
        The dynamics of the model. For some time t and state vector x, returns the
        derivative of the state, xdot. We'll use state-space form for this model,
        so this method returns the state derivative.

        Note that in this particular model, it's independent of t. We still
        need to include it in the signature because the solver expects it.
        """
        # Define dynamics matrix
        A = np.array([[0.0, 1.0],
                      [-self.k / self.m, -self.b / self.m]])
        # Define control matrix
        B = np.array([0.0, 1.0 / self.m])
        u = self.u
        
        return A.dot(x) + B.dot(u)
    
    def compute_outputs(self):
        """
        Post-integration step of update().

        Ordinarily, this is where we define y = Cx + Du. In this case, our output is
        just our state, so C = 1 and D = 0. We've actually defined it thusly in our base
        class, but let's do it again here for clarity.
        """
        print(f"msd: compute_outputs set y to {self.x}")
        self.y = np.array(self.x)
    

