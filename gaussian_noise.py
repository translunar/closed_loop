from discrete_model import DiscreteModel

class GaussianNoise(DiscreteModel):
    """
    A scalar, zero-mean, first-order Gaussian noise model.
    """

    def __init__(
            self, 
            world,
            name: str,
            dt: float = 0.1,
            sigma: float = 1.0,
    ):
        super().__init__(world, name, dt=dt)
        
        # set the noise parameters
        self.sigma = sigma
        self.generate() # first initialization of mu

    def generate(self):
        self.mu = self.world.rng.normal(0.0, self.sigma)

    def compute_inputs(self):
        self.x = self.get_input('process')
        self.generate()

    def compute_outputs(self):
        # add noise to the process
        self.y = self.x + self.mu



