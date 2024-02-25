from discrete_model import DiscreteModel

class GaussianNoise(DiscreteModel):
    """
    A scalar, zero-mean, first-order Gaussian noise model.
    """

    def __init__(
            self, 
            world,
            sigma: float = 1.0,
            dt: float = 0.1,
    ):
        super().__init__(world, dt=dt)
        
        # set the noise parameters
        self.sigma = sigma

    def compute_inputs(self):
        self.y = self.get_input('process')
        self.mu = self.world.rng.normal(0.0, self.sigma)

    def compute_outputs(self):
        # add noise to the process
        self.u = self.y + self.mu



