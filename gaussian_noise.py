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

    def update(self, t):
        process = self.get_input('process')

        self.t = t # update the time

        # add noise to the process
        self.mu = self.world.rng.normal(0.0, self.sigma)
        self.u = process + self.mu

