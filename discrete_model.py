from model import Model

class DiscreteModel(Model):
    """
    A model that performs a discrete computation. This is mostly a base
    class from which we can derive other discrete models, but you can
    instantiate it if you really want.
    """

    def __init__(
            self, 
            world, 
            dt: float = 0.1
    ):
        super().__init__(world, dt=dt)
        
        # initialize the outputs
        self.u = 0.0

    def update(self, t):
        process = self.get_input('process')[0]

        self.t = t # update the time
        self.u = process + 1.0 # do a very basic calculation and update output
