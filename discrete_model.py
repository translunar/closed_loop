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
            name: str,
            dt: float = 0.1
    ):
        super().__init__(world, name, dt=dt)
        
        # initialize the outputs
        self.u = 0.0
        self.x = 0.0
        self.y = 0.0

    def compute_inputs(self):
        print("called compute_inputs on DiscreteModel")
        self.x = self.get_input('process')
        self.u = 1.0

    def compute_outputs(self):
        self.y = self.x + self.u # do a very basic calculation and update output

    def update(self, t):
        self.compute_inputs()
        self.t = t
        self.compute_outputs()
