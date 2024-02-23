

class Model:
    """
    Pure base class for all models.
    """

    def __init__(
            self,
            world,
            dt: float = 0.1,
    ):
        self.inputs = {}
        self.world = world
        self.t = 0.0
        self.dt = dt # inverse of the model's update frequency
        self.valid = False

    @property
    def t_next(self):
        return self.t + self.dt

    def ready_at(self, t):
        return t - self.t >= self.dt

    def add_input(self, local_name, world_key, attribute):
        self.inputs[local_name] = (world_key, attribute)

    def get_input(self, local_name):
        world_key, attribute = self.inputs[local_name]
        return getattr(self.world.models[world_key], attribute)

    def update(self, t):
        """
        This is where we do the computation to update the model's outputs from its inputs.
        For the base class, we do nothing at all.
        """
        pass