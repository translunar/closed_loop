

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

    def add_input(self, local_name, model_id_attribute, index=None):
        model_id, attribute = model_id_attribute.split('.')
        if local_name in self.inputs:
            raise ValueError(f"input {local_name} already exists")
        self.inputs[local_name] = (model_id, attribute, index)

    def get_input(self, local_name):
        model_id, attribute, index = self.inputs[local_name]
        value = getattr(self.world.models[model_id], attribute)
        if index is None:
            return value
        else:
            return value[index]

    def update(self, t):
        """
        This is where we do the computation to update the model's outputs from its inputs.
        For the base class, we do nothing at all.
        """
        pass