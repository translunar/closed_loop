

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
        """Get the next time the model needs updating."""
        return self.t + self.dt

    def ready_at(self, t):
        """
        Given the time-step-size of this model, is it time to update it again?
        """
        return t - self.t >= self.dt

    def add_input(self, local_name, model_id_attribute, index=None):
        """
        This method connects the output of another model to this model's input.

        Args:
        * local_name: the name of the input in this model
        * model_id_attribute: a string of the form "model_id.attribute" giving the outputting
            model and its attribute
        """
        model_id, attribute = model_id_attribute.split('.')
        if local_name in self.inputs:
            raise ValueError(f"input {local_name} already exists")
        self.inputs[local_name] = (model_id, attribute, index)

    def get_input(self, local_name):
        """
        Locally-called function which retrieves an input value from the outputting model.
        """
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
        raise NotImplementedError("attempted to instantiate pure base class or expected model to have update() method defined")