
class Model:
    """
    Pure base class for all models.
    """

    def __init__(
            self,
            world,
            name: str,
            dt: float = 0.1,
    ):
        if "." in name: # need this for when we setup inputs, which use '.' as a delimiter
            raise ValueError(f"model name {name} contains illegal character '.'")

        self.inputs = {}                
        self.name = name
        self.world = world
        self.t = 0.0
        self.dt = dt # inverse of the model's update frequency
        self.valid = False

        # Add the model to the registry
        self.world.add_model(self)

    @property
    def t_next(self):
        """Get the next time the model needs updating."""
        return self.t + self.dt

    def ready_at(self, t):
        """
        Given the time-step-size of this model, is it time to update it again?
        """
        return t - self.t >= self.dt

    def add_input(self, input_name, model_id_attribute, index=None):
        """
        This method connects the output of another model to this model's input.

        Args:
        * input_name: the name of the input in this model
        * model_id_attribute: a string of the form "model_id.attribute" giving the outputting
            model and its attribute
        """
        model_id, attribute = model_id_attribute.split('.')

        if model_id not in self.world.models:
            raise ValueError(f"model {model_id} does not exist")

        if input_name in self.inputs:
            raise ValueError(f"input {input_name} already exists")
        self.inputs[input_name] = (model_id, attribute, index)

    def get_input(self, input_name):
        """
        Locally-called function which retrieves an input value from the outputting model.
        """
        model_id, attribute, index = self.inputs[input_name]
        value = getattr(self.world.models[model_id], attribute)
        if index is None:
            return value
        else:
            return value[index]
        
    def compute_inputs(self):
        """
        Setup all inputs for the model. When this method is finished, self.u should be
        defined if it's a dynamic model. For a non-state space model,
        """
        raise NotImplementedError("expected model to have setup_inputs method defined")

    def update(self):
        """Update the model's state."""
        raise NotImplementedError("expected model to have update method defined")