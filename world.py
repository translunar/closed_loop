import h5py
import numpy.random as npr
import sys

from mass_spring_damper import MassSpringDamper
from pid_controller import PIDController
from gaussian_noise import GaussianNoise
from logger import Logger

BIGGEST_STEP = 1000000.0

class World:
    """
    This is the registry of all models in the world. It is responsible for
    making sure everything is synchronized and stepping all of the different 
    processes.
    """
    def __init__(self, basename: str = "data", rng=npr.default_rng()):
        self.models = {}
        self.order = []
        self.t = 0.0

        self.f = h5py.File(f'{basename}.h5', 'w')
        self.logging_ready = False

        # Only needed if we use the noise model:
        self.rng = rng

    def add_model(self, model):
        """
        Add a model to the world (do this in order of model priority,
        so dynamic models first generally).
        """
        if model.name in self.models:
            raise KeyError(f"model {model.name} already exists")

        self.models[model.name] = model
        self.order.append(model.name)

    def setup_logging(self):
        """
        Create datasets for each of the loggers.
        """
        if self.logging_ready:
            raise ValueError("logging already set up, please only call once")
        
        for model_id in self.models:
            if isinstance(self.models[model_id], Logger):
                # Create a dataset with the same model_id as the logger
                self.models[model_id].create_log(model_id)

        self.logging_ready = True

    def finish_logging(self):
        """
        Close the file and clean up.
        """
        for model_id in self.models:
            if isinstance(self.models[model_id], Logger):
                self.models[model_id].finalize()
        self.f.close()

    def cycle(self):
        """
        Cycle through the models in the world and update them. In practice,
        this picks the model with the earliest next update time and updates
        all models whose frequencies make them eligible at that time.
        """

        # Calculate the next update time for each model individually.
        min_dt = float('inf')
        for model in self.models.values():
            if model.t_next < min_dt:
                min_dt = min(min_dt, model.t_next - self.t)

        if min_dt == float('inf'):
            raise ValueError("something is wrong with model time")

        # Calculate the proposed update time.
        proposed_t = self.t + min_dt

        # Update models that are ready.
        for name, model in self.models.items():
            if proposed_t >= model.t_next:
                model.update(proposed_t)

        # Update the world clock to the proposed time.
        self.t = proposed_t


if __name__ == "__main__":
    SEED = 0 # makes this run reproducible (which it will be anyway, if we
             # don't use the noise model)
    
    # Get the desired position of the mass from the command line
    if len(sys.argv) != 3:
        raise SyntaxError("Usage: python world.py <duration> <desired_x>")
    duration = float(sys.argv[1])
    desired_x = float(sys.argv[2])
    if duration <= 0.0:
        raise ValueError("duration must be positive")

    # Create a world (our "prime mover")
    world = World(rng=npr.default_rng(seed=SEED))

    # Add the models to the world in the order they should be updated
    msd = MassSpringDamper(world, 'mass_spring_damper', m=4.0, k=2.0, b=3.0, dt=0.01)
    sensor = GaussianNoise(world, 'sensor', sigma=0.01, dt=0.1)
    pid = PIDController(world, 'pid', dt=0.1, kp=40.0, ki=2.0, kd=4.0, setpoint=desired_x)

    # Now add the loggers
    high_rate_log = Logger(world, 'high_rate_log', dt=0.01)
    low_rate_log  = Logger(world, 'low_rate_log', dt=0.1)

    # Define the connections between the model inputs and outputs
    msd.add_input('force', 'pid.y')
    sensor.add_input('process', 'mass_spring_damper.y', 0)
    pid.add_input('process', 'sensor.y')

    # Now define logging inputs
    high_rate_log.add_input('position', 'mass_spring_damper.y', 0)
    high_rate_log.add_input('velocity', 'mass_spring_damper.x', 1)
    high_rate_log.add_input('force', 'mass_spring_damper.u')
    low_rate_log.add_input('E', 'pid.E', label="E(t)")
    low_rate_log.add_input('e', 'pid.e', label="e(t)")
    low_rate_log.add_input('de', 'pid.de', label="de(t)/dt")
    low_rate_log.add_input('pid_y', 'pid.y', label="PID output")
    low_rate_log.add_input('mu', 'sensor.mu', label="sensor noise")
    low_rate_log.add_input('sensor_y', 'sensor.y', label="sensor output")
    low_rate_log.add_input('setpoint', 'pid.setpoint')

    # This should be the final call in the setup phase.
    world.setup_logging()

    # Cycle the world
    while world.t < duration:
        world.cycle()

    world.finish_logging()
