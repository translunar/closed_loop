from dynamic_model import DynamicModel, MassSpringDamper
from discrete_model import PIDController

BIGGEST_STEP = 1000000.0

class World:
    """
    This is the registry of all models in the world. It is responsible for
    making sure everything is synchronized.
    """
    def __init__(self):
        self.models = {}
        self.order = []
        self.t = 0.0

    def add_model(self, name, model):
        """
        Add a model to the world (do this in order of model priority,
        so dynamic models first generally).
        """
        self.models[name] = model
        self.order.append(name)

    def cycle(self):
        """
        Cycle through the models in the world and update them. In practice,
        this picks the model with the earliest next update time and updates
        all models whose frequencies make them eligible at that time.
        """
        # Go through each of the models and figure out the earliest possible
        # dt to which an update is needed.
        dt = BIGGEST_STEP
        for name in self.order:
            t_next = self.models[name].t_next
            print(f"model {name} wants {t_next}")
            model_delta = t_next - self.t
            if model_delta > 0.0:
                dt = min(dt, model_delta)

        # Time of validity
        t = self.t + dt
        print(f"Proposed time {t}")

        # Perform updates and integrations
        for name in self.order:
            if self.models[name].ready_at(t):
                print(f"updating model {name}")
                if isinstance(self.models[name], DynamicModel):
                    # Integrate also calls update
                    self.models[name].integrate(t)
                else:
                    self.models[name].update(t)

        # Update the world clock
        self.t = t


if __name__ == "__main__":
    # Create a world
    world = World()

    # Create models
    msd = MassSpringDamper(world, m=1.0, k=1.0, b=1.0, dt=0.01)
    pid = PIDController(world, dt=0.1)

    # Add the models to the world in the order they should be updated
    world.add_model('mass_spring_damper', msd)
    world.add_model('pid', pid)

    # Define the connections between the model inputs and outputs
    msd.add_input('force', 'pid', 'u')
    pid.add_input('process', 'mass_spring_damper', 'x')

    ts = []
    xs = []
    xds = []
    fs = []
    # Cycle the world
    while world.t < 10.0:
        world.cycle()
        ts.append(msd.t)
        xs.append(msd.x[0])
        xds.append(msd.x[1])
        fs.append(msd.x[2])
        #print(f"{msd.t}: {msd.x}")

    import matplotlib.pyplot as plt
    # Plot t against x and xds on two axes with the same t axis
    fig = plt.figure()
    axes = fig.subplots(3, 1)
    axes[0].plot(ts, xs)
    axes[1].plot(ts, xds)
    axes[2].plot(ts, fs)
    plt.show()

