from model import Model

class DiscreteModel(Model):
    """
    A model that performs a discrete computation.
    """

    def __init__(
            self, 
            world, 
            dt: float = 0.1
    ):
        super().__init__(world, dt)
        
        # initialize the outputs
        self.u = 0.0

    def update(self, t):
        process = self.get_input('process')[0]

        self.t = t # update the time
        self.u = process + 1.0 # do a very basic calculation and update output

class PIDController(DiscreteModel):
    """
    A discrete model that implements a PID controller.
    """

    def __init__(
            self, 
            world, 
            kp: float = 1.0,
            ki: float = 1.0,
            kd: float = 1.0,
            setpoint: float = 1.0,
            dt: float = 0.1,
    ):
        super().__init__(world, dt)
        
        # set the controller parameters
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint

        # initialize the error, its integral, and its derivative.
        # We don't have to save the derivative as a state variable, but it's useful
        # when we inevitably want to debug our PID controller without restarting the
        # whole simulation (particularly if we're doing an end-to-end mission).
        self.E = 0.0
        self.ep = 0.0 # previous error
        self.e = 0.0
        self.de = 0.0
        self.valid = False

    def update(self, t):
        process = self.get_input('process')[0]

        if self.valid: # This prevents weird stuff from happening with the derivative term.
            # calculate the error
            e = self.setpoint - process

            # calculate the integral (trapezoidal approximation)
            E = self.E + e * self.dt

            # calculate the derivative
            de = (self.e - self.ep) / self.dt

            # calculate the control signal and save the temporary state
            self.e = e
            self.E = E
            self.de = de
            self.u = self.kp * e + self.ki * E + self.kd * de
        else:
            self.u = 0.0 # zero control signal until we're valid

            # Once ep gets set at the end of this function, we're valid
            self.valid = True

        # save for the next time step
        self.t = t
        self.ep = self.e