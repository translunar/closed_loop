from discrete_model import DiscreteModel

class PIDController(DiscreteModel):
    """
    A discrete model that implements a PID controller.
    """

    def __init__(
            self, 
            world, 
            name: str,
            kp: float = 1.0,
            ki: float = 1.0,
            kd: float = 1.0,
            setpoint: float = 1.0,
            dt: float = 0.1,
    ):
        super().__init__(world, name, dt=dt)
        
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

    def compute_outputs(self):
        self.ep = self.e
        self.e = self.setpoint - self.x # calculate the error
        self.y = 0.0 # always zero until we're valid

        if self.valid: # This prevents weird stuff from happening with the derivative term.
            
            # calculate the integral (trapezoidal approximation)
            self.E += self.e * self.dt

            # calculate the derivative
            self.de = (self.e - self.ep) / self.dt

            # calculate the control signal and save the temporary state            
            self.y = self.kp * self.e + self.ki * self.E + self.kd * self.de
        else:
            # Once we've run this function once, we're valid.
            self.valid = True

