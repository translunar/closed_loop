import h5py
import sys
import numpy as np

import matplotlib.pyplot as plt

class Plotter:
    """
    Loads logged hdf5 data for plotting.
    """
    def __init__(self, file_id):
        self.f = h5py.File(f"{file_id}.h5", 'r')

    def plot(self, group):
        """
        Plot a single logger.
        """
        # Get columns for group
        if group not in self.f:
            raise ValueError(f"no such group {group}")
        
        # Get the display names out of the group attributes
        labels = {}
        for key, value in self.f[group].attrs.items():
            labels[key] = value
        
        columns = list(self.f[group].keys())
        columns.remove('t')

        if len(columns) == 0:
            raise IOError(f"dataset {group} contained no columns except time")
        
        # Create subplots for each column except t
        self.fig, self.ax = plt.subplots(len(columns), 1)
        self.fig.suptitle(group)

        t = self.f[group]['t'][:]
        for i, column in enumerate(columns):
            if column == 't':
                continue
            self.ax[i].plot(t, self.f[group][column][:])

            # Optionally set the label (if a label is given)
            if column in labels:
                self.ax[i].set_ylabel(labels[column])
            else:
                self.ax[i].set_ylabel(column)

        self.ax[-1].set_xlabel('t')

    def show(self):
        plt.show()

    def analyze(
            self,
            group='low_rate_log',
            error_column='e',
            setpoint_column='setpoint',
            settling_threshold=0.05,
        ):
        """
        Compute and display performance metrics on an error signal.
        
        This method is a bit of a hack. It would be nice 
        """
        e = self.f[group][error_column][:]  # Assuming 'e' is the error signal
        setpoint = self.f[group][setpoint_column][:]  # Assuming 'setpoint' is the setpoint signal
        t = self.f[group]['t'][:]
        
        # Calculate RMSE from error
        rmse = np.sqrt(np.mean(e ** 2))
        
        # Settling time: find the time it takes for the error to fall to within settling_threshold
        # percent of the setpoint.
        threshold = np.max(np.abs(setpoint)) * settling_threshold
        within_threshold_indices = np.abs(e) <= threshold
        # Start at the end of within_threshold_indices and work backwards until we find
        # one that isn't True
        for index in range(len(within_threshold_indices) - 1, -1, -1):
            if not within_threshold_indices[index]:
                break
            else:
                threshold_index = index

        settling_time = t[threshold_index] if len(within_threshold_indices) > 0 else None

        # Peak response time: find first peak in error signal
        peak_response_time = t[np.argmax(np.abs(e))]
        
        # Overshoot: find percentage of time points where e is less than zero
        overshoot = np.sum(e < 0) / len(e)
        
        # Print the metrics
        print(f"RMSE: {rmse}")
        print(f"Peak Response Time: {peak_response_time} time units")
        print(f"Settling Time: {settling_time} time units (within {100 * settling_threshold}% of setpoint)")
        print(f"Overshoot: {100 * overshoot}%")


if __name__ == "__main__":
    p = Plotter('data')

    for log in ['low_rate_log', 'high_rate_log']:
        print(f"plotting {log}")
        p.plot(log)

    # Can also print out a few performance metrics for the PID controller.
    p.analyze('low_rate_log', 'e')

    p.show()
