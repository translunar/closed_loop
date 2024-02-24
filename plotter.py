import h5py

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
            self.ax[i].set_ylabel(column)

        self.ax[-1].set_xlabel('t')

    def show(self):
        plt.show()


if __name__ == "__main__":
    p = Plotter('data')
    
    for log in ['low_rate_log', 'high_rate_log']:
        print(f"plotting {log}")
        p.plot(log)

    p.show()