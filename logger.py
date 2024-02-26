from typing import Dict
import numpy as np
from model import Model

import re

class Logger(Model):
    """
    This is a type of model that logs data from other models. It uses hdf5, and buffers
    data to avoid writing to disk too often.

    The important functions here are the constructor and the update() method, as with
    most of the other models I've created.
    """
    def __init__(self, world, name: str, buffer_size: int = 10000, dt: float = 0.1):
        super().__init__(world, name, dt=dt)

        self.buffer_size = buffer_size
        self.buffer = {}
        self.index = 0  # Index in the current buffer
        self.total_logged = 0  # Total number of points logged
        self.labels = {} # Maps input names to display names

    def add_input(self, input_name, model_id_attribute, index=None, label=None):
        """
        This method adds an input to the logger, and works just like Model.add_input,
        except that it also stores the display name in an hdf5 attribute.
        """
        # Check that input_name matches the hdf5 column name regex:
        if not re.match(r'^[a-zA-Z0-9_\-]+$', input_name):
            raise ValueError(f"input name {input_name} is not a valid hdf5 column name")
        
        super().add_input(input_name, model_id_attribute, index)

        if label is not None:
            self.labels[input_name] = label
        

    def create_log(self, model_id):
        """
        Make some room in the file for logging data under this model.
        """
        self.group = self.world.f.create_group(model_id)
        
        # Store the display names in an attribute of the group
        for input_name, display_name in self.labels.items():
            self.group.attrs[input_name] = display_name

        # Initialize buffers for 't' and each input
        self.buffer['t'] = np.zeros(self.buffer_size, dtype='f')
        for column in self.inputs:
            self.buffer[column] = np.zeros(self.buffer_size, dtype='f')

    def dump_buffers(self):
        # Resize datasets if total logged exceeds current size
        new_size = self.total_logged + self.index  # New size after dump
        for name, buffer in self.buffer.items():
            dataset = self.group.require_dataset(name, shape=(new_size,), dtype='f', maxshape=(None,))
            dataset[self.total_logged:new_size] = buffer[:self.index]

        self.total_logged += self.index
        self.index = 0  # Reset buffer index

    def append_frame(self, t: float, frame: Dict[str, float]):
        if self.index == self.buffer_size:
            self.dump_buffers()  # Dump and resize if buffer is full

        self.buffer['t'][self.index] = t
        for column, value in frame.items():
            self.buffer[column][self.index] = value

        self.index += 1

    def update(self, t):
        frame = {}  # Collect data for this timestep

        for column in self.inputs:
            model_id, attribute, index = self.inputs[column]
            value = getattr(self.world.models[model_id], attribute)

            if index is None:
                frame[column] = value
            else:
                frame[column] = value[index]

        self.append_frame(t, frame)
        self.t = t

    def finalize(self):
        """Call this method when logging is complete to dump any remaining data."""
        if self.index > 0:
            self.dump_buffers()