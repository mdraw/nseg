import numpy as np
import gunpowder as gp


class AddBoundaryLabels(gp.BatchFilter):
    """Add a binary boundary label array based on instance labels to the batch."""

    def __init__(self, instance_labels, boundary_labels, dtype=np.int64):
        self.instance_labels = instance_labels
        self.boundary_labels = boundary_labels
        self.dtype = dtype

    def setup(self):
        # tell downstream nodes about the new array
        spec = self.spec[self.instance_labels].copy()
        spec.dtype = self.dtype
        self.provides(
            self.boundary_labels,
            spec
        )

    def prepare(self, request):
        deps = gp.BatchRequest()
        deps[self.instance_labels] = request[self.boundary_labels].copy()
        return deps

    def process(self, batch, request):
        np_boundaries = batch[self.instance_labels].data == 0
        np_boundaries = np_boundaries.astype(self.dtype)

        # create the array spec for the new array
        spec = batch[self.instance_labels].spec.copy()
        spec.roi = request[self.boundary_labels].roi.copy()
        spec.dtype = self.dtype

        # create a new batch to hold the new array
        batch = gp.Batch()

        # create a new array
        boundaries = gp.Array(np_boundaries, spec)

        # store it in the batch
        batch[self.boundary_labels] = boundaries

        # return the new batch
        return batch



class ArgMax(gp.BatchFilter):
    def __init__(self, array, axis=1, keepdims=True):
        self.array = array
        self.axis = axis
        self.keepdims = keepdims

    def process(self, batch, request):

        if self.array not in batch.arrays:
            return

        arr = batch.arrays[self.array]
        arr.data = np.argmax(arr.data, self.axis, keepdims=self.keepdims)

