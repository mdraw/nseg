import gunpowder as gp
import numpy as np
import torch


class ToTorch(gp.BatchFilter):
    def process(self, batch, request):
        for k, v in batch.arrays.items():
            vtensor = torch.as_tensor(v.data)
            # vtensor.pin_memory()
            batch.arrays[k].data = vtensor


class ToNumpy(gp.BatchFilter):
    def process(self, batch, request):
        for k, v in batch.arrays.items():
            if isinstance(v.data, torch.Tensor):
                vtensor = v.data.numpy()
                # vtensor.pin_memory()
                batch.arrays[k].data = vtensor


class Cast(gp.BatchFilter):
    def __init__(self, key, dtype):
        self.key = key
        self.dtype = dtype

    def process(self, batch, request):
        arr = batch.arrays[self.key]
        arr.data = arr.data.astype(self.dtype)
        arr.spec.dtype = self.dtype
        batch.arrays[self.key] = arr
