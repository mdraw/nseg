from typing import Callable, Optional

import numpy as np
from scipy.ndimage import distance_transform_edt
import gunpowder as gp


# Adapted from https://github.com/ELEKTRONN/elektronn3/blob/a8b8fa0b/elektronn3/data/transforms/transforms.py
class AddDistMap(gp.BatchFilter):
    """Converts discrete instance- or binary label target tensors to their (signed)
    euclidean distance transform (EDT) representation.

    Based on the method proposed in https://arxiv.org/abs/1805.02718.

    Args:
        scale: Scalar value to divide distances before applying normalization
        normalize_fn: Function to apply to distance map for normalization.
        inverted: Invert target labels before computing transform if ``True``.
             This means the distance map will show the distance to the nearest
             foreground pixel at each background pixel location (which is the
             opposite behavior of standard distance transform).
        signed: Compute signed distance transform (SEDT), where foreground
            regions are not 0 but the negative distance to the nearest
            foreground border.
        vector: Return distance vector map instead of scalars.
    """
    def __init__(
            self,
            instance_labels,
            distmap,
            vector_enabled: bool = False,
            inverted: bool = True,
            signed: bool = True,
            normalize_fn: Optional[Callable[[np.ndarray], np.ndarray]] = np.tanh,
            scale: Optional[float] = 40.,
            dtype=np.float32
    ):
        self.instance_labels = instance_labels
        self.distmap = distmap
        self.vector_enabled = vector_enabled
        self.inverted = inverted
        self.signed = signed
        self.normalize_fn = normalize_fn
        self.scale = scale
        self.dtype = dtype

    def setup(self):
        # tell downstream nodes about the new array
        spec = self.spec[self.instance_labels].copy()
        spec.dtype = self.dtype
        self.provides(
            self.distmap,
            spec
        )

    def prepare(self, request):
        deps = gp.BatchRequest()
        deps[self.instance_labels] = request[self.distmap].copy()
        return deps

    def _edt(self, target: np.ndarray) -> np.ndarray:
        sh = target.shape
        if target.min() == 1:  # If everything is 1, the EDT should be inf for every pixel
            nc = target.ndim if self.vector_enabled else 1
            return np.full((nc, *sh), np.inf, dtype=self.dtype)

        if self.vector_enabled:
            if target.ndim == 2:
                coords = np.mgrid[:sh[0], :sh[1]]
            elif target.ndim == 3:
                coords = np.mgrid[:sh[0], :sh[1], :sh[2]]
            else:
                raise RuntimeError(f'Target shape {sh} not understood.')
            inds = distance_transform_edt(
                target, return_distances=False, return_indices=True
            ).astype(self.dtype)
            dist = inds - coords
            # assert np.isclose(np.sqrt(dist[0] ** 2 + dist[1] ** 2), distance_transform_edt(target))
            return dist
        else:  # Regular scalar edt
            dist = distance_transform_edt(target).astype(np.float32)[None]
            return dist

    def _get_distance_map(self, target: np.ndarray) -> np.ndarray:
        # Ensure np.bool dtype, invert if needed
        if self.inverted:
            target = target == 0
        else:
            target = target > 0
        dist = self._edt(target)
        if self.signed:
            # Compute same transform on the inverted target. The inverse transform can be
            #  subtracted from the original transform to get a signed distance transform.
            invdist = self._edt(~target)
            dist -= invdist
        if self.normalize_fn is not None:
            dist = self.normalize_fn(dist / self.scale)
        return dist

    def process(self, batch, request):
        np_boundaries = batch[self.instance_labels].data == 0

        np_distmap = self._get_distance_map(np_boundaries)

        np_distmap = np_distmap.astype(self.dtype)

        # create the array spec for the new array
        spec = batch[self.instance_labels].spec.copy()
        spec.roi = request[self.distmap].roi.copy()
        spec.dtype = self.dtype

        # create a new batch to hold the new array
        batch = gp.Batch()

        # create a new array
        distmap = gp.Array(np_distmap, spec)

        # store it in the batch
        batch[self.distmap] = distmap

        # return the new batch
        return batch
