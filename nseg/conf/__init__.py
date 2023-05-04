"""
Import OmegaConf from this module to get custom resolvers
"""

import randomname
from omegaconf import OmegaConf, DictConfig

# use_cache=True is necessary for node interpolation - otherwise the name is randomized on each access
OmegaConf.register_new_resolver('randomname', randomname.get_name, use_cache=True)
