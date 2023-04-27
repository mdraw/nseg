"""
Import OmegaConf from this module to get custom resolvers
"""

import randomname
from omegaconf import OmegaConf, DictConfig


OmegaConf.register_new_resolver('randomname', randomname.get_name)
