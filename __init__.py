from omegaconf import OmegaConf

import randomname


OmegaConf.register_new_resolver('randomname', randomname.get_name)
