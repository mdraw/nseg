"""
Set up config system based on OmegaConf and Hydra.
"""

from typing import Any, Optional
import hydra
import randomname
import yaml
from omegaconf._utils import _ensure_container, get_omega_conf_dumper
from omegaconf import OmegaConf, DictConfig


class NConf(OmegaConf):
    """OmegaConf subclass with minor modifications"""

    @staticmethod
    def to_yaml(cfg: Any, *, resolve: bool = False, sort_keys: bool = False, default_flow_style: Optional[bool] = False) -> str:
        """
        returns a yaml dump of this config object.

        :param cfg: Config object, Structured Config type or instance
        :param resolve: if True, will return a string with the interpolations resolved, otherwise
            interpolations are preserved
        :param sort_keys: If True, will print dict keys in sorted order. default False.
        :param default_flow_style: Set default_flow_style option for PyYAML dump.
            Choices (default is False):
             - False: Always use block style for collections
             - True: Always use flow style for collections
             - None: Use block style for nested collections, otherwise use flow style
        :return: A string containing the yaml representation.
        """
        cfg = _ensure_container(cfg)
        container = OmegaConf.to_container(cfg, resolve=resolve, enum_to_str=True)
        return yaml.dump(  # type: ignore
            container,
            default_flow_style=default_flow_style,
            allow_unicode=True,
            sort_keys=sort_keys,
            Dumper=get_omega_conf_dumper(),
        )


# Register custom resolvers to NConf

# use_cache=True is necessary for node interpolation - otherwise the name is randomized on each access
NConf.register_new_resolver('randomname', randomname.get_name, use_cache=True)


__all__ = [
    'NConf',
    'DictConfig',
    'hydra'
]
