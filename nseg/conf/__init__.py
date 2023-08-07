"""
Set up config system based on OmegaConf and Hydra.
"""
from pathlib import Path
from typing import Any, Optional, Sequence, TypeVar
import hydra
import randomname
import yaml
from omegaconf._utils import _ensure_container, get_omega_conf_dumper
from omegaconf import OmegaConf, DictConfig


T = TypeVar('T')


def unwind_dict(cfg_dict: dict[str, T], keys: Sequence[str], init_empty: bool = False) -> dict[str, T]:
    cfg_dict = cfg_dict.copy()  # Avoid side effects on original dict
    unw = {} if init_empty else cfg_dict
    for key in keys:
        unw.update(cfg_dict.pop(key))
    return unw


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

def get_setup_name(model_path: str | Path, short=True) -> str:
    """String representation to identify the model checkpoint"""
    model_path = Path(model_path)
    model_name = model_path.parent.name
    if short:
        model_name = model_name[:20]  # Limit to first 20 chars to avoid overly long names
        # Just the step number
        checkpoint_name = model_path.stem.split('_')[-1]
        assert checkpoint_name.isdigit(), f'Can\'t parse step number from {model_path.name}'
        assert checkpoint_name[-3:] == '000', f'Can\'t parse step number from {model_path.name} - not a multiple of 1000'
        checkpoint_name = f'{checkpoint_name[:-3]}k'  # trailing 000 -> k
    else:
        checkpoint_name = model_path.name
    setup_name = f'{model_name}__{checkpoint_name}'
    return setup_name


# use_cache=True is necessary for node interpolation - otherwise the name is randomized on each access
NConf.register_new_resolver('randomname', randomname.get_name, use_cache=True)

NConf.register_new_resolver('setupname', get_setup_name, use_cache=True)


__all__ = [
    'NConf',
    'DictConfig',
    'hydra',
    'unwind_dict',
]
