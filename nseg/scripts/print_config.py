""" Script to debug / test hydra configs """

import logging
from pathlib import Path
from nseg.conf import NConf, DictConfig, hydra


@hydra.main(version_base='1.3', config_path='../conf', config_name='config')
def main(cfg: DictConfig) -> None:
    logging.info(f'{Path(__file__).stem} config:\n==\n{NConf.to_yaml(cfg, resolve=True)}\n==')


if __name__ == '__main__':
    main()
