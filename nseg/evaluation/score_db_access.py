from typing import Literal

import numpy as np
import pandas as pd
import pymongo
import matplotlib.pyplot as plt
import wandb
import sys
import logging
from pathlib import Path


# TODO: Use new score summaries (thresh_results, best_thresh_results) instead of raw score collections?

_default_proj_keys = [
    'threshold',
    'voi_split',
    'voi_merge',
    'voi',
    'erl',
    # 'max_erl',
    # 'total path length',
    # 'normalized_erl',
    # 'number_of_merging_segments',
    # 'number_of_split_skeletons',
]


def get_scores_df(collection_name: str, db_name: str = 'scores', db_host='cajalg001', proj_keys=None):
    with pymongo.MongoClient(db_host) as client:
        db = client[db_name]
        coll = db[collection_name]
        projection = {k: 1 for k in proj_keys} | {'_id': 0} if proj_keys else None
        cursor = coll.find({}, projection)
        df = pd.DataFrame(cursor)
        assert df.size > 0, f'No scores found in {db_name}.{collection_name}'
    return df


def mpl_plot(df, collection_name, markers, plot_dir):
    marker_size = 100
    marker_color = 'black'

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    erl_ref = 12_000
    voi_ref = 3.3

    fig, ax = plt.subplots(figsize=(10, 10))

    df.plot(
        x='threshold',
        y='voi',
        # y=['voi_split', 'voi_merge', 'voi'],
        ylim=[0, 10],
        yticks=range(11),
        title=f'{collection_name}: VOI',
        ax=ax,
        legend=False,
    )
    ax.scatter(**markers['voi'], marker='v', s=marker_size, c=marker_color)

    if collection_name.endswith('test'):
        ax.scatter(markers['voi']['x'], voi_ref, marker='x', s=marker_size, c='red')

    fig.savefig(f'{plot_dir}/{collection_name}_voi.png')

    fig, ax = plt.subplots(figsize=(10, 10))
    df.plot(
        x='threshold',
        y=['erl'],
        ylim=[0, 40_000],
        yticks=[0, 5000, 10_000, 15_000, 20_000, 25_000, 30_000, 35_000, 40_000],
        title=f'{collection_name}: ERL',
        ax=ax,
        legend=False,
    )
    ax.scatter(**markers['erl'], marker='v', s=marker_size, c=marker_color)
    if collection_name.endswith('test'):
        ax.scatter(markers['erl']['x'], erl_ref, marker='x', s=marker_size, c='red')


    fig.savefig(f'{plot_dir}/{collection_name}_erl.png')


# def query_scores(setup_name: str, annos: Literal['val', 'test'] = 'val'):
def query_scores(
        collection_name: str,
        db_name: str = 'scores',
        db_host: str = 'cajalg001',
        proj_keys=None,
        plot_dir='/u/mdraw/lsdex/tmp/scores_plots',
        markers=None,
        enable_mpl=True,
        enable_wandb=True,
):
    logging.info(f'Querying scores from {db_name}{collection_name}')
    df = get_scores_df(collection_name=collection_name, db_name=db_name, db_host=db_host, proj_keys=proj_keys)
    best_voi_row = df.loc[df['voi'].idxmin()]
    best_erl_row = df.loc[df['erl'].idxmax()]

    print(df)
    print()
    print(f'Best VOI: {best_voi_row.voi:.4f} with threshold {best_voi_row.threshold}')
    print(f'Best ERL: {best_erl_row.erl:.0f} with threshold {best_erl_row.threshold}')

    if markers is None:
        markers = {
            'voi': {'x': best_voi_row.threshold, 'y': best_voi_row.voi},
            'erl': {'x': best_erl_row.threshold, 'y': best_erl_row.erl},
        }
    else:
        # Keep thresholds (x) but update score values (y) with current eval values
        markers['voi']['y'] = best_voi_row.voi
        # markers['voi']['y'] = 3.081
        markers['erl']['y'] = best_erl_row.erl
        # markers['erl']['y'] = 10349

    if enable_mpl:
        mpl_plot(df, collection_name, markers, plot_dir)

    if enable_wandb:
        wdf = wandb.Table(dataframe=df)
        voi_line = wandb.plot.line(wdf, x='threshold', y='voi', title='VOI')
        erl_line = wandb.plot.line(wdf, x='threshold', y='erl', title='ERL')
        anno_str = db_name.split('_')[-1]
        log_data = {
            f'{anno_str}/scores_table': wdf,
            f'{anno_str}/voi_line': voi_line,
            f'{anno_str}/erl_line': erl_line,
        }
        wandb.log(log_data)

    return markers


def query_and_log_scores(
    setup_name: str,
    plot_dir: str = '/cajal/scratch/projects/misc/mdraw/lsdex/tmp/scores_plots',
    wandb_dir: str = '/cajal/scratch/projects/misc/mdraw/lsdex/tmp/scores_wandb',
    db_name: str = 'scores',
    db_host: str = 'cajalg001',
    enable_mpl: bool = True,
    enable_wandb: bool = True,
):
    # A totally safe way to get model_name and roi_name, no way this could go wrong
    model_name = setup_name[:-10]
    roi_name = setup_name[-9:]
    # model_name = setup_name[:26]
    # roi_name = setup_name[27:]

    val_collection_name = f'{setup_name}_val'
    test_collection_name = f'{setup_name}_test'

    if enable_wandb:
        wandb.init(
            entity='mdth',
            project='nseg_scores',
            dir=wandb_dir,
            name=f'{setup_name}',
            tags=[model_name, roi_name]
        )

    val_markers = query_scores(
        collection_name=val_collection_name,
        db_name=db_name,
        db_host=db_host,
        proj_keys=_default_proj_keys,
        plot_dir=plot_dir,
        enable_mpl=enable_mpl,
        enable_wandb=enable_wandb,
    )
    query_scores(
        collection_name=test_collection_name,
        db_name=db_name,
        db_host=db_host,
        proj_keys=_default_proj_keys,
        plot_dir=plot_dir,
        enable_mpl=enable_mpl,
        enable_wandb=enable_wandb,
        markers=val_markers,
    )
