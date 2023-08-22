from typing import Literal

import numpy as np
import pandas as pd
import pymongo
import matplotlib.pyplot as plt
import wandb
import sys


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


def get_scores_df(db_name, db_host='cajalg001', proj_keys=None):
    client = pymongo.MongoClient(db_host)
    db = client[db_name]
    coll = db['scores']
    projection = {k: 1 for k in proj_keys} | {'_id': 0} if proj_keys else None
    cursor = coll.find({}, projection)
    df = pd.DataFrame(cursor)
    return df


def mpl_plot(df, db_name, markers, plot_dir):
    marker_size = 100
    marker_color = 'black'

    fig, ax = plt.subplots(figsize=(10, 10))

    df.plot(
        x='threshold',
        y='voi',
        # y=['voi_split', 'voi_merge', 'voi'],
        ylim=[0, 10],
        yticks=range(11),
        title=f'{plot_dir}/{db_name}: VOI',
        ax=ax
    )
    ax.scatter(**markers['voi'], marker='v', s=marker_size, c=marker_color)
    fig.savefig(f'{db_name}_voi.png')

    fig, ax = plt.subplots(figsize=(10, 10))
    df.plot(
        x='threshold',
        y=['erl'],
        ylim=[0, 40_000],
        # yticks=range(11),
        title=f'{db_name}: ERL'
    )
    ax.scatter(**markers['erl'], marker='v', s=marker_size, c=marker_color)
    fig.savefig(f'{plot_dir}/{db_name}_erl.png')


# def query_scores(setup_name: str, annos: Literal['val', 'test'] = 'val'):
def query_scores(
        db_name: str,
        db_host: str = 'cajalg001',
        proj_keys=None,
        plot_dir='/u/mdraw/lsdex/tmp/scores_plots',
        markers=None,
        enable_mpl=True,
        enable_wandb=True,
):
    df = get_scores_df(db_name=db_name, db_host=db_host, proj_keys=proj_keys)
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
        markers['erl']['y'] = best_erl_row.erl

    if enable_mpl:
        mpl_plot(df, db_name, markers, plot_dir)

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


def main():
    plot_dir = '/u/mdraw/tmp/scores_plots'
    scores_db_host = 'cajalg001'
    db_prefix = 'scores'
    # setup_name = '08-19_01-34_absolute__400k_benchmark'
    # setup_name = '08-20_00-26_fat-micr__400k_benchmark'
    # setup_name = '08-20_01-10_muted-mo__400k_benchmark'
    # setup_name = '08-20_00-23_plastic-__395k_benchmark'
    setup_name = '08-18_17-42_glum-cof__400k_11_micron'

    if len(sys.argv) == 1:
        setup_name = sys.argv[1]

    model_name = setup_name[:-10]
    roi_name = setup_name[-9:]

    val_db_name = f'{db_prefix}_{setup_name}_val'
    test_db_name = f'{db_prefix}_{setup_name}_test'

    enable_mpl = True
    enable_wandb = True

    if enable_wandb:
        wandb.init(
            entity='mdth',
            project='nseg_scores',
            dir='/u/mdraw/lsdex/tmp/scores_plots',
            name=f'{setup_name.split()}',
            tags=[model_name, roi_name]
        )

    query_scores(
        val_db_name,
        db_host=scores_db_host,
        proj_keys=_default_proj_keys,
        plot_dir=plot_dir,
        enable_mpl=enable_mpl,
        enable_wandb=enable_wandb,
    )
    query_scores(
        test_db_name,
        db_host=scores_db_host,
        proj_keys=_default_proj_keys,
        plot_dir=plot_dir,
        enable_mpl=enable_mpl,
        enable_wandb=enable_wandb,
    )


if __name__ == '__main__':
    main()
