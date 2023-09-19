import argparse
from pathlib import Path

from nseg.evaluation.score_db_access import query_and_log_scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Display and optionally log and plot scores from a given setup name')
    parser.add_argument('setup_name', type=str, help='Name of the experiment setup')
    parser.add_argument('--enable-mpl', action='store_true', help='Enable plot')
    parser.add_argument('--enable-wandb', action='store_true', help='Enable wandb upload')
    args = parser.parse_args()

    query_and_log_scores(
        setup_name=args.setup_name,
        enable_mpl=args.enable_mpl,
        enable_wandb=args.enable_wandb
    )
