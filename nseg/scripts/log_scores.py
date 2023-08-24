
import sys
from pathlib import Path

from nseg.evaluation.score_db_access import query_and_log_scores

if __name__ == '__main__':
    assert len(sys.argv) == 2, f'Usage: {Path(__file__).name} <setup_name>'
    # e.g. setup_name = '08-18_17-42_glum-cof__400k_11_micron'
    setup_name = sys.argv[1]
    query_and_log_scores(setup_name=setup_name)
