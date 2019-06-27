import sys

sys.path.append(".")

from rllab.misc.ext import is_iterable, set_seed
from rllab.misc.instrument import concretize
from rllab import config
import rllab.misc.logger as logger
import argparse
import os.path as osp
import datetime
import dateutil.tz
import ast
import uuid
import pickle as pickle
import base64
import joblib

import logging


def run_experiment(args_data,
                   variant_data=None,
                   seed=None,
                   n_parallel=1,
                   exp_name=None,
                   log_dir=None,
                   snapshot_mode='all',
                   snapshot_gap=1,
                   tabular_log_file='progress.csv',
                   text_log_file='debug.log',
                    params_log_file='params.json',
                    variant_log_file='variant.json',
                    resume_from=None,
                    plot=False,
                    log_tabular_only=False,
                   log_debug_log_only=False,
                   ):
    default_log_dir = config.LOG_DIR
    now = datetime.datetime.now(dateutil.tz.tzlocal())

    # avoid name clashes when running distributed jobs
    rand_id = str(uuid.uuid4())[:5]
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S_%f_%Z')

    default_exp_name = 'experiment_%s_%s' % (timestamp, rand_id)
    if exp_name is None:
        exp_name = default_exp_name

    if seed is not None:
        set_seed(seed)

    if n_parallel > 0:
        from rllab.sampler import parallel_sampler
        parallel_sampler.initialize(n_parallel=n_parallel)
        if seed is not None:
            parallel_sampler.set_seed(seed)

    if plot:
        from rllab.plotter import plotter
        plotter.init_worker()

    if log_dir is None:
        log_dir = osp.join(default_log_dir, exp_name)
    else:
        log_dir = log_dir
    tabular_log_file = osp.join(log_dir, tabular_log_file)
    text_log_file = osp.join(log_dir, text_log_file)
    params_log_file = osp.join(log_dir, params_log_file)

    if variant_data is not None:
        variant_data = variant_data
        variant_log_file = osp.join(log_dir, variant_log_file)
        # print(variant_log_file)
        # print(variant_data)
        logger.log_variant(variant_log_file, variant_data)
    else:
        variant_data = None

    logger.add_text_output(text_log_file)
    logger.add_tabular_output(tabular_log_file)
    prev_snapshot_dir = logger.get_snapshot_dir()
    prev_mode = logger.get_snapshot_mode()
    logger.set_snapshot_dir(log_dir)
    logger.set_snapshot_mode(snapshot_mode)
    logger.set_snapshot_gap(snapshot_gap)
    logger.set_log_tabular_only(log_tabular_only)
    logger.set_debug_log_only(log_debug_log_only)
    logger.push_prefix("[%s] " % exp_name)

    if resume_from is not None:
        data = joblib.load(resume_from)
        assert 'algo' in data
        algo = data['algo']
        algo.train()
    else:
        args_data(variant_data)

    logger.set_snapshot_mode(prev_mode)
    logger.set_snapshot_dir(prev_snapshot_dir)
    logger.remove_tabular_output(tabular_log_file)
    logger.remove_text_output(text_log_file)
    logger.pop_prefix()


if __name__ == "__main__":
    run_experiment(sys.argv)
