# Copyright (c) 2021 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


# from posixpath import split
import sys
import argparse
import json
import os
import random
import logging

import numpy as np
import pandas as pd
import tensorflow as tf

project_base = "/fs/scratch/rng_cr_bcai_dl_students/r26/patent_classification/code/bosch-patent/"
sys.path.append(os.path.join(project_base, "src"))

from experiments_pl.data_utils import load_embeddings, get_dataset_ids
from generic_classification.experiments_patent_landscape import load_and_train

tf.random.set_seed(42)
random.seed(42)
np.random.seed(42)


def get_logger(filename):
    """Get a logger object.

    Args:
        filename (str): The filename for a logger object.

    Returns:
        logger: logger object
    """
    logger = logging.getLogger('server_logger')
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(filename)
    fh.setLevel(logging.INFO)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    # add the handlers to logger
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


def get_exp_dir_name(config, run_prefix, train_sample_count=None):
    """Get the experiment directoru name specific to a given experiment configuration.

    Args:
        config (dict): configuration for an experiment.
        run_prefix (str): run prefix provided for the experiment. This will be appended to the directory name.
        train_sample_count (int, optional): Number of training examples used to train a model. Defaults to None.

    Returns:
        tuple: A two-valued tuple. The first string is the path to the experiment directory whereas the second string is the experiment run string.
    """
    exp_run_string = list()

    if "model" in config:
        exp_run_string.append(config["model"])

    if "embs" in config["doc_rep_params"]:
        exp_run_string.extend([emb_info["type"]
                              for emb_info in config["doc_rep_params"]["embs"]])

    if "emb_agg" in config["model_params"]:
        exp_run_string.append(config["model_params"]["emb_agg"])

    exp_dir = os.path.join(
        config["exp_dir"], config["dataset"], "_".join(exp_run_string) + run_prefix)

    if train_sample_count:
        exp_dir += "_tsc-%s" % train_sample_count

    if "exp_dir_prefix" in config:
        exp_dir += "_" + config["exp_dir_prefix"]

    return exp_dir, exp_run_string


def run_config(config, split_index=None, mode="debug", cpc_code_filter_list=list(), train_sample_count=None):
    """Run the experiment for a given configuration.

    Args:
        config (dict): configuration for an experiment
        split_index (int, optional): The split_index to use as test dataset. Defaults to None.
        mode (str, optional): the model to run the application as dev or debug. Defaults to "debug".
        cpc_code_filter_list (list, optional): TODO: how is this list used in practice? Defaults to list().
        train_sample_count (int, optional): Number of training examples to be sampled to train a classifier. None indicates that the model is trained over all the training examples.
    """
    emb_dict = load_embeddings(config["doc_rep_params"]["embs"])

    k = config["split-k"]

    data = pd.read_csv(os.path.join(
        config["input_dir_path"], config["dataset_filename"]))
    data.labels = [eval(label) for label in data.labels.tolist()]

    run_prefix = ""
    if split_index is not None:
        run_prefix = "_split-%s" % split_index

    exp_dir, exp_run_string = get_exp_dir_name(config, run_prefix, train_sample_count)

    os.makedirs(exp_dir, exist_ok=True)
    open(os.path.join(exp_dir, "_".join(exp_run_string) + ".log"), "w").close()
    open(os.path.join(exp_dir, "config.json"), "w").write(json.dumps(config))

    logging_filename = os.path.join(exp_dir, "_".join(exp_run_string) + ".log")
    print(logging_filename)
    
    logger = get_logger(logging_filename)

    logger.info("logging_filename: %s" % logging_filename)
    logger.info("exp_dir : %s" % exp_dir)
    logger.info(json.dumps(config))

    logger.info("mode: %s " % mode)
    logger.info("split_index: %s " % split_index)
    logger.info("len(cpc_code_filter_list): %s " % len(cpc_code_filter_list))

    metrics_heldout = list()
    metrics = list()

    content_fields = ["Title", "Abstract", "Claims", "Description"]

    doc_rep_prop = config["doc_rep_params"]["doc_rep_prop"]

    train_ids, dev_ids, test_ids, final_hold_out_set_ids = get_dataset_ids(split_index, config, k, logger=logger)

    load_and_train(config, data, train_ids, test_ids, dev_ids, final_hold_out_set_ids, 
                doc_rep_prop, emb_dict, exp_dir, metrics, metrics_heldout, 
                split_index, mode=mode, content_fields=content_fields, logger=logger)


def main():
    """
    This is the starting point for all the experiments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--option', type=str,
                        help="all_cfg|1_cfg|1_split", required=True)
    parser.add_argument('--cfg', type=str, required=True)
    parser.add_argument('--split', type=int, required=False)
    parser.add_argument('--mode', type=str, default="debug", required=False)
    parser.add_argument('--train_sample_count', type=int, required=False)

    args = parser.parse_args()

    print("args.option: ", args.option)
    print("args.cfg: ", args.cfg)
    print("args.split: ", args.split)
    print("args.mode: ", args.mode)
    print("args.train_sample_count: ", args.train_sample_count)

    if args.option == "1_split" and args.split >= 0:
        run_config(json.loads(open(args.cfg).read()),
                   split_index=args.split, 
                   mode=args.mode,
                   train_sample_count=args.train_sample_count)

    elif args.option == "1_cfg":
        run_config(json.loads(open(args.cfg).read()),
                   split_index=None, 
                   mode=args.mode,
                   train_sample_count=args.train_sample_count)

    elif args.option == "all_cfg":
        for config_json in open(args.cfg):
            config = json.loads(config_json.strip())
            run_config(config, 
                    split_index=None,
                    mode=args.mode,
                    train_sample_count=args.train_sample_count)

    elif args.option == "partial_split_cfg":
        run_config(json.loads(open(args.cfg).read()),
                   split_index=None, 
                    mode=args.mode,
                    train_sample_count=args.train_sample_count)
    else:
        ValueError("Unkown option : %s " % args.option)


if __name__ == "__main__":
    main()
