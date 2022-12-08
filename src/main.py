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

# TODO: remove this path variable
project_base = "/fs/scratch/rng_cr_bcai_dl_students/r26/patent_classification/code/bosch-patent/"
sys.path.append(os.path.join(project_base, "src"))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)).split("bosch-patent")[0],
                             "bosch-patent/baseline/sklearn-hierarchical-classification"))


from experiments_pl.data_utils import load_embeddings, get_dataset_ids
from generic_classification.experiments_patent_landscape import load_and_train

tf.random.set_seed(42)
random.seed(42)
np.random.seed(42)

# emb_filename_type_dict = {
#     "l-g-n2v": "label_emb/graph_emb/sg_norm_weight_graph_emb.txt",
#     "l-t-bert": "label_emb/text_emb/sg-all-SciBERT-ft-THMM.pkl"
# }

# dataset_bert_emb_filenames_dict = {
#     "atz": "cont_embs/sci-ft-THMM/atz_txt_emb.txt",
#     "rito": "cont_embs/sci-ft-THMM/rito_txt_emb.txt",
#     "fi": "cont_embs/sci-ft-THMM/fi_txt_emb.txt"
# }

# dataset_partial_dict_map = {
#     "atz": (50, None, 25),
#     "fi": (50, 2001, 50),
#     "rito": (50, None, 25)
# }


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

    if train_sample_count:
        #load train ids from the file and replace the train_ids.
        random_files_dir = "/fs/scratch/rng_cr_bcai_dl_students/r26/patent_classification/dataset/patent_ls/random-ids-pl"
        filename = os.path.join(random_files_dir, "rand_train_ids_%s_%s_%s.txt" % (config["dataset"], k, train_sample_count))
        train_ids = [int(item.strip()) for item in open(filename).readlines()]

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
                #    option=args.option, 
                   mode=args.mode,
                   train_sample_count=args.train_sample_count)

    elif args.option == "1_cfg":
        run_config(json.loads(open(args.cfg).read()),
                   split_index=None, 
                #    option=args.option, 
                   mode=args.mode,
                   train_sample_count=args.train_sample_count)

    elif args.option == "all_cfg":
        for config_json in open(args.cfg):
            config = json.loads(config_json.strip())
            run_config(config, 
                    split_index=None,
                    # option=args.option, 
                    mode=args.mode,
                    train_sample_count=args.train_sample_count)

    elif args.option == "partial_split_cfg":
        run_config(json.loads(open(args.cfg).read()),
                   split_index=None, 
                #    option=args.option, 
                    mode=args.mode,
                    train_sample_count=args.train_sample_count)
    else:
        ValueError("Unkown option : %s " % args.option)

# TODO: remove the configuration from here at the end
config = {
    "input_dir_path": "/fs/scratch/rng_cr_bcai_dl_students/r26/patent_classification/dataset/patent_ls",
    "output_dir_path": "/fs/scratch/rng_cr_bcai_dl_students/r26/patent_classification/output_dir",
    "dataset_filename": "rito_2021-08-04_filtered.csv",
    "split_dir": "splits-5/splits-rito",
    "split-k": 5,
    "dataset": "rito",
    "doc_rep_params": {
        "embs": [
            {
                "type": "bert",
                "fields": [
                    "Title",
                    "Abstract"
                ],
                "path": "/fs/scratch/rng_cr_bcai_dl_students/r26/patent_classification/scibert-test",
                "max_len": 512,
                "label_text": "False",
                "trainable": "True"
            },
            {
                "type": "bert",
                "fields": [
                    "Claims"
                ],
                "path": "/fs/scratch/rng_cr_bcai_dl_students/r26/patent_classification/scibert-test",
                "max_len": 512,
                "label_text": "False",
                "trainable": "True"
            },
            {
                "type": "bert",
                "fields": [
                    "Description"
                ],
                "path": "/fs/scratch/rng_cr_bcai_dl_students/r26/patent_classification/scibert-test",
                "max_len": 512,
                "label_text": "False",
                "trainable": "True"
            },
            # {
            #     "type": "l-t-tfidf",
            #     "path": "/fs/scratch/rng_cr_bcai_dl_students/r26/patent_classification/output_dir/label_emb/text_emb/tf_idf_label.pkl"
            # }
        ],
        "doc_rep_prop": {
            "cf_agg": "sum",
            "mf_agg": "sum",
            "doc_agg": "concat",
            "cf_agg_layer_norm": "False",
            "mf_agg_layer_norm": "False"
        }
    },
    "model": "TMM",
    "model_params": {
        "model": "TMM",
        "dense_layer_size": 50,
        "dropout_rate": 0.25,
        "learning_rate": 3e-05,
        "epochs": 50,
        "batch_size": 4,
        "emb_agg": "concat",
        "encoder_size": 768,
        "kernel": "rbf"
    },
    "label_desc_file_path": "/fs/scratch/rng_cr_bcai_dl_students/r26/patent_classification/code/bosch-patent/output/hier_cpc/label.csv",
    "exp_dir_prefix": "tacd",
    "exp_dir": "/fs/scratch/rng_cr_bcai_dl/ujp5kor/output_dir/experiment-pl-code-release-emnlp-2022/experiments-bert"
}


if __name__ == "__main__":
    # run_config(config, split_index=0, mode="debug", train_sample_count=50)
    main()
