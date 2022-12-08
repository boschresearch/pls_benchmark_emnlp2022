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

import json
import os
import pandas as pd

from application.classification.label_binarizer import get_multitask_output
from experiments_pl.data.utils import DataLoader, FeatureGeneratorPatentLandscape, PrePorcessTargetLabelPatentLandscape
from application.classification.aggregation_utils import training


def get_exp_dir_name(config, run_prefix):
    """Get the experiment directory.

    Args:
        config (dict): 
        run_prefix (str): to be added to the experiment directory

    Returns:
        tuple: experiment directory, a text to refer to the experiments
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

    if "exp_dir_prefix" in config:
        exp_dir += "_" + config["exp_dir_prefix"]

    return exp_dir, exp_run_string


def load_and_train(config, data, 
                train_ids, test_ids, dev_ids, heldout_ids,
                doc_rep_prop,
                emb_dict, 
                exp_dir, 
                metrics, 
                metrics_heldout, 
                split_index, 
                mode, 
                content_fields, 
                logger=None):
    """Load the dataset, create features and train the model.

    Args:
        config (dict): 
        data (data frame): 
        train_ids (list): ids for the train set
        test_ids (list): ids for the test set
        dev_ids (list): ids for the dev set
        heldout_ids (list): ids for the heldout set
        doc_rep_prop (dict): dictionary storing the settings for document representation
        emb_dict (dict): information on embeddings
        exp_dir (str): path to the experiment directory
        metrics (list): list of metrics for test set
        metrics_heldout (list): list of metrics for the heldout set
        split_index (int): split index
        mode (str): debug or dev -- the dev is for actual training and debug for debugging the code
        content_fields (list): content fields to train a model
        logger (logger, optional): logger. Defaults to None.

    Raises:
        Exception: 
    """

    logger.info("loading dataset ... ")

    # is their any special processing required for the target label.
    process_label = PrePorcessTargetLabelPatentLandscape(exp_dir, is_flat=True, logger=logger)

    # # load the ids for each split.
    # train_ids, test_ids, dev_ids, heldout_ids = get_dataset_ids(0, config, 5)

    # load the dataset
    data_loader = DataLoader(process_label, "Family", "labels", logger=logger)
    train, test, dev, heldout, y_train, y_test, y_dev, y_heldout, mlb, hier_tree = data_loader.load_train_test(
        data, exp_dir, train_ids, test_ids, dev_ids, heldout_ids, mode=mode, split_index=split_index)

    logger.info("train.shape: %s " % str(train.shape))
    logger.info("test.shape: %s" % str(test.shape))
    logger.info("dev.shape:  %s" % str(dev.shape))
    
    if heldout is not None:
        logger.info("train.shape:  %s" % str(heldout.shape))

    # get the feature vector
    feature_generator = FeatureGeneratorPatentLandscape(train, test, dev, heldout, exp_dir, "debug", None,
                                                        config["label_desc_file_path"], emb_dict, logger=logger)
    logger.info("Model - %s" % config["model"])
    if config["model"] == "SVM":
        if config["doc_rep_params"]["embs"]:
            fields = config["doc_rep_params"]["embs"][0]["fields"]
            label =  config["doc_rep_params"]["embs"][0]["label_text"]
            X_train, X_test, X_dev, X_heldout = feature_generator.get_split_text(content_fields, fields, label=label)
            y_train_mt = y_train
            emb_info_list = None
            embedding_weights = None
        else:
            raise Exception("Fields information missing for SVM model.")

    elif config["model"] == "TMM" or config["model"] == "THMM" or config["model"] == "SingleTask":

        emb_info_list = feature_generator.generate_feature_vector(config["doc_rep_params"]["embs"], content_fields)
        X_train, X_test, X_dev, X_heldout = feature_generator.X_train, feature_generator.X_test, feature_generator.X_dev, feature_generator.X_heldout
        y_train_mt = get_multitask_output(y_train)
        embedding_weights = feature_generator.embedding_weights

        for emb_info in emb_info_list:
            logger.info(json.dumps(emb_info))

        for emb in X_train:
            logger.info("X_train -- emb size: %s" % str(emb.shape))

        for emb in X_test:
            logger.info("X_test -- emb size: %s" % str(emb.shape))

        for emb in X_dev:
            logger.info("X_dev -- emb size: %s" % str(emb.shape))

        if X_heldout:
            for emb in X_heldout:
                logger.info("X_train -- emb size: %s" % str(emb.shape))

    # train the model
    perf_test, perf_test_heldout, callback = training(X_train, y_train_mt,
                                            X_test, y_test,
                                            X_dev, y_dev,
                                            X_heldout, y_heldout,
                                            train_ids, test_ids, dev_ids, heldout_ids,
                                            doc_rep_prop=doc_rep_prop,
                                            model_config=config["model_params"],
                                            exp_dir=exp_dir,
                                            mlb=mlb,
                                            emb_info_list=emb_info_list,
                                            embedding_weights=embedding_weights,
                                            # hierarchical_label_tree=hier_tree,
                                            label="_".join(sorted([emb_info["type"] for emb_info in config["doc_rep_params"]["embs"]])),
                                            split_index=split_index, 
                                            logger=logger)


    logger.info("perf_test : %s" % json.dumps(perf_test))
    logger.info("perf_test_heldout : %s" % json.dumps(perf_test_heldout))

    # metrics = list()
    # metrics_heldout = list()
    iter_index = 0

    perf = dict()
    perf["split"] = split_index

    if iter_index:
        perf["iter_index"] = iter_index

    perf["model"] = config.get("model")
    perf["agg-type"] = config["model_params"].get("emb_agg")
    if config["model"] == "TMM":
        if "embs" in config.get("doc_rep_params"):
            for emb_info in config["doc_rep_params"].get("embs"):
                perf[emb_info["type"]] = True

    if perf_test_heldout:
        perf_test_heldout = {**perf, **perf_test_heldout}
        metrics_heldout.append(perf_test_heldout)
        df = pd.DataFrame(metrics_heldout)
        df.to_csv(os.path.join(exp_dir, "metrics_heldout_cv.csv"), index=False)

    if perf_test:
        perf_test = {**perf, **perf_test}
        metrics.append(perf_test)
        df = pd.DataFrame(metrics)
        df.to_csv(os.path.join(exp_dir, "metrics_cv.csv"), index=False)

        metrics_avg = list()
        for run in df.run.unique().tolist():
            df_run = df[df.run == run]
            perf_avg = {
                "run": run,
                "precision_macro": round(df_run.precision_macro.mean(), 3),
                "recall_macro": round(df_run.recall_macro.mean(), 3),
                "f1_macro": round(df_run.f1_macro.mean(), 3),
                "precision_micro": round(df_run.precision_micro.mean(), 3),
                "recall_micro": round(df_run.recall_micro.mean(), 3),
                "f1_micro": round(df_run.f1_micro.mean(), 3)
            }
            metrics_avg.append({**perf, **perf_avg})
        pd.DataFrame(metrics_avg).to_csv(os.path.join(exp_dir, "metrics_avg.tsv"), index=False)

    del callback