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

import logging
import os
import pickle

import matplotlib.pyplot as plt
from networkx.algorithms.euler import eulerize
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from transformers.utils.dummy_pt_objects import BertGenerationDecoder

from application.classification.label_binarizer import get_mlb_output
from evaluations.utils import calc_metrics


class CallbackBase(tf.keras.callbacks.Callback):

    def __init__(self,
                 X_train, y_train,
                 X_test, y_test,
                 X_dev, y_dev,
                 X_test_holdout, y_test_holdout,
                 train_ids, test_ids, dev_ids, heldout_ids,
                 mlb, exp_dir, split_id, patience=7, logger=None):

        super(CallbackBase, self).__init__()

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.X_dev = X_dev
        self.y_dev = y_dev
        self.X_heldout = X_test_holdout
        self.y_heldout = y_test_holdout

        self.train_ids = train_ids
        self.test_ids = test_ids
        self.dev_ids = dev_ids
        self.heldout_ids = heldout_ids

        self.mlb = mlb
        self.exp_dir = exp_dir
        self.best_f1 = 0
        self.best_model = None
        self.best_model_threshold = None
        self.best_model_threshold_list = list()
        self.epochs_since_best_f1 = 0
        self.epoch_threshold = None
        self.metrics = list()
        self.split_id = split_id
        self.patience = patience
        self.stopped_epoch = 0
        self.dev_threshold = None
        self.train_ids = train_ids
        self.dev_ids = dev_ids
        self.logger = logger
        self.epoch_done = 1
        self.load_previous_run()

    def load_previous_run(self):
        if os.path.exists(os.path.join(self.exp_dir, "metrics_%s.csv" % self.split_id)):
            self.logger.info("found previous run %s " % os.path.join(
                self.exp_dir, "metrics_%s.csv" % self.split_id))

            df_metrics = pd.read_csv(os.path.join(
                self.exp_dir, "metrics_%s.csv" % self.split_id))
            self.metrics = list(df_metrics.T.to_dict().values())
            df_metrics_dev = df_metrics[df_metrics["mode"] == "dev"]
            max_value = max(df_metrics_dev.f1_macro.tolist())
            self.best_f1 = max_value

            self.epoch_done = max(df_metrics.epoch.tolist()) + self.epoch_done

            if os.path.exists(os.path.join(self.exp_dir, "last-model.h5")):
                self.epochs_since_best_f1 = 0
            elif os.path.exists(os.path.join(self.exp_dir, "best-model.h5")):
                max_index = df_metrics_dev.f1_macro.tolist().index(max_value)
                num_of_epochs = df_metrics_dev.shape[0]
                self.epochs_since_best_f1 = num_of_epochs - max_index - 1

            self.logger.info("init self.epochs_since_best_f1 - %s " %
                             self.epochs_since_best_f1)
            self.logger.info(
                "init self.epochs_since_best_f1 - %s " % self.best_f1)
            self.logger.info("init self.epoch_done - %s " % self.epoch_done)
            self.logger.info("initialization done ...")

    def on_epoch_begin(self, epoch, logs=None):
        epoch = epoch + self.epoch_done
        self.logger.info("epoch: %s starts." % epoch)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            self.logger.info("Epoch %s : early stopping" %
                             (self.stopped_epoch + 1))

    def plot_metrics(self, df_metrics):
        f, axes = plt.subplots(1, 6, figsize=(15, 5))
        for ax in axes:
            ax.set_ylim(0, 1.0)

        sns.lineplot(data=df_metrics, x='epoch',
                     y='precision_macro', hue='mode', ax=axes[0])
        axes[0].set_ylabel('precision_macro')
        sns.lineplot(data=df_metrics, x='epoch',
                     y='recall_macro', hue='mode', ax=axes[1])
        axes[1].set_ylabel('recall_macro')
        sns.lineplot(data=df_metrics, x='epoch',
                     y='f1_macro', hue='mode', ax=axes[2])
        axes[2].set_ylabel('f1_macro')

        sns.lineplot(data=df_metrics, x='epoch',
                     y='precision_micro', hue='mode', ax=axes[3])
        axes[3].set_ylabel('precision_micro')
        sns.lineplot(data=df_metrics, x='epoch',
                     y='recall_micro', hue='mode', ax=axes[4])
        axes[4].set_ylabel('recall_micro')
        sns.lineplot(data=df_metrics, x='epoch',
                     y='f1_micro', hue='mode', ax=axes[5])
        axes[5].set_ylabel('f1_micro')
        plt.tight_layout()
        plt.savefig(os.path.join(
            self.exp_dir, 'metrics_%s.png' % self.split_id))

    def save_for_split(self, X, y_actual, ids, split_name, epoch):

        os.makedirs(os.path.join(self.exp_dir, "pred-output"), exist_ok=True)

        columns = ["id"] + list(self.mlb.classes_)

        y_pred = self.best_model.predict(X)
        y_pred = self.transform_pred(y_pred)

        df_pred = pd.DataFrame(y_pred, columns=self.mlb.classes_)
        df_pred["id"] = ids[:y_pred.shape[0]]
        df_pred[columns].to_csv(os.path.join(
            self.exp_dir, "pred-output", "best_pred_%s.csv" % split_name), index=False)
        df_pred[columns].to_csv(os.path.join(
            self.exp_dir, "pred-output", "epoch_pred_%s_%s.csv" % (split_name, epoch)), index=False)

        df_actual = pd.DataFrame(y_actual, columns=self.mlb.classes_)
        df_actual["id"] = ids[:y_pred.shape[0]]
        df_actual[columns].to_csv(os.path.join(
            self.exp_dir, "pred-output", "actual_%s.csv" % (split_name)), index=False)

    def save_epoch_results(self, epoch):

        self.save_for_split(self.X_test, self.y_test,
                            self.test_ids, "test", epoch)

        if self.heldout_ids is not None:
            self.save_for_split(self.X_heldout, self.y_heldout,
                                self.heldout_ids, "heldout", epoch)

    def on_epoch_end(self, epoch, logs=None):

        epoch = epoch + self.epoch_done

        self.logger.info("epoch: %s ends." % (epoch))

        bert_layer_list = [
            layer for layer in self.model.layers if "bert" in layer.name]

        if len(bert_layer_list):
            bert_layer_list[0].save_pretrained(
                os.path.join(self.exp_dir, "bert_model"))

        self.model.save_weights(os.path.join(self.exp_dir, "last-model.h5"))

        os.makedirs(os.path.join(
            self.exp_dir, "predicts-training"), exist_ok=True)

        y_dev_pred = self.model.predict(self.X_dev)
        y_dev_pred = self.transform_pred(y_dev_pred)
        pickle.dump(y_dev_pred, open(os.path.join(
            self.exp_dir, "predicts-training", "dev_prob_%s_%s.pkl" % (self.split_id, epoch)), "wb"))

        self.set_best_threshold(self.y_dev, y_dev_pred)
        y_dev_pred = y_dev_pred > self.epoch_threshold

        metric_dev = calc_metrics(
            self.y_dev, y_dev_pred, 'dev-macro-calc', index=0)
        metric_dev["mode"] = "dev"
        metric_dev["epoch"] = epoch
        self.metrics.append(metric_dev)

        with open(os.path.join(self.exp_dir, "predicts-training", "dev_%s_%s.txt" % (self.split_id, epoch)),
                  "w") as fileW:
            fileW.write(
                "\n".join([":".join([str(item[0]), ",".join(item[1]), ",".join(item[2])])
                           for item in zip(self.dev_ids,
                                           self.mlb.inverse_transform(
                                               self.y_dev),
                                           self.mlb.inverse_transform(
                                               y_dev_pred)
                                           )])
            )

        if metric_dev['f1_macro'] > self.best_f1:
            self.logger.info('CHANGE: old score: %s < new score: %s.' %
                             (self.best_f1, metric_dev['f1_macro']))
            self.best_model = self.model
            self.best_f1 = metric_dev['f1_macro']
            self.best_model_threshold = self.epoch_threshold
            self.epochs_since_best_f1 = 0
            self.best_model_threshold_list.append(
                {"epoch": epoch, "threshold": round(self.epoch_threshold, 2)})
            pd.DataFrame(self.best_model_threshold_list) \
                .to_csv(os.path.join(self.exp_dir, "best_threshold_%s.csv" % self.split_id), index=False)
        else:
            self.logger.info('NO CHANGE: old score: %s > new score: %s.' %
                             (self.best_f1, metric_dev['f1_macro']))
            if self.epochs_since_best_f1 == self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
            else:
                self.epochs_since_best_f1 += 1

        if self.best_model is None:
            self.best_model = self.model
            self.best_model_threshold = self.epoch_threshold

        self.best_model.save_weights(
            os.path.join(self.exp_dir, "best-model.h5"))

        y_train_pred = self.model.predict(self.X_train)
        y_train_pred = self.transform_pred(y_train_pred)
        y_train_pred = y_train_pred > self.epoch_threshold

        with open(os.path.join(self.exp_dir, "predicts-training", "train_%s_%s.txt" % (self.split_id, epoch)),
                  "w") as fileW:
            fileW.write(
                "\n".join([":".join([str(item[0]), ",".join(item[1]), ",".join(item[2])])
                           for item in zip(self.train_ids,
                                           self.mlb.inverse_transform(
                                               self.transform_pred(self.y_train)),
                                           self.mlb.inverse_transform(
                                               y_train_pred)
                                           )])
            )

        metric_train = calc_metrics(self.transform_pred(
            self.y_train), y_train_pred, 'train-macro-calc', index=0)
        metric_train["mode"] = "train"
        metric_train["epoch"] = epoch
        self.metrics.append(metric_train)
        df_metrics = pd.DataFrame(self.metrics)
        df_metrics.to_csv(os.path.join(
            self.exp_dir, "metrics_%s.csv" % self.split_id), index=False)
        self.plot_metrics(df_metrics)

        # save the results for test and heldout.
        self.save_epoch_results(epoch)

    def transform_pred(self, y):
        raise NotImplementedError("the function is not implemented.")

    def set_best_threshold(self, y, y_pred_prob):
        raise NotImplementedError("needs to implemented in child classes.")


class CallbackTMM(CallbackBase):

    def __init__(self,
                 X_train, y_train,
                 X_test, y_test,
                 X_dev, y_dev,
                 X_test_holdout, y_test_holdout,
                 train_ids, test_ids, dev_ids, heldout_ids,
                 mlb, exp_dir, split_id, patience=5, logger=None):
        super(CallbackTMM, self).__init__(
            X_train, y_train,
            X_test, y_test,
            X_dev, y_dev,
            X_test_holdout, y_test_holdout,
            train_ids, test_ids, dev_ids, heldout_ids,
            mlb, exp_dir, split_id, patience=patience, logger=logger)

    def transform_pred(self, y):
        return get_mlb_output(y)

    def set_best_threshold(self, y, y_pred_prob):
        self.epoch_threshold = 0.50
