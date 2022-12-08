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
import pickle
from collections import Counter

import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline
from sklearn.svm import SVC
from tensorflow.python.keras.layers import Embedding
from transformers import TFAutoModel

from application.classification.callbacks import CallbackTMM
from application.classification.label_binarizer import get_mlb_output
from application.classification.model.tmm import TMM
from evaluations.utils import calc_metrics
from experiments_pl import constants


# def get_model(model_config,
#              exp_dir,
#              mlb,
#              emb_info_list,
#              embedding_weights,
#             #  hierarchical_label_tree,
#              logger=None):
    
#     logger.info("defining a classifier for %s model ... " % model_config["model"])
#     model = define_classifier(
#             model_config, 
#             mlb.classes_, 
#             emb_info_list, 
#             embedding_weights, 
#             # hierarchical_label_tree, 
#             logger)

#     if os.path.exists(os.path.join(exp_dir, "last-model.h5")):
#         logger.info("last model found")
#         model.load_weights(os.path.join(exp_dir, "last-model.h5"))
#         logger.info("last model found - model weights loaded.")
#     elif os.path.exists(os.path.join(exp_dir, "best-model.h5")):
#         logger.info("best model found")
#         model.load_weights(os.path.join(exp_dir, "best-model.h5"))
#         logger.info("best model found - model weights loaded.")

#     return model


# def training(X_train, y_train,
#              X_test, y_test,
#              X_dev, y_dev,
#              X_heldout, y_heldout,
#              train_ids, test_ids, dev_ids, heldout_ids,
#              model_config,
#              exp_dir,
#              mlb,
#              emb_info_list,
#              embedding_weights,
#             #  hierarchical_label_tree,
#              label="NA",
#              split_index=0,
#              logger=None):

#     if model_config["model"] == "SVM":
#         logger.info("training a SVM model ... ")
#         return train_svm(X_train, y_train,
#                          X_test, y_test,
#                          X_dev, y_dev,
#                          X_heldout, y_heldout,
#                          model_config,
#                          split_index)

#     else:
#         model = get_model(model_config, 
#                     exp_dir, 
#                     mlb, 
#                     emb_info_list, 
#                     embedding_weights, 
#                     # hierarchical_label_tree, 
#                     logger=logger)
                    
#         logger.info("training a classifier for %s model ... " % model_config["model"])
#         if model_config["model"] == "TMM" or model_config["model"] == "THMM":
#             callback = CallbackTMM(
#                                     X_train, y_train, 
#                                     X_test, y_test,
#                                     X_dev, y_dev,
#                                     X_heldout, y_heldout,
#                                     train_ids, test_ids, dev_ids, heldout_ids, 
#                                     mlb, exp_dir, split_index, logger=logger
#                                 )
#             model.fit(X_train, y_train,
#                       batch_size=model_config["batch_size"],
#                       epochs=model_config["epochs"],
#                       callbacks=callback,
#                       shuffle=True)

#         # elif model_config["model"] == "SingleTask":
#         #     callback = CallbackSingleTask(
#         #                             X_train, get_mlb_output(y_train), 
#         #                             X_test, y_test,
#         #                             X_dev, y_dev,
#         #                             X_heldout, y_heldout,
#         #                             train_ids, test_ids, dev_ids, heldout_ids, 
#         #                             mlb, exp_dir, split_index, logger=logger)
#         #     model.fit(X_train, get_mlb_output(y_train),
#         #               batch_size=model_config["batch_size"],
#         #               epochs=model_config["epochs"],
#         #               callbacks=callback,
#         #               shuffle=True)

#         logger.info("training completed for classifier for %s model ... " % model_config["model"])
#         if X_heldout:
#             y_pred = callback.best_model.predict(X_heldout)
#             y_pred = callback.transform_pred(y_pred)
            
#             os.makedirs(os.path.join(exp_dir, "predicts-test-final"), exist_ok=True)
#             pickle.dump(y_pred, open(os.path.join(exp_dir, "predicts-test-final", "prob_%s.pkl" % split_index), "wb"))
            
#             y_pred = y_pred > callback.best_model_threshold
#             with open(os.path.join(exp_dir, "predicts-test-final", "%s.txt" % split_index), "w") as fileW:
#                 fileW.write(
#                     "\n".join([":".join([str(item[0]), ",".join(item[1]), ",".join(item[2])])
#                                for item in zip(heldout_ids,
#                                                mlb.inverse_transform(
#                                                    y_heldout),
#                                                mlb.inverse_transform(y_pred))])
#                 )
#             perf_final = calc_metrics(y_heldout, y_pred, label=label, index=split_index)
#         else:
#             perf_final = None

#         if X_test:
#             y_pred = callback.best_model.predict(X_test)
#             y_pred = callback.transform_pred(y_pred)
#             os.makedirs(os.path.join(exp_dir, "predicts-test"), exist_ok=True)

#             pickle.dump(y_pred, open(os.path.join(
#                 exp_dir, "predicts-test", "prob_%s.pkl" % split_index), "wb"))
#             y_pred = y_pred > callback.best_model_threshold

#             with open(os.path.join(exp_dir, "predicts-test", "%s.txt" % split_index), "w") as fileW:
#                 fileW.write(
#                     "\n".join([":".join([str(item[0]), ",".join(item[1]), ",".join(item[2])])
#                                for item in zip(test_ids,
#                                                mlb.inverse_transform(y_test),
#                                                mlb.inverse_transform(y_pred))])
#                 )
#             perf_test = calc_metrics(y_test, y_pred, label=label, index=split_index)
#         else:
#             perf_test = None

#         del model
#         # del callback

#         return perf_test, perf_final, callback


def train_svm(X_train_tfidf, y_train,
              X_test_tfidf, y_test,
              X_dev_tfidf, y_dev,
              X_heldout_tfidf, y_heldout,
              config,
              index):
    context_features = FeatureUnion(
        transformer_list=[
            ('word', TfidfVectorizer(
                strip_accents=None,
                lowercase=True,
                analyzer='word',
                ngram_range=(1, 2),
                max_df=1.0,
                min_df=0.0,
                binary=False,
                use_idf=True,
                smooth_idf=True,
                sublinear_tf=True,
                max_features=70000
            )),
            ('char', TfidfVectorizer(
                strip_accents=None,
                lowercase=False,
                analyzer='char',
                ngram_range=(3, 6),
                max_df=1.0,
                min_df=0.0,
                binary=False,
                use_idf=True,
                smooth_idf=True,
                sublinear_tf=True,
                max_features=70000
            )),
        ]
    )

    vectorizer = FeatureUnion(
        transformer_list=[
            ('context', Pipeline(
                steps=[('vect', context_features)]
            )),
        ]
    )

    bclf = OneVsRestClassifier(SVC(kernel=config["kernel"]))
    clf = make_pipeline(vectorizer, bclf)

    clf.fit(X_train_tfidf, y_train)

    metrics_dev = list()
    thresholds = []
    threshold = -1.0
    for index in range(1, 20):
        thresholds.append(threshold)
        threshold += 0.1
    for threshold in thresholds:
        y_dev_pred = clf.decision_function(X_dev_tfidf)
        y_dev_pred = y_dev_pred > threshold
        perf = calc_metrics(y_dev, y_dev_pred, 'svc+tf-idf')
        perf["threshold"] = threshold
        metrics_dev.append(perf)
    df_metrics_dev = pd.DataFrame(metrics_dev)
    threshold = df_metrics_dev.threshold.iloc[df_metrics_dev.f1_macro.idxmax()]

    pred_test = clf.decision_function(X_test_tfidf)
    pred_test = pred_test > threshold

    pred_heldout = clf.decision_function(X_heldout_tfidf)
    pred_heldout = pred_heldout > threshold

    return calc_metrics(y_test, pred_test, 'svc+tf-idf', index), \
        calc_metrics(y_heldout, pred_heldout, 'svc+tf-idf', index)


def init_model(doc_rep, 
        inputs, 
        config, 
        label_list, 
        # hierarchical_label_tree=None, 
        logger=None):
    if config["model"] == "TMM":
        logger.info("intitalizing TMM model.")
        outputs = TMM(config, len(label_list))(doc_rep)

    else:
        raise Exception("Unknown model %s . " % config["model"])

    for index, input in enumerate(inputs):
        logger.info("index - %s - shape - %s" % (index, input.shape))

    model = tf.keras.models.Model(inputs, outputs)
    model.summary()
    adam_optimizer = tfa.optimizers.AdamW(
        weight_decay=0, learning_rate=config["learning_rate"])

    logger.info("Model compilation start ... ")
    model.compile(optimizer=adam_optimizer,
                  loss='binary_crossentropy', metrics=['categorical_accuracy'])
    logger.info("Model compiled")
    return model


# def init_doc_rep(config, emb_info_list, embedding_matrix_ft, logger=None):
#     inputs = list()
#     encoded_embs = list()

#     has_transformer = False
#     trainable = False
#     transformer_path = None

#     for emb_info in emb_info_list:
#         if emb_info["type"] in [constants.CONTENT_EMB_SCIBERT, 
#                                 constants.CONTENT_EMB_LONGFORMER, 
#                                 constants.LABEL_EMB_TEXT_BERT_TRAINABLE]:
#             has_transformer = True
#             if emb_info["trainable"]:
#                 trainable = True
#             transformer_path = emb_info["path"]

#     if has_transformer:
#         bert_layer = TFAutoModel.from_pretrained(transformer_path,
#                                                  output_hidden_states=True,
#                                                  from_pt=True)
#         bert_layer.trainable = trainable

#     # generate embeddings
#     for index, emb_info in enumerate(emb_info_list):
#         if emb_info["type"] in [constants.CONTENT_EMB_SCIBERT, 
#                                 constants.CONTENT_EMB_LONGFORMER, 
#                                 constants.LABEL_EMB_TEXT_BERT_TRAINABLE]:
#             token_inputs = tf.keras.layers.Input(
#                 emb_info["emb_size"], dtype=tf.int32, name='input_word_ids_%s' % index)
#             mask_inputs = tf.keras.layers.Input(
#                 emb_info["emb_size"], dtype=tf.int32, name='input_masks_%s' % index)

#             inputs = inputs + [token_inputs, mask_inputs]

#             print(emb_info["path"])

#             transformer_output = bert_layer([token_inputs, mask_inputs])
#             doc_rep_bert = transformer_output["last_hidden_state"][:, 0, :]
#             encoded_embs.append(doc_rep_bert)

#             logger.info("initialized bert layers ")
#             logger.info("inputs size: %s  " % len(inputs))
#             logger.info("input embedding size: %s " % len(encoded_embs))

#         elif emb_info["type"] == constants.CONTENT_EMB_CNN:

#             input = tf.keras.layers.Input(emb_info["emb_size"], dtype=tf.int32)

#             emb = Embedding(embedding_matrix_ft.shape[0],
#                             embedding_matrix_ft.shape[1],
#                             weights=[embedding_matrix_ft],
#                             trainable=False)(input)

#             conv_layers = []
#             for index, kernel in enumerate(config["kernel"]):
#                 conv = tf.keras.layers.Conv1D(filters=config["filter_size"],
#                                               kernel_size=kernel,
#                                               padding="valid",
#                                               activation="relu",
#                                               strides=1)(emb)
#                 pool = tf.keras.layers.GlobalMaxPooling1D()(conv)
#                 flat = tf.keras.layers.Flatten()(pool)
#                 conv_layers.append(flat)

#             doc_rep_cnn = tf.keras.layers.concatenate(conv_layers)

#             inputs.append(input)
#             encoded_embs.append(doc_rep_cnn)

#         else:
#             input = tf.keras.layers.Input(emb_info["emb_size"], dtype=tf.float64,
#                                           name='embedding_layer_%s' % index)
#             inputs.append(input)
#             encoded_embs.append(input)
#             logger.info("added all input embedding layers")
#             logger.info("inputs size: %s  " % len(inputs))
#             logger.info("input embedding size: %s " % len(encoded_embs))

#     # Apply aggregation if the number of embeddings is more than 1.
#     if len(encoded_embs) > 1:
#         logger.info("number of input embeddings > 1: %s " % len(encoded_embs))
#         logger.info("initializing mapping layer for %s with encoder-size %s " %
#                      (index, config["encoder_size"]))

#         encoding_req = False

#         # apply encoder, if we have embeddings of different size.
#         if len(Counter([emb_info["emb_size"] for emb_info in emb_info_list]).most_common()) > 1:
#             encoding_req = True

#         # or if we have a very large embedding, for example tf-idf.
#         for emb_info in emb_info_list:
#             if emb_info["emb_size"] > 2000:
#                 encoding_req = True

#         # apply encoding layer
#         if encoding_req:
#             embs = list()
#             for index, emb in enumerate(encoded_embs):
#                 embs.append(tf.keras.layers.Dense(config["encoder_size"])(tf.math.l2_normalize(emb, axis=0, epsilon=1e-12, name=None)))

#         # without encoding layer
#         else:
#             embs = encoded_embs

#         if config["emb_agg"] == "concat":
#             doc_rep = tf.concat(embs, 1)
#             logger.info("emb_agg: concat : %s " % str(doc_rep.shape))

#         # sum of the vectors
#         elif config["emb_agg"] == "sum":
#             doc_rep = tf.reduce_sum(embs, 0)
#             logger.info("emb_agg: sum")

#         else:
#             logger.info("unkown emb-agg %s " % config["emb_agg"])
#             raise Exception("unkown emb-agg %s " % config["emb_agg"])

#     # incase of a single embedding, put it directly into the THMM model
#     else:

#         is_tfidf = False
#         for emb_info in emb_info_list:
#             if emb_info["type"] == "l-t-tfidf" or emb_info["type"] == "tf-idf":
#                 is_tfidf = True

#         # if tf-idf, reduce the dimension for efficiency reasons.
#         if is_tfidf:
#             tfidf_lm = tf.keras.layers.Dense(config["encoder_size"])
#             doc_rep = tfidf_lm(encoded_embs[0])
#         else:
#             doc_rep = encoded_embs[0]

#     logger.info("doc_rep.shape : %s " % str(doc_rep.shape))
#     return doc_rep, inputs


# def define_classifier(config, 
#         label_list, 
#         emb_info_list, 
#         embedding_matrix_ft, 
#         # hierarchical_label_tree=None, 
#         logger=None):
#     logger.info("defining classifier ... ")
#     logger.info("config: %s " % json.dumps(config))
#     # logging.info("emb size list:  %s " % str(emb_size_list))
#     logger.info("labels shape:  %s " % str(label_list))

#     doc_rep, inputs = init_doc_rep(config, emb_info_list, embedding_matrix_ft, logger)
#     model = init_model(doc_rep, 
#                     inputs, 
#                     config, 
#                     label_list, 
#                     # hierarchical_label_tree, 
#                     logger)

#     return model
