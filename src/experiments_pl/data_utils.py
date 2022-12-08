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
import logging
import os
import numpy as np
import pickle
import re

import gensim
import pandas as pd
import tensorflow as tf
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import AutoTokenizer

from analysis.networks.edge_list_gen import get_CPC_IPC_list
from application.classification.label_binarizer import get_multitask_output
from document_rep.doc_rep_utils import load_docrep_label, get_doc_rep_from_label_emb
from experiments_pl.features.vectorizer import TFIDF_Vectorizer, HuggingFaceTokenizer, LabelEmbedding, CNNTokenizer
from experiments_pl import constants

def get_CPC_text(cpcs_list, cpc_code_filter_list, label_details_file_path):
    pd_label_details = pd.read_csv(label_details_file_path, sep="\t")
    descs = pd_label_details.desc.tolist()
    descs = [item.replace("-", " ").lower() if type(item) is str else "" for item in descs]
    label_desc_dict = dict(zip(pd_label_details.label.tolist(), descs))

    cpcs_text_list = list()
    for cpcs in cpcs_list:
        text = ""
        for cpc in cpcs:
            if cpc_code_filter_list:
                if cpc in cpc_code_filter_list and cpc in label_desc_dict:
                    text += " " + label_desc_dict[cpc]
            else:
                if cpc in label_desc_dict:
                    text += " " + label_desc_dict[cpc]
        cpcs_text_list.append(text)
    return cpcs_text_list


def get_xml_rem_text(text):
    return re.sub('<[^<]+>', "", text)


def get_field_documents(fields, df):
    docs = list()
    for index, field in enumerate(fields):
        for row_index, text in enumerate(df[field].tolist()):
            
            if type(text) is not str:
                text = ""
            else:
                text = get_xml_rem_text(text.lower())

            if index == 0:
                docs.append({field: text})
            else:
                docs[row_index][field] = text
    return docs


def get_str_list(_list):
    return [str(number) for number in _list]


def get_train_test_data(data, config, exp_dir, train_ids, test_ids, dev_ids, heldout_ids, split_index,
                        emb_dict, mode="debug", cpc_code_filter_list=None, text_only=False):
    train = data[data.Family.isin(set(train_ids))]
    test = data[data.Family.isin(set(test_ids))]
    dev = data[data.Family.isin(set(dev_ids))]

    train = train.sort_values(by="Family", ascending=True)
    test = test.sort_values(by="Family", ascending=True)
    dev = dev.sort_values(by="Family", ascending=True)
    

    if heldout_ids:
        data_heldout = data[data.Family.isin(set(heldout_ids))]
        data_heldout = data_heldout.sort_values(by="Family", ascending=True)


    if mode == "debug":
        train = train[:10]
        test = test[:10]
        dev = dev[:10]
        if heldout_ids:
            data_heldout = data_heldout[:10]
            logging.info("data_heldout len: %s" % len(data_heldout))

    print(train.shape)
    open(os.path.join(exp_dir, "ids_train_%s.txt" % split_index), "w").write("\n".join(get_str_list(train.Family.tolist())))
    open(os.path.join(exp_dir, "ids_test_%s.txt" % split_index), "w").write("\n".join(get_str_list(test.Family.tolist())))
    open(os.path.join(exp_dir, "ids_dev_%s.txt" % split_index), "w").write("\n".join(get_str_list(dev.Family.tolist())))


    if heldout_ids:
        open(os.path.join(exp_dir, "ids_heldout_%s.txt" % split_index), "w").write("\n".join(get_str_list(data_heldout.Family.tolist())))


    logging.info("data_utils train ids : %s " % str(train.Family.tolist()[:10]) )
    logging.info("data_utils test ids : %s " % str(test.Family.tolist()[:10]) )
    logging.info("data_utils dev ids : %s " % str(dev.Family.tolist()[:10]) )
    logging.info("data_utils heldout ids : %s " % str(data_heldout.Family.tolist()[:10]) )

    logging.info("train len: %s " % len(train))
    logging.info("test len: %s " % len(test))
    logging.info("dev len: %s " % len(dev))

    y_train = train.labels.tolist()
    y_test = test.labels.tolist()
    y_dev = dev.labels.tolist()

    mlb = MultiLabelBinarizer()
    mlb.fit(y_train)
    pickle.dump(mlb, open(os.path.join(exp_dir, "mlb.pkl"), "wb"))

    logging.info("classes in mlb : %s" % "|".join(mlb.classes_))

    y_train = mlb.transform(y_train)
    y_test = mlb.transform(y_test)
    y_dev = mlb.transform(y_dev)
    logging.info("y_train shape: %s" % str(y_train.shape))
    logging.info("y_test shape: %s " % str(y_test.shape))
    logging.info("y_dev shape: %s " % str(y_dev.shape))

    if heldout_ids:
        y_test_heldout = data_heldout.labels.tolist()
        y_test_heldout = mlb.transform(y_test_heldout)
        logging.info("y_test_heldout shape: %s" % str(y_test_heldout.shape))

    train_cls_codes = get_CPC_IPC_list(train.CPC.tolist(), train.IPC.tolist())
    test_cls_codes = get_CPC_IPC_list(test.CPC.tolist(), test.IPC.tolist())
    dev_cls_codes = get_CPC_IPC_list(dev.CPC.tolist(), dev.IPC.tolist())
    if heldout_ids:
        heldout_cls_codes = get_CPC_IPC_list(data_heldout.CPC.tolist(), data_heldout.IPC.tolist())

    # load the documents
    FIELDS = ['Title', 'Abstract', 'ft_claims', 'ft_description']
    train_documents = get_field_documents(FIELDS, train)
    test_documents = get_field_documents(FIELDS, test)
    dev_documents = get_field_documents(FIELDS, dev)
    if heldout_ids:
        heldout_documents = get_field_documents(FIELDS, data_heldout)

    # Load label text.
    X_train_label_text = get_CPC_text(get_CPC_IPC_list(train.CPC.tolist(), train.IPC.tolist()), cpc_code_filter_list, config["label_desc_file_path"])
    X_dev_label_text = get_CPC_text(get_CPC_IPC_list(dev.CPC.tolist(), dev.IPC.tolist()), cpc_code_filter_list, config["label_desc_file_path"])
    X_test_label_text = get_CPC_text(get_CPC_IPC_list(test.CPC.tolist(), test.IPC.tolist()), cpc_code_filter_list, config["label_desc_file_path"])

    # load the heldout dataset in case heldout ids exist.
    if heldout_ids:
        X_test_heldout_label_text = get_CPC_text(get_CPC_IPC_list(data_heldout.CPC.tolist(), data_heldout.IPC.tolist()),
                                                 cpc_code_filter_list, config["label_desc_file_path"])

    # These variables contains the embeddings which will be used for feature vector generation in next step.
    X_train = list()
    X_test = list()
    X_dev = list()
    X_heldout = list()

    # Embedding info list.
    emb_info_list = list()

    # Embedding weights are loaded from the text embedding matrix.
    embedding_weights = None

    # if we want to load only the text, with no pretrained embedding.
    if text_only:

        X_train_text = X_test_text = X_dev_text = X_test_heldout_text = None
        return X_train_text, y_train, X_test_text, y_test, X_dev_text, y_dev, X_test_heldout_text, y_test_heldout, mlb

    else:

        for emb_info in config["doc_rep_params"]["embs"]:

            logging.info("emb_type: %s " % json.dumps(emb_info))
            emb_size = None

            X_train_text = X_test_text = X_dev_text = X_test_heldout_text = None

            if emb_info.get("fields"):
                # For each embedding, we can define the fields from which the data should be included.
                X_train_text = get_text(train_documents, emb_info["fields"])
                X_test_text = get_text(test_documents, emb_info["fields"])
                X_dev_text = get_text(dev_documents, emb_info["fields"])
                X_test_heldout_text = get_text(heldout_documents, emb_info["fields"])

            """
            Is the label-text included in embedding?
            """
            if emb_info.get("label_text"):

                if X_train_text:
                    X_train_text = [" ".join(list(item)) for item in zip(X_train_text, X_train_label_text)]
                else:
                    X_train_text = X_train_label_text
                
                if X_test_text:
                    X_test_text = [" ".join(list(item)) for item in zip(X_test_text, X_test_label_text)]
                else:
                    X_test_text = X_test_label_text

                if X_dev_text:
                    X_dev_text = [" ".join(list(item)) for item in zip(X_dev_text, X_dev_label_text)]
                else:
                    X_dev_text = X_dev_label_text
                

                if heldout_ids:
                    if X_test_heldout_text:
                        X_test_heldout_text = [" ".join(list(item)) for item in zip(X_test_heldout_text, X_test_heldout_label_text)]
                    else:
                        X_test_heldout_text = X_test_heldout_label_text

            # bert: label embedding
            # longformer: text feature vector
            if emb_info["type"] == constants.CONTENT_EMB_SCIBERT or emb_info["type"] == constants.CONTENT_EMB_LONGFORMER:
                hugging_face_tokenizer = HuggingFaceTokenizer(emb_info["path"], emb_info["max_len"])
                X_train.extend(hugging_face_tokenizer.transform(X_train_text))
                X_test.extend(hugging_face_tokenizer.transform(X_test_text))
                X_dev.extend(hugging_face_tokenizer.transform(X_dev_text))
                if heldout_ids:
                    X_heldout.extend(hugging_face_tokenizer.transform(X_test_heldout_text))

            # l-t-bert: label embedding
            # l-g-n2v: label embedding from the co-occurrence graph
            elif emb_info["type"] == constants.LABEL_EMB_TEXT_BERT or emb_info["type"] == constants.LABEL_EMB_GRAPH:

                if emb_info["type"] == constants.LABEL_EMB_TEXT_BERT:
                    emb_size = 768
                elif emb_info["type"] == constants.LABEL_EMB_GRAPH:
                    emb_size = 128

                label_embedding_vectorizer = LabelEmbedding(emb_dict[emb_info["type"]], cpc_code_filter_list, emb_size)
                X_train.append(label_embedding_vectorizer.transform(train_cls_codes))
                X_test.append(label_embedding_vectorizer.transform(test_cls_codes))
                X_dev.append(label_embedding_vectorizer.transform(dev_cls_codes))


                if heldout_ids:
                    X_heldout.append(label_embedding_vectorizer.transform(heldout_cls_codes))

            # tf-idf : Embeddings for the content fields.
            elif emb_info["type"] == constants.CONTENT_EMB_TFIDF:

                tfidf = TFIDF_Vectorizer()
                tfidf.fit(X_train_text)
                tfidf.save(os.path.join(exp_dir, "tf_idf.pkl"))

                X_train.append(tfidf.transform(X_train_text))
                X_test.append(tfidf.transform(X_test_text))
                X_dev.append(tfidf.transform(X_dev_text))

                if heldout_ids:
                    X_heldout.append(tfidf.transform(X_test_heldout_text))

            # Label embedding based on tf-idf feature vector.
            elif emb_info["type"] == constants.LABEL_EMB_TEXT_TFIDF:
                tfidf = TFIDF_Vectorizer(emb_info["path"])
                X_train.append(tfidf.transform(X_train_label_text))
                X_test.append(tfidf.transform(X_test_label_text))
                X_dev.append(tfidf.transform(X_dev_label_text))

                if heldout_ids:
                    X_heldout.append(tfidf.transform(X_test_heldout_text))

            # CNN based feature vector.
            elif emb_info["type"] == constants.CONTENT_EMB_CNN:

                cnn_tokenizer = CNNTokenizer(emb_info["max_len"])
                cnn_tokenizer.fit(X_train_text)
                if mode == "debug" and os.path.exists(os.path.join(exp_dir, "embedding_weights.pkl")):
                    embedding_weights = pickle.load(open(os.path.join(exp_dir, "embedding_weights.pkl"), "rb"))
                else:
                    embedding_weights = cnn_tokenizer.load_embeddings_weights(emb_info["path"])
                    pickle.dump(embedding_weights, open(os.path.join(exp_dir, "embedding_weights.pkl"), "wb"))
                
                X_train.append(cnn_tokenizer.transform(X_train_text))
                X_test.append(cnn_tokenizer.transform(X_test_text))
                X_dev.append(cnn_tokenizer.transform(X_dev_text))

                if heldout_ids:
                    X_heldout.append(cnn_tokenizer.transform(X_test_heldout_text))

            else:
                raise Exception("unknown emb type : %s " % emb_info["type"])

            logging.info("embedding type: %s and embedding size: %s " % (emb_info["type"], X_train[-1].shape[1]))
            emb_info_list.append({**emb_info, **{"emb_size": X_train[-1].shape[1]}})

        for index, emb_info in enumerate(emb_info_list):
            print("emb info index - %s - value - %s " % (index, json.dumps(emb_info)))

        if heldout_ids:
            return X_train, y_train, X_test, y_test, X_dev, y_dev, X_heldout, y_test_heldout, mlb, emb_info_list, embedding_weights
        else:
            return X_train, y_train, X_test, y_test, X_dev, y_dev, None, None, mlb, emb_info_list, embedding_weights


def load_embeddings(emb_info_list):
    emb_type_doc_emb_dict = dict()
    for emb_info in emb_info_list:
        if emb_info["type"] in [constants.LABEL_EMB_GRAPH, constants.LABEL_EMB_TEXT_BERT]:
            if "txt" in emb_info["path"]:
                label_emb_dict = KeyedVectors.load_word2vec_format(emb_info["path"], binary=False)
            elif "pkl" in emb_info["path"]:
                label_emb_dict = pickle.load(open(emb_info["path"], "rb"))
                for label in label_emb_dict:
                    label_emb_dict[label] = label_emb_dict[label][0]
            emb_type_doc_emb_dict[emb_info["type"]] = label_emb_dict
    return emb_type_doc_emb_dict


def get_dataset_ids(split_id, config, k, logger):

    logger.info("retrieving dataset ids.")
    heldout_ids = [int(item.strip()) for item in
                       open(os.path.join(config["input_dir_path"], config["split_dir"], "test-split.txt"))]

    splits = [[int(item.strip()) for item in
               open(os.path.join(config["input_dir_path"], config["split_dir"], "split-%s.txt" % index))]
              for index in range(1, k + 1)]

    splits_index = list()
    for index in range(1, k + 1):
        if index == k:
            splits_index.append([index, 1, set([item for item in range(1, k + 1)]).difference(set([index, 1]))])
        else:
            splits_index.append(
                [index, index + 1, set([item for item in range(1, k + 1)]).difference(set([index, index + 1]))])

    test_split_index = splits_index[split_id][0]
    dev_split_index = splits_index[split_id][1]
    train_split_index_list = splits_index[split_id][2]

    train_ids = list()
    _ = [train_ids.extend(splits[index - 1]) for index in train_split_index_list]
    test_ids = splits[test_split_index - 1]
    dev_ids = splits[dev_split_index - 1]

    logger.info("num of train_ids: %s" % len(train_ids))
    logger.info("num of test_ids: %s" % len(test_ids))
    logger.info("num of dev_ids: %s" % len(dev_ids))
    logger.info("num of heldout_ids: %s" % len(heldout_ids))

    return sorted(train_ids), sorted(dev_ids), sorted(test_ids), sorted(heldout_ids)


def get_text(documents, fields):
    """Generate the text for the fields.

    Args:
        train_documents (list): train documents
        test_documents (list): test documents
        dev_documents (list): dev documents
        heldout_documents (list): heldout documents
        fields (list): The fields 
    """

    texts = ["" for index in range(len(documents))]
    for field in fields:
        for document_index, document in enumerate(documents):
            texts[document_index] += " " + document[field]
    return texts
