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

import os
from collections import Counter

import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer


def agg_avg(emb_list, emb_size):
    if len(emb_list):
        emb_avg = np.mean(emb_list, 0)
        return emb_avg
    else:
        emb_avg = np.asarray([0.0 for index in range(emb_size)])
        return emb_avg


def get_doc_rep_from_label_emb(labels, label_emb, cpc_not_found, emb_size=128, cpc_code_filter_list=None):
    doc_rep_list = list()

    len_count = list()

    for cpcs in labels:
        cpcs = [cpc.replace(' ', '') for cpc in cpcs]

        if cpc_code_filter_list:
            cpcs = [cpc for cpc in cpcs if cpc in cpc_code_filter_list]

        len_count.append(len(cpcs))

        emb_list = list()
        for cpc in cpcs:
            if cpc in label_emb:
                emb_list.append(label_emb[cpc])
            else:
                cpc_not_found.add(cpc)
        doc_rep_list.append(agg_avg(emb_list, emb_size))

    print(np.mean(len_count))
    print(Counter(len_count).most_common())

    return doc_rep_list


def load_docrep_label(train_cpcs, test_cpcs, dev_cpcs, label_emb_keyvector, emb_size=128, cpc_code_filter_list=None):
    cpc_not_found = set()
    return tf.cast(
        get_doc_rep_from_label_emb(train_cpcs, label_emb_keyvector, cpc_not_found, emb_size,
                                   cpc_code_filter_list=cpc_code_filter_list),
        'float32'), \
        tf.cast(get_doc_rep_from_label_emb(test_cpcs, label_emb_keyvector, cpc_not_found, emb_size,
                                           cpc_code_filter_list=cpc_code_filter_list), 'float32'), \
        tf.cast(get_doc_rep_from_label_emb(dev_cpcs, label_emb_keyvector, cpc_not_found, emb_size,
                                           cpc_code_filter_list=cpc_code_filter_list), 'float32'), \
        cpc_not_found


def load_docrep_tfidf(X_train_text, X_test_text, X_dev_text):
    tf_idf = TfidfVectorizer(lowercase=True, stop_words={'english'})
    tf_idf.fit(X_train_text)
    return tf.cast(tf_idf.transform(X_train_text).toarray(), 'float32'), \
        tf.cast(tf_idf.transform(X_test_text).toarray(), 'float32'), \
        tf.cast(tf_idf.transform(X_dev_text).toarray(), 'float32')


def get_embs(_ids, embs):
    return np.asarray([embs[_id] for _id in _ids if _id in embs])


def load_docrep(train_ids, test_ids, dev_ids, embs):
    return tf.cast(get_embs(train_ids, embs), 'float32'), \
        tf.cast(get_embs(test_ids, embs), 'float32'), \
        tf.cast(get_embs(dev_ids, embs), 'float32')


def load_splits(splits_dir):
    metrics = list()
    splits = [[int(line.strip()) for line in open(os.path.join(splits_dir, filename)).readlines()]
              for filename in os.listdir(splits_dir)]

    splits_index_list = list()
    for index in range(10):
        if index == 9:
            test = index
            dev = 0
        else:
            test = index
            dev = index + 1

        train = set(list(range(10))).difference(set([test, dev]))
        splits_index_list.append((train, test, dev))

    return splits_index_list, splits
