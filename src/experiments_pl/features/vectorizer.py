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

import pickle
import numpy as np

import gensim
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline, FeatureUnion, Pipeline
from transformers import BertTokenizer, LongformerTokenizer

from document_rep.doc_rep_utils import get_doc_rep_from_label_emb
from experiments_pl import constants


class FeatureVectorizer:

    def __init__(self, filename):
        pass

    def fit(self):
        raise NotImplementedError("This class needs to be implemented in the Child class.")

    def transform(self):
        raise NotImplementedError("This class needs to be implemented in the Child class.")


class TFIDF_Vectorizer(FeatureVectorizer):

    def __init__(self, filename=None):
        if filename:
            self.tf_idf = pickle.load(open(filename, "rb"))
        else:
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
                transformer_list=[('context', Pipeline(steps=[('vect', context_features)]))])
            self.tf_idf = make_pipeline(vectorizer)

    def fit(self, text):
        self.tf_idf.fit(text)

    def transform(self, text):
        return tf.cast(self.tf_idf.transform(text).toarray(), "float32")

    def save(self, filename):
        pickle.dump(self.tf_idf, open(filename, "wb"))


class LabelEmbedding(FeatureVectorizer):

    def __init__(self, label_embedding_filename, cpc_code_filter_list, emb_size):
        self.label_embedding_filename = label_embedding_filename
        self.cpc_code_filter_list = cpc_code_filter_list
        self.emb_size = emb_size

    def transform(self, labels):
        embs = get_doc_rep_from_label_emb(labels,
                                          self.label_embedding_filename,
                                          set(),
                                          self.emb_size,
                                          self.cpc_code_filter_list)
        return tf.cast(embs, "float32")


class InputTokenizer:

    def __init__(self):
        pass

    def fit(self, docs):
        raise NotImplementedError("The docs.")

    def transform(self, docs):
        raise NotImplementedError("")


class HuggingFaceTokenizer(InputTokenizer):

    def __init__(self, filename, max_len, emb_type):
        if emb_type == constants.CONTENT_EMB_LONGFORMER:
            self.tokenizer = LongformerTokenizer.from_pretrained(filename, do_lower_case=True)
        elif emb_type == constants.CONTENT_EMB_SCIBERT:
            self.tokenizer = BertTokenizer.from_pretrained(filename, do_lower_case=True)
        elif emb_type == constants.LABEL_EMB_TEXT_BERT_TRAINABLE:
            self.tokenizer = BertTokenizer.from_pretrained(filename, do_lower_case=True)
        self.max_len = max_len

    def fit(self, docs):
        raise NotImplementedError("not required")

    def transform(self, docs):
        tokenize = self.tokenizer(docs, return_tensors="tf", max_length=self.max_len, truncation=True,
                                  padding='max_length')
        return [tokenize["input_ids"], tokenize["attention_mask"]]


class CNNTokenizer(InputTokenizer):

    def __init__(self, max_len):
        self.max_len = max_len

    def fit(self, docs):
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(lower=True)
        self.tokenizer.fit_on_texts(docs)
        # self.embedding_weights = self.__load_embeddings_weights(self.embedding_filename)

    def transform(self, docs):
        emb = self.tokenizer.texts_to_sequences(docs)
        emb = tf.keras.preprocessing.sequence.pad_sequences(emb, maxlen=self.max_len)
        return tf.cast(emb, "int32")

    def load_embeddings_weights(self, embedding_file_path):
        # initialize the embedding matrix
        fasttext_emb = gensim.models.KeyedVectors.load_word2vec_format(embedding_file_path, binary=False,
                                                                       encoding='utf8')
        embedding_matrix_ft = np.random.random((len(self.tokenizer.word_index) + 1, fasttext_emb.vector_size))
        pas = 0
        for word, i in self.tokenizer.word_index.items():
            try:
                embedding_matrix_ft[i] = fasttext_emb.wv[word]
            except:
                pas += 1
        del fasttext_emb
        return embedding_matrix_ft


class PrecomputedEmbedding(InputTokenizer):

    def __init__(self, path):
        self.path = path
        self.doc_vector = dict()
        with open(self.path, "r") as fileR:
            for line in fileR:
                tokens = line.strip().split(" ")
                doc_id = int(tokens[0])
                vector = [float(item) for item in tokens[1:]]
                self.doc_vector[doc_id] = vector

    def fit(self, docs):
        raise NotImplementedError("this function is not applicable for this class.")

    def transform(self, ids):
        vectors = list()
        for _id in ids:
            vectors.append(self.doc_vector[_id])
        return tf.cast(vectors, "float32")