import os
import pickle
import logging
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.python.framework.constant_op import constant
from tensorflow.python.ops.gen_math_ops import cos
from analysis.networks.edge_list_gen import get_CPC_IPC_list
from experiments_pl.data_utils import get_CPC_text, get_xml_rem_text
from experiments_pl.features.vectorizer import CNNTokenizer, HuggingFaceTokenizer, LabelEmbedding, PrecomputedEmbedding, TFIDF_Vectorizer
from experiments_pl import constants


GENERIC_FEATURES = [constants.CONTENT_EMB_SCIBERT, 
                    constants.CONTENT_EMB_LONGFORMER, 
                    constants.CONTENT_EMB_TFIDF, 
                    constants.CONTENT_EMB_CNN,
                    constants.CONTENT_EMB_SCIBERT_PRECOMPUTED
                ]


def get_str_list(_list):
    return [str(number) for number in _list]


class PreprocessTargetLabel:

    def __init__(self, logger):
        self.logger = logger

    def pre_process_label():
        raise NotImplementedError("this method needs to implemented in the child class.")


class PrePorcessTargetLabelPatentLandscape(PreprocessTargetLabel):

    def __init__(self, exp_dir, is_flat=True, logger=None):
        super().__init__(logger)
        self.is_flat = is_flat
        self.exp_dir = exp_dir

    def pre_process_label(self, y_train, y_test, y_dev, y_heldout=None):

        hierarchical_tree = None
        y_train = [item for item in y_train]
        y_test = [item for item in y_test]
        y_dev = [item for item in y_dev]

        mlb = MultiLabelBinarizer()
        mlb.fit(y_train)
        pickle.dump(mlb, open(os.path.join(self.exp_dir, "mlb.pkl"), "wb"))
        self.logger.info("classes in mlb : %s" % "|".join(mlb.classes_))

        y_train = mlb.transform(y_train)
        y_test = mlb.transform(y_test)
        y_dev = mlb.transform(y_dev)
        self.logger.info("y_train shape: %s" % str(y_train.shape))
        self.logger.info("y_test shape: %s " % str(y_test.shape))
        self.logger.info("y_dev shape: %s " % str(y_dev.shape))

        if y_heldout:
            y_heldout = [item for item in y_heldout]
            y_heldout = mlb.transform(y_heldout)
            self.logger.info("y_test_heldout shape: %s" % str(y_heldout.shape))

        return y_train, y_test, y_dev, y_heldout, mlb, hierarchical_tree


class DataLoader:

    def __init__(self, preprocess_label, unique_id_label, target_label_column, logger):
        self.preprocess_label = preprocess_label
        self.target_label_column = target_label_column
        self.unique_id_label = unique_id_label
        self.hierarchical_tree = None
        self.logger = logger

    @staticmethod
    def get_reduced_dataset(train, test, dev, heldout):
        train = train[:10]
        test = test[:10]
        dev = dev[:10]
        if heldout is not None:
            heldout = heldout[:10]
        return train, test, dev, heldout

    def load_train_test(self, data, exp_dir, train_ids, test_ids, dev_ids, heldout_ids=None, mode="debug", split_index=0):

        train = data[data[self.unique_id_label].isin(set(train_ids))]
        test = data[data[self.unique_id_label].isin(set(test_ids))]
        dev = data[data[self.unique_id_label].isin(set(dev_ids))]

        train = train.sort_values(by=self.unique_id_label, ascending=True)
        test = test.sort_values(by=self.unique_id_label, ascending=True)
        dev = dev.sort_values(by=self.unique_id_label, ascending=True)
        
        if heldout_ids:
            heldout = data[data[self.unique_id_label].isin(set(heldout_ids))]
            heldout = heldout.sort_values(by=self.unique_id_label, ascending=True)
            self.logger.info("heldout.shape - %s " % str(heldout.shape))
        else:
            heldout = None

        if mode == "debug":
            train, test, dev, heldout = DataLoader.get_reduced_dataset(train, test, dev, heldout)

        print(train.shape)
        open(os.path.join(exp_dir, "ids_train_%s.txt" % split_index), "w").write("\n".join(get_str_list(train[self.unique_id_label].tolist())))
        open(os.path.join(exp_dir, "ids_test_%s.txt" % split_index), "w").write("\n".join(get_str_list(test[self.unique_id_label].tolist())))
        open(os.path.join(exp_dir, "ids_dev_%s.txt" % split_index), "w").write("\n".join(get_str_list(dev[self.unique_id_label].tolist())))

        if heldout_ids:
            open(os.path.join(exp_dir, "ids_heldout_%s.txt" % split_index), "w").write("\n".join(get_str_list(heldout[self.unique_id_label].tolist())))
            self.logger.info("data_utils heldout ids : %s " % str(heldout[self.unique_id_label].tolist()[:10]) )

        self.logger.info("data_utils train ids : %s " % str(train[self.unique_id_label].tolist()[:10]) )
        self.logger.info("data_utils test ids : %s " % str(test[self.unique_id_label].tolist()[:10]) )
        self.logger.info("data_utils dev ids : %s " % str(dev[self.unique_id_label].tolist()[:10]) )
    
        self.logger.info("train len: %s " % len(train))
        self.logger.info("test len: %s " % len(test))
        self.logger.info("dev len: %s " % len(dev))

        y_train = train[self.target_label_column].tolist()
        y_test = test[self.target_label_column].tolist()
        y_dev = dev[self.target_label_column].tolist()

        # preprocess_label = PrePorcessTargetLabelPatentLandscape(exp_dir, True)
        if heldout_ids:
            y_test_heldout = heldout[self.target_label_column].tolist()
            y_train, y_test, y_dev, y_heldout, mlb, hier_tree = self.preprocess_label.pre_process_label(y_train, y_test, y_dev, y_test_heldout)
        else:
            y_train, y_test, y_dev, y_heldout, mlb, hier_tree = self.preprocess_label.pre_process_label(y_train, y_test, y_dev, None)

        return train, test, dev, heldout, y_train, y_test, y_dev, y_heldout, mlb, hier_tree


class FeatureGenerator:

    def __init__(self, train, test, dev, heldout=None, exp_dir=None, mode="debug", logger=None, id_column=None):

        self.train = train
        self.test = test
        self.dev = dev
        self.heldout = heldout

        self.X_train = list()
        self.X_dev = list()
        self.X_test = list()
        self.X_heldout = list()

        self.exp_dir = exp_dir
        self.mode = mode
        
        self.embedding_weights = None
        self.logger = logger
        self.id_column = id_column


    def generate_generic_features(self, emb_info, X_train_text, X_test_text, X_dev_text, X_heldout_text):

        if emb_info["type"] == constants.CONTENT_EMB_SCIBERT or emb_info["type"] == constants.CONTENT_EMB_LONGFORMER:
            hugging_face_tokenizer = HuggingFaceTokenizer(emb_info["path"], emb_info["max_len"], emb_info["type"])
            self.X_train.extend(hugging_face_tokenizer.transform(X_train_text))
            self.X_test.extend(hugging_face_tokenizer.transform(X_test_text))
            self.X_dev.extend(hugging_face_tokenizer.transform(X_dev_text))

            if X_heldout_text:
                self.X_heldout.extend(hugging_face_tokenizer.transform(X_heldout_text))

        elif emb_info["type"] == constants.CONTENT_EMB_TFIDF:

            tfidf = TFIDF_Vectorizer()
            tfidf.fit(X_train_text)
            tfidf.save(os.path.join(self.exp_dir, "tf_idf.pkl"))

            self.X_train.append(tfidf.transform(X_train_text))
            self.X_test.append(tfidf.transform(X_test_text))
            self.X_dev.append(tfidf.transform(X_dev_text))

            if self.heldout is not None:
                self.X_heldout.append(tfidf.transform(X_heldout_text))
        
        elif emb_info["type"] == constants.CONTENT_EMB_CNN:

            cnn_tokenizer = CNNTokenizer(emb_info["max_len"])
            cnn_tokenizer.fit(X_train_text)
            if self.mode == "debug" and os.path.exists(os.path.join(self.exp_dir, "embedding_weights.pkl")):
                self.embedding_weights = pickle.load(open(os.path.join(self.exp_dir, "embedding_weights.pkl"), "rb"))
            else:
                self.embedding_weights = cnn_tokenizer.load_embeddings_weights(emb_info["path"])
                pickle.dump(self.embedding_weights, open(os.path.join(self.exp_dir, "embedding_weights.pkl"), "wb"))
            
            self.X_train.append(cnn_tokenizer.transform(X_train_text))
            self.X_test.append(cnn_tokenizer.transform(X_test_text))
            self.X_dev.append(cnn_tokenizer.transform(X_dev_text))

            if self.heldout is not None:
                self.X_heldout.append(cnn_tokenizer.transform(X_heldout_text))
        
        elif emb_info["type"] == constants.CONTENT_EMB_SCIBERT_PRECOMPUTED:
            precomputed_embedding = PrecomputedEmbedding(emb_info["path"])
            self.X_train.append(precomputed_embedding.transform(self.train[self.id_column]))
            self.X_test.append(precomputed_embedding.transform(self.test[self.id_column]))
            self.X_dev.append(precomputed_embedding.transform(self.dev[self.id_column]))
            if self.heldout is not None:
                self.X_heldout.append(precomputed_embedding.transform(self.heldout[self.id_column]))

        else:
            raise Exception("unknown emb type : %s " % emb_info["type"])

    def generate_specific_features(emb_info, X_train_text, X_test_text, X_dev_text, X_heldout_text):
        pass

    def get_split_text(self, content_fields, sel_fields, label=False):
        train_documents = FeatureGenerator.get_field_documents(content_fields, self.train)
        test_documents = FeatureGenerator.get_field_documents(content_fields, self.test)
        dev_documents = FeatureGenerator.get_field_documents(content_fields, self.dev)
        if self.heldout is not None:
            heldout_documents = FeatureGenerator.get_field_documents(content_fields, self.heldout)

        X_train_text = FeatureGenerator.get_text(train_documents, sel_fields)
        X_test_text = FeatureGenerator.get_text(test_documents, sel_fields)
        X_dev_text = FeatureGenerator.get_text(dev_documents, sel_fields)
        X_heldout_text = None
        if self.heldout is not None:
            X_heldout_text = FeatureGenerator.get_text(heldout_documents, sel_fields)

        if label:
            X_train_text, X_test_text, X_dev_text, X_heldout_text = self.append_additional_text(X_train_text, X_test_text, X_dev_text, X_heldout_text)
        return X_train_text, X_test_text, X_dev_text, X_heldout_text

    def generate_feature_vector(self, emb_info_list, fields):
        
        train_documents = FeatureGenerator.get_field_documents(fields, self.train)
        test_documents = FeatureGenerator.get_field_documents(fields, self.test)
        dev_documents = FeatureGenerator.get_field_documents(fields, self.dev)
        if self.heldout is not None:
            heldout_documents = FeatureGenerator.get_field_documents(fields, self.heldout)

        emb_info_list_return = list()
        for emb_info in emb_info_list:
            self.logger.info("generating embedding for : %s " % emb_info["type"])

            X_train_text = X_test_text = X_dev_text = X_heldout_text = None

            if emb_info.get("fields"):
                X_train_text = FeatureGenerator.get_text(train_documents, emb_info["fields"])
                X_test_text = FeatureGenerator.get_text(test_documents, emb_info["fields"])
                X_dev_text = FeatureGenerator.get_text(dev_documents, emb_info["fields"])
                if self.heldout is not None:
                    X_heldout_text = FeatureGenerator.get_text(heldout_documents, emb_info["fields"])

            if emb_info["type"] in GENERIC_FEATURES:
                self.generate_generic_features(emb_info, X_train_text, X_test_text, X_dev_text, X_heldout_text)
            else:
                self.generate_specific_features(emb_info, X_train_text, X_test_text, X_dev_text, X_heldout_text)

            self.logger.info("embedding type: %s and embedding size: %s " % (emb_info["type"], self.X_train[-1].shape[1]))
            emb_info_list_return.append({**emb_info, **{"emb_size": self.X_train[-1].shape[1]}})
        return emb_info_list_return

    @staticmethod
    def get_text(documents, fields):
        texts = ["" for index in range(len(documents))]
        for field in fields:
            for document_index, document in enumerate(documents):
                texts[document_index] += " " + document[field]
        return texts

    @staticmethod
    def get_field_documents(fields, df):
        documents = list()
        for index, field in enumerate(fields):
            for row_index, text in enumerate(df[field].tolist()):
                
                if type(text) is not str:
                    text = ""
                else:
                    text = get_xml_rem_text(text.lower())

                if index == 0:
                    documents.append({field: text})
                else:
                    documents[row_index][field] = text
        return documents

    def append_additional_text(self, X_train_text, X_test_text, X_train_dev, X_heldout_text):
        raise NotImplementedError


class FeatureGeneratorPatentLandscape(FeatureGenerator):

    def __init__(self, train, test, dev, heldout, exp_dir, mode, cpc_code_filter_list, label_desc_file_path, emb_dict, logger=None):
        super().__init__(train, test, dev, heldout=heldout, exp_dir=exp_dir, mode=mode, logger=logger)

        self.cpc_code_filter_list = cpc_code_filter_list
        self.label_desc_file_path = label_desc_file_path
        self.emb_dict = emb_dict

        # load cls code for label embedding.
        self.train_cls_codes = get_CPC_IPC_list(self.train.CPC.tolist(), self.train.IPC.tolist())
        self.test_cls_codes = get_CPC_IPC_list(self.test.CPC.tolist(), self.test.IPC.tolist())
        self.dev_cls_codes = get_CPC_IPC_list(self.dev.CPC.tolist(), self.dev.IPC.tolist())
        if self.heldout is not None:
            self.heldout_cls_codes = get_CPC_IPC_list(self.heldout.CPC.tolist(), self.heldout.IPC.tolist())

        # Load label text.
        self.X_train_label_text = get_CPC_text(get_CPC_IPC_list(self.train.CPC.tolist(), self.train.IPC.tolist()), self.cpc_code_filter_list, self.label_desc_file_path)
        self.X_dev_label_text = get_CPC_text(get_CPC_IPC_list(self.dev.CPC.tolist(), self.dev.IPC.tolist()), self.cpc_code_filter_list, self.label_desc_file_path)
        self.X_test_label_text = get_CPC_text(get_CPC_IPC_list(self.test.CPC.tolist(), self.test.IPC.tolist()), self.cpc_code_filter_list, self.label_desc_file_path)
        if self.heldout is not None:
            self.X_test_heldout_label_text = get_CPC_text(get_CPC_IPC_list(self.heldout.CPC.tolist(), self.heldout.IPC.tolist()), self.cpc_code_filter_list, self.label_desc_file_path)

    def append_additional_text(self, X_train_text, X_test_text, X_dev_text, X_heldout_text):
        # Load label text.
        X_train_label_text = get_CPC_text(get_CPC_IPC_list(self.train.CPC.tolist(), self.train.IPC.tolist()), self.cpc_code_filter_list, self.label_desc_file_path)
        X_dev_label_text = get_CPC_text(get_CPC_IPC_list(self.dev.CPC.tolist(), self.dev.IPC.tolist()), self.cpc_code_filter_list, self.label_desc_file_path)
        X_test_label_text = get_CPC_text(get_CPC_IPC_list(self.test.CPC.tolist(), self.test.IPC.tolist()), self.cpc_code_filter_list, self.label_desc_file_path)

        if self.heldout is not None:
            X_test_heldout_label_text = get_CPC_text(get_CPC_IPC_list(self.heldout.CPC.tolist(), self.heldout.IPC.tolist()),
                                                    self.cpc_code_filter_list, self.label_desc_file_path)

        # if emb_info.get("label_text"):
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
        
        if self.heldout is not None:
            if X_heldout_text:
                X_heldout_text = [" ".join(list(item)) for item in zip(X_heldout_text, X_test_heldout_label_text)]
            else:
                X_heldout_text = X_heldout_text

        return X_train_text, X_test_text, X_dev_text, X_heldout_text

    def generate_specific_features(self, emb_info, X_train_text, X_test_text, X_dev_text, X_heldout_text):

        if emb_info["type"] == constants.LABEL_EMB_TEXT_BERT or emb_info["type"] == constants.LABEL_EMB_GRAPH:

                if emb_info["type"] == constants.LABEL_EMB_TEXT_BERT:
                    emb_size = 768
                elif emb_info["type"] == constants.LABEL_EMB_GRAPH:
                    emb_size = 128

                label_embedding_vectorizer = LabelEmbedding(self.emb_dict[emb_info["type"]], self.cpc_code_filter_list, emb_size)
                self.X_train.append(label_embedding_vectorizer.transform(self.train_cls_codes))
                self.X_test.append(label_embedding_vectorizer.transform(self.test_cls_codes))
                self.X_dev.append(label_embedding_vectorizer.transform(self.dev_cls_codes))
                if self.heldout is not None:
                    self.X_heldout.append(label_embedding_vectorizer.transform(self.heldout_cls_codes))


        # Label embedding based on tf-idf feature vector.
        elif emb_info["type"] == constants.LABEL_EMB_TEXT_TFIDF:
            tfidf = TFIDF_Vectorizer(emb_info["path"])
            self.X_train.append(tfidf.transform(self.X_train_label_text))
            self.X_test.append(tfidf.transform(self.X_test_label_text))
            self.X_dev.append(tfidf.transform(self.X_dev_label_text))
            if self.heldout is not None:
                self.X_heldout.append(tfidf.transform(self.X_test_heldout_label_text))


