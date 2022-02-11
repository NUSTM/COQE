import torch
from transformers import BertTokenizer


class BaseConfig(object):
    def __init__(self, args):
        # common model mode setting
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.crf_type = args.crf_type

        # camera-CQE: 4, car-CQE: 5, ele-CQE: 6
        self.csr_radius = args.csr_radius

        self.generate_method = args.generate_method
        self.classification_model_type = args.classification_model_type

        self.file_type = args.file_type
        self.premodel_path = args.premodel_path

        self.path = PathConfig(
            self.csr_radius, self.file_type, self.premodel_path, self.generate_method
        )

        self.bert_tokenizer = BertTokenizer.from_pretrained(self.path.bert_model_path)


class PathConfig(object):
    def __init__(self, csr_radius, file_type, premodel_path, generate_method):
        """
        :param csr_radius:
        :param file_type:
        :param premodel_path:
        :param generate_method:
        """
        # store split train and test data file path
        dir_name = file_type if file_type == "kesserl14" else "coae2013/" + file_type

        self.standard_path = {
            "train": "../data/{}/train.txt".format(dir_name),
            "dev": "../data/{}/dev.txt".format(dir_name),
            "test": "../data/{}/test.txt".format(dir_name)
        }

        self.pre_process_path = {
            "train": "./rule_data/{}/train_data.txt".format(dir_name),
            "dev": "./rule_data/{}/dev_data.txt".format(dir_name),
            "test": "./rule_data/{}/test_data.txt".format(dir_name)
        }

        self.database_path = {
            "train": "./rule_data/{}/train_database_{}.txt".format(dir_name, str(csr_radius)),
            "dev": "./rule_data/{}/dev_database_{}.txt".format(dir_name, str(csr_radius)),
            "test": "./rule_data/{}/test_database_{}.txt".format(dir_name, str(csr_radius))
        }

        self.feature_embed_path = {
            "train": "./data/{}/train_feature.txt".format(dir_name),
            "dev": "./data/{}/dev_feature.txt".format(dir_name),
            "test": "./data/{}/test_feature.txt".format(dir_name)
        }

        self.stanford_path = "{}/stanford-corenlp-full-2018-02-27".format(premodel_path)

        if file_type == "kesserl14":
            self.bert_model_path = premodel_path + "base_uncased/"
        else:
            self.bert_model_path = premodel_path + "base_chinese/"

        lang = "zh" if file_type in {"car", "ele"} else "en"

        self.keyword_path = "./rule_data/{}_keyword_vocab.txt".format(lang)

        self.csr_path = "./rule_data/{}/csr_{}_{}.txt".format(dir_name, str(csr_radius), generate_method)
        self.csr_model_path = "./rule_data/{}/model_{}_{}.pkl".format(dir_name, str(csr_radius), generate_method)
