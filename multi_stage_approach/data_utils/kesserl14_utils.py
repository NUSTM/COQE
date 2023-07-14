import os
import numpy as np
from data_utils.label_parse import LabelParser
from data_utils import shared_utils
from data_utils import current_program_code as cpc
from open_source_utils import stanford_utils
from transformers import BertTokenizer


class DataGenerator(object):
    def __init__(self, config):
        """
        :param config: a program configure
        :return: input_ids, attn_mask, pos_ids, dep_matrix, dep_label_matrix, label_ids
        """
        self.config = config
        self.vocab, self.pos_dict = {"pad": 0, "[CLS]": 1, "[SEP]": 2}, {"pad": 0}
        self.vocab_index, self.pos_index = 5, 5
        self.token_max_len, self.char_max_len = -1, -1

        # store some data using in model
        self.train_data_dict, self.dev_data_dict, self.test_data_dict = {}, {}, {}
        self.bert_tokenizer = BertTokenizer.from_pretrained(config.path.bert_model_path)

        self.elem_col = ["entity_1", "entity_2", "aspect", "result"]

    def create_data_dict(self, data_path, data_type, label_path=None):
        """
        :param data_path: sentence file path
        :param data_type:
        :param label_path: label file path
        :return: a data dict with many parameters
        """
        data_dict = {}

        sent_col, sent_label_col, label_col = cpc.read_standard_file(data_path)

        LP = LabelParser(label_col, ["entity_1", "entity_2", "aspect", "result"])
        label_col, tuple_pair_col = LP.parse_sequence_label("&&", sent_col, file_type="eng")

        # using stanford tool to get some feature data.
        if not os.path.exists(self.config.path.pre_process_data[data_type]):
            sf = stanford_utils.stanfordFeature(sent_col, self.config.path.stanford_path)
            data_dict['standard_token'] = sf.get_tokenizer()
            shared_utils.write_pickle(data_dict, self.config.path.pre_process_data[data_type])

        else:
            data_dict = shared_utils.read_pickle(self.config.path.pre_process_data[data_type])

        self.token_max_len = max(self.token_max_len, shared_utils.get_max_token_length(data_dict['standard_token']))

        data_dict['label_col'] = label_col
        data_dict['comparative_label'] = sent_label_col

        if self.config.model_mode == "bert":
            data_dict['bert_token'] = shared_utils.get_token_col(sent_col, bert_tokenizer=self.bert_tokenizer, dim=1)

            mapping_col = shared_utils.token_mapping_bert(data_dict['bert_token'], data_dict['standard_token'])

            label_col = cpc.convert_eng_label_dict_by_mapping(label_col, mapping_col)

            tuple_pair_col = cpc.convert_eng_tuple_pair_by_mapping(tuple_pair_col, mapping_col)

            data_dict['input_ids'] = shared_utils.bert_data_transfer(
                self.bert_tokenizer,
                data_dict['bert_token'],
                "tokens"
            )

            self.char_max_len = max(self.char_max_len, shared_utils.get_max_token_length(data_dict['input_ids'])) + 2

        else:

            self.vocab, self.vocab_index = shared_utils.update_vocab(
                data_dict['standard_token'],
                self.vocab,
                self.vocab_index,
                dim=2
            )

            data_dict['input_ids'] = shared_utils.transfer_data(data_dict['standard_token'], self.vocab, dim=1)

            self.char_max_len = max(self.char_max_len, shared_utils.get_max_token_length(data_dict['input_ids'])) + 2

        data_dict['tuple_pair_col'] = tuple_pair_col
        print("convert pair number: ", cpc.get_tuple_pair_num(data_dict['tuple_pair_col']))

        token_col = data_dict['standard_token'] if self.config.model_mode == "norm" else data_dict['bert_token']

        data_dict['attn_mask'] = shared_utils.get_mask(token_col, dim=1)

        special_symbol = False

        # multi-label: a sentence denote four sequence-label. [N, 3, sequence_length]
        # result_label: [N, sequence_length] polarity-col: [N, pair_num]
        data_dict['multi_label'], data_dict['result_label'], data_dict['polarity_label'] = \
            cpc.elem_dict_convert_to_multi_sequence_label(
                token_col, label_col, special_symbol=special_symbol
            )

        ################################################################################################################
        # tags to ids
        ################################################################################################################

        data_dict['multi_label'] = shared_utils.transfer_data(
            data_dict['multi_label'],
            self.config.val.norm_id_map,
            dim=2
        )

        data_dict['result_label'] = shared_utils.transfer_data(
            data_dict['result_label'],
            self.config.val.norm_id_map,
            dim=1
        )

        return data_dict

    def generate_data(self):
        self.train_data_dict = self.create_data_dict(
            self.config.path.standard_path['train'],
            "train"
        )

        self.dev_data_dict = self.create_data_dict(
            self.config.path.standard_path['dev'],
            "dev"
        )

        self.test_data_dict = self.create_data_dict(
            self.config.path.standard_path['test'],
            "test"
        )

        self.train_data_dict = self.padding_data_dict(self.train_data_dict)
        self.dev_data_dict = self.padding_data_dict(self.dev_data_dict)
        self.test_data_dict = self.padding_data_dict(self.test_data_dict)

        self.train_data_dict = self.data_dict_to_numpy(self.train_data_dict)
        self.dev_data_dict = self.data_dict_to_numpy(self.dev_data_dict)
        self.test_data_dict = self.data_dict_to_numpy(self.test_data_dict)

    def padding_data_dict(self, data_dict):
        """
        :param data_dict:
        :return:
        """
        pad_key_ids = {0: ["input_ids", "attn_mask", "result_label"],
                       1: ["multi_label"]}

        cur_max_len = self.char_max_len

        param = [{"max_len": cur_max_len, "dim": 1, "pad_num": 0, "data_type": "norm"},
                 {"max_len": cur_max_len, "dim": 2, "pad_num": 0, "data_type": "norm"}]

        for index, key_col in pad_key_ids.items():
            for key in key_col:
                data_dict[key] = shared_utils.padding_data(
                    data_dict[key],
                    max_len=param[index]['max_len'],
                    dim=param[index]['dim'],
                    padding_num=param[index]['pad_num'],
                    data_type=param[index]['data_type']
                )

        return data_dict

    @staticmethod
    def data_dict_to_numpy(data_dict):
        """
        :param data_dict:
        :return:
        """
        key_col = ["input_ids", "attn_mask", "tuple_pair_col", "result_label", "multi_label", "comparative_label"]

        # Find the maximum length in the array
        max_length = max(len(x) for x in data_dict["tuple_pair_col"])
        print(max_length)
        # Pad each element with the default value (0, 0)
        data_dict["tuple_pair_col"] = [x + [[(0, 0), (0, 0), (0, 0), (0, 0), (0, 0)]] * (max_length - len(x)) for x in
                                       data_dict["tuple_pair_col"]]

        for key in key_col:
            data_dict[key] = np.array(data_dict[key])
            print(key, data_dict[key].shape)

        data_dict['comparative_label'] = np.array(data_dict['comparative_label']).reshape(-1, 1)

        return data_dict
