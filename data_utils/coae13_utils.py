import copy, os
import numpy as np
from data_utils.label_parse import LabelParser
from data_utils import shared_utils, check_utils
from data_utils import current_program_code as cpc
from open_source_utils import ddparser_utils
from transformers import BertTokenizer
import pickle


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

    def create_data_dict(self, data_path, data_type):
        """
        :param data_path: sentence file path
        :param data_type:
        :param label_path: label file path
        :return: a data dict with many parameters
        """
        data_dict = {}

        sent_col, sent_label_col, label_col = cpc.read_standard_file(data_path)

        LP = LabelParser(label_col, ["entity_1", "entity_2", "aspect", "result"])
        label_col, tuple_pair_col = LP.parse_sequence_label("&", sent_col)

        # check_utils.check_data_elem_pair_number(label_col)

        # using stanford tool to get some feature data.
        if not os.path.exists(self.config.path.pre_process_data[data_type]):
            # using DDParser to get chinese feature data.
            dp = ddparser_utils.DDParserFeature(sent_col)
            data_dict['standard_token'], data_dict['token_char'] = dp.get_tokenizer()
            data_dict['pos_ids'], self.pos_dict, self.pos_index = dp.get_pos_feature(self.pos_dict, self.pos_index)
            data_dict['dep_matrix'], data_dict['dep_label_matrix'] = dp.get_dep_feature()
            data_dict['pos_dict'], data_dict['pos_index'] = self.pos_dict, self.pos_index

            shared_utils.write_pickle(data_dict, self.config.path.pre_process_data[data_type])
        else:
            data_dict = shared_utils.read_pickle(self.config.path.pre_process_data[data_type])
            self.pos_dict = data_dict['pos_dict']
            self.pos_index = data_dict['pos_index']

        self.token_max_len = max(self.token_max_len, shared_utils.get_max_token_length(data_dict['standard_token']))

        data_dict['label_col'] = label_col
        data_dict['comparative_label'] = sent_label_col

        # chinese label is based on BERTTokenizer.
        data_dict['bert_token'] = shared_utils.get_token_col(sent_col, bert_tokenizer=self.bert_tokenizer, dim=1)
        data_dict['standard_char'] = shared_utils.get_char_col(sent_col, 1)

        # print(data_dict['bert_token'][2160])
        # print(data_dict['standard_char'][2160])

        # {bert_index: [char_index]}
        mapping_col = cpc.bert_mapping_char(data_dict['bert_token'], data_dict['standard_char'])

        if self.config.model_mode == "norm":
            # label_col = shared_utils.convert_label_dict_by_mapping(label_col, mapping_col)

            self.vocab, self.vocab_index = shared_utils.update_vocab(
                data_dict['standard_char'],
                self.vocab,
                self.vocab_index,
                dim=2,
            )

            data_dict['input_ids'] = shared_utils.transfer_data(
                data_dict['standard_char'],
                self.vocab,
                dim=1
            )

            self.char_max_len = max(self.char_max_len, shared_utils.get_max_token_length(data_dict['standard_char'])) + 2

        else:
            label_col = cpc.convert_label_dict_by_mapping(label_col, mapping_col)
            tuple_pair_col = cpc.convert_tuple_pair_by_mapping(tuple_pair_col, mapping_col)

            data_dict['input_ids'] = shared_utils.bert_data_transfer(
                self.bert_tokenizer,
                data_dict['bert_token'],
                data_type='tokens'
            )


            # check_utils.check_token_to_bert_char(
            #     data_dict['standard_token'], data_dict['bert_token'], data_dict['token_char'], dim=1
            # )

            self.char_max_len = max(self.char_max_len, shared_utils.get_max_token_length(data_dict['bert_token'])) + 2

        data_dict['tuple_pair_col'] = tuple_pair_col

        print("convert pair number: ", cpc.get_tuple_pair_num(data_dict['tuple_pair_col']))
        token_col = data_dict['standard_char'] if self.config.model_mode == "norm" else data_dict['bert_token']

        data_dict['attn_mask'] = shared_utils.get_mask(token_col, dim=1)

        ################################################################################################################
        # using label col to get predicate label, pair label and index_col.
        ################################################################################################################
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

        for key in key_col:
            data_dict[key] = np.array(data_dict[key])
            print(key, data_dict[key].shape)

        data_dict['comparative_label'] = np.array(data_dict['comparative_label']).reshape(-1, 1)

        return data_dict

