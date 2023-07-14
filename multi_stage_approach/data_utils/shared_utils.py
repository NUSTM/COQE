import os
import copy
import json
import torch
import pickle
import numpy as np
from tqdm import tqdm

from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

########################################################################################################################
# configure table change part
########################################################################################################################


# using elem_col and position_sys to get {tag: id}
def create_tag_mapping_ids(elem_col, position_sys, other_flag=True):
    """
    :param elem_col: like: ["entity_1", "entity_2", "aspect", "scale", "predicate"]
    :param position_sys: like ["B", "M", "E", "S"], ["B", "I"], ["B", "I", "E", "S"]
    :param other_flag: true denote {"O": 0}, false denote {}
    :return:
    """
    assert "".join(position_sys) in {"BMES", "BI"}, "[ERROR] position system error!"

    tags_map_ids = {"O": 0} if other_flag else {}

    if elem_col is None or len(elem_col) == 0:
        for i, pos in enumerate(position_sys):
            tags_map_ids[pos] = i + 1 if other_flag else i

    else:
        for i, elem in enumerate(elem_col):
            for j, pos in enumerate(position_sys):
                tags_map_ids[pos + "-" + elem] = i * len(position_sys) + ((j + 1) if other_flag else j)

    return tags_map_ids


# using parameters dict to change configure table
def set_config_parameters(args, parameters_dict):
    """
    :param args:
    :param parameters_dict:
    :return:
    """
    for param, value in parameters_dict.items():
        if param == "file_type":
            args.file_type = value

        elif param == "model_mode":
            args.model_mode = value

        elif param == "stage_model":
            args.stage_model = value

        elif param == "model_type":
            args.model_type = value

        elif param == "epoch":
            args.epoch = value

        elif param == "batch_size":
            args.batch_size = value

        elif param == "program_mode":
            args.program_mode = value

        elif param == "result_representation":
            args.result_representation = value

        elif param == "window_size":
            args.window_size = value

        elif param == "matrix_level":
            args.matrix_level = value

    return args


def combine_program_configure(config_dict, model_dict, optimizer_dict):
    """
    :param config_dict:
    :param model_dict:
    :param optimizer_dict:
    :return:
    """

    return {"config": config_dict, "model": model_dict, "optimizer": optimizer_dict}


def read_grid_search_parameters(path):
    """
    :param path: file path of parameters setting table.
    :return: a list of dict, {"config": {}, "model": {}}
    """
    with open(path, "r", encoding='gb18030', errors='ignore') as f:
        param_dict = json.load(f)

    return param_dict


def parameters_to_model_name(param_dict):
    """
    :param param_dict: {"config": {}, "model": {}}
    :return:
    """
    assert "config" in param_dict, "must need config parameters."

    result_file, model_file = "./ModelResult/", "./PreTrainModel/"

    config_param = param_dict['config']
    model_param = param_dict['model'] if "model" in param_dict else None
    optimizer_param = param_dict['optimizer'] if "optimizer" in param_dict else None

    model_name = ""

    for index, (key, value) in enumerate(config_param.items()):
        model_name += str(value) if isinstance(value, int) else value
        model_name += "_" if index != len(config_param.keys()) - 1 else ""

    model_name = model_name.replace("second", "first")
    model_name = model_name.replace("test", "run")

    if not os.path.exists(result_file):
        os.mkdir(result_file)
    if not os.path.exists(model_file):
        os.mkdir(model_file)
    if not os.path.exists(os.path.join(result_file, model_name)):
        os.mkdir(os.path.join(result_file, model_name))
    if not os.path.exists(os.path.join(model_file, model_name)):
        os.mkdir(os.path.join(model_file, model_name))

    model_name += "/"
    if model_param is not None:
        model_param_col = []
        for index, (key, value) in enumerate(model_param.items()):
            if key == "first_stage" or key == "factor":
                continue

            if isinstance(value, float) or isinstance(value, int):
                value = str(int(value * 10))

            model_param_col.append(key[:4] + "_" + value)

        model_name += "_".join(model_param_col)

    result_file, model_file = os.path.join(result_file, model_name), os.path.join(model_file, model_name)

    if not os.path.exists(result_file):
        os.mkdir(result_file)
    if not os.path.exists(model_file):
        os.mkdir(model_file)
    #
    # if optimizer_param is not None:
    #     model_name += "_"
    #     for index, (key, value) in enumerate(optimizer_param.items()):
    #         model_name += key + "_" + (str(value) if isinstance(value, int) else value)
    #         model_name += "_" if index != len(optimizer_param.keys()) - 1 else ""

    return model_name


########################################################################################################################
# data process part
########################################################################################################################

# clear the string of special symbol.
def clear_string(line, strip_symbol=None, replace_symbol=None):
    """
    :param line: a string
    :param strip_symbol:
    :param replace_symbol: a list of special symbol, need replace.
    :return:
    """
    if strip_symbol is not None:
        for sym in strip_symbol:
            line = line.strip(sym)

    if replace_symbol is not None:
        for sym in replace_symbol:
            line = line.replace(sym, "")

    return line


# using split symbol get a list of string.
def split_string(line, split_symbol):
    """
    :param line: a string need be split
    :param split_symbol: a string: split symbol
    :return:
    """
    return list(filter(None, line.split(split_symbol)))


# access file to get sent(label and un-label)
def get_sent(sent_path, encode='utf-8', label_type=0, strip_symbol=None, replace_symbol=None):
    """
    :param sent_path: a file path, store sent in each line.
    :param label_type: 0 denote without label. 1 denote label first. 2 denote label second
    :param encode: a type of file encode
    :param replace_symbol: a list of special symbol need be replace
    :param strip_symbol: a list of symbol need be strip
    :return: a sent_col or sent_col and sent_label_col
    """
    assert os.path.exists(sent_path), "maybe send error file path"

    sent_col, label_col = [], []
    with open(sent_path, "r", encoding=encode, errors='ignore') as f:

        # traverse each line to parse sent
        for line in f.readlines():
            if line == "\n":
                continue

            # without label, only parse sentence.
            if label_type == 0:
                sent_col.append(clear_string(line, strip_symbol=strip_symbol, replace_symbol=replace_symbol))

            # label first, label + '\t' + sent.
            elif label_type == 1:
                clear_line = clear_string(line, strip_symbol=strip_symbol, replace_symbol=replace_symbol)
                cur_label, cur_sent = split_string(clear_line, split_symbol='\t')

                sent_col.append(cur_sent)
                label_col.append(int(cur_label))

            elif label_type == 2:
                clear_line = clear_string(line, strip_symbol=strip_symbol, replace_symbol=replace_symbol)
                cur_sent, cur_label = split_string(clear_line, split_symbol='\t')

                sent_col.append(cur_sent)
                label_col.append(int(cur_label))

    return sent_col if label_type == 0 else sent_col, label_col


# using a list of token to get max token length.
def get_max_token_length(token_col):
    """
    :param token_col: a list of list token. shape: [n, each_token_num]
    :return:
    """
    token_len = -1
    for index in range(len(token_col)):
        token_len = max(token_len, len(token_col[index]))
    return token_len


# using data to update vocab
def update_vocab(data, elem_dict, elem_index=0, dim=1):
    """
    :param data:
    :param dim:
    :param elem_dict:
    :param elem_index:
    :return:
    """
    if dim == 0:
        if data not in elem_dict:
            elem_dict[data] = elem_index
            elem_index += 1

    else:
        for index in range(len(data)):
            elem_dict, elem_index = update_vocab(data[index], elem_dict, elem_index, dim - 1)
    return elem_dict, elem_index


# through sentences to get token col by norm-split or bert-split
def get_token_col(sent_col, split_symbol=None, bert_tokenizer=None, dim=1, add_next_sent=None):
    """
    :param sent_col: a shape string (finish strip symbol and replace symbol).
    :param split_symbol: split token by split symbol.
    :param bert_tokenizer: a object of BERTTokenizer, return bert type token.
    :param dim: 0 denote a string of sentence. 1 denote a list of sentences.
    :param add_next_sent:
    :return: a token_col, shape-like sent_col.
    """
    assert split_symbol is not None or bert_tokenizer is not None, "you need send split symbol or bert tokenizer."

    if dim == 0:
        # using split symbol to split sentence
        if split_symbol is not None:
            return split_string(sent_col, split_symbol)
        # using bert tokenizer to get bert token
        else:
            if add_next_sent is None:
                return bert_tokenizer.tokenize('[CLS] ' + sent_col + ' [SEP]')
            else:
                return bert_tokenizer.tokenize('[CLS] ' + sent_col + ' [SEP] ' + add_next_sent + ' [SEP]')
    else:
        token_col = []

        for index in range(len(sent_col)):
            token_col.append(get_token_col(sent_col[index], split_symbol, bert_tokenizer, dim - 1))

        return token_col


# Chinese version split char.
def get_char_col(sent_col, dim=1):
    """
    :param sent_col: a list of sent.
    :param dim: 0 denote a sent.
    :param add_hidden: a flag to add [CLS] and [SEP]
    :return:
    """
    if dim == 0:
        return list(sent_col)
    else:
        char_col = []
        for index in range(len(sent_col)):
            char_col.append(get_char_col(sent_col[index], dim - 1))
        return char_col


# get data correspond attn-mask
def get_mask(input_ids, dim=1):
    """
    :param input_ids: a input ids
    :param dim: create mask in which mask
    :return: a attn mask co-respond input_ids
    """
    if dim == 0:
        return len(input_ids) * [1]

    else:
        attn_mask = []
        for index in range(len(input_ids)):
            attn_mask.append(get_mask(input_ids[index], dim=dim-1))
        return attn_mask


def get_stanford_char_bert_char(stanford_token_col, bert_token_col, mapping_col):
    """
    :param stanford_token_col:
    :param bert_token_col:
    :param mapping_col:
    :return:
    """
    final_data = []
    for index in range(len(mapping_col)):
        sf_sequence_token = stanford_token_col[index]
        bert_sequence_token = bert_token_col[index]
        sequence_map = mapping_col[index]

        sf_length, bert_length = len(sf_sequence_token), len(bert_sequence_token)

        sf_to_bert = [[0 for _ in range(bert_length)] for _ in range(sf_length)]

        for i in range(sf_length):
            for j in sequence_map[i]:
                sf_to_bert[i][j] = 1

        final_data.append(sf_to_bert)

    return final_data


# data (token to id) or (id to token)
def transfer_data(data, convert_dict, dim=1):
    """
    :param data: a data need be convert to ids
    :param convert_dict: a dict that token => id
    :param dim: process on which dim
    :return:
    """
    data_ids = copy.deepcopy(data)
    if dim == 0:
        for i in range(len(data_ids)):
            assert data_ids[i] in convert_dict, "data error or convert dict error!"
            data_ids[i] = convert_dict[data_ids[i]]

    else:
        for i in range(len(data_ids)):
            data_ids[i] = transfer_data(data_ids[i], convert_dict, dim=dim-1)

    return data_ids


# bert type data (token to id) or (id to token)
def bert_data_transfer(bert_tokenizer, input_tokens, data_type="tokens"):
    """
    :param bert_tokenizer: a object of BERTTokenizer
    :param input_tokens: a list of token or ids
    :param data_type: "tokens" denote tokens to ids, "ids" denote ids to tokens
    :return: a list of token or ids
    """
    result_data = []

    if not isinstance(input_tokens, list):
        input_tokens = input_tokens.tolist()

    for seq_tokens in input_tokens:
        if data_type == "tokens":
            result_data.append(bert_tokenizer.convert_tokens_to_ids(seq_tokens))
        else:
            result_data.append(bert_tokenizer.convert_ids_to_tokens(seq_tokens))

    return result_data


# chinese version mapping, {bert_index: [char_index]}
def bert_mapping_char(bert_token_col, gold_char_col):
    """
    :param bert_token_col: a list of token list by BertTokenizer (with [cls] and [sep])
    :param gold_char_col: a list char list
    :return: a map: {bert_index: [char_index]}
    """
    assert len(bert_token_col) == len(gold_char_col), "bert data length not equal to char data length"

    mapping_col = []
    for index in range(len(bert_token_col)):
        seq_map, bert_index, char_index = {}, 1, 0
        seq_bert_token, seq_gold_char = bert_token_col[index], gold_char_col[index]

        while bert_index < len(seq_bert_token) and char_index < len(seq_gold_char):
            seq_map[bert_index] = [char_index]

            # [UNK] denote special symbol
            if seq_bert_token[bert_index] == "[UNK]":
                bert_index = bert_index + 1
                char_index = char_index + 1
                continue

            # get cur index correspond length
            char_length = len(seq_gold_char[char_index])
            bert_length = len(seq_bert_token[bert_index])

            # drop "##" prefix
            if seq_bert_token[bert_index].find("##") != -1:
                bert_length = len(seq_bert_token[bert_index]) - 2

            while char_length < bert_length:
                char_index = char_index + 1
                seq_map[bert_index].append(char_index)
                char_length += len(seq_gold_char[char_index])

            # print(seq_bert_token[bert_index], end=" || ")
            # print(seq_gold_char[seq_map[bert_index][0]: seq_map[bert_index][-1] + 1])
            char_index = char_index + 1
            bert_index = bert_index + 1

        # check_utils.check_mapping_process(seq_map, bert_token_col[index], gold_char_col[index])
        mapping_col.append(seq_map)

    return mapping_col


# english version mapping, {token_index: [bert_index]}
def token_mapping_bert(bert_token_col, gold_token_col):
    """
    :param bert_token_col: a list of token list by BertTokenizer (with [cls] and [sep])
    :param gold_token_col: a list char list
    :return: a map: {bert_index: [char_index]}
    """
    assert len(bert_token_col) == len(gold_token_col), "bert data length not equal to char data length"

    mapping_col = []
    for index in range(len(bert_token_col)):
        seq_map, bert_index, token_index = {}, 1, 0
        seq_bert_token, seq_gold_token = bert_token_col[index], gold_token_col[index]

        while bert_index < len(seq_bert_token) and token_index < len(seq_gold_token):
            seq_map[token_index] = [bert_index]

            # [UNK] denote special symbol
            if seq_bert_token[bert_index] == "[UNK]":
                bert_index = bert_index + 1
                token_index = token_index + 1
                continue

            # get cur index correspond length
            token_length = len(seq_gold_token[token_index])
            bert_length = len(seq_bert_token[bert_index])

            # drop "##" prefix
            if seq_bert_token[bert_index].find("##") != -1:
                bert_length = len(seq_bert_token[bert_index]) - 2

            while token_length > bert_length:
                bert_index = bert_index + 1
                seq_map[token_index].append(bert_index)
                bert_length += len(seq_bert_token[bert_index])

                if seq_bert_token[bert_index].find("##") != -1:
                    bert_length -= 2

            assert bert_length == token_length, "appear mapping error!"
            # check_utils.check_mapping_process(seq_map, seq_gold_token, seq_bert_token)

            token_index = token_index + 1
            bert_index = bert_index + 1

        # 为了处理 e_index 为最后一位的特殊情况
        seq_map[token_index] = [bert_index]

        # check_utils.check_mapping_process(seq_map, bert_token_col[index], gold_char_col[index])
        mapping_col.append(seq_map)

    return mapping_col


# padding data by different data_type and max_len
def padding_data(data, max_len, dim=2, padding_num=0, data_type="norm"):
    """
    :param data: a list of matrix or a list of list data
    :param max_len: integer for norm data, a tuple (n, m) for matrix
    :param dim: denote which dim will padding
    :param padding_num: padding number default is 0
    :param data_type: "norm" or "matrix"
    :return: a data of padding
    """
    assert data_type == "norm" or data_type == "matrix", "you need send truth data type, {norm or matrix}"

    if data_type == "norm":
        assert data_type == "norm" and isinstance(max_len, int), "you need sent the integer padding length"

        if dim == 0:
            return data + [padding_num] * (max_len - len(data))

        else:
            pad_data = []
            for index in range(len(data)):
                pad_data.append(
                    padding_data(data[index], max_len, dim=dim-1, padding_num=padding_num, data_type=data_type)
                )
            return pad_data

    # padding a list of matrix by max_len
    else:
        assert data_type == "matrix" and isinstance(max_len, tuple), "you need sent the tuple padding length"
        n, m = max_len

        if dim == 0:
            pad_data = [line + [padding_num] * (m - len(line)) for line in data]
            padding_length = n - len(pad_data)

            for i in range(padding_length):
                pad_data.append([padding_num] * m)

        else:
            pad_data = []
            for index in range(len(data)):
                pad_data.append(
                    padding_data(data[index], max_len, dim - 1, padding_num, data_type)
                )

        return pad_data


def cartesian_product(init_elem_col, add_elem_list):
    """
    :param init_elem_col: a list of [(s_index1, e_index1), (s_index2, e_index2)], length is n
    :param add_elem_list: a list of elem: (s_index3, e_index3), length is m
    :return: a list of [(s_index1, e_index1), (s_index2, e_index2), (s_index3, e_index3)], length is n * m
    """
    result_elem_data_col = []

    if len(init_elem_col) == 0:
        for add_elem in add_elem_list:
            result_elem_data_col.append([add_elem])
        return result_elem_data_col

    for index in range(len(init_elem_col)):
        for add_elem in add_elem_list:
            result_elem_data_col.append(init_elem_col[index] + [add_elem])

    return result_elem_data_col


# sequence label => elem_dict_v2: {elem: [(s_index, e_index)]}.
def sequence_label_convert_to_elem_dict(sequence_label, elem_col, position_sys):
    """
    :param sequence_label:
    :param elem_col:
    :param position_sys:
    :return:
    """
    s_index, elem_set = -1, {elem for elem in elem_col}
    sequence_elem_dict = {elem: [] for elem in elem_col}

    # check sequence label is legitimate.
    # assert check_utils.check_correct_sequence_label(sequence_label, position_sys), "Error sequence label."

    for index in range(len(sequence_label)):
        cur_pos, cur_tag = get_label_pos_tag(sequence_label[index])

        assert cur_tag == "O" or cur_tag in elem_set, "[ERROR] appear error elem type!"

        if position_sys == "BMES":
            if cur_pos in {"S", "B"}:
                s_index = index

            if cur_pos in {"S", "E"} and s_index != -1:
                e_index = index
                sequence_elem_dict[cur_tag].append((s_index, e_index + 1))
                s_index = -1

        else:
            if index == 0:
                continue

            last_pos, last_tag = get_label_pos_tag(sequence_label[index - 1])
            if cur_pos in {"B", "O"} and last_pos in {"B", "I"} and s_index != -1:
                sequence_elem_dict[last_tag].append((s_index, index))

            if cur_pos in {"B"}:
                s_index = index

    if position_sys == "BI":
        last_pos, last_tag = get_label_pos_tag(sequence_label[-1])

        if last_pos in {"B", "I"} and s_index != -1:
            sequence_elem_dict[last_tag].append((s_index, len(sequence_label)))

    return sequence_elem_dict


# elem_dict_v2: {elem: [(s_index, e_index)]}.
def elem_dict_convert_to_pair_col(elem_dict):
    """
    :param elem_dict: {elem: [(s_index, e_index)]}
    :return:
    """
    key_col = ["entity_1", "entity_2", "aspect", "scale"]
    final_pair_col = []

    for elem in key_col:
        if len(elem_dict[elem]) == 0:
            cur_elem_represent = [(-1, -1)]
        else:
            cur_elem_represent = elem_dict[elem]
        final_pair_col = cartesian_product(final_pair_col, cur_elem_represent)

    return final_pair_col


def elem_dict_v1_convert_to_elem_dict_v2(elem_dict):
    """
    :param elem_dict: {elem: {s_index: length}}
    :return: {elem: [(s_index, e_index)]}
    """
    final_elem_dict = {}

    for elem in elem_dict.keys():
        if elem == "predicate":
            continue

        cur_elem_representation = []

        for s_index, length in elem_dict[elem].items():
            cur_elem_representation.append((s_index, s_index + length))

        final_elem_dict[elem] = cur_elem_representation

    return final_elem_dict


def invert_dict(data_dict):
    """
    :param data_dict:
    :return:
    """
    return {v: k for k, v in data_dict.items()}


def cover_rate(g_interval, p_interval, intermittent=None, proportion=True):
    """
    :param g_interval: a tuple like [s_index, e_index)
    :param p_interval: a tuple like [s_index, e_index)
    :param proportion: True: denote return proportion, False denote return length.
    :param intermittent:
    :return: proportional of cover
    """
    l_board = max(p_interval[0], g_interval[0])
    r_board = min(p_interval[1], g_interval[1])

    gold_length = (g_interval[1] - g_interval[0])
    cover_length = max(0, (r_board - l_board))

    if not proportion:
        return cover_length

    if intermittent is None:
        return cover_length / float(gold_length)
    else:
        return cover_length / intermittent


def parse_interval(interval):
    """
    :param interval: (s_index, e_index) or (s_index1, e_index1, s_index2, e_index2)
    :return:
    """
    assert len(interval) > 0, "interval length must > 0."

    interval_col = [(interval[0], interval[1])]

    if len(interval) == 4:
        interval_col.append((interval[2], interval[3]))

    return interval_col


def interval_length(interval):
    """
    :param interval: (s_index, e_index) or (s_index1, e_index1, s_index2, e_index2)
    :return:
    """
    if len(interval) == 2:
        return interval[1] - interval[0]

    return interval[3] - interval[2] + interval[1] - interval[0]


def add_list(a, b):
    """
    :param a:
    :param b:
    :return:
    """
    assert len(a) == len(b), "[ERROR] data length don't equal."

    return [a[i] + b[i] for i in range(len(a))]



########################################################################################################################
# current program need part
########################################################################################################################

# using elem dict to get sequence label.
def elem_dict_convert_to_sequence_label(token_col, label_col, label_type="predicate", position_sys="BMES"):
    """
    :param token_col: a list of token list.
    :param label_col: a list of [elem dict]. elem_dict: {elem: {s_index: length}}
    :param label_type: contain "predicate" and "elem".
    :param position_sys: contain "BMES" and "BI"
    :return: a sequence label: [n, each_sequence_length]
    """
    assert len(token_col) == len(label_col), "label length need equal to data length."

    def elem_dict_to_sequence_label(sequence_label, elem_dict, elem_type):
        for s_index, length in elem_dict.items():
            if length == 1:
                if position_sys == "BMES":
                    sequence_label[s_index] = "S-" + elem_type
                else:
                    sequence_label[s_index] = "B-" + elem_type
                continue

            if position_sys == "BMES":
                sequence_label[s_index] = "B-" + elem_type
                sequence_label[s_index + length - 1] = "E-" + elem_type

                for k in range(s_index + 1, s_index + length - 1):
                    sequence_label[k] = "M-" + elem_type
            else:
                sequence_label[s_index] = "B-" + elem_type
                for k in range(s_index + 1, s_index + length):
                    sequence_label[k] = "I-" + elem_type

        return sequence_label

    def create_sequence_label(sequence_token, pair_label_col, elem_col):
        each_pair_col, sequence_label = [], ["O"] * len(sequence_token)
        for pair_index in range(len(pair_label_col)):
            for elem in elem_col:
                sequence_label = elem_dict_to_sequence_label(sequence_label, pair_label_col[pair_index][elem], elem)

            if label_type == "elem":
                each_pair_col.append(sequence_label)
                sequence_label = ["O"] * len(sequence_token)
        assert len(each_pair_col) != 0 or label_type == "predicate", "data error!"
        return sequence_label if label_type == "predicate" else each_pair_col

    sequence_label_col = []
    for index in range(len(token_col)):
        if label_type == "predicate":
            sequence_label_col.append(create_sequence_label(token_col[index], label_col[index], ["predicate"]))

        elif label_type == "elem":
            elem_col = ["entity_1", "entity_2", "aspect", "scale"]
            sequence_label_col.append(create_sequence_label(token_col[index], label_col[index], elem_col))

    return sequence_label_col


def elem_dict_convert_to_multi_sequence_label(token_col, label_col):
    """
    :param token_col: a list of token list.
    :param label_col: a elem dict like: {elem: {s_index: length}}
    :return:
    """
    elem_pair_col, polarity_col = [], []
    elem_col = ["entity_1", "entity_2", "aspect", "result"]

    for index in range(len(token_col)):
        sent_multi_col = []
        for elem in elem_col:
            sequence_label = ["O"] * len(token_col[index])

            for s_index, length in label_col[index][elem].items():
                if length == 1:
                    sequence_label[s_index] = "S"
                    continue

                e_index = s_index + length - 1
                sequence_label[s_index] = "B"
                sequence_label[e_index] = "E"

                for k in range(s_index + 1, e_index):
                    sequence_label[k] = "M"

            sent_multi_col.append(sequence_label)

        elem_pair_col.append(sent_multi_col)

    return elem_pair_col


def predicate_convert_to_index_col(label_col):
    """
    :param label_col: a list of [elem dict]. elem_dict: {elem: {s_index: length}}
    :return: a list of [(s_index, e_index)]
    """
    predicate_index_col = []

    for index in range(len(label_col)):
        each_sent_index_col = []
        for pair_index in range(len(label_col[index])):
            elem_dict = label_col[index][pair_index]['predicate']

            assert len(elem_dict) <= 2, "predicate data error"

            predicate_index = None
            for s_index, length in elem_dict.items():
                cur_predicate_index = (s_index, s_index + length)

                if predicate_index is None:
                    predicate_index = cur_predicate_index
                else:
                    if cur_predicate_index[0] < predicate_index[0]:
                        predicate_index = cur_predicate_index + predicate_index
                    else:
                        predicate_index = predicate_index + cur_predicate_index

            if predicate_index is not None:
                each_sent_index_col.append(predicate_index)

        predicate_index_col.append(each_sent_index_col)

    return predicate_index_col


def get_label_pos_tag(cur_label):
    """
    :param cur_label:
    :return:
    """
    if cur_label.find("-") == -1:
        return cur_label, "NULL"
    else:
        return split_string(cur_label, "-")


def convert_label_dict_by_mapping(label_col, mapping_col):
    """
    :param label_col: [n, pair_num, elem_dict], elem_dict: {elem: {s_index: length}}.
    :param mapping_col: [n, index_dict], index_dict: {bert_index: [char_index]} or {token_index: [bert_index]}.
    :return: new type label col.
    """
    elem_col = ["entity_1", "entity_2", "aspect", "scale", "predicate"]
    assert len(label_col) == len(mapping_col), "mapping_col length equal to label length."
    final_label_col = copy.deepcopy(label_col)

    for index in range(len(final_label_col)):

        # get each pair elem dict representation
        for pair_index in range(len(final_label_col[index])):

            init_elem_dict = {k: {} for k in elem_col}
            print(final_label_col[index][pair_index])
            for elem in final_label_col[index][pair_index].keys():

                for s_index, length in final_label_col[index][pair_index][elem].items():
                    e_index = length + s_index - 1

                    new_s_index = mapping_col[index][s_index][0]
                    new_e_index = mapping_col[index][e_index][-1]
                    init_elem_dict[elem][new_s_index] = (new_e_index - new_s_index) + 1

            final_label_col[index][pair_index] = copy.deepcopy(init_elem_dict)

    return final_label_col


# chinese version need combine the same predicates.
def combine_predicate_label_col(label_col, gold_tuple_pair_col):
    """
    :param label_col: [n, predicate_num, elem_dict]
    :param gold_tuple_pair_col: [n, predicate_num, pair_num, tuple_pair]
    :return:
    """
    def combine_elem_dict(source_elem_dict, add_elem_dict):
        final_elem_dict = copy.deepcopy(source_elem_dict)
        for elem in source_elem_dict['label'].keys():
            for add_s_index, add_length in add_elem_dict['label'][elem].items():
                if add_s_index not in source_elem_dict['label'][elem]:
                    final_elem_dict['label'][elem][add_s_index] = add_length
        final_elem_dict['tuple_pair'].extend(add_elem_dict['tuple_pair'])

        return final_elem_dict

    final_label_col, final_tuple_pair_col = [], []

    for index in range(len(label_col)):

        each_sent_predicate_dict = {}
        for pair_index in range(len(label_col[index])):

            for s_index, length in label_col[index][pair_index]["predicate"].items():
                e_index = s_index + length - 1

                if (s_index, e_index) not in each_sent_predicate_dict:
                    each_sent_predicate_dict[(s_index, e_index)] = \
                        {"label": label_col[index][pair_index], "tuple_pair": gold_tuple_pair_col[index][pair_index]}

                else:
                    each_sent_predicate_dict[(s_index, e_index)] = combine_elem_dict(
                        each_sent_predicate_dict[(s_index, e_index)],
                        {"label": label_col[index][pair_index], "tuple_pair": gold_tuple_pair_col[index][pair_index]}
                    )

                # four tuple only calculate once.
                break

        cur_pair_elem_dict_col, cur_tuple_pair_col = [], []
        for predicate_index, elem_dict in each_sent_predicate_dict.items():
            cur_pair_elem_dict_col.append(elem_dict['label'])
            cur_tuple_pair_col.append(elem_dict['tuple_pair'])

        if len(cur_pair_elem_dict_col) == 0:
            cur_pair_elem_dict_col = [{"entity_1": {}, "entity_2": {}, "aspect": {}, "scale": {}, "predicate": {}}]
            cur_tuple_pair_col = [[[(-1, -1)] * 4]]
        final_label_col.append(cur_pair_elem_dict_col)
        final_tuple_pair_col.append(cur_tuple_pair_col)

    return final_label_col, final_tuple_pair_col


def calculate_average_measure(add_eval, global_eval):
    """
    :param add_eval:
    :param global_eval:
    :return:
    """
    global_eval.avg_exact_measure = global_eval.add_fold_measure(
        global_eval.avg_exact_measure, add_eval.optimize_exact_measure, fold_num=1
    )

    global_eval.avg_prop_measure = global_eval.add_fold_measure(
        global_eval.avg_prop_measure, add_eval.optimize_prop_measure, fold_num=1
    )

    global_eval.avg_binary_measure = global_eval.add_fold_measure(
        global_eval.avg_binary_measure, add_eval.optimize_binary_measure, fold_num=1
    )


def read_pickle(path):
    """
    :param path:
    :return:
    """
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def write_pickle(data_dict, path):
    """
    :param data_dict:
    :param path:
    :return:
    """
    with open(path, "wb") as f:
        pickle.dump(data_dict, f)


def convert_to_tuple(elem_dict):
    """
    :param elem_dict: a list elem_dict: {elem: [(s_index, e_index)]}
    :return:
    """
    pair_col = []
    for elem, value in elem_dict.items():
        pair_col = cartesian_product(pair_col, value if len(value) != 0 else [(-1, -1)])

    return pair_col


def get_init_pair_num(label_col):
    """
    :param label_col: [n, pair_num, elem_dict], elem_dict: {elem: {s_index: length}}
    :return:
    """
    elem_col = ["entity_1", "entity_2", "aspect", "scale"]

    init_pair_num = 0
    for index in range(len(label_col)):
        for pair_index in range(len(label_col[index])):
            is_pair, cur_pair_num = False, 1

            for elem in elem_col:
                cur_elem_num = len(label_col[index][pair_index][elem])

                if cur_elem_num != 0:
                    is_pair = True

                cur_pair_num *= max(1, cur_elem_num)

            if is_pair:
                init_pair_num += cur_pair_num

    return init_pair_num


def get_tuple_pair_num(gold_tuple_pair):
    """
    :param gold_tuple_pair: [n, predicate_num, pair_num, tuple]
    :return:
    """
    pair_num = 0
    for index in range(len(gold_tuple_pair)):
        for predicate_index in range(len(gold_tuple_pair[index])):
            if gold_tuple_pair[index][predicate_index] != [[(-1, -1)] * 4]:
                pair_num += len(gold_tuple_pair[index][predicate_index])

    return pair_num


def create_matrix(n, m, default_value=0):
    return [[default_value for _ in range(m)] for _ in range(n)]


def index_if_out_bound(lower_bound, upper_bound, index):
    """
    :param lower_bound:
    :param upper_bound:
    :param index:
    :return:
    """
    if lower_bound <= index < upper_bound:
        return True

    return False


########################################################################################################################
# model feature process part
########################################################################################################################


# get Glove embedding or word2vec embedding to nn.Embedding
def get_pretrain_weight(glove_path, word2vec_path, vocab):
    """
    :param glove_path: glove file need to convert word2vec file type.
    :param word2vec_path: word2vec to get word embedding.
    :param vocab: lstm vocab = {token: index}
    :return: weight: shape is [vocab_size * dim_size]
    """
    # transfer GloVe to word2vec
    if not os.path.exists(word2vec_path):
        glove2word2vec(glove_path, word2vec_path)

    # loading vector model
    vec_model = KeyedVectors.load_word2vec_format(word2vec_path, limit=100000)

    # define weight to store vector embedding
    weight = torch.zeros(len(vocab) + 5, 300)

    # updating weight by vec_model embedding
    for i in tqdm(range(len(vec_model.index2word))):
        if vec_model.index2word[i] not in vocab:
            continue

        index = vocab[vec_model.index2word[i]]

        weight[index, :] = torch.from_numpy(vec_model.get_vector(vec_model.index2word[i]))

    return weight


# generate predicate-based direction (sentence level)
def generate_predicate_direction(n, predicate_index):
    """
    :param n: the length of sentence
    :param predicate_index: (s_index, e_index) or (s_index1, e_index1, s_index2, e_index2)
    :return: a sequence predicate direction
    """
    seq_direction = [0] * n

    assert len(predicate_index) == 2 or len(predicate_index) == 4, "predicate index error!"

    if len(predicate_index) == 2:
        s_index, e_index = predicate_index

        # create predicate direction info
        for k in range(0, s_index):
            seq_direction[k] = 1
        for k in range(e_index, len(seq_direction)):
            seq_direction[k] = 2

    else:
        s_index1, e_index1, s_index2, e_index2 = predicate_index

        # create predicate direction info
        for k in range(0, s_index1):
            seq_direction[k] = 1

        for k in range(e_index1, s_index2):
            seq_direction[k] = 2

        for k in range(e_index2, n):
            seq_direction[k] = 1

    return seq_direction


# generate predicate-based position info (sentence level)
def generate_predicate_position(n, predicate_index):
    """
    :param n: the length of sentence
    :param predicate_index: (s_index, length) or (s_index1, length1, s_index2, length2)
    :return: a sequence predicate direction
    """
    seq_position = [0] * n

    assert len(predicate_index) == 2 or len(predicate_index) == 4, "predicate index error!"

    if len(predicate_index) == 2:
        s_index, e_index = predicate_index
    else:
        s_index, e_index, s_index1, e_index1 = predicate_index

    # create predicate distance info
    distance, index = 1, s_index - 1
    while index >= 0:
        seq_position[index] = distance
        index, distance = index - 1, distance + 1

    distance, index = 1, e_index
    while index < len(seq_position):
        seq_position[index] = distance
        index, distance = index + 1, distance + 1

    if len(predicate_index) == 4:
        distance, index = 1, s_index1 - 1
        while index >= 0:
            seq_position[index] = min(seq_position[index], distance)
            index, distance = index - 1, distance + 1

        distance, index = 1, e_index1
        while index < len(seq_position):
            seq_position[index] = min(seq_position[index], distance)
            index, distance = index + 1, distance + 1

    return seq_position


# get predicate-based info (all sentence level)
def generate_position_direction_embed(index_col, attn_mask, hidden_instance=False):
    """
    :param index_col: a list of [(s_index, e_index), (s_index, e_index)......]
    :param attn_mask: a list of mask
    :param hidden_instance:
    :return: return a list of each predicate distance
    """
    predicate_position = []
    predicate_direction = []

    for i in range(len(index_col)):
        seq_mask = np.array(copy.deepcopy(attn_mask[i]))

        # create fake data on un-comparative sentence
        if len(index_col[i]) == 0 and hidden_instance:
            seq_position = [t for t in range(len(attn_mask[i]))]
            seq_position = np.array(seq_position)
            seq_position[seq_mask == 0] = 0
            predicate_position.append(seq_position.tolist())

            seq_direction = [1] * len(attn_mask[i])
            seq_direction = np.array(seq_direction)
            seq_direction[seq_mask == 0] = 0
            predicate_direction.append(seq_direction)
            continue

        for j in range(len(index_col[i])):
            seq_position = generate_predicate_position(len(attn_mask[i]), index_col[i][j])
            seq_direction = generate_predicate_direction(len(attn_mask[i]), index_col[i][j])

            seq_position = np.array(seq_position)
            seq_direction = np.array(seq_direction)

            seq_position[seq_mask == 0] = 0
            seq_direction[seq_mask == 0] = 0

            predicate_position.append(seq_position.tolist())
            predicate_direction.append(seq_direction.tolist())
    return predicate_position, predicate_direction

########################################################################################################################
# Token to Char or Bert Char matrix. and convert predicate index representation part.
########################################################################################################################


def token_char_convert_to_token_bert(bert_token_col, token_char, mapping_col):
    """
    :param bert_token_col: [n, sequence_length]
    :param token_char: [token_length, char_length]
    :param mapping_col: {bert_index: [char_index]}
    :return: [token_length, bert_char_length]
    """
    assert len(token_char) == len(mapping_col), "[ERROR] data length error!"

    bert_token_char = []
    for index in range(len(token_char)):
        seq_bert_token = bert_token_col[index]
        seq_token_char = token_char[index]
        seq_map = mapping_col[index]

        # token length denote ddparser tokenizer length
        # char length denote bert_tokenizer char length
        token_length, char_length = len(seq_token_char), len(seq_bert_token)
        seq_matrix = [[0 for _ in range(char_length)] for _ in range(token_length)]

        for i in range(token_length):
            for bert_index, char_index_col in seq_map.items():
                max_val = 0
                for k in range(len(char_index_col)):
                    max_val = max(max_val, seq_token_char[i][char_index_col[k]])

                seq_matrix[i][bert_index] = max_val

        bert_token_char.append(seq_matrix)

    return bert_token_char


# index_col denote bert sequence position representation.
def char_index_col_convert_to_token_index_col(index_col, token_char):
    """
    :param index_col: [n, predicate_num, representation]. representation: (s_index, e_index)
    :param mapping_col:
    :param token_char: [n, token_length, bert_char_length]
    :param lang: "eng" or "cn"
    :return:
    """
    assert len(index_col) == len(token_char), "[ERROR] data length error!"

    final_convert_index_col = []
    for index in range(len(index_col)):
        sequence_index_col = []
        for predicate_index in range(len(index_col[index])):
            cur_token_index = modify_chinese_index_col(index_col[index][predicate_index], token_char[index])
            if len(cur_token_index) > 0:
                sequence_index_col.append(cur_token_index)

        assert len(sequence_index_col) == len(index_col[index])
        final_convert_index_col.append(sequence_index_col)

    return final_convert_index_col


def modify_chinese_index_col(predicate_index, sequence_token_char):
    """
    :param predicate_index: (s_index, e_index) or (s_index1, e_index1, s_index2, e_index2)
    :param sequence_token_char: [token_length, bert_char_length]
    :return:
    """
    if predicate_index == (0, 1):
        return predicate_index

    if len(predicate_index) == 2:
        s_index, e_index = predicate_index
        s_index1 = None
    else:
        s_index, e_index, s_index1, e_index1 = predicate_index

    final_predicate_index = ()

    # i denote each token.
    for i in range(len(sequence_token_char)):
        if sequence_token_char[i][s_index] == 1:
            final_predicate_index += (i, i + 1)
        if s_index1 is not None and sequence_token_char[i][s_index1] == 1:
            final_predicate_index += (i, i + 1)

    return final_predicate_index


def bert_token_equal_standard_token(bert_token, standard_token, lang="english"):
    """
    :param bert_token:
    :param standard_token:
    :param lang:
    :return:
    """
    assert lang in {"english", "chinese"}, "[ERROR] Language only support \"chinese\" and \"english\""

    if bert_token == "[UNK]":
        return True

    bert_token = bert_token.replace("##", "")

    bert_token = bert_token.lower()
    standard_token = standard_token.lower()

    if bert_token == standard_token:
        return True

    return False


def clear_optimize_measure(pair_eval):
    """
    :param pair_eval:
    :return:
    """
    pair_eval.optimize_exact_measure = {}
    pair_eval.optimize_prop_measure = {}
    pair_eval.optimize_binary_measure = {}


def clear_global_measure(pair_eval):
    """
    :param pair_eval:
    :return:
    """
    pair_eval.avg_exact_measure = {}
    pair_eval.avg_prop_measure = {}
    pair_eval.avg_binary_measure = {}