import numpy as np
from data_utils import shared_utils


# support "BMES" mode sequence label parser.
def check_sequence_label_bmes(token_list, sequence_label):
    """
    :param token_list: a list of token.
    :param sequence_label: a list of tag.
    :return:
    """
    data_dict, s_index = {}, -1
    for index in range(len(sequence_label)):
        if sequence_label[index] == "O":
            continue

        cur_pos, cur_tag = shared_utils.get_label_pos_tag(sequence_label[index])
        if cur_pos == "B":
            s_index = index

        if cur_pos in {"S", "E"}:
            if cur_tag not in data_dict:
                data_dict[cur_tag] = []

            s_index = index if cur_pos == "S" else s_index
            e_index = index + 1
            data_dict[cur_tag].append(" ".join(token_list[s_index: e_index]))

    if len(data_dict) == 0:
        return

    for key in data_dict.keys():
        print(key, data_dict[key], end=' ')
    print()


# support "BIO" mode sequence label parser.
def check_sequence_label_bio(token_list, sequence_label):
    """
    :param token_list: a list of token.
    :param sequence_label: a list of tag.
    :return:
    """
    data_dict, s_index = {}, -1
    for index in range(len(sequence_label)):
        if sequence_label[index] == "O":
            cur_pos, cur_tag = "O", "O"
        else:
            cur_pos, cur_tag = shared_utils.split_string(sequence_label[index], "-")

        if index > 0 and sequence_label[index - 1][0] in {"B", "I"} and cur_pos in {"B", "O"}:
            last_pos, last_tag = shared_utils.split_string(sequence_label[index - 1], "-")

            if last_tag not in data_dict:
                data_dict[last_tag] = []

            data_dict[last_tag].append(" ".join(token_list[s_index: index]))

        if cur_pos == "B":
            s_index = index

    if data_dict != {}:
        print(" ".join(token_list))
        print(sequence_label)

    for key in data_dict.keys():
        print(key, data_dict[key], end=' ')
    print()


# support "kesserl14" and "coae13" data set predicate index check.
def check_predicate_index_col(token_list, predicate_index_col):
    """
    :param token_list:
    :param predicate_index_col:
    :return:
    """
    predicate_str = ""
    for predicate_index in predicate_index_col:
        if len(predicate_index) == 2:
            predicate_str += " " + " ".join(token_list[predicate_index[0]: predicate_index[1]])
        elif len(predicate_index) == 4:
            predicate_str += " " + " ".join(token_list[predicate_index[0]: predicate_index[1]])
            predicate_str += "..." + " ".join(token_list[predicate_index[2]: predicate_index[3]])
        print(predicate_index)

    if predicate_str != "":
        print(predicate_str)


def check_correct_sequence_label(sequence_label, position_sys="BMES"):
    if position_sys == "BMES":
        correct_map = {"O": {"B", "S", "O"},
                       "B": {"M", "E"},
                       "M": {"M", "E"},
                       "E": {"B", "S", "O"},
                       "S": {"B", "S", "O"}}
    else:
        correct_map = {"O": {"B", "O"},
                       "B": {"I", "B", "O"},
                       "I": {"B", "I", "O"}}

    check_flag = True
    for index in range(1, len(sequence_label)):
        last_pos, _ = shared_utils.get_label_pos_tag(sequence_label[index - 1])
        cur_pos, _ = shared_utils.get_label_pos_tag(sequence_label[index])

        if last_pos not in correct_map or cur_pos not in correct_map[last_pos]:
            check_flag = False
            print("[ERROR] appear sequence label! {} -> {}".format(last_pos, cur_pos))
            print("[ERROR] appear sequence label! {} -> {}".format(last_pos, cur_pos))
            print("[ERROR] appear sequence label! {} -> {}".format(last_pos, cur_pos))
            print("[ERROR] appear sequence label! {} -> {}".format(last_pos, cur_pos))
            print("[ERROR] appear sequence label! {} -> {}".format(last_pos, cur_pos))

    return check_flag


def print_matrix(matrix, dim=1):
    """
    :param matrix:
    :param dim:
    :return:
    """
    if dim == 0:
        for row in matrix:
            for k in range(len(row)):
                print("{:.2f}".format(row[k]), end=" ")
            print()

    else:
        for index in range(len(matrix)):
            print_matrix(matrix[index], dim=dim-1)


# check chinese version.
def check_mapping_process(map_dict, gold_token, bert_token):
    """
    :param map_dict:
    :param gold_token:
    :param bert_token:
    :return:
    """
    for gold_index, bert_index_col in map_dict.items():

        cur_gold_token = gold_token[gold_index]
        cur_bert_token = "".join(bert_token[bert_index_col[0]: bert_index_col[-1] + 1])

        if cur_gold_token.find("##") != -1:
            cur_gold_token = cur_gold_token[2:]

        if cur_bert_token.lower() == cur_gold_token.lower() or cur_gold_token == "[UNK]":
            continue

        print("{} || {}".format(cur_bert_token, cur_gold_token))


def print_config(config):
    print("===========================")
    print(config.model_mode)
    print(config.file_type)
    print(config.stage_model)
    print(config.path.standard_path['train'])
    print(config.path.standard_path['test'])
    print(config.path.bert_model_path)
    for index in range(len(config.path.predicate_model_path)):
        print(config.path.predicate_model_path[index])
    print("==========================")


def check_predicate_info(predicate_info, sequence_token):
    """
    :param predicate_info:
    :param sequence_token:
    :return:
    """
    for index in range(len(predicate_info)):
        if predicate_info[index] == 1:
            print(sequence_token[index], end=' ')
    print()


def check_data_elem_pair_number(label_col):
    """
    :param label_col: [n, pair_num, elem_dict], elem_dict: {elem: {s_index: length}}.
    :return:
    """
    cnt_dict = {"pair": 0}
    elem_col = ["entity_1", "entity_2", "aspect", "scale", "predicate"]
    for index in range(len(label_col)):
        for pair_index in range(len(label_col[index])):
            pair_num, is_pair = 1, False
            for elem in elem_col:
                elem_num = len(label_col[index][pair_index][elem])

                if elem not in cnt_dict:
                    cnt_dict[elem] = 0

                cnt_dict[elem] += elem_num

                if elem_num != 0:
                    is_pair = True
                pair_num *= max(1, elem_num) if elem != "predicate" else 1

            if is_pair:
                cnt_dict['pair'] += pair_num

    print(cnt_dict)


def check_index_correspond_pair_label(token_col, pair_label, predicate_index):
    """
    :param token_col: [sequence_length]
    :param pair_label: [sequence_length]
    :param predicate_index: (s_index, e_index) or (s_index1, e_index1, s_index2, e_index2)
    :return:
    """
    elem_col = ["entity_1", "entity_2", "aspect", "scale"]

    print(predicate_index)
    if len(predicate_index) == 0:
        return

    elem_label_dict = shared_utils.sequence_label_convert_to_elem_dict(pair_label, elem_col, "BMES")

    write_str = "{"
    for elem_index, elem in enumerate(elem_col):
        write_str += elem + ":["
        for k, (s_index, e_index) in enumerate(elem_label_dict[elem]):
            write_str += "".join(token_col[s_index: e_index])

            if k != len(elem_label_dict[elem]) - 1:
                write_str += " , "
        write_str += "]"

        write_str += ", "

    write_str += "predicate:["

    if len(predicate_index) == 2:
        s_index, e_index = predicate_index
        write_str += "".join(token_col[s_index: e_index]) + "]}\n"
    elif len(predicate_index) == 4:
        s_index1, e_index1, s_index2, e_index2 = predicate_index
        write_str += "".join(token_col[s_index1: e_index1]) + "..."
        write_str += "".join(token_col[s_index2: e_index2]) + "]}\n"

    print(write_str)


def check_token_to_char(token_col, token_char_matrix, dim=1):
    """
    :param token_col: a list of sentences by DDParser tokenizer
    :param token_char_matrix: a list of matrix or matrix, shape is [token_length, char_length]
    :param dim: dim >= 0
    :return:
    """
    assert dim >= 0, "dim need >= 0"

    if dim == 0:
        sentence = "".join(token_col)
        token_length, char_length = len(token_char_matrix), len(token_char_matrix[0])

        print(token_col)
        for token_index in range(token_length):
            cur_char = ""
            for char_index in range(char_length):
                if token_char_matrix[token_index][char_index] == 1:
                    cur_char += sentence[char_index]
            print(cur_char, end=' / ')
        print()

    else:
        for index in range(len(token_col)):
            check_token_to_char(token_col[index], token_char_matrix[index], dim=dim-1)


def check_token_to_bert_char(token_col, bert_char_col, token_char_matrix, dim=1):
    """
    :param token_col: a list of sentences by DDParser tokenizer
    :param token_char_matrix: a list of matrix or matrix, shape is [token_length, char_length]
    :param dim: dim >= 0
    :return:
    """
    assert dim >= 0, "dim need >= 0"

    if dim == 0:
        token_length, char_length = len(token_char_matrix), len(token_char_matrix[0])

        print("....................")
        print(token_col)
        print(bert_char_col)
        for token_index in range(token_length):
            cur_char = ""
            for char_index in range(char_length):
                if token_char_matrix[token_index][char_index] == 1:
                    cur_char += bert_char_col[char_index]

            print(cur_char, end=' / ')
        print()

    else:
        for index in range(len(token_col)):
            check_token_to_bert_char(token_col[index], bert_char_col[index], token_char_matrix[index], dim=dim-1)


def check_token_index_col(token_col, index_col):
    """
    :param token_col: a list of token list
    :param index_col: a token_level list.
    :return:
    """
    for index in range(len(token_col)):
        sequence_token = token_col[index]

        for pair_index in range(len(index_col[index])):
            if len(index_col[index][pair_index]) == 2:
                s_index, e_index = index_col[index][pair_index]
                print("".join(sequence_token[s_index: e_index]))
            elif len(index_col[index][pair_index]) == 4:
                s_index1, e_index1, s_index2, e_index2 = index_col[index][pair_index]
                print("".join(sequence_token[s_index1: e_index1]) + "..." + "".join(sequence_token[s_index2: e_index2]))


def check_input_data(config, input_ids, attn_mask, write_path):
    """
    :param config:
    :param input_ids:
    :param attn_mask:
    :param write_path:
    :return:
    """
    input_col = []
    for index in range(input_ids.size(0)):
        input_col.append(input_ids[index][attn_mask[index] == 1].cpu().numpy().tolist())
    input_token = shared_utils.bert_data_transfer(config.bert_tokenizer, input_col, "ids")

    write_str = ""

    for index in range(len(input_token)):
        write_str += "".join(input_token[index]) + "\n"

    with open(write_path, "a", encoding='utf-8') as f:
        f.write(write_str)


def check_feature_data(input_ids, attn_mask, position_ids, predict_out, truth_label, write_path):
    write_str = ""
    for index in range(input_ids.size(0)):
        cur_input_ids = [str(x) for x in input_ids[index].cpu().numpy().tolist()]
        cur_attn_mask = [str(x) for x in attn_mask[index].cpu().numpy().tolist()]
        cur_position_ids = [str(x) for x in position_ids[index].cpu().numpy().tolist()]
        cur_predict_out = [str(x) for x in predict_out[index].cpu().numpy().tolist()]
        cur_truth_label = [str(x) for x in truth_label[index].cpu().numpy().tolist()]

        write_str += " ".join(cur_input_ids) + "\n"
        write_str += " ".join(cur_attn_mask) + "\n"
        write_str += " ".join(cur_position_ids) + "\n"
        write_str += " ".join(cur_predict_out) + "\n"
        write_str += " ".join(cur_truth_label) + "\n"

    with open(write_path, "a", encoding="utf-8") as f:
        f.write(write_str)


def check_output_feature_data(sequence_label, dim, write_path):
    write_str = ""
    if dim == 1:
        for index in range(sequence_label.size(0)):
            cur_input_ids = [str(x) for x in sequence_label[index].cpu().numpy().tolist()]
            write_str += " ".join(cur_input_ids) + "\n"
    elif dim == 2:
        for index in range(sequence_label.size(0)):
            for s_index in range(sequence_label.size(1)):
                cur_input_ids = [str(x) for x in sequence_label[index][s_index].cpu().numpy().tolist()]
                write_str += " ".join(cur_input_ids) + "\n"

    with open(write_path, "w", encoding="utf-8") as f:
        f.write(write_str)


# 检查经过 mapping 之后的要素与原要素是否一致
def check_elem_position(standard_char_col, standard_sequence_label, bert_token_col, bert_sequence_label, elem_col):
    """
    :param standard_char_col: a list of char.
    :param standard_sequence_label: {"elem": [(s_index, e_index)]}
    :param bert_token_col: a list of token by BertTokenizer.
    :param bert_sequence_label: {"elem": [(s_index, e_index)]}
    :param elem_col: ["entity_1", "entity_2", "aspect", "result"]
    :return:
    """
    for elem in elem_col:
        standard_token, bert_token = [], []
        for elem_position in standard_sequence_label[elem]:
            cur_token = standard_char_col[elem_position[0]: elem_position[1]]
            standard_token.append(cur_token)

        for elem_position in bert_sequence_label[elem]:
            cur_token = bert_token_col[elem_position[0]: elem_position[1]]
            bert_token.append(cur_token)

        for bt in bert_token:
            appear_flag = False
            for st in standard_token:
                if shared_utils.bert_token_equal_standard_token("".join(bt), "".join(st), "chinese"):
                    appear_flag = True
                    break

            if not appear_flag:
                print("++++++++++++++++++++")
                print(bt)
                # print(standard_sequence_label)
                # print(bert_sequence_label)
                print(bert_token, standard_token)


def check_position_ids(sequence_position_ids, sequence_token):
    """
    :param sequence_position_ids:
    :param sequence_token:
    :return:
    """

    predicate_token_col = []
    for index in range(len(sequence_position_ids)):
        if sequence_position_ids[index] == 1:
            predicate_token_col.append(sequence_token[index])
    if len(predicate_token_col) != 0:
        print("".join(predicate_token_col))


def check_midden_output(feature):
    with open("./midden_output.txt", "a", encoding="gb18030") as f:
        print(feature, file=f)


def check_keyword_correct(sequence_token, sequence_keyword):
    keyword_col = []
    for index in range(len(sequence_keyword)):
        if sequence_keyword[index] == "NO":
            continue

        keyword_col.append(sequence_token[index])

    print(keyword_col)


def check_position_correct(sequence_token, sequence_position):
    keyword_col = []
    for index in range(len(sequence_position)):
        if sequence_position[index] == "LBF":
            continue

        keyword_col.append(sequence_token[index])

    print(keyword_col)

