from data_utils import shared_utils
import numpy as np
import copy
import torch


# read file to get sentence and label
def read_standard_file(path):
    """
    :param path:
    :return: sent_col, sent_label_col and label_col
    """
    sent_col, sent_label_col, final_label_col = [], [], []
    last_sentence = ""
    with open(path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            line = line.rstrip('\n')

            # "[[" denote the begin of sequence label.
            if line[:2] == "[[":
                label_col.append(line)

            else:
                if last_sentence != "":
                    cur_sent, cur_sent_label = shared_utils.split_string(last_sentence, "\t")
                    sent_col.append(cur_sent)
                    sent_label_col.append(int(cur_sent_label))
                    final_label_col.append(label_col)

                last_sentence = shared_utils.clear_string(line, replace_symbol={u'\u3000': u""})
                label_col = []

        cur_sent, cur_sent_label = shared_utils.split_string(last_sentence, "\t")
        sent_col.append(cur_sent)
        sent_label_col.append(int(cur_sent_label))
        final_label_col.append(label_col)

        return sent_col, sent_label_col, final_label_col


########################################################################################################################
# Create Mapping col {char_index: bert_index} and Convert Label col Part
########################################################################################################################

# chinese version mapping, {char_index: bert_index}
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
            seq_map[char_index] = bert_index

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
                seq_map[char_index] = bert_index
                char_length += len(seq_gold_char[char_index])

            # print(seq_bert_token[bert_index], end=" || ")
            # print(seq_gold_char[seq_map[bert_index][0]: seq_map[bert_index][-1] + 1])
            char_index = char_index + 1
            bert_index = bert_index + 1

        seq_map[char_index] = bert_index
        # check_utils.check_mapping_process(seq_map, bert_token_col[index], gold_char_col[index])
        mapping_col.append(seq_map)

    return mapping_col


def convert_label_dict_by_mapping(label_col, mapping_col):
    """
    :param label_col: [{"entity_1": {(s_index, e_index)}}]
    :param mapping_col: [{char_index: bert_index]}]
    :return:
    """
    assert len(label_col) == len(mapping_col)

    convert_label_col = []
    for index in range(len(label_col)):
        sequence_label, sequence_map = copy.deepcopy(label_col[index]), mapping_col[index]

        for key in sequence_label.keys():
            sequence_label[key] = sorted(list(sequence_label[key]), key=lambda x: x[0])

            for k in range(len(sequence_label[key])):
                sequence_label[key][k] = list(sequence_label[key][k])

        for key, elem_position_col in sequence_label.items():
            for elem_index, elem_position in enumerate(elem_position_col):
                s_index = elem_position[0]
                e_index = elem_position[1]

                sequence_label[key][elem_index] = [sequence_map[s_index], sequence_map[e_index]]

                if key == "result":
                    sequence_label[key][elem_index].append(elem_position[-1])

        for key in sequence_label.keys():
            for k in range(len(sequence_label[key])):
                sequence_label[key][k] = tuple(sequence_label[key][k])

        convert_label_col.append(sequence_label)

    return convert_label_col


def convert_eng_label_dict_by_mapping(label_col, mapping_col):
    """
    :param label_col: [{"entity_1": {(s_index, e_index)}}]
    :param mapping_col: {bert_index: [char_index]}
    :return:
    """
    assert len(label_col) == len(mapping_col)

    convert_label_col = []
    for index in range(len(label_col)):
        sequence_label, sequence_map = copy.deepcopy(label_col[index]), mapping_col[index]

        for key in sequence_label.keys():
            sequence_label[key] = sorted(list(sequence_label[key]), key=lambda x: x[0])

            for k in range(len(sequence_label[key])):
                sequence_label[key][k] = list(sequence_label[key][k])

        for key, elem_position_col in sequence_label.items():
            for elem_index, elem_position in enumerate(elem_position_col):
                s_index = elem_position[0]
                e_index = elem_position[1]

                # 针对英文数据集可能存在空的情况
                if s_index == -1 or e_index == -1:
                    sequence_label[key][elem_index] = [-1, -1]
                else:
                    sequence_label[key][elem_index] = [sequence_map[s_index][0], sequence_map[e_index][-1]]

                if key == "result":
                    sequence_label[key][elem_index].append(elem_position[-1])

        for key in sequence_label.keys():
            for k in range(len(sequence_label[key])):
                sequence_label[key][k] = tuple(sequence_label[key][k])

        convert_label_col.append(sequence_label)

    return convert_label_col


def convert_tuple_pair_by_mapping(tuple_pair_col, mapping_col):
    """
    :param tuple_pair_col:
    :param mapping_col:
    :return:
    """
    convert_tuple_pair_col = []

    for index in range(len(tuple_pair_col)):
        sequence_tuple_pair, sequence_map = tuple_pair_col[index], mapping_col[index]

        new_sequence_tuple_pair = []
        for pair_index in range(len(sequence_tuple_pair)):
            new_tuple_pair = []
            for k in range(4):
                s_index = sequence_tuple_pair[pair_index][k][0]
                e_index = sequence_tuple_pair[pair_index][k][1]

                if s_index == -1 or e_index == -1:
                    new_tuple_pair.append((-1, -1))
                    continue

                new_s_index, new_e_index = sequence_map[s_index], sequence_map[e_index]
                new_tuple_pair.append((new_s_index, new_e_index))

            # add polarity.
            new_tuple_pair.append(sequence_tuple_pair[pair_index][4])

            new_sequence_tuple_pair.append(new_tuple_pair)

        convert_tuple_pair_col.append(new_sequence_tuple_pair)

    return convert_tuple_pair_col


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
            for char_index, bert_index in seq_map.items():
                # seq_map 中存在{char_index: bert_index} char_index == len(seq_token_char[i])的情况
                # char_index == len(seq_token_char[i])的情况，是为了处理 （s_index, e_index）中e_index 越界的情况
                if char_index > len(seq_token_char[i]) - 1:
                    continue
                seq_matrix[i][bert_index] = max(seq_token_char[i][char_index], seq_matrix[i][bert_index])

        bert_token_char.append(seq_matrix)

    return bert_token_char

########################################################################################################################
# Multi-Sequence Label Generate Code
########################################################################################################################


def get_sequence_label_item(position_symbol, polarity, elem_type, special_symbol):
    """
    :param position_symbol:
    :param polarity:
    :param elem_type:
    :param special_symbol:
    :return:
    """
    polarity_dict = {-1: "Negative", 0: "Equal", 1: "Positive", 2: "None"}
    if elem_type == "result" and special_symbol:
        return position_symbol + "-" + polarity_dict[polarity]
    else:
        return position_symbol


def each_elem_convert_to_multi_sequence_label(sequence_token, each_elem, elem_type, special_symbol):
    """
    :param sequence_token:
    :param each_elem:
    :param elem_type:
    :param special_symbol:
    :return:
    """
    polarity_col = []
    sequence_label = ["O"] * len(sequence_token)

    for elem_position in each_elem:
        s_index, e_index = elem_position[0], elem_position[1]

        if elem_type == "result":
            polarity = elem_position[-1]
            polarity_col.append(polarity)
        else:
            polarity = None

        if e_index == s_index + 1:
            sequence_label[s_index] = get_sequence_label_item("S", polarity, elem_type, special_symbol)
            continue

        sequence_label[s_index] = get_sequence_label_item("B", polarity, elem_type, special_symbol)
        sequence_label[e_index - 1] = get_sequence_label_item("E", polarity, elem_type, special_symbol)

        for k in range(s_index + 1, e_index - 1):
            sequence_label[k] = get_sequence_label_item("M", polarity, elem_type, special_symbol)

    return sequence_label, polarity_col


def elem_dict_convert_to_multi_sequence_label(token_col, label_col, special_symbol=False):
    """
    :param token_col: a list of token list.
    :param label_col: a elem dict like: {elem: [(s_index, e_index)]}
    :param special_symbol: True denote using "B-NEG" sequence label system.
    :return:
    """
    elem_pair_col, polarity_col, result_sequence_label_col = [], [], []
    elem_col = ["entity_1", "entity_2", "aspect", "result"]

    for index in range(len(token_col)):
        sent_multi_col = []
        for elem_index, elem in enumerate(elem_col):
            if elem_index < 3:
                sequence_label, _ = each_elem_convert_to_multi_sequence_label(
                    token_col[index], label_col[index][elem], "norm", special_symbol
                )
                sent_multi_col.append(sequence_label)

            # result may be add special symbol label system.
            else:
                sequence_label, cur_polarity = each_elem_convert_to_multi_sequence_label(
                    token_col[index], label_col[index][elem], "result", special_symbol
                )
                polarity_col.append(cur_polarity)
                result_sequence_label_col.append(sequence_label)

        elem_pair_col.append(sent_multi_col)

    return elem_pair_col, result_sequence_label_col, polarity_col

########################################################################################################################
# Count the number of element or pair Code
########################################################################################################################


def get_tuple_pair_num(tuple_pair_col):
    """
    :param tuple_pair_col:
    :return:
    """
    pair_num, null_tuple_pair = 0, [(-1, -1)] * 5

    for index in range(len(tuple_pair_col)):
        # traverse each pair.
        for pair_index in range(len(tuple_pair_col[index])):
            # skip null tuple pair.
            if tuple_pair_col[index][pair_index] == null_tuple_pair:
                continue

            # print(tuple_pair_col[index][pair_index])
            pair_num += 1

    return pair_num


########################################################################################################################
# Predicate-aware Code Part.
########################################################################################################################

def create_predicate_position_embedding(sequence_length, predicate_index_col, window_size=6, direction=False):
    """
    :param sequence_length: a length of sequence token.
    :param predicate_index_col: a list of predicate index, [] denote none predicate
    :param window_size:
    :param direction: True denote left is -k, right is k. False denote left is k, right is k.
    :return:
    """
    sequence_position_embed = [0] * sequence_length

    for index_col in predicate_index_col:
        # predicate embed denote 1.
        for k in range(index_col[0], index_col[-1] + 1):
            assert 0 <= k < sequence_length, "[ERROR] predicate index error"
            sequence_position_embed[k] = 1

        dist, s_index, e_index = 1, index_col[0], index_col[-1]
        while dist < window_size:
            if shared_utils.index_if_out_bound(0, sequence_length, e_index + dist):
                if sequence_position_embed[e_index + dist] == 0:
                    sequence_position_embed[e_index + dist] = dist + 1
                else:
                    sequence_position_embed[e_index + dist] = min(sequence_position_embed[e_index + dist], dist + 1)
            if shared_utils.index_if_out_bound(0, sequence_length, s_index - dist):
                if sequence_position_embed[s_index - dist] == 0:
                    sequence_position_embed[s_index - dist] = dist + 1
                else:
                    sequence_position_embed[s_index - dist] = min(sequence_position_embed[s_index - dist], dist + 1)
            dist += 1

    return sequence_position_embed


def combine_sub_matrix(a, b, default_val=0, default_type="min"):
    """
    :param a:
    :param b:
    :param default_val:
    :param default_type:
    :return:
    """
    a, b = np.array(a), np.array(b)

    assert a.shape == b.shape, "[ERROR] Matrix A shape must be equal to Matrix B's."
    assert default_type in {"max", "min"}, "[Parameter ERROR] must be \"max\" or \"min\"."

    add_mask = np.bitwise_or((a == default_val), (b == default_val))
    operator_mask = np.invert(add_mask)

    add_result = np.where(add_mask, a + b, 0)

    if default_type == "max":
        return np.where(operator_mask, np.maximum(a, b), add_result).tolist()
    else:
        return np.where(operator_mask, np.minimum(a, b), add_result).tolist()


def combine_sequence_matrix(matrix_col, default_val=0, default_type="min"):
    assert len(matrix_col) >= 1, "[ERROR] The length of matrix col must >= 1."

    last_matrix = matrix_col[0]
    for index in range(1, len(matrix_col)):
        last_matrix = combine_sub_matrix(last_matrix, matrix_col[index], default_val, default_type)

    return last_matrix


def add_norm_matrix(matrix, predicate_index, window_size):
    """
    :param matrix:
    :param predicate_index:
    :param window_size:
    :return:
    """
    s_index, e_index = predicate_index[0], predicate_index[-1]

    l_board, r_board = max(0, s_index - window_size), min(len(matrix), e_index + window_size)

    adj_matrix = copy.deepcopy(matrix)
    for index in range(l_board, r_board):
        adj_matrix[index][index] = 1

    adj_matrix = np.array(adj_matrix, dtype=float)

    # norm.
    D = np.sum(adj_matrix, axis=1)
    D = np.where(D, np.power(D, -1 / 2), D)
    D = np.diag(D)
    A_hat = np.matmul(np.matmul(D, adj_matrix), D)

    return A_hat.tolist()


def create_predicate_position_distance_matrix(
        sequence_length, predicate_index_col, window_size=6, bi_direction=False, combine=False):
    """
    :param sequence_length:
    :param predicate_index_col:
    :param window_size:
    :param bi_direction:
    :param combine:
    :return:
    """
    position_distance_matrix_col = []

    for index_col in predicate_index_col:
        position_distance_matrix = copy.deepcopy(
            shared_utils.create_matrix(sequence_length, sequence_length, default_value=0.0)
        )

        dist, s_index, e_index = 0, index_col[0], index_col[-1]
        while dist < window_size:
            if shared_utils.index_if_out_bound(0, sequence_length, e_index + dist):
                position_distance_matrix[e_index][e_index + dist] = 1 / (dist + 1)
                if bi_direction:
                    position_distance_matrix[e_index + dist][e_index] = 1 / (dist + 1)

            if shared_utils.index_if_out_bound(0, sequence_length, s_index - dist):
                position_distance_matrix[s_index - dist][s_index] = 1 / (dist + 1)
                if bi_direction:
                    position_distance_matrix[s_index][s_index - dist] = 1 / (dist + 1)

            dist += 1

        position_distance_matrix_col.append(
            add_norm_matrix(position_distance_matrix, index_col, window_size)
        )

    if len(position_distance_matrix_col) == 0:
        position_distance_matrix_col.append(shared_utils.create_matrix(
            sequence_length, sequence_length, default_value=0.0
        ))

    # return a global sequence_distance_matrix.
    if combine:
        return [combine_sequence_matrix(position_distance_matrix_col)]

    return position_distance_matrix_col


def get_predicate_vocab(path):
    """
    :param path: predicate vocab file path
    :return: a set of candidate predicate
    """
    predicate_vocab = set()
    with open(path, "r", encoding='utf-8', errors='ignore') as f:
        for line in f.readlines():
            predicate_vocab.add(line.rstrip('\n'))

    return predicate_vocab


def create_predicate_info(predicate_vocab, token_col):
    """
    :param predicate_vocab:
    :param token_col:
    :return:
    """
    predicate_index_col = []
    for i in range(len(token_col)):
        sequence_predicate_index_col = []

        for token in predicate_vocab:
            cur_token_length = len(token)
            for j in range(len(token_col[i])):
                if "".join(token_col[i][j: j + cur_token_length]) == token:
                    sequence_predicate_index_col.append([t for t in range(j, j + cur_token_length)])
        sequence_predicate_index_col = sorted(sequence_predicate_index_col, key=lambda x:x[0])
        predicate_index_col.append(sequence_predicate_index_col)

    return predicate_index_col


def generate_train_pair_data(data_representation, data_label):
    assert len(data_representation) == len(data_label), "[ERROR] Data Length Error."

    feature_dim = len(data_representation[0][0])
    final_representation, final_label = [], []

    for index in range(len(data_representation)):
        if data_representation[index] == [[0] * feature_dim]:
            continue

        for pair_index in range(len(data_representation[index])):
            final_representation.append(data_representation[index][pair_index])
            final_label.append([data_label[index][pair_index]])

    return final_representation, final_label


def create_polarity_train_data(tuple_pair_col, feature_out, bert_feature_out, feature_type=1):
    """
    :param feature_out:
    :param tuple_pair_col:
    :param bert_feature_out:
    :param feature_type:
    :return:
    """
    representation_col, polarity_col, hidden_size = [], [], 5

    for index in range(len(tuple_pair_col)):
        for pair_index in range(len(tuple_pair_col[index])):
            each_pair_representation = []
            for elem_index in range(4):
                s, e = tuple_pair_col[index][pair_index][elem_index]
                if s == -1:
                    # 采用5维 + 768维
                    if feature_type == 0:
                        each_pair_representation.append(torch.zeros(1, hidden_size).cpu())
                        each_pair_representation.append(torch.zeros(1, 768).cpu())

                    # 采用 5维
                    elif feature_type == 1:
                        each_pair_representation.append(torch.zeros(1, hidden_size).cpu())

                    # 采用 768维
                    elif feature_type == 2:
                        each_pair_representation.append(torch.zeros(1, 768).cpu())

                else:
                    # 采用5维 + 768维
                    if feature_type == 0:
                        each_pair_representation.append(
                            torch.mean(feature_out[index][elem_index][s: e], dim=0).cpu().view(-1, hidden_size)
                        )
                        each_pair_representation.append(
                            torch.mean(bert_feature_out[index][s: e], dim=0).cpu().view(-1, 768)
                        )

                    # 采用 5维
                    elif feature_type == 1:
                        each_pair_representation.append(
                            torch.mean(feature_out[index][elem_index][s: e], dim=0).cpu().view(-1, hidden_size)
                        )

                    # 采用 768维
                    elif feature_type == 2:
                        each_pair_representation.append(
                            torch.mean(bert_feature_out[index][s: e], dim=0).cpu().view(-1, 768)
                        )

            if torch.cuda.is_available():
                cur_representation = torch.cat(each_pair_representation, dim=-1).view(-1).cpu().numpy().tolist()
            else:
                cur_representation = torch.cat(each_pair_representation, dim=-1).view(-1).numpy().tolist()

            representation_col.append(cur_representation)

            assert tuple_pair_col[index][pair_index][-1][0] in {-1, 0, 1, 2}, "[ERROR] Tuple Pair Col Error."
            polarity_col.append([tuple_pair_col[index][pair_index][-1][0] + 1])

    return representation_col, polarity_col


def get_after_pair_representation(pair_hat, representation):
    """
    :param pair_hat:
    :param representation:
    :return:
    """
    feature_dim = len(representation[0][0])

    if len(pair_hat) == 0:
        return representation

    for index in range(len(representation)):
        assert len(pair_hat[index]) == len(representation[index]), "[ERROR] Param error or Data process error."

        for pair_index in range(len(representation[index])):
            if pair_hat[index][pair_index] == 0:
                representation[index][pair_index] = [0] * feature_dim

    return representation


def create_polarity_multi_label(token_col, tuple_pair_col):
    """
    :param token_col:
    :param tuple_pair_col:
    :return:
    """
    final_label_col = []

    for index in range(len(token_col)):
        final_label_col.append(create_sequence_polarity_multi_label(
            len(token_col[index]), tuple_pair_col[index]
        ))

    return final_label_col


def create_sequence_polarity_multi_label(sequence_length, tuple_pair_col):
    """
    :param sequence_length:
    :param tuple_pair_col: [pair_num, tuple_pair]
    :return: [polarity_num, elem_num, sequence_length]
    """
    sequence_polarity_multi_label = [copy.deepcopy(create_init_multi_label(sequence_length)) for _ in range(4)]

    for index in range(len(tuple_pair_col)):
        sequence_polarity_multi_label = change_sequence_label_by_tuple_pair(
            sequence_polarity_multi_label, tuple_pair_col[index]
        )

    return sequence_polarity_multi_label


def create_init_multi_label(sequence_length):
    """
    :param sequence_length:
    :return: [4, sequence_length]
    """
    return shared_utils.create_matrix(4, sequence_length, default_value='O')


def change_sequence_label_by_tuple_pair(sequence_multi_label, tuple_pair):
    """
    :param sequence_multi_label: [4, 4, sequence_label]
    :param tuple_pair: [(s_index, e_index)]
    :return:
    """
    polarity_index = tuple_pair[-1][0] + 1

    for index in range(len(tuple_pair) - 1):
        s_index, e_index = tuple_pair[index]
        if s_index == -1 or e_index == -1:
            continue

        if s_index == e_index - 1:
            sequence_multi_label[polarity_index][index][s_index] = "S"
        else:
            sequence_multi_label[polarity_index][index][s_index] = "B"
            sequence_multi_label[polarity_index][index][e_index - 1] = "E"

            for t in range(s_index + 1, e_index - 1):
                sequence_multi_label[polarity_index][index][t] = "M"

    return sequence_multi_label




def create_polarity_label(tuple_pair_col):
    """
    :param tuple_pair_col:
    :return:
    """
    polarity_col = []
    for index in range(len(tuple_pair_col)):
        sequence_polarity = [0] * 4
        for pair_index in range(len(tuple_pair_col[index])):
            sequence_polarity[tuple_pair_col[index][pair_index][-1][0] + 1] = 1

        polarity_col.append(sequence_polarity)

    return polarity_col


def convert_eng_tuple_pair_by_mapping(tuple_pair_col, mapping_col):
    """
    :param tuple_pair_col:
    :param mapping_col: {token_index: [bert_index]}
    :return:
    """
    convert_tuple_pair_col = []

    for index in range(len(tuple_pair_col)):
        sequence_tuple_pair, sequence_map = tuple_pair_col[index], mapping_col[index]

        new_sequence_tuple_pair = []
        for pair_index in range(len(sequence_tuple_pair)):
            new_tuple_pair = []
            for k in range(4):
                s_index = sequence_tuple_pair[pair_index][k][0]
                e_index = sequence_tuple_pair[pair_index][k][1]

                if s_index == -1 or e_index == -1:
                    new_tuple_pair.append((-1, -1))
                    continue

                new_s_index, new_e_index = sequence_map[s_index][0], sequence_map[e_index][-1]
                new_tuple_pair.append((new_s_index, new_e_index))

            # add polarity.
            new_tuple_pair.append(sequence_tuple_pair[pair_index][4])

            new_sequence_tuple_pair.append(new_tuple_pair)

        convert_tuple_pair_col.append(new_sequence_tuple_pair)

    return convert_tuple_pair_col


def get_tuple_pair_col_board(current_tuple_pair):
    s_index, e_index = 1000, -1
    for elem_index in range(4):
        if current_tuple_pair[elem_index] == (-1, -1):
            continue

        # 得到当前要素的左右边界
        elem_s_index = current_tuple_pair[elem_index][0]
        elem_e_index = current_tuple_pair[elem_index][1] - 1

        s_index = min(s_index, elem_s_index)
        e_index = max(e_index, elem_e_index)

    if s_index == 1000 or e_index == -1:
        return -1, -1

    return s_index, e_index


def create_comparative_board_index_col(token_col, tuple_pair_col):
    """
    :param token_col:
    :param tuple_pair_col: [n, pair_num, tuple_pair], tuple_pair = [(s_index, e_index)] * elem_num
    :return:
    """
    assert len(token_col) == len(tuple_pair_col)

    start_index_col, end_index_col = [], []
    for index in range(len(token_col)):
        start_index, end_index = copy.deepcopy([0] * len(token_col)), copy.deepcopy([0] * len(token_col))

        for pair_index in range(len(tuple_pair_col[index])):
            print(tuple_pair_col[index][pair_index])
            cur_s_index, cur_e_index = get_tuple_pair_col_board(tuple_pair_col[index][pair_index])
            print(cur_s_index, cur_e_index)

            start_index[cur_s_index], end_index[cur_e_index] = 1, 1
        print("----------------------")
        start_index_col.append(start_index)
        end_index_col.append(end_index)

    return start_index_col, end_index_col

