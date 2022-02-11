import os
import copy
from tqdm import tqdm
from data_utils.label_parse import LabelParser
from data_utils import current_program_code as cpc
from data_utils import shared_utils, check_utils
from Baseline_Systems.crf_utils import create_feature_file
from eval_utils import extern_eval
from multiprocessing.pool import Pool
from functools import partial


def read_keyword_vocab(vocab_path):
    keyword_vocab = []
    with open(vocab_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            if line == "\n":
                continue

            keyword_vocab.append(shared_utils.split_string(line.strip('\n'), " "))

    return keyword_vocab


def create_sequence_keyword_feature(sequence_token, sequence_pos, keyword_vocab, lang="en"):
    """
    :param sequence_token:
    :param sequence_pos:
    :param keyword_vocab: a list of list
    :param lang:
    :return:
    """
    comparative_dict = {"JJR", "JJS", "RBR", "RBS"}
    sequence_keyword = ["NO"] * len(sequence_token)
    for index in range(len(sequence_token)):
        if lang == "en" and sequence_pos[index] in comparative_dict:
            sequence_keyword[index] = "YES"

    for keyword in keyword_vocab:
        for index in range(len(sequence_token)):
            error_flag = False
            if sequence_token[index] == keyword[0]:
                for t in range(len(keyword)):
                    if (index + t) < len(sequence_token) and keyword[t] != sequence_token[index + t] and keyword[t] != "<word>":
                        error_flag = True
            else:
                error_flag = True

            if error_flag:
                continue

            for t in range(len(keyword)):
                sequence_keyword[index + t] = "YES"

    return sequence_keyword


def create_keyword_feature(token_col, pos_col, keyword_path, lang="en"):
    keyword_vocab = read_keyword_vocab(keyword_path)

    keyword_col = []
    for index in range(len(token_col)):
        sequence_keyword = create_sequence_keyword_feature(token_col[index], pos_col[index], keyword_vocab, lang=lang)

        keyword_col.append(sequence_keyword)

    return keyword_col


def get_all_csr_by_keyword(sequence_token, sequence_pos, keyword, radius=3, pos_tag=True):
    rule_col = []
    for index in range(len(sequence_token)):
        cur_span = sequence_token[index] if not pos_tag else sequence_pos[index]
        error_flag = False
        if cur_span == keyword[0]:
            for t in range(len(keyword)):
                if (index + t) < len(sequence_token) and keyword[t] != sequence_token[index + t] and keyword[t] != "<word>":
                    error_flag = True
        else:
            error_flag = True

        if error_flag:
            continue

        l_board, r_board = max(0, index - radius), min(len(sequence_token), radius + len(keyword) + index)

        class_sequence_rule = []
        for t in range(l_board, r_board):
            if index <= t < index + len(keyword):
                if pos_tag:
                    class_sequence_rule.append(sequence_token[t] + sequence_pos[t])
                else:
                    class_sequence_rule.append(keyword[t - index] + sequence_pos[t])
            else:
                class_sequence_rule.append(sequence_pos[t])

        rule_col.append(class_sequence_rule)

    return list(rule_col)


def get_feature_data(sentence_col, stanford_path, pre_process_path, keyword_path, lang="en"):
    # if os.path.exists(pre_process_path):
    #     token_col, pos_col, keyword_col = shared_utils.read_pickle(pre_process_path)
    #
    # else:
    sf = create_feature_file.stanfordFeature(sentence_col, stanford_path, lang)
    token_col = sf.get_tokenizer()
    pos_col = sf.get_pos_feature()

    keyword_col = create_keyword_feature(token_col, pos_col, keyword_path, lang=lang)

        # data_col = [token_col, pos_col, keyword_col]
        # shared_utils.write_pickle(data_col, pre_process_path)

    return token_col, pos_col, keyword_col


def label_col_convert_to_elem_char_label(sentence_col, label_col, label_type):
    """
    :param label_col: [n, pair_num, elem_dict], elem_dict denote {"entity_1": [(s_index, e_index)]}
    :return:
    """
    elem_char_label_col, label_feature_col = [], []
    for index in range(len(sentence_col)):
        sequence_label = copy.deepcopy(['O'] * len(sentence_col[index]))
        sequence_label_feature = copy.deepcopy(['O'] * len(sentence_col[index]))
        for elem_representation in label_col[index][label_type]:
            if len(elem_representation) == 2:
                s_index, e_index = elem_representation
            else:
                s_index, e_index, _ = elem_representation

            if s_index == e_index - 1:
                sequence_label[s_index] = "S"
                sequence_label_feature[s_index] = "$" + label_type
                continue
            print(sentence_col[index])
            print(s_index, e_index)
            sequence_label[s_index] = "B"
            sequence_label[e_index - 1] = "E"

            for t in range(s_index + 1, e_index - 1):
                sequence_label[t] = "M"

            for t in range(s_index, e_index):
                sequence_label_feature[t] = "$" + label_type

        elem_char_label_col.append(sequence_label)
        label_feature_col.append(sequence_label_feature)

    return elem_char_label_col, label_feature_col


def char_label_convert_to_token_label(token_char_col, char_label_col, char_feature_label_col):
    token_label_col, label_feature_col = [], []
    for index in range(len(token_char_col)):
        sequence_token_label = ["O"] * len(token_char_col[index])
        sequence_label_feature = ["O"] * len(token_char_col[index])

        char_index = 0
        for token_index in range(len(token_char_col[index])):
            position_dict = {"O": 0, "B": 0, "M": 0, "E": 0, "S": 0}
            while char_index < len(token_char_col[index][token_index]) and token_char_col[index][token_index][char_index] == 1:
                position_dict[char_label_col[index][char_index]] += 1
                if char_feature_label_col[index][char_index] != "O":
                    sequence_label_feature[token_index] = char_feature_label_col[index][char_index]
                char_index += 1

            if (position_dict["B"] != 0 and position_dict["E"] != 0) or position_dict['S'] != 0:
                sequence_token_label[token_index] = "S"

            elif position_dict["B"] != 0:
                sequence_token_label[token_index] = "B"

            elif position_dict["E"] != 0:
                sequence_token_label[token_index] = "E"

            elif position_dict["M"] != 0:
                sequence_token_label[token_index] = "M"

            else:
                sequence_token_label[token_index] = "O"

        label_feature_col.append(sequence_label_feature)
        token_label_col.append(sequence_token_label)

    return token_label_col, label_feature_col


def generate_radius_sequence(sequence, s_index, radius=4):
    e_index = s_index
    while sequence[e_index][0] == sequence[s_index][0]:
        e_index += 1

    init_sequence_col = [sequence[t] for t in range(s_index, e_index)]
    cnt, l, r = 1, s_index - 1, e_index

    l_board, r_board = 0, len(sequence) - 1
    while r < r_board and cnt <= radius:
        init_sequence_col.append(["r" + str(cnt)])
        init_sequence_col.append(sequence[r])
        cnt += 1
        r += 1

    cnt = 1
    while l >= l_board and cnt <= radius:
        init_sequence_col.insert(0, ["l" + str(cnt)])
        init_sequence_col.insert(0, sequence[l])
        cnt += 1
        l -= 1

    return init_sequence_col


def generate_radius_sequence_without_distance(sequence, s_index, radius=4):
    e_index = s_index
    while sequence[e_index][0] == sequence[s_index][0]:
        e_index += 1

    init_sequence_col = [sequence[t] for t in range(s_index, e_index)]
    cnt, l, r = 1, s_index - 1, e_index

    l_board, r_board = 0, len(sequence) - 1
    while r < r_board and cnt <= radius:
        init_sequence_col.append(sequence[r])
        cnt += 1
        r += 1

    cnt = 1
    while l >= l_board and cnt <= radius:
        init_sequence_col.insert(0, sequence[l])
        cnt += 1
        l -= 1

    return init_sequence_col


def create_init_database(token_col, pos_col, label_col, keyword_col, elem_type, radius):
    sequence_col = []
    for index in range(len(token_col)):
        cur_sequence = [["#start"]]
        # cur_sequence = []
        for t in range(len(token_col[index])):
            if keyword_col[index][t] == "YES":
                cur_sequence.append([token_col[index][t] + pos_col[index][t]])
            else:
                if label_col[index][t] != "O":
                    cur_sequence.append([elem_type, pos_col[index][t]])
                    # cur_sequence.append([elem_type])
                else:
                    cur_sequence.append([token_col[index][t], pos_col[index][t]])
                    # cur_sequence.append([pos_col[index][t]])
        cur_sequence.append(["#end"])
        sequence_col.append(cur_sequence)

    database = []
    for index in range(len(sequence_col)):
        for t in range(len(sequence_col[index])):
            if sequence_col[index][t][0] == elem_type and (index - 1 >= 0 and sequence_col[index][t - 1][0] != elem_type):
                database.append(generate_radius_sequence_without_distance(sequence_col[index], t, radius))
                # database.append(generate_radius_sequence(sequence_col[index], t, radius))
    return database


def select_comparative_sentence(data_col):
    token_col, pos_col, keyword_col, elem_token_label_col, label_feature_col, sent_label_col = data_col

    new_token_col, new_pos_col, new_keyword_col = [], [], []
    new_elem_token_label_col, new_label_feature_col = [], []
    for index in range(len(sent_label_col)):
        if sent_label_col[index] == 1:
            new_token_col.append(token_col[index])
            new_pos_col.append(pos_col[index])
            new_keyword_col.append(keyword_col[index])
            new_elem_token_label_col.append(elem_token_label_col[index])
            new_label_feature_col.append(label_feature_col[index])

    return new_token_col, new_pos_col, new_keyword_col, new_elem_token_label_col, new_label_feature_col


def create_pre_process_data(
        init_file_path, pre_process_path, stanford_path, keyword_path, elem_type, lang="en", comp_flag=False
):
    if os.path.exists(pre_process_path):
        token_col, pos_col, keyword_col, elem_token_label_col, label_feature_col = \
            shared_utils.read_pickle(pre_process_path)

    else:
        sent_col, sent_label_col, label_col = cpc.read_standard_file(init_file_path)
        LP = LabelParser(label_col, ["entity_1", "entity_2", "aspect", "result"])

        if lang == "en":
            label_col, _ = LP.parse_sequence_label("&&", sent_col, file_type="eng")
        else:
            label_col, _ = LP.parse_sequence_label("&", sent_col)

        token_col, pos_col, keyword_col = get_feature_data(
            sent_col, stanford_path, pre_process_path, keyword_path, lang
        )

        # char_label convert to token_label.
        if lang == "en":
            elem_token_label_col, label_feature_col = label_col_convert_to_elem_char_label(
                token_col, label_col, elem_type
            )
        else:
            elem_char_label_col, label_feature_col = label_col_convert_to_elem_char_label(
                sent_col, label_col, elem_type
            )
            token_char_col = create_feature_file.create_token_to_char_feature(token_col)
            elem_token_label_col, label_feature_col = char_label_convert_to_token_label(
                token_char_col, elem_char_label_col, label_feature_col
            )

        data_col = [token_col, pos_col, keyword_col, elem_token_label_col, label_feature_col, sent_label_col]

        if comp_flag:
            token_col, pos_col, keyword_col, elem_token_label_col, label_feature_col = \
                select_comparative_sentence(data_col)

        data_col = [token_col, pos_col, keyword_col, elem_token_label_col, label_feature_col]
        shared_utils.write_pickle(data_col, pre_process_path)

    return token_col, pos_col, keyword_col, elem_token_label_col, label_feature_col


def get_init_database(database_path, init_file_path, stanford_path, pre_process_path, keyword_path, elem_type, radius, lang="en"):
    if os.path.exists(database_path):
        database, elem_token_label_col = shared_utils.read_pickle(database_path)
    else:
        token_col, pos_col, keyword_col, elem_token_label_col, label_feature_col = create_pre_process_data(
            init_file_path, pre_process_path, stanford_path, keyword_path, elem_type, lang
        )

        database = create_init_database(token_col, pos_col, label_feature_col, keyword_col, elem_type, radius)

        shared_utils.write_pickle([database, elem_token_label_col], database_path)

    return database, elem_token_label_col


def create_final_csr_col(database, min_support):
    from pyspark import SparkContext
    from pyspark import SparkConf
    from pyspark.mllib.fpm import PrefixSpan

    sc = SparkContext("local", "testing")
    rdd = sc.parallelize(database, 2)
    model = PrefixSpan.train(rdd, minSupport=min_support)
    csr_col = model.freqSequences().collect()
    final_csr_col = [x.sequence for x in csr_col]

    # final_csr_col = label_sequential_rule.prefix_span(
    #     database, min_support
    # )
    return final_csr_col


def drop_without_label_rules(rule_col, goal_item):
    final_rule_col = []
    for index in range(len(rule_col)):
        appear_flag, error_flag = False, False
        for t in range(len(rule_col[index])):
            if rule_col[index][t] == goal_item:
                appear_flag = True

            if len(rule_col[index][t]) >= 2 and rule_col[index][t][0] == goal_item[0]:
                error_flag = True

        if appear_flag and not error_flag:
            final_rule_col.append(rule_col[index])

    return final_rule_col


def drop_simple_label(rule_col, goal_item):
    final_rule_col = []
    for index in range(len(rule_col)):
        appear_flag = False
        for t in range(len(rule_col[index])):
            if rule_col[index][t] != goal_item:
                appear_flag = True

        if appear_flag:
            final_rule_col.append(rule_col[index])

    return final_rule_col


def drop_position_symbol(rule_col, radius):
    position_item, new_rule_col = set(), []
    for i in range(1, radius + 1):
        position_item.add("l" + str(i))
        position_item.add("r" + str(i))

    for index in range(len(rule_col)):
        drop_rule = []
        for t in range(len(rule_col[index])):
            if rule_col[index][t][0] in position_item:
                continue
            drop_rule.append(rule_col[index][t])

        new_rule_col.append(drop_rule)
    return new_rule_col


def add_pos_to_label(rule_col, goal_item):
    add_pos_col = ["NN", "NR"]

    final_rule_col = []
    for index in range(len(rule_col)):
        for pos in add_pos_col:
            cur_rule = copy.deepcopy(rule_col[index])
            for t in range(len(cur_rule)):
                if cur_rule[t] == goal_item:
                    cur_rule[t] = [goal_item[0], pos]
            final_rule_col.append(cur_rule)

    return final_rule_col


def create_init_sequence(token_col, pos_col, keyword_col, label_feature_col, elem_type):
    init_sequence_col = []

    for index in range(len(token_col)):
        # cur_sequence = []
        cur_sequence = [["#start"]]
        for t in range(len(token_col[index])):
            if keyword_col[index][t] == "YES":
                cur_sequence.append([token_col[index][t] + pos_col[index][t]])
            else:
                if label_feature_col[index][t] != "O":
                    cur_sequence.append([elem_type, pos_col[index][t]])
                    # cur_sequence.append([elem_type])
                else:
                    cur_sequence.append([token_col[index][t], pos_col[index][t]])
                    # cur_sequence.append([pos_col[index][t]])
        cur_sequence.append(['#end'])
        init_sequence_col.append(cur_sequence)

    return init_sequence_col


########################################################################################################################
# test part
########################################################################################################################

def create_test_database(
        database_path, init_file_path, stanford_path, pre_process_path, keyword_path, elem, lang="en", comp_flag=False
):
    if os.path.exists(database_path):
        database, elem_token_label_col = shared_utils.read_pickle(database_path)
    else:
        token_col, pos_col, keyword_col, elem_token_label_col, label_feature_col = create_pre_process_data(
            init_file_path, pre_process_path, stanford_path, keyword_path, elem, lang, comp_flag
        )

        database = create_init_sequence(token_col, pos_col, keyword_col, label_feature_col, elem)

        shared_utils.write_pickle([database, elem_token_label_col], database_path)

    return database, elem_token_label_col


def simple_sequence_is_contain_lsr(sequence, rule, elem, pre_elem_dict, frequently_pos_set):
    def is_contain(a, b):
        if len(b) > len(a):
            return False

        elif len(a) == len(b):
            for t in range(len(a)):
                if b[t] == a[t] or b[t] == elem:
                    continue
                else:
                    return False
            return True

        else:
            if a[0] == b[0] or a[1] == b[0] or (b[0] == elem and a[1] in frequently_pos_set):
                return True
            return False

    for index in range(len(sequence) - len(rule) + 1):
        extract_flag = True
        if is_contain(sequence[index], rule[0]):
            for t in range(len(rule)):
                if not is_contain(sequence[index + t], rule[t]):
                    extract_flag = False
                    break

        else:
            extract_flag = False

        if not extract_flag:
            continue
        # print(rule)
        t, s_index = 0, -1
        while t < len(rule):
            if rule[t][0] == elem:
                s_index = t
                while t < len(rule) and rule[t][0] == elem:
                    t += 1
                # print(sequence[index + s_index: index + t])

                pre_elem_dict[elem].add((index + s_index, index + t))
            t += 1


def parse_sequence_by_lsr(test_sequence, lsr_col, elem, pre_dict_col, frequently_pos_set):
    label_dict_col = copy.deepcopy(pre_dict_col)
    for index in range(len(test_sequence)):
        cur_elem_dict = {elem: set()}
        # print(test_sequence[index])
        for rule in lsr_col:
            simple_sequence_is_contain_lsr(test_sequence[index], rule, elem, cur_elem_dict, frequently_pos_set)

        cur_elem_dict[elem] = list(cur_elem_dict[elem])
        if len(pre_dict_col) == 0:
            label_dict_col.append(cur_elem_dict)
        else:
            label_dict_col[index][elem] = cur_elem_dict[elem]
        # print(cur_elem_dict)
        # print("-------------------------")

    return label_dict_col


def count_pos_tag_number(sequence_col, gold_dict, elem):
    pos_dict = {}
    for index in range(len(gold_dict)):
        for t in range(len(gold_dict[index][elem])):
            s, e = gold_dict[index][elem][t]

            s, e = s + 1, e + 1
            # print(sequence_col[index][s: e])
            for k in range(s, e):
                if len(sequence_col[index][k]) == 1:
                    continue
                cur_pos = sequence_col[index][k][1]
                if cur_pos not in pos_dict:
                    pos_dict[cur_pos] = 0
                pos_dict[cur_pos] += 1

    frequently_pos_set, pos_item_sum = set(), 0

    for key, value in pos_dict.items():
        pos_item_sum += value

    for key, value in pos_dict.items():
        if value / pos_item_sum > 0.1:
            frequently_pos_set.add(key)
            print("{}: {}".format(key, value))
    print("---------------------------------")

    return frequently_pos_set

########################################################################################################################
# select label sequential rule by high confidence
########################################################################################################################


def is_contain_star(rule_item):
    assert len(rule_item) == 1 or len(rule_item) == 2, "[ERROR] rule item length error"

    if len(rule_item) == 1:
        if rule_item[0] == "*":
            return True
        return False

    else:
        if rule_item[0] == "*" or rule_item[1] == "*":
            return True
        return False


def generate_match_rule(rule, elem):
    pos_rule = copy.deepcopy(rule)
    elem_pos_rule = copy.deepcopy(rule)

    for index in range(len(rule)):
        if rule[index] == [elem]:
            pos_rule[index] = ["*", "*"]
            elem_pos_rule[index] = [elem, "*"]

    return pos_rule, elem_pos_rule


def is_match_items(sequence_item, rule_item, frequency_pos_set):
    assert len(sequence_item) == 1 or len(sequence_item) == 2, "[ERROR] item length error!"
    assert len(rule_item) == 1 or len(rule_item) == 2, "[ERROR] item length error!"

    if len(sequence_item) < len(rule_item):
        return False

    elif len(sequence_item) == len(rule_item):
        if is_contain_star(rule_item):
            if rule_item[0] == "*" and rule_item[1] == "*":
                if sequence_item[0] in frequency_pos_set or sequence_item[1] in frequency_pos_set:
                    return True
                return False

            else:
                if sequence_item[0] in frequency_pos_set and sequence_item[1] == rule_item[0]:
                    return True
                elif sequence_item[1] in frequency_pos_set and sequence_item[0] == rule_item[0]:
                    return True
                return False


        else:
            if len(rule_item) == 1:
                if sequence_item == rule_item:
                    return True
                return False
            else:
                if sequence_item[0] == rule_item[1] and sequence_item[1] == rule_item[0]:
                    return True
                return False

    else:
        if sequence_item[0] == rule_item[0] or sequence_item[1] == rule_item[0]:
            return True
        return False


def is_label_item(rule_item):
    if len(rule_item) == 1:
        return False

    if rule_item[1] == "*" and rule_item[0] != "*":
        return True
    return False


def check_label_is_continue(rule, sequence_index_col):
    error_flag = False
    for index in range(len(rule)):
        if is_label_item(rule[index]):
            t = index + 1
            while t < len(rule) and is_label_item(rule[t]):
                if sequence_index_col[t] != sequence_index_col[t-1] + 1:
                    error_flag = True
                t += 1
    return not error_flag


def sequence_is_match_rule(sequence, rule, frequency_pos_set, radius):
    dp = [[0 for _ in range(len(sequence))] for _ in range(len(rule))]
    last_min_index = []

    for i in range(len(rule)):
        min_index = len(sequence)

        s_index = 0 if i == 0 else last_min_index[-1] + 1
        for j in range(s_index, len(sequence)):
            if is_match_items(sequence[j], rule[i], frequency_pos_set):
                min_index = min(min_index, j)
                dp[i][j] = 1

        if min_index == len(sequence):
            break

        last_min_index.append(min_index)

    if len(last_min_index) == len(rule):
        if check_label_is_continue(rule, last_min_index) and (last_min_index[-1] - last_min_index[0]) < 2 * radius:
            return True, last_min_index
        return False, None

    return False, None


def parse_sequence_by_rule(sequence, rule, frequency_pos_set, radius):
    dp = [[0 for _ in range(len(sequence))] for _ in range(len(rule))]
    last_min_index = []

    for i in range(len(rule)):
        min_index = len(sequence)

        s_index = 0 if i == 0 else last_min_index[-1] + 1
        for j in range(s_index, len(sequence)):
            if is_match_items(sequence[j], rule[i], frequency_pos_set):
                min_index = min(min_index, j)
                dp[i][j] = 1

        if min_index == len(sequence):
            break

        last_min_index.append(min_index)

    position_set = set()
    if len(last_min_index) == len(rule):
        if check_label_is_continue(rule, last_min_index) and (last_min_index[-1] - last_min_index[0]) < 2 * radius:
            for t in range(len(rule)):
                if is_label_item(rule[t]):
                    s_index, e_index = t, t
                    while e_index < len(rule) and is_label_item(rule[e_index]):
                        e_index += 1
                    position_set.add((s_index, e_index))

    return position_set


def parse_test_sequence_lsr_col(sequence_col, rule_col, pre_elem_dict, elem, frequency_pos_set, radius):
    final_elem_col = copy.deepcopy(pre_elem_dict)
    for index in range(len(sequence_col)):
        cur_position_set = set()

        for t in range(len(rule_col)):
            _, elem_pos_rule = generate_match_rule(rule_col[t], elem)
            cur_rule_parse_set = parse_sequence_by_rule(
                sequence_col[index], elem_pos_rule, frequency_pos_set, radius
            )

            cur_position_set = cur_position_set | cur_rule_parse_set

        if len(pre_elem_dict) == 0:
            final_elem_col.append({elem: []})

        final_elem_col[index][elem] = list(combine_interval(cur_position_set))

    return final_elem_col


def calculate_rule_confidence(rule, sequence_col, elem, frequency_pos_set, radius):
    rule_dict = {"rule": rule, "seq_id": {}, "seq_cnt": []}
    for index in range(len(sequence_col)):
        pos_rule, pos_elem_rule = generate_match_rule(rule, elem)
        pos_elem_match_flag, pos_elem_index_col = sequence_is_match_rule(
            sequence_col[index], pos_elem_rule, frequency_pos_set, radius
        )

        pos_match_flag, _ = sequence_is_match_rule(
            sequence_col[index], pos_rule, frequency_pos_set, radius
        )
        if pos_elem_match_flag:
            rule_dict["seq_id"][index] = pos_elem_index_col
            rule_dict["seq_cnt"].append(index)

        elif pos_match_flag:
            rule_dict["seq_cnt"].append(index)

    if len(rule_dict['seq_cnt']) == 0:
        rule_dict['confidence'] = 0
    else:
        rule_dict['confidence'] = len(rule_dict['seq_id']) / len(rule_dict['seq_cnt'])

    return rule_dict


def calculate_label_sequential_rule_confidence(sequence_col, rule_col, elem, frequency_pos_set, radius):
    # rule_id_confidence_dict: {rule_id: {"seq_id": [], "seq_cnt": 0}}
    pool = Pool(processes=20)
    rule_id_confidence_dict = pool.map(
        partial(calculate_rule_confidence,
                sequence_col=sequence_col,
                elem=elem,
                frequency_pos_set=frequency_pos_set,
                radius=radius),
        rule_col
    )
    pool.close()
    pool.join()

    rule_id_confidence_dict = sorted(rule_id_confidence_dict, key=lambda x: -x['confidence'])

    return rule_id_confidence_dict


def repeat_calculate_rule_confidence(rule, sequence_col, info_dict, elem, frequency_pos_set, radius):
    rule_dict = {"seq_id": {}, "seq_cnt": []}

    for index in range(len(info_dict['seq_cnt'])):
        pos_rule, pos_elem_rule = generate_match_rule(rule, elem)
        pos_elem_match_flag, pos_elem_index_col = sequence_is_match_rule(
            sequence_col[info_dict['seq_cnt'][index]], pos_elem_rule, frequency_pos_set, radius
        )
        pos_match_flag, _ = sequence_is_match_rule(
            sequence_col[info_dict['seq_cnt'][index]], pos_rule, frequency_pos_set, radius
        )
        if pos_elem_match_flag:
            rule_dict["seq_id"][index] = pos_elem_index_col
            rule_dict["seq_cnt"].append(info_dict['seq_cnt'][index])

        elif pos_match_flag:
            rule_dict["seq_cnt"].append(info_dict['seq_cnt'][index])

    return rule_dict


def repeat_calculate_label_sequential_rule_confidence(
        cur_rule_id_confidence_dict, sequence_col, elem, frequency_pos_set, radius
):
    rule_dict = {"rule": cur_rule_id_confidence_dict['rule'], "seq_id": {}, "seq_cnt": []}

    for index in range(len(cur_rule_id_confidence_dict['seq_cnt'])):
        pos_rule, pos_elem_rule = generate_match_rule(cur_rule_id_confidence_dict['rule'], elem)
        pos_elem_match_flag, pos_elem_index_col = sequence_is_match_rule(
            sequence_col[cur_rule_id_confidence_dict['seq_cnt'][index]], pos_elem_rule, frequency_pos_set, radius
        )
        pos_match_flag, _ = sequence_is_match_rule(
            sequence_col[cur_rule_id_confidence_dict['seq_cnt'][index]], pos_rule, frequency_pos_set, radius
        )
        if pos_elem_match_flag:
            rule_dict["seq_id"][cur_rule_id_confidence_dict['seq_cnt'][index]] = pos_elem_index_col
            rule_dict["seq_cnt"].append(cur_rule_id_confidence_dict['seq_cnt'][index])

        elif pos_match_flag:
            rule_dict["seq_cnt"].append(cur_rule_id_confidence_dict['seq_cnt'][index])

    if len(rule_dict['seq_cnt']) == 0:
        rule_dict["confidence"] = 0
    else:
        rule_dict["confidence"] = len(rule_dict['seq_id']) / len(rule_dict['seq_cnt'])

    return rule_dict


def mask_sequence_by_highest_confidence_rule(sequence_col, rule_info_dict, elem):
    """
    :param sequence_col:
    :param rule:
    :param rule_info_dict: {"seq_id": {sequence_id: rule_to_sequence_index}}
    :param elem:
    :param frequency_pos_set:
    :param radius:
    :return:
    """
    _, pos_elem_rule = generate_match_rule(rule_info_dict['rule'], elem)
    mask_sequence_col = copy.deepcopy(sequence_col)

    for seq_id, rule_to_sequence_index in rule_info_dict['seq_id'].items():
        print(mask_sequence_col[seq_id])
        print(pos_elem_rule, rule_to_sequence_index)
        for t in range(len(pos_elem_rule)):
            if is_label_item(pos_elem_rule[t]):
                mask_sequence_col[seq_id][rule_to_sequence_index[t]] = [elem]

    return mask_sequence_col


def update_rule_confidence_col(rule_id_confidence_dict_col):
    rule_ids_confidence_col = []
    for ids, info_dict in rule_id_confidence_dict_col.items():
        if len(info_dict['seq_cnt']) == 0:
            rule_ids_confidence_col.append((ids, 0))
        else:
            rule_ids_confidence_col.append((ids, len(info_dict['seq_id']) / len(info_dict['seq_cnt'])))
    rule_ids_confidence_col = sorted(rule_ids_confidence_col, key=lambda x: -x[1])

    return rule_ids_confidence_col


def select_valid_rule(sequence_col, rule_col, elem, frequency_pos_set, radius, min_confidence):
    # generate first support and confidence.
    cur_sequence_col = copy.deepcopy(sequence_col)
    cur_rule_col = copy.deepcopy(rule_col)
    final_rule_col = []
    while len(cur_rule_col) > 0:
        rule_id_confidence_dict, rule_ids_confidence_col = calculate_label_sequential_rule_confidence(
            cur_sequence_col, cur_rule_col, elem, frequency_pos_set, radius
        )
        print(len(cur_rule_col), rule_ids_confidence_col[0][1])
        if rule_ids_confidence_col[0][1] >= min_confidence:
            rule_id = rule_ids_confidence_col[0][0]
            cur_sequence_col = mask_sequence_by_highest_confidence_rule(
                cur_sequence_col, cur_rule_col[rule_id], rule_id_confidence_dict[rule_id], elem
            )
            final_rule_col.append(cur_rule_col[rule_id])
            del cur_rule_col[rule_id]
        else:
            break

    return final_rule_col


def check_sequence_rule(sequence_col, rule_id_confidence_dict_col):
    for index in range(10):
        print(rule_id_confidence_dict_col[index]['rule'])
        for seq_id in rule_id_confidence_dict_col[index]['seq_id']:
            print(sequence_col[seq_id])


def select_valid_rule_slower(sequence_col, rule_col, elem, frequency_pos_set, radius, min_confidence):
    # generate first support and confidence.
    rule_id_confidence_dict_col = calculate_label_sequential_rule_confidence(
        sequence_col, rule_col, elem, frequency_pos_set, radius
    )

    cur_sequence_col = copy.deepcopy(sequence_col)
    cur_rule_id_confidence_dict_col = copy.deepcopy(rule_id_confidence_dict_col)
    final_rule_col = []

    while len(cur_rule_id_confidence_dict_col) > 0:
        pool = Pool(processes=20)
        cur_rule_id_confidence_dict_col = pool.map(
            partial(repeat_calculate_label_sequential_rule_confidence,
                    sequence_col=cur_sequence_col,
                    elem=elem,
                    frequency_pos_set=frequency_pos_set,
                    radius=radius),
            cur_rule_id_confidence_dict_col
        )
        pool.close()
        pool.join()

        cur_rule_id_confidence_dict_col = sorted(cur_rule_id_confidence_dict_col, key=lambda x: -x['confidence'])

        print(len(final_rule_col), cur_rule_id_confidence_dict_col[0]['confidence'])
        if cur_rule_id_confidence_dict_col[0]['confidence'] >= min_confidence:
            cur_sequence_col = mask_sequence_by_highest_confidence_rule(
                cur_sequence_col, cur_rule_id_confidence_dict_col[0], elem
            )
            final_rule_col.append(cur_rule_id_confidence_dict_col[0]['rule'])
            cur_rule_id_confidence_dict_col[0] = \
                {"rule": final_rule_col[-1], "seq_id": {}, "seq_cnt": [], "confidence": 0}

        else:
            break

    return final_rule_col


def drop_low_confidence_rule(id_conf_col, lsr_col, min_confidence):
    final_lsr_col = []
    id_conf_col = sorted(id_conf_col, key=lambda x: -x[1])

    for index in range(len(id_conf_col)):
        if id_conf_col[index][1] < min_confidence:
            break

        final_lsr_col.append(lsr_col[id_conf_col[index][0]])

    return final_lsr_col


def drop_cover_rule(lsr_col):
    final_lsr_col = []
    for index in range(len(lsr_col)):
        appear_flag = False
        for t in range(len(final_lsr_col)):
            if final_lsr_col[t] == lsr_col[index]:
                appear_flag = True

        if appear_flag:
            continue

        final_lsr_col.append(lsr_col[index])

    return final_lsr_col


def combine_interval(position_set):
    if len(position_set) == 0:
        return position_set

    position_col = sorted(list(position_set), key=lambda x: x[0])

    s_index, index = 0, 1
    final_position_col, end_position = [], position_col[0][1]
    while index < len(position_col):
        if position_col[index][0] <= end_position:
            end_position = max(end_position, position_col[index][1])
        else:
            final_position_col.append((position_col[s_index][0], end_position))
            s_index = index
            end_position = position_col[index][1]
        index += 1

    final_position_col.append((position_col[s_index][0], end_position))

    return final_position_col


def train_lsr():
    pass


def get_final_lsr_col(file_type, lsr_radius):
    # file_type, radius = "car", 6
    dir_name = file_type if file_type == "kesserl14" else "coae2013/" + file_type

    standard_path = {"train": "./data/" + dir_name + "/train.txt",
                     "dev": "./data" + dir_name + "dev.txt",
                     "test": "./data/" + dir_name + "/test.txt"}

    # stanford_path = r"D:/stanford-corenlp-full-2018-10-05"
    stanford_path = "/home/zhliu/1080ti/stanford-corenlp-full-2018-02-27"

    lang = "zh" if file_type in {"car", "ele"} else "en"
    keyword_path = "./lsr_data/" + lang + "_keyword_vocab.txt"

    elem_col = ["entity_1", "entity_2", "aspect", "result"]
    # elem_col = ["entity_1"]
    gold_dict, predict_dict, train_gold_dict = [], [], []
    elem_lsr_col, elem_frequency_pos_set = [], []

    for elem in elem_col:
        pre_process_path = {"train": "./lsr_data/" + dir_name + "/" + elem + "/train_data.txt",
                            "test": "./lsr_data/" + dir_name + "/" + elem + "/test_data.txt"}

        database_path = {"train": "./lsr_data/" + dir_name + "/" + elem + "/train_database_" + str(lsr_radius) + ".txt",
                         "tt": "./lsr_data/" + dir_name + "/" + elem + "/test_train_database_" + str(lsr_radius) + ".txt",
                         "test": "./lsr_data/" + dir_name + "/" + elem + "/test_database_" + str(lsr_radius) + ".txt"}

        database, _ = get_init_database(
            database_path['train'],
            standard_path['train'],
            stanford_path,
            pre_process_path['train'],
            keyword_path,
            elem,
            lsr_radius,
            lang=lang
        )

        final_csr_path = "./lsr_data/" + dir_name + "/" + elem + "/final_csr_" + str(lsr_radius) + ".txt"

        if os.path.exists(final_csr_path):
            lsr_col = shared_utils.read_pickle(final_csr_path)
        else:
            csr_path = "./lsr_data/" + dir_name + "/" + elem + "/csr_" + str(lsr_radius) + ".txt"
            if os.path.exists(csr_path):
                lsr_col = shared_utils.read_pickle(csr_path)
            else:
                lsr_col = create_final_csr_col(database, min_support=0.01)
                shared_utils.write_pickle(lsr_col, csr_path)

        lsr_col = drop_without_label_rules(lsr_col, [elem])
        lsr_col = drop_simple_label(lsr_col, [elem])

        train_sequence_col, train_elem_label_col = create_test_database(
            database_path['tt'],
            standard_path['train'],
            stanford_path,
            pre_process_path['train'],
            keyword_path, elem, lang=lang
        )

        for index in range(len(train_elem_label_col)):
            if elem == elem_col[0]:
                train_gold_dict.append(extern_eval.sequence_label_convert_dict(
                    train_elem_label_col[index], {}, elem
                ))
            else:
                train_gold_dict[index] = extern_eval.sequence_label_convert_dict(
                    train_elem_label_col[index], train_gold_dict[index], elem
                )

        frequency_pos_set = count_pos_tag_number(train_sequence_col, train_gold_dict, elem)

        id_conf_path = "./lsr_data/" + dir_name + "/" + elem + "/id_conf_" + str(lsr_radius) + ".txt"

        if os.path.exists(id_conf_path):
            final_lsr_col = shared_utils.read_pickle(id_conf_path)
        else:
            final_lsr_col = select_valid_rule_slower(
                database, lsr_col, elem, frequency_pos_set, lsr_radius, min_confidence=0.9
            )
            shared_utils.write_pickle(final_lsr_col, id_conf_path)

        elem_lsr_col.append(final_lsr_col)
        elem_frequency_pos_set.append(frequency_pos_set)

    return elem_lsr_col, elem_frequency_pos_set


def predict_elem_dict_by_lsr(
        elem_lsr_col, elem_frequency_pos_set, standard_path, file_type, radius, data_type
):
    elem_col = ["entity_1", "entity_2", "aspect", "result"]
    dir_name = file_type if file_type == "kesserl14" else "coae2013/" + file_type

    # stanford_path = r"D:/stanford-corenlp-full-2018-10-05"
    stanford_path = "/home/zhliu/1080ti/stanford-corenlp-full-2018-02-27"

    lang = "zh" if standard_path.find("kesserl14") == -1 else "en"
    keyword_path = "./lsr_data/" + lang + "_keyword_vocab.txt"

    gold_dict, predict_dict = [], []
    for index, elem in enumerate(elem_col):
        database_path = {"train": "./lsr_data/" + dir_name + "/" + elem + "/test_train_database_" + str(radius) + ".txt",
                         "dev": "./lsr_data/" + dir_name + "/" + elem + "/dev_database_" + str(radius) + ".txt",
                         "test": "./lsr_data/" + dir_name + "/" + elem + "/test_database_" + str(radius) + ".txt"}

        pre_process_path = {"train": "./lsr_data/" + dir_name + "/" + elem + "/train_data.txt",
                            "dev": "./lsr_data/" + dir_name + "/" + elem + "/dev_data.txt",
                            "test": "./lsr_data/" + dir_name + "/" + elem + "/test_data.txt"}

        test_sequence_col, test_elem_label_col = create_test_database(
            database_path[data_type],
            standard_path,
            stanford_path,
            pre_process_path[data_type],
            keyword_path, elem, lang=lang, comp_flag=False
        )
        print(data_type)
        print(len(test_sequence_col))

        # for index in range(len(test_elem_label_col)):
        #     if elem == elem_col[0]:
        #         gold_dict.append(extern_eval.sequence_label_convert_dict(
        #             test_elem_label_col[index], {}, elem
        #         ))
        #     else:
        #         gold_dict[index] = extern_eval.sequence_label_convert_dict(
        #             test_elem_label_col[index], gold_dict[index], elem
        #         )

        predict_dict = parse_test_sequence_lsr_col(
            test_sequence_col, elem_lsr_col[index], predict_dict, elem, elem_frequency_pos_set[index], radius
        )

    return predict_dict
#    extern_eval.evaluate_three_measure_to_predict_label(gold_dict, predict_dict, elem_col)




# if __name__ == "__main__":
#     file_type, radius = "car", 6
#     dir_name = file_type if file_type == "kesserl14" else "coae2013/" + file_type
#
#     standard_path = {"train": "./data/" + dir_name + "/train.txt",
#                      "test": "./data/" + dir_name + "/test.txt"}
#
#     # stanford_path = r"D:/stanford-corenlp-full-2018-10-05"
#     stanford_path = "/home/zhliu/1080ti/stanford-corenlp-full-2018-02-27"
#
#     lang = "zh" if file_type in {"car", "ele"} else "en"
#     keyword_path = "./lsr_data/" + lang + "_keyword_vocab.txt"
#
#     # elem_col = ["entity_1", "entity_2", "aspect", "result"]
#     elem_col = ["entity_1"]
#     gold_dict, predict_dict, train_gold_dict = [], [], []
#     for elem in elem_col:
#         pre_process_path = {"train": "./lsr_data/" + dir_name + "/" + elem + "/train_data.txt",
#                             "test": "./lsr_data/" + dir_name + "/" + elem + "/test_data.txt"}
#
#         database_path = {"train": "./lsr_data/" + dir_name + "/" + elem + "/train_database_" + str(radius) + ".txt",
#                          "tt": "./lsr_data/" + dir_name + "/" + elem + "/test_train_database_" + str(radius) + ".txt",
#                          "test": "./lsr_data/" + dir_name + "/" + elem + "/test_database_" + str(radius) + ".txt"}
#
#         database, _ = get_init_database(
#             database_path['train'],
#             standard_path['train'],
#             stanford_path,
#             pre_process_path['train'],
#             keyword_path,
#             elem,
#             lang=lang
#         )
#
#         final_csr_path = "./lsr_data/" + dir_name + "/" + elem + "/final_csr_" + str(radius) + ".txt"
#
#         if os.path.exists(final_csr_path):
#             lsr_col = shared_utils.read_pickle(final_csr_path)
#         else:
#             csr_path = "./lsr_data/" + dir_name + "/" + elem + "/csr_" + str(radius) + ".txt"
#             if os.path.exists(csr_path):
#                 lsr_col = shared_utils.read_pickle(csr_path)
#             else:
#                 lsr_col = create_final_csr_col(database, min_support=0.01)
#                 shared_utils.write_pickle(lsr_col, csr_path)
#
#         lsr_col = drop_without_label_rules(lsr_col, [elem])
#         lsr_col = drop_simple_label(lsr_col, [elem])
#
#         train_sequence_col, train_elem_label_col = create_test_database(
#             database_path['tt'],
#             standard_path['train'],
#             stanford_path,
#             pre_process_path['train'],
#             keyword_path, elem, lang=lang
#         )
#
#         for index in range(len(train_elem_label_col)):
#             if elem == elem_col[0]:
#                 train_gold_dict.append(extern_eval.sequence_label_convert_dict(
#                     train_elem_label_col[index], {}, elem
#                 ))
#             else:
#                 train_gold_dict[index] = extern_eval.sequence_label_convert_dict(
#                     train_elem_label_col[index], train_gold_dict[index], elem
#                 )
#
#         frequency_pos_set = count_pos_tag_number(train_sequence_col, train_gold_dict, elem)
#
#         id_conf_path = "./lsr_data/" + dir_name + "/" + elem + "/id_conf_" + str(radius) + ".txt"
#
#         if os.path.exists(id_conf_path):
#             final_lsr_col = shared_utils.read_pickle(id_conf_path)
#         else:
#             final_lsr_col = select_valid_rule_slower(
#                 database, lsr_col, elem, frequency_pos_set, radius, min_confidence=0.9
#             )
#             shared_utils.write_pickle(final_lsr_col, id_conf_path)
#
#         for index in range(len(final_lsr_col)):
#             print(final_lsr_col[index])
#
#         test_sequence_col, test_elem_label_col = create_test_database(
#             database_path['test'],
#             standard_path['test'],
#             stanford_path,
#             pre_process_path['test'],
#             keyword_path, elem, lang=lang, comp_flag=True
#         )
#
#         for index in range(len(test_elem_label_col)):
#             if elem == elem_col[0]:
#                 gold_dict.append(extern_eval.sequence_label_convert_dict(
#                     test_elem_label_col[index], {}, elem
#                 ))
#             else:
#                 gold_dict[index] = extern_eval.sequence_label_convert_dict(
#                     test_elem_label_col[index], gold_dict[index], elem
#                 )
#
#         predict_dict = parse_test_sequence_lsr_col(
#             test_sequence_col, final_lsr_col, predict_dict, elem, frequency_pos_set, radius
#         )
#
#     extern_eval.evaluate_three_measure_to_predict_label(gold_dict, predict_dict, elem_col)
#
#
