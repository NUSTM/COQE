import os
import copy
import numpy as np
from data_utils.label_parse import LabelParser
from data_utils import current_program_code as cpc
from data_utils import shared_utils, check_utils
from crf_utils import create_feature_file, create_template_file
from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.mllib.fpm import PrefixSpan
from sklearn.metrics import accuracy_score
comparative_dict = {"JJR", "JJS", "RBR", "RBS"}


def drop_repeat_vocab(vocab_path):
    vocab_set = set()
    with open(vocab_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            if line == "\n":
                continue

            vocab_set.add(line.rstrip("\n"))

    write_str = "\n".join([x for x in vocab_set])

    with open(vocab_path, "w", encoding="utf-8") as f:
        f.write(write_str)


def get_feature_data(sentence_col, stanford_path, pre_process_path, keyword_path, lang="en"):
    if os.path.exists(pre_process_path):
        token_col, pos_col = shared_utils.read_pickle(pre_process_path)

    else:
        sf = create_feature_file.stanfordFeature(sentence_col, stanford_path, lang)
        token_col = sf.get_tokenizer()
        pos_col = sf.get_pos_feature()

        data_col = [token_col, pos_col]
        shared_utils.write_pickle(data_col, pre_process_path)

    return token_col, pos_col


def read_keyword_vocab(vocab_path):
    keyword_vocab = []
    with open(vocab_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            if line == "\n":
                continue

            keyword_vocab.append(shared_utils.split_string(line.strip('\n'), " "))

    return keyword_vocab


# 根据单个序列上单个keyword得到CSR的初始序列
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


# 得到最初的序列数据
def create_init_dataset(token_col, pos_col, sent_label_col, keyword_vocab, radius=3):
    rule_dict, label_col = [], []
    for index in range(len(token_col)):
        cur_rule_col = []
        for keyword in keyword_vocab:
            pos_tag = True if keyword[0] in comparative_dict else False

            cur_kw_rule_col = get_all_csr_by_keyword(
                token_col[index], pos_col[index], keyword, radius, pos_tag
            )

            cur_rule_col.extend(cur_kw_rule_col)

        label_col.append(sent_label_col[index])
        rule_dict.append(cur_rule_col)

    return rule_dict, label_col


def create_init_database(data_set, label_col):
    database, y = [], []
    for index in range(len(data_set)):
        for t in range(len(data_set[index])):
            database.append(data_set[index][t])
            y.append(label_col[index])

    return database, y


def sequence_is_satisfy_csr(sequence_data, csr):
    for index in range(len(sequence_data)):
        if sequence_data[index] == csr[0][0]:
            t, k = index, 0

            while t < len(sequence_data) and k < len(csr):
                if sequence_data[t] == csr[k][0]:
                    k += 1
                t += 1

            if k == len(csr):
                return True

    return False


def create_classification_feature(csr_col, data_set):
    feature_col = []
    cnt = 0
    for index in range(len(data_set)):
        cur_feature = copy.deepcopy([0] * len(csr_col))

        for seq_id in range(len(data_set[index])):
            for t in range(len(csr_col)):
                if sequence_is_satisfy_csr(data_set[index][seq_id], csr_col[t]):
                    cur_feature[t] = 1
        if cur_feature != [0] * len(csr_col):
            cnt += 1
        feature_col.append(cur_feature)
    return feature_col


def create_sequence_database_label(
        data_path, stanford_path, pre_process_path, keyword_path, init_dataset_path, lang="en", radius=3
):
    sent_col, sent_label_col, label_col = cpc.read_standard_file(data_path)
    LP = LabelParser(label_col, ["entity_1", "entity_2", "aspect", "result"])
    label_col, _ = LP.parse_sequence_label("&", sent_col)

    token_col, pos_col = get_feature_data(
        sent_col, stanford_path, pre_process_path, keyword_path, lang
    )

    keyword_vocab = read_keyword_vocab(keyword_path)

    # 读取初始序列数据库
    if not os.path.exists(init_dataset_path):
        data_set, sequence_label_col = create_init_dataset(token_col, pos_col, sent_label_col, keyword_vocab, radius)
        shared_utils.write_pickle([data_set, sequence_label_col], init_dataset_path)
    else:
        data_set, sequence_label_col = shared_utils.read_pickle(init_dataset_path)

    return data_set, sequence_label_col


def create_mis_csr_col(data_set, sequence_label_col):
    database, y = create_init_database(data_set, sequence_label_col)

    for index in range(len(database)):
        database[index] = [[x if x not in {",", ".", ":"} else "PU"] for x in database[index]]

    final_csr_col = final_csrs_utils.multi_minimum_support_prefix_span(
        database, y, min_confidence=0.7, class_num=1
    )

    return final_csr_col


def create_spark_csr_col(data_set, sequence_label_col, min_support):
    database, y = create_init_database(data_set, sequence_label_col)

    for index in range(len(database)):
        database[index] = [[x if x not in {",", ".", ":"} else "PU"] for x in database[index]]

    sc = SparkContext("local", "testing")
    rdd = sc.parallelize(database, 2)
    model = PrefixSpan.train(rdd, minSupport=min_support)
    csr_col = model.freqSequences().collect()
    final_csr_col = [x.sequence for x in csr_col]

    return final_csr_col


def train_classification_model(config):
    """
    :param config:
    :return:
    """
    lang = "zh" if config.file_type in {"car", "ele"} else "en"

    # create train, dev and test data.
    train_data_set, train_label_col = create_sequence_database_label(
        config.path.standard_path['train'],
        config.path.stanford_path,
        config.path.pre_process_path['train'],
        config.path.keyword_path,
        config.path.database_path['train'],
        lang,
        config.csr_radius
    )

    dev_data_set, dev_label_col = create_sequence_database_label(
        config.path.standard_path['dev'],
        config.path.stanford_path,
        config.path.pre_process_path['dev'],
        config.path.keyword_path,
        config.path.database_path['dev'],
        lang,
        config.csr_radius
    )

    test_data_set, test_label_col = create_sequence_database_label(
        config.path.standard_path['test'],
        config.path.stanford_path,
        config.path.pre_process_path['test'],
        config.path.keyword_path,
        config.path.database_path['test'],
        lang,
        config.csr_radius
    )

    # create csr.
    if os.path.exists(config.path.csr_path):
        csr_col = shared_utils.read_pickle(config.path.csr_path)
    else:
        if config.generate_method == "mis_csr":
            csr_col = create_mis_csr_col(train_data_set, train_label_col)
        else:
            csr_col = create_spark_csr_col(train_data_set, train_label_col, min_support=0.1)
        shared_utils.write_pickle(csr_col, config.path.csr_path)

    # create classification feature.
    train_feature_col = np.array(create_classification_feature(csr_col, train_data_set))
    train_label_col = np.array(copy.deepcopy(train_label_col))

    dev_feature_col = np.array(create_classification_feature(csr_col, dev_data_set))
    dev_label_col = np.array(copy.deepcopy(dev_label_col))

    # train and test by train dataset and dev dataset.
    if os.path.exists(config.path.csr_model_path):
        model = shared_utils.read_pickle(config.path.csr_model_path)
    else:
        model = shared_utils.train_commonly_used_classification_model(
            config.classification_model_type, train_feature_col, train_label_col, dev_feature_col, dev_label_col
        )
        shared_utils.write_pickle(model, config.classification_model_type)

    # test part.
    test_feature_col = np.array(create_classification_feature(csr_col, test_data_set))
    test_label_col = np.array(copy.deepcopy(test_label_col))
    y_hat = model.predict(test_feature_col)
    accuracy = accuracy_score(test_label_col, y_hat)

    print("comparative sentence accuracy is {:.2f}%".format(accuracy * 100))

    return y_hat

