from data_utils import shared_utils


def sequence_label_convert_dict(sequence_label, pre_elem_dict, cur_tag):
    """
    :param sequence_label:
    :param pre_elem_dict:
    :param cur_tag:
    :return:
    """
    if cur_tag not in pre_elem_dict:
        pre_elem_dict[cur_tag] = []

    s_index = -1
    for index in range(len(sequence_label)):
        cur_position, cur_emotion = shared_utils.get_label_pos_tag(sequence_label[index])

        # "O" denote "Others", skip this token
        if cur_position == "O":
            continue

        # "S" means alone element
        if cur_position == "S":
            s_index, pre_tag = -1, ""
            pre_elem_dict[cur_tag].append((index, index + 1))

        elif cur_position == "B":
            s_index = index

        elif cur_position == "E" and s_index != -1:
            pre_elem_dict[cur_tag].append((s_index, index + 1))
            s_index = -1

    return pre_elem_dict


def get_predict_label(file_path, file_type="predict"):
    token_col, label_col = [], []
    sequence_token, sequence_label = [], []

    encode = "utf-8" if file_type == "gold" else "utf-16"
    with open(file_path, "r", encoding=encode, errors="ignore") as f:
        for line in f.readlines():
            if line == "\n":
                if len(sequence_label) != 0:
                    label_col.append(sequence_label)

                    if file_type == "gold":
                        token_col.append(sequence_token)

                sequence_token, sequence_label = [], []
                continue

            data_col = shared_utils.split_string(line.rstrip('\n'), "\t")

            if file_type == "gold":
                sequence_token.append(data_col[0])

            sequence_label.append(data_col[-1])

    return token_col, label_col


def init_elem(key_col=None):
    return {elem: 0.0 for elem in key_col}

def get_elem_num(gold_elem_dict, predict_elem_dict):
    """
    :param gold_elem_dict:
    :param predict_elem_dict:
    :return:
    """
    exact_num = get_exact_num(gold_elem_dict, predict_elem_dict)
    prop_num = get_cover_num(gold_elem_dict, predict_elem_dict, "prop")
    binary_num = get_cover_num(gold_elem_dict, predict_elem_dict, "binary")

    return exact_num, prop_num, binary_num


def get_exact_num(gold_col, predict_col):
    """
    :param gold_col: [(s_index, e_index)]
    :param predict_col: [(s_index, e_index)]
    :param elem_type:
    :return:
    """
    correct_num = 0.0

    result_polarity_correct_num, vis, polarity_correct_num_col = 0.0, set(), [0] * 4

    if len(predict_col) == 0:
        return [correct_num, result_polarity_correct_num, polarity_correct_num_col]
    for pi in range(len(predict_col)):
        for gi in range(len(gold_col)):
            if gi in vis:
                continue

            if is_position_tuple_equal(predict_col[pi], gold_col[gi]):
                correct_num = correct_num + 1
                vis.add(gi)

                # calculate result element with polarity
                if len(predict_col[pi]) == 3 and predict_col[pi] == gold_col[gi]:
                    result_polarity_correct_num += 1
                    polarity_correct_num_col[predict_col[pi][2] + 1] += 1

                break

    return [correct_num, result_polarity_correct_num, polarity_correct_num_col]


def get_cover_num(gold_col, predict_col, measure_type="binary"):
    """
    :param gold_col: [(s_index, e_index)]
    :param predict_col: [(s_index, e_index)]
    :param measure_type: "prop" or "binary"
    :return: correct extract elements.
    """
    correct_num = 0.0
    result_polarity_correct_num, vis, polarity_correct_num_col = 0.0, set(), [0] * 4

    if len(predict_col) == 0:
        return [correct_num, result_polarity_correct_num, polarity_correct_num_col]

    for pi in range(len(predict_col)):
        for gi in range(len(gold_col)):
            # skip used gold index
            if gi in vis:
                continue

            cover_prop = shared_utils.cover_rate(gold_col[gi], predict_col[pi], proportion=True)

            if cover_prop > 0 and measure_type == "binary":
                vis.add(gi)
                correct_num += 1

                # calculate result element with polarity
                if len(predict_col[pi]) == 3 and predict_col[pi][2] == gold_col[gi][2]:
                    result_polarity_correct_num += 1
                    polarity_correct_num_col[predict_col[pi][2] + 1] += 1
                break

            elif cover_prop > 0 and measure_type == "prop":
                vis.add(gi)
                correct_num += cover_prop

                # calculate result element with polarity
                if len(predict_col[pi]) == 3 and predict_col[pi][2] == gold_col[gi][2]:
                    result_polarity_correct_num += cover_prop
                    polarity_correct_num_col[predict_col[pi][2] + 1] += 1
                break

    return [correct_num, result_polarity_correct_num, polarity_correct_num_col]


def is_position_tuple_equal(a, b):
    """
    :param a:
    :param b:
    :return:
    """
    if a[0] == b[0] and a[1] == b[1]:
        return True
    return False


def get_f_score(gold_num, predict_num, correct_num, elem_col, multi_elem_score=False):
    """
    :param gold_num: {elem: number......}
    :param predict_num: {elem: number....}
    :param correct_num: {elem: number.....}
    :param multi_elem_score: True denote calculate macro and micro, False denote don't calculate.
    :return: {elem: {P:num, R:num, F:num}.....}
    """
    result_dict = {}

    # maybe elem_col and "pair"
    for elem in predict_num.keys():
        try:
            precision = correct_num[elem] / float(predict_num[elem]) * 100
        except ZeroDivisionError:
            precision = 0.0

        try:
            recall = correct_num[elem] / float(gold_num[elem]) * 100
        except ZeroDivisionError:
            recall = 0.0

        try:
            f_score = 2 * precision * recall / (precision + recall)
        except ZeroDivisionError:
            f_score = 0.0

        result_dict[elem] = {"P": precision, "R": recall, "F": f_score}

    if not multi_elem_score:
        return result_dict

    base_elem_col = ["entity_1", "entity_2", "aspect", "result"]

    result_dict = get_macro_measure(result_dict, base_elem_col, elem_col, elem_name="macro")
    result_dict = get_micro_measure(
        result_dict, gold_num, predict_num, correct_num, base_elem_col, elem_name="micro"
    )

    return result_dict


def get_macro_measure(result_dict, multi_key_col, elem_col, elem_name="macro"):
    """
    :param result_dict:
    :param multi_key_col:
    :param elem_name:
    :return:
    """
    # calculate macro-average F-Measure
    macro_precision, macro_recall, macro_f_score = 0.0, 0.0, 0.0

    for key in multi_key_col:
        macro_precision += result_dict[key]['P']
        macro_recall += result_dict[key]['R']
    macro_precision = macro_precision / float(len(elem_col))
    macro_recall = macro_recall / float(len(elem_col))

    try:
        macro_f_score = 2 * macro_precision * macro_recall / (macro_precision + macro_recall)
    except ZeroDivisionError:
        macro_f_score = 0.0

    result_dict[elem_name] = {"P": macro_precision, "R": macro_recall, "F": macro_f_score}

    return result_dict


def get_micro_measure(result_dict, gold_num, predict_num, correct_num, multi_key_col, elem_name="micro"):
    """
    :param result_dict:
    :param gold_num:
    :param predict_num:
    :param correct_num:
    :param multi_key_col:
    :param elem_name:
    :return:
    """
    # calculate micro-average F-Measure
    micro_gold_num, micro_predict_num, micro_correct_num = 0.0, 0.0, 0.0

    for elem in multi_key_col:
        micro_gold_num += gold_num[elem]
        micro_predict_num += predict_num[elem]
        micro_correct_num += correct_num[elem]

    try:
        micro_precision = micro_correct_num / float(micro_predict_num) * 100
    except ZeroDivisionError:
        micro_precision = 0.0

    try:
        micro_recall = micro_correct_num / float(micro_gold_num) * 100
    except ZeroDivisionError:
        micro_recall = 0.0

    try:
        micro_f_score = 2 * micro_precision * micro_recall / (micro_precision + micro_recall)
    except ZeroDivisionError:
        micro_f_score = 0.0

    result_dict[elem_name] = {"P": micro_precision, "R": micro_recall, "F": micro_f_score}

    return result_dict


def get_sentence_label(data_dict):
    null_data_dict = {'entity_1': [], 'entity_2': [], 'aspect': [], 'result': []}

    sentence_label = [0] * len(data_dict)
    for index in range(len(data_dict)):
        if data_dict[index] != null_data_dict:
            sentence_label[index] = 1

    return sentence_label


def evaluate_three_measure_to_predict_label(gold_dict, predict_dict, elem_col):
    """
    :param gold_dict: {elem: [(s_index, e_index)]}
    :param predict_dict: {elem: [(s_index, e_index)]}
    :return:
    """
    print(len(gold_dict), len(predict_dict))

    gold_sent_label = get_sentence_label(gold_dict)
    predict_sent_label = get_sentence_label(predict_dict)

    accuracy_num = 0
    for index in range(len(gold_sent_label)):
        if gold_sent_label[index] == predict_sent_label[index]:
            accuracy_num += 1
    print("Comparative Sentence Accuracy is {}%".format(accuracy_num / len(gold_sent_label) * 100))

    gold_num = init_elem(elem_col)
    predict_num = init_elem(elem_col)

    exact_correct_num, prop_correct_num, binary_correct_num = \
        init_elem(elem_col), init_elem(elem_col), init_elem(elem_col)

    for index in range(len(gold_dict)):
        # sequence elem dict: {elem: {s_index: length}}
        gold_sequence_elem_dict = gold_dict[index]
        predict_sequence_elem_dict = predict_dict[index]

        # print(gold_sequence_elem_dict)
        # print(predict_sequence_elem_dict)
        for elem in elem_col:
            gold_num[elem] += len(gold_sequence_elem_dict[elem])
            predict_num[elem] += len(predict_sequence_elem_dict[elem])

            cur_exact_num, cur_prop_num, cur_binary_num = get_elem_num(
                gold_sequence_elem_dict[elem], predict_sequence_elem_dict[elem]
            )

            exact_correct_num[elem] += cur_exact_num[0]
            prop_correct_num[elem] += cur_prop_num[0]
            binary_correct_num[elem] += cur_binary_num[0]

    print(gold_num)
    print(predict_num)
    print(exact_correct_num)
    print(prop_correct_num)
    print(binary_correct_num)
    exact_measure = get_f_score(gold_num, predict_num, exact_correct_num, elem_col, True)
    prop_measure = get_f_score(gold_num, predict_num, prop_correct_num, elem_col, True)
    binary_measure = get_f_score(gold_num, predict_num, binary_correct_num, elem_col, True)

    print(gold_num, predict_num)
    print(exact_measure)
    print(prop_measure)
    print(binary_measure)
    print("----------------------------------")


# if __name__ == "__main__":
#     file_type = "car"
#     elem_col = ["entity_1", "entity_2", "aspect", "result"]
#     gold_dict, predict_dict = [], []
#     for elem in elem_col:
#         test_data_path = "../crfpp_data/" + file_type + "/" + elem + "_test.txt"
#         output_data_path = "../crf_result/" + file_type + "/" + elem + "_output.txt"
#
#         token_col, gold_label = get_predict_label(test_data_path, "gold")
#         _, predict_label = get_predict_label(output_data_path, "predict")
#         print(len(gold_label), len(predict_label))
#
#         for index in range(len(gold_label)):
#             if elem == "entity_1":
#                 gold_dict.append(sequence_label_convert_dict(gold_label[index], {}, elem))
#                 predict_dict.append(sequence_label_convert_dict(predict_label[index], {}, elem))
#             else:
#                 gold_dict[index] = sequence_label_convert_dict(gold_label[index], gold_dict[index], elem)
#                 predict_dict[index] = sequence_label_convert_dict(predict_label[index], predict_dict[index], elem)
