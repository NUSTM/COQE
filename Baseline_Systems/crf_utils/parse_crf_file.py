from crf_utils import evaluate_test_file as eval


def parse_crf_model_output_file(file_type, feature_type="enhanced", data_set_type="test"):
    """
    :param file_type:
    :param feature_type:
    :param data_set_type:
    :return:
    """
    assert file_type in {"car", "ele", "kesserl14"}, "[ERROR] file type error."
    assert feature_type in {"standard", "enhanced"}, "[ERROR] feature type error."

    # get predict label.
    predict_dict, elem_col = [], ["entity_1", "entity_2", "aspect", "result"]
    for elem in elem_col:
        ground_truth_data_path = "./crfpp_data/" + file_type + "/" + elem + "_" + data_set_type + ".txt"
        output_data_path = "./crf_result/" + file_type + "/" + feature_type + "/" + data_set_type + "_" + elem + "_output.txt"

        token_col, gold_label = eval.get_predict_label(ground_truth_data_path, "gold")
        _, predict_label = eval.get_predict_label(output_data_path, "predict")

        # convert token level label to char level label.
        if file_type != "kesserl14":
            for index in range(len(token_col)):
                predict_label[index] = eval.token_label_convert_to_char_label(token_col[index], predict_label[index])

        for index in range(len(predict_label)):
            if elem == "entity_1":
                predict_dict.append(eval.sequence_label_convert_dict(predict_label[index], {}, elem))
            else:
                predict_dict[index] = eval.sequence_label_convert_dict(predict_label[index], predict_dict[index], elem)

    # predict_sent_label = eval.get_sentence_label(predict_dict)

    return predict_dict


def evaluate_element_performance(gold_dict, predict_dict):
    """
    :param gold_dict:
    :param predict_dict:
    :return:
    """
    elem_col = ["entity_1", "entity_2", "aspect", "result"]
    gold_num = eval.init_elem(elem_col)
    predict_num = eval.init_elem(elem_col)

    exact_correct_num, prop_correct_num, binary_correct_num = \
        eval.init_elem(elem_col), eval.init_elem(elem_col), eval.init_elem(elem_col)

    for index in range(len(gold_dict)):
        # sequence elem dict: {elem: {s_index: length}}
        gold_sequence_elem_dict = gold_dict[index]
        predict_sequence_elem_dict = predict_dict[index]

        # print(gold_sequence_elem_dict)
        # print(predict_sequence_elem_dict)
        for elem in elem_col:
            gold_num[elem] += len(gold_sequence_elem_dict[elem])
            predict_num[elem] += len(predict_sequence_elem_dict[elem])

            cur_exact_num, cur_prop_num, cur_binary_num = eval.get_elem_num(
                gold_sequence_elem_dict[elem], predict_sequence_elem_dict[elem]
            )

            exact_correct_num[elem] += cur_exact_num[0]
            prop_correct_num[elem] += cur_prop_num[0]
            binary_correct_num[elem] += cur_binary_num[0]

    exact_measure = eval.get_f_score(gold_num, predict_num, exact_correct_num, True)
    prop_measure = eval.get_f_score(gold_num, predict_num, prop_correct_num, True)
    binary_measure = eval.get_f_score(gold_num, predict_num, binary_correct_num, True)

    print(gold_num, predict_num)
    print(exact_measure)
    print(prop_measure)
    print(binary_measure)
    print("----------------------------------")

