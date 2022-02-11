import os
import torch
import argparse
import Config

from data_utils.label_parse import LabelParser
from data_utils import current_program_code as cpc
from transformers import BertModel, BertTokenizer
from sklearn.metrics import accuracy_score

from csr_utils import mis_csrs_utils
from crf_utils import parse_crf_file
from crf_utils import evaluate_test_file as eval
from data_generator_utils import pair_representation_generator as prg


# 返回当前sentences对应的predicate label.
def comparative_sentence_identification(file_type, radius, generate_method="mis_csr", model_type="svm"):
    predict_sentence_label = mis_csrs_utils.train_classification_model(
        file_type, radius, generate_method=generate_method, model_type=model_type
    )

    return predict_sentence_label


# 返回当前sentence_col对应的elem_dict
def comparative_element_extraction(file_type, feature_type="enhanced", extraction_type="crf", data_set_type="test"):
    if extraction_type == "crf":
        return parse_crf_file.parse_crf_model_output_file(file_type, feature_type, data_set_type)


def comparative_valid_elem_dict(comparative_sentence_label, comparative_element_dict_col):
    assert len(comparative_sentence_label) == len(comparative_element_dict_col), "[ERROR] data error."

    null_elem_dict = {'entity_1': [], 'entity_2': [], 'aspect': [], 'result': []}
    for index in range(len(comparative_sentence_label)):
        if comparative_sentence_label[index] == 0:
            comparative_element_dict_col[index] = null_elem_dict

    return comparative_element_dict_col


def parse_by_label_col(label_col):
    final_ground_truth = []
    for index in range(len(label_col)):
        cur_elem_dict = {'entity_1': [], 'entity_2': [], 'aspect': [], 'result': []}
        for elem, elem_set in label_col[index].items():
            cur_elem_dict[elem] = [(x[0], x[1]) for x in elem_set]
        final_ground_truth.append(cur_elem_dict)

    return final_ground_truth


def get_sent_and_label_by_file(file_path):
    if file_path.find("kesserl14") != -1:
        split_symbol, lang = "&&", "eng"
    else:
        split_symbol, lang = "&", "cn"

    sent_col, sent_label_col, label_col = cpc.read_standard_file(file_path)
    LP = LabelParser(label_col, ["entity_1", "entity_2", "aspect", "result"])

    label_col, tuple_pair_col = LP.parse_sequence_label(split_symbol, sent_col, file_type=lang)

    return sent_col, sent_label_col, label_col, tuple_pair_col


def parameters_to_model_name(model_param):
    """
    :param model_param
    :return:
    """
    result_file, model_file = "./ModelResult/", "./PreTrainModel/"

    model_name = ""

    if not os.path.exists(os.path.join(result_file, model_name)):
        os.mkdir(os.path.join(result_file, model_name))
    if not os.path.exists(os.path.join(model_file, model_name)):
        os.mkdir(os.path.join(model_file, model_name))

    model_name = ""
    if model_param is not None:
        model_param_col = []
        for index, (key, value) in enumerate(model_param.items()):
            if isinstance(value, float) or isinstance(value, int):
                value = str(int(value * 10))

            model_param_col.append(key[:4] + "_" + value)

        model_name += "_".join(model_param_col)

    result_file, model_file = os.path.join(result_file, model_name), os.path.join(model_file, model_name)

    if not os.path.exists(result_file):
        os.mkdir(result_file)
    if not os.path.exists(model_file):
        os.mkdir(model_file)

    return model_name


def TerminalParser():
    # define parse parameters
    parser = argparse.ArgumentParser()
    parser.description = 'choose train data and test data file path'

    # program mode choose.
    parser.add_argument('--crf_type', help='standard or enhanced', default='standard')
    parser.add_argument('--csr_radius', help='the radius of csr', default=4)
    parser.add_argument('--generate_method', help='the type of csr', default="mis_csr")
    parser.add_argument('--classification_model_type', help='the type of model', default="svm")

    parser.add_argument('--device', help='run program in device type',
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--file_type', help='the type of data set', default='car')
    parser.add_argument('--premodel_path', help='the file store pretrain model', default="/home/gj/pretrain_model/")

    args = parser.parse_args()

    return args


def main():
    args = TerminalParser()
    config = Config.BaseConfig(args)

    # print("finish comparative sentence identification")

    # feature enhanced crf part.
    test_predict_elem_dict = comparative_element_extraction(
        args.file_type, feature_type=args.crf_type, extraction_type="crf", data_set_type="test"
    )

    # comparative element extraction
    if args.crf_type == "enhanced" or args.file_type == "kesserl14":
        predict_sent_label = eval.get_sentence_label(test_predict_elem_dict)
    else:
        predict_sent_label = comparative_sentence_identification(
            args.file_type, radius=args.csr_radius, generate_method="mis_csr", model_type="svm"
        )

    dev_predict_elem_dict = comparative_element_extraction(
        args.file_type, feature_type=args.crf_type, extraction_type="crf", data_set_type="dev"
    )

    train_predict_elem_dict = comparative_element_extraction(
        args.file_type, feature_type=args.crf_type, extraction_type="crf", data_set_type="train"
    )

    # get element ground truth.
    sent_col, sent_label_col, label_col, tuple_pair_col = get_sent_and_label_by_file(
        config.path.standard_path['test']
    )

    gold_dict = parse_by_label_col(label_col)

    accuracy = accuracy_score(sent_label_col, predict_sent_label)
    print(accuracy * 100)

    # evaluate comparative element extraction.
    parse_crf_file.evaluate_element_performance(gold_dict, test_predict_elem_dict)

    # pairing and filtering configure table.
    main_dir, device = args.premodel_path, "cuda" if torch.cuda.is_available() else "cpu"

    model_path = main_dir + "base_uncased/" if args.file_type == "kesserl14" else main_dir + "base_chinese/"

    model = BertModel.from_pretrained(model_path).to(device)

    predict_elem_dict_col = [train_predict_elem_dict, dev_predict_elem_dict, test_predict_elem_dict]

    model_parameters = {"file": args.file_type, "type": args.crf_type, "factor": 0.1}
    model_name = parameters_to_model_name(model_parameters)
    prg.pair_representation_generator_main(
        config.path.standard_path,
        config.path.feature_embed_path,
        predict_elem_dict_col,
        model,
        config.bert_tokenizer,
        config.device,
        model_name,
        model_parameters
    )


if __name__ == "__main__":
    main()
