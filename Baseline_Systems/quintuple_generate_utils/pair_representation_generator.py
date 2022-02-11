import torch
import torch.nn as nn
import os, copy
import numpy as np
from tqdm import tqdm
from model_utils import optimizer_utils, train_test_utils
from quintuple_generate_utils.pair_eval import PairEvaluation
from data_utils import shared_utils, data_loader_utils
from data_utils import current_program_code as cpc
from data_utils.label_parse import LabelParser
from model_utils import pipeline_model_utils
from quintuple_generate_utils import pair_eval

elem_col = ["entity_1", "entity_2", "aspect", "result"]


def create_bert_feature_embed(model, sentence_col, bert_tokenizer, device):
    # model_path = r"D:/base_uncased/" if file_type == "kesserl14" else r"D:/base_chinese/"
    # bert_tokenizer = BertTokenizer.from_pretrained(model_path)
    # model = BertModel.from_pretrained(model_path)
    model.eval()
    bert_token = shared_utils.get_token_col(sentence_col, bert_tokenizer=bert_tokenizer, dim=1)
    input_ids = shared_utils.bert_data_transfer(bert_tokenizer, bert_token, data_type='tokens')
    attn_mask = shared_utils.get_mask(input_ids, dim=1)

    max_len = shared_utils.get_max_token_length(input_ids)

    input_ids = shared_utils.padding_data(
        input_ids, max_len=max_len, dim=1, padding_num=0, data_type="norm"
    )

    attn_mask = shared_utils.padding_data(
        attn_mask, max_len=max_len, dim=1, padding_num=0, data_type="norm"
    )

    input_ids = np.array(input_ids)
    attn_mask = np.array(attn_mask)

    data_loader = data_loader_utils.get_loader([input_ids, attn_mask], 16)

    bert_feature_embedding = []
    with torch.no_grad():
        for index, data in tqdm(enumerate(data_loader)):
            cur_input_ids, cur_attn_mask = data

            # qargs = ['index', 'gpu_name', 'memory.free', 'memory.total', 'power.draw', 'power.limit', 'utilization.gpu']
            # cmd = 'nvidia-smi --query-gpu={} --format=csv,noheader'.format(','.join(qargs))
            # results = os.popen(cmd).readlines()
            # print(results)

            cur_input_ids = torch.tensor(cur_input_ids).int().to(device)
            cur_attn_mask = torch.tensor(cur_attn_mask).int().to(device)

            # batch_size, sequence_length, feature_dim
            # drop [CLS] and [SEP]
            token_embedding = model(cur_input_ids, cur_attn_mask)[0][:, 1: , :]
            bert_feature_embedding.append(token_embedding.cpu())

    return bert_feature_embedding


def generate_elem_representation(predict_dict, gold_pair_label, bert_feature_embed):
    """
    :param predict_dict:{elem: [(s_index, e_index)]}
    :param gold_pair_label: [(s_index, e_index)] * 5
    :param feature_embed: [N, 3, sequence_length, feature_dim], feature_dim=5
    :param bert_feature_embed:
    :param feature_type：
    :return:
    """
    candidate_pair_col = []

    # elem_col = {"entity_1", "entity_2", "aspect", "result"}
    for index in range(len(predict_dict)):
        cur_candidate_pair_col = []
        cur_predict_elem_dict = predict_dict[index]

        for elem in elem_col:
            if len(cur_predict_elem_dict[elem]) != 0:
                cur_elem = cur_predict_elem_dict[elem]
            else:
                cur_elem = [(-1, -1)]

            cur_candidate_pair_col = shared_utils.cartesian_product(cur_candidate_pair_col, cur_elem)

        candidate_pair_col.append(cur_candidate_pair_col)

    pair_representation = create_pair_representation(
        candidate_pair_col, bert_feature_embed
    )

    make_pair_label = create_pair_label(candidate_pair_col, gold_pair_label)

    return [candidate_pair_col, pair_representation, make_pair_label, bert_feature_embed]


def create_pair_representation(candidate_col, bert_feature_out):
    """
    :param candidate_col: [n, tuple_pair_num, tuple_pair], tuple_pair: [(s_index, e_index)]
    :param bert_feature_out: [n, sequence_length, feature_dim]
    :return:
    """
    pair_input, hidden_size = [], 5

    for index in range(len(candidate_col)):
        pair_representation = []
        for pair_index in range(len(candidate_col[index])):
            each_pair_representation = []
            for elem_index in range(4):
                s, e = candidate_col[index][pair_index][elem_index]
                if s == -1:
                    each_pair_representation.append(torch.zeros(1, 768).cpu())

                else:
                    each_pair_representation.append(
                        torch.mean(bert_feature_out[index][s: e], dim=0).cpu().view(-1, 768)
                    )

            if torch.cuda.is_available():
                cur_representation = torch.cat(each_pair_representation, dim=-1).view(-1).cpu().numpy().tolist()
            else:
                cur_representation = torch.cat(each_pair_representation, dim=-1).view(-1).numpy().tolist()

            pair_representation.append(cur_representation)

        pair_input.append(pair_representation)

    return pair_input


def is_equal_tuple_pair(candidate_tuple_col, truth_tuple_col, null_pair):
    if truth_tuple_col == null_pair:
        return False

    if len(candidate_tuple_col) != len(truth_tuple_col):
        if candidate_tuple_col == truth_tuple_col[:-1]:
            return True
        return False
    else:
        if candidate_tuple_col == truth_tuple_col:
            return True
        return False


def create_pair_label(candidate_col, truth_pair_label):
    """
    :param candidate_col: shape is [n, tuple_pair_num, tuple_pair]
    :param truth_pair_label: shape is [n, tuple_pair_num, tuple_pair]
    :return:
    """
    pair_label_col, null_pair = [], [(-1, -1)] * 5
    for i in range(len(candidate_col)):
        # cartesian product pair num
        is_pair_label = []
        for j in range(len(candidate_col[i])):
            # truth predicate pair num
            isExist = False
            for k in range(len(truth_pair_label[i])):
                if is_equal_tuple_pair(candidate_col[i][j], truth_pair_label[i][k], null_pair):
                    isExist = True

            is_pair_label.append(1 if isExist else 0)

        pair_label_col.append(is_pair_label)

    return pair_label_col


def get_sent_and_label_by_file(file_path):
    if file_path.find("kesserl14") != -1:
        split_symbol, lang = "&&", "eng"
    else:
        split_symbol, lang = "&", "cn"

    sent_col, sent_label_col, label_col = cpc.read_standard_file(file_path)
    LP = LabelParser(label_col, ["entity_1", "entity_2", "aspect", "result"])

    label_col, tuple_pair_col = LP.parse_sequence_label(split_symbol, sent_col, file_type=lang)

    return sent_col, sent_label_col, label_col, tuple_pair_col


def create_comparative_relation_representation_label(
        predict_dict, cur_standard_path, cur_feature_path, model, bert_tokenizer, device
):
    cur_sent_col, cur_sent_label_col, cur_label_col, cur_tuple_pair_col = get_sent_and_label_by_file(
        cur_standard_path
    )
    if os.path.exists(cur_feature_path):
        cur_bert_feature = shared_utils.read_pickle(cur_feature_path)
    else:
        cur_bert_feature = create_bert_feature_embed(model, cur_sent_col, bert_tokenizer, device)
        shared_utils.write_pickle(cur_bert_feature, cur_feature_path)
    cur_bert_feature_embed = torch.cat(cur_bert_feature, dim=0)
    data_col = generate_elem_representation(predict_dict, cur_tuple_pair_col, cur_bert_feature_embed)
    data_col.append(cur_tuple_pair_col)
    return data_col


def pair_stage_model_train(model, optimizer, train_loader, device, epoch):
    """
    :param model:
    :param optimizer:
    :param train_loader:
    :param device:
    :param epoch:
    :return:
    """
    model.train()
    epoch_loss, t = 0, 0
    for index, data in tqdm(enumerate(train_loader)):
        pair_representation, pair_label = data
        pair_representation = torch.tensor(pair_representation).float().to(device)
        pair_label = torch.tensor(pair_label).long().to(device)

        if torch.equal(pair_representation, torch.zeros_like(pair_representation)):
            continue

        loss = model(pair_representation, pair_label)

        loss = torch.sum(loss)
        epoch_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("epoch is {} and Loss: {:.2f}".format(epoch, epoch_loss))


def pair_stage_model_test(
        model, device, test_loader, res_eval, eval_parameters=None, mode="pair", polarity=False, initialize=(False, False)):
    """
    :param model: the model
    :param test_loader: test data loader: [input_ids, attn_mask, pos_ids, predicate_label]
    :param device:
    :param res_eval: a Evaluation object
    :param eval_parameters:
    :param mode:
    :param polarity:
    :param initialize:
    :return:
    """
    model.eval()
    measure_file, model_path = eval_parameters

    with torch.no_grad():
        for index, data in tqdm(enumerate(test_loader)):
            pair_representation = data

            pair_representation = torch.tensor(pair_representation).float().to(device)

            pair_out = model(pair_representation).view(-1)

            if torch.equal(pair_representation, torch.zeros_like(pair_representation)):
                pair_out = torch.zeros(pair_out.size())

            if mode == "pair":
                res_eval.add_pair_data(pair_out)
            else:
                res_eval.add_polarity_data(pair_out)

    res_eval.eval_model(measure_file, model, model_path, polarity=polarity, initialize=initialize)


def pair_stage_model_main(pair_representation, make_pair_label, pair_eval, polarity_col,
                          model_parameters, optimizer_parameters, model_name, feature_type):
    """
    :param config:
    :param pair_representation:
    :param make_pair_label:
    :param pair_eval:
    :param polarity_col:
    :param model_parameters:
    :param optimizer_parameters:
    :param model_name:
    :param feature_type:
    :return:
    """
    train_pair_representation, dev_pair_representation, test_pair_representation = pair_representation
    train_make_pair_label, dev_make_pair_label, test_make_pair_label = make_pair_label
    dev_pair_eval, test_pair_eval, global_pair_eval = pair_eval
    train_polarity_representation, train_polarity_col = polarity_col
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("finish second model data generate")

    # get pair loader
    train_pair_loader = data_loader_utils.get_loader([train_pair_representation, train_make_pair_label], 16)
    dev_pair_loader = data_loader_utils.get_loader([dev_pair_representation], 1)
    test_pair_loader = data_loader_utils.get_loader([test_pair_representation], 1)

    # get polarity data loader.
    train_polarity_loader = data_loader_utils.get_loader([train_polarity_representation, train_polarity_col], 16)

    pair_weight = torch.tensor([model_parameters['factor'], 1]).float().to(device)

    feature_dim = [4 * (5 + 768), 4 * 5, 4 * 768]
    pair_feature_dim = feature_dim[feature_type]

    # define pair and polarity model.
    pair_model = copy.deepcopy(
        pipeline_model_utils.LogisticClassifier(None, pair_feature_dim, 2, weight=pair_weight).to(device)
    )
    polarity_model = copy.deepcopy(
        pipeline_model_utils.LogisticClassifier(None, pair_feature_dim, 4).to(device)
    )

    # if torch.cuda.device_count() > 1:
    #     pair_model = nn.DataParallel(pair_model)
    #     polarity_model = nn.DataParallel(polarity_model)
    #     pair_optimizer = optimizer_utils.Logistic_Optim(pair_model.module, optimizer_parameters)
    #     polarity_optimizer = optimizer_utils.Logistic_Optim(polarity_model.module, optimizer_parameters)
    # else:
    pair_optimizer = optimizer_utils.Logistic_Optim(pair_model, optimizer_parameters)
    polarity_optimizer = optimizer_utils.Logistic_Optim(polarity_model, optimizer_parameters)

    dev_pair_parameters = ["./ModelResult/" + model_name + "/dev_pair_result.txt",
                           "./PreTrainModel/" + model_name + "/dev_pair_model"]

    dev_polarity_parameters = ["./ModelResult/" + model_name + "/dev_polarity_result.txt",
                               "./PreTrainModel/" + model_name + "/dev_polarity_model"]

    for epoch in range(50):
        pair_stage_model_train(pair_model, pair_optimizer, train_pair_loader, device, epoch)
        pair_stage_model_test(
            pair_model, device, dev_pair_loader, dev_pair_eval,
            dev_pair_parameters, mode="pair", polarity=False, initialize=(False, True)
        )

    # get optimize pair model.
    predict_pair_model = torch.load(dev_pair_parameters[1])
    test_pair_parameters = ["./ModelResult/" + model_name + "/test_pair_result.txt", None]
    pair_stage_model_test(
        predict_pair_model, device, dev_pair_loader, dev_pair_eval,
        test_pair_parameters, mode="pair", polarity=False, initialize=(False, False)
    )

    # get representation by is_pair label filter.
    dev_polarity_representation = cpc.get_after_pair_representation(dev_pair_eval.y_hat, dev_pair_representation)
    dev_polarity_loader = data_loader_utils.get_loader([dev_polarity_representation], 1)

    shared_utils.clear_optimize_measure(dev_pair_eval)

    for epoch in range(50):
        pair_stage_model_train(polarity_model, polarity_optimizer, train_polarity_loader, device, epoch)
        pair_stage_model_test(
            polarity_model, device, dev_polarity_loader, dev_pair_eval,
            dev_polarity_parameters, mode="polarity", polarity=True, initialize=(True, False)
        )

    print("==================test================")
    predict_pair_model = torch.load(dev_pair_parameters[1])
    predict_polarity_model = torch.load(dev_polarity_parameters[1])

    test_pair_parameters = ["./ModelResult/" + model_name + "/test_pair_result.txt", None]
    test_polarity_parameters = ["./ModelResult/" + model_name + "/test_pair_result.txt", None]

    pair_stage_model_test(
        predict_pair_model, device, test_pair_loader, test_pair_eval,
        test_pair_parameters, mode="pair", polarity=False, initialize=(False, False)
    )

    shared_utils.calculate_average_measure(test_pair_eval, global_pair_eval)
    global_pair_eval.avg_model("./ModelResult/" + model_name + "/test_pair_result.txt")
    global_pair_eval.store_result_to_csv([model_name], "result.csv")

    shared_utils.clear_global_measure(global_pair_eval)
    shared_utils.clear_optimize_measure(test_pair_eval)

    # create polarity representation and data loader.
    test_polarity_representation = cpc.get_after_pair_representation(test_pair_eval.y_hat, test_pair_representation)
    test_polarity_loader = data_loader_utils.get_loader([test_polarity_representation], 1)

    pair_stage_model_test(
        predict_polarity_model, device, test_polarity_loader, test_pair_eval,
        test_polarity_parameters, mode="polarity", polarity=True, initialize=(True, True)
    )

    # add average measure.
    shared_utils.calculate_average_measure(test_pair_eval, global_pair_eval)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.nn.Module.dump_patches = True


def pair_representation_generator_main(
        standard_path, feature_embed_path, predict_elem_dict_col, model, bert_tokenizer, device, model_name, model_parameters
):
    set_seed(2021)
    norm_id_map = shared_utils.create_tag_mapping_ids([], "BMES", other_flag=True)
    train_predict_elem_dict, dev_predict_elem_dict, test_predict_elem_dict = predict_elem_dict_col

    _, train_pair_representation, train_make_pair_label, train_bert_feature_out, train_tuple_pair_col = \
        create_comparative_relation_representation_label(
            train_predict_elem_dict, standard_path['train'], feature_embed_path['train'], model, bert_tokenizer, device
        )

    dev_candidate_pair_col, dev_pair_representation, dev_make_pair_label, dev_bert_feature_out, dev_tuple_pair_col = \
        create_comparative_relation_representation_label(
            dev_predict_elem_dict, standard_path['dev'], feature_embed_path['dev'], model, bert_tokenizer, device
        )

    test_candidate_pair_col, test_pair_representation, test_make_pair_label, test_bert_feature_out, test_tuple_pair_col = \
        create_comparative_relation_representation_label(
            test_predict_elem_dict, standard_path['test'], feature_embed_path['test'], model, bert_tokenizer, device
        )

    train_pair_representation, train_make_pair_label = cpc.generate_train_pair_data(
        train_pair_representation, train_make_pair_label
    )

    train_polarity_representation, train_polarity_label = cpc.create_polarity_train_data(
        train_tuple_pair_col, train_bert_feature_out,
        train_bert_feature_out, feature_type=2
    )

    pair_representation = [train_pair_representation, dev_pair_representation, test_pair_representation]
    make_pair_label = [train_make_pair_label, dev_make_pair_label, test_make_pair_label]

    dev_pair_eval = PairEvaluation(
        gold_pair_col=dev_tuple_pair_col,
        candidate_pair_col=dev_candidate_pair_col,
        elem_col=elem_col,
        ids_to_tags=norm_id_map,
        save_model=True
    )

    test_pair_eval = PairEvaluation(
        gold_pair_col=test_tuple_pair_col,
        candidate_pair_col=test_candidate_pair_col,
        elem_col=elem_col,
        ids_to_tags=norm_id_map,
        save_model=False
    )

    import json

    global_pair_eval = pair_eval.BaseEvaluation()

    optimizer_parameters = None

    pair_stage_model_main(
        pair_representation, make_pair_label,
        [dev_pair_eval, test_pair_eval, global_pair_eval],
        [train_polarity_representation, train_polarity_label],
        model_parameters, optimizer_parameters, model_name, feature_type=2
    )

