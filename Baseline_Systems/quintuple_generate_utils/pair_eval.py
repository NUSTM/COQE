import torch, copy, csv
from data_utils import shared_utils


class PairEvaluation(object):
    def __init__(self, candidate_pair_col, gold_pair_col, elem_col, ids_to_tags, save_model=False):
        self.elem_col = elem_col
        self.ids_to_tags = ids_to_tags
        self.save_model = save_model
        self.candidate_pair_col = candidate_pair_col
        self.gold_pair_col = gold_pair_col

        self.y_hat = []
        self.polarity_hat = []

        # store optimize measure.
        self.optimize_exact_measure = {}
        self.optimize_prop_measure = {}
        self.optimize_binary_measure = {}

    def eval_model(self, measure_file, model=None, model_path=None, polarity=False, initialize=(False, False)):
        """
        :param measure_file: a file path to write result.
        :param model: use to save.
        :param model_path: a file path to save model.
        :param polarity:
        :param initialize: (polarity, pair)
        :return:
        """
        assert (self.save_model and model is not None) or not self.save_model

        # using exact, proportional and binary measure model.
        predict_num, gold_num = {"init_pair": 0.0, "pair": 0.0}, {"init_pair": 0.0, "pair": 0.0}
        exact_correct_num, prop_correct_num = {"init_pair": 0.0, "pair": 0.0}, {"init_pair": 0.0, "pair": 0.0}
        binary_correct_num = {"init_pair": 0.0, "pair": 0.0}

        predict_tuple_pair_col = self.get_predict_truth_tuple_pair(self.candidate_pair_col)

        assert len(self.gold_pair_col) == len(predict_tuple_pair_col), "data length error!"

        # calculate elem dict.
        # tuple_str = ""
        for index in range(len(self.gold_pair_col)):
            gold_sequence_pair_col = self.gold_pair_col[index]
            predict_sequence_pair_col = predict_tuple_pair_col[index]

            gold_num['pair'] += self.get_effective_pair_num(gold_sequence_pair_col)
            predict_num['pair'] += self.get_effective_pair_num(predict_sequence_pair_col)

            gold_num['init_pair'] += self.get_effective_pair_num(gold_sequence_pair_col)
            predict_num['init_pair'] += self.get_effective_pair_num(self.candidate_pair_col[index])

            cur_exact_num, cur_prop_num, cur_binary_num = self.get_pair_num(
                gold_sequence_pair_col, predict_sequence_pair_col, polarity=polarity
            )

            cur_fake_exact_num, cur_fake_prop_num, cur_fake_binary_num = self.get_pair_num(
                gold_sequence_pair_col, self.candidate_pair_col[index], polarity=False
            )

            assert cur_exact_num <= cur_prop_num <= cur_binary_num, "eval calculate error!"
            assert cur_fake_exact_num <= cur_fake_prop_num <= cur_fake_binary_num, "eval calculate error!"

            # tuple_str += self.print_tuple_pair(
            #     gold_sequence_pair_col, predict_sequence_pair_col, [cur_exact_num, cur_binary_num]
            # )

            exact_correct_num['pair'] += cur_exact_num
            prop_correct_num['pair'] += cur_prop_num
            binary_correct_num['pair'] += cur_binary_num

            exact_correct_num['init_pair'] += cur_fake_exact_num
            prop_correct_num['init_pair'] += cur_fake_prop_num
            binary_correct_num['init_pair'] += cur_fake_binary_num

        # with open("./tuple_pair_output.txt", "w", encoding='utf-8') as f:
        #     f.write(tuple_str)

        print(gold_num, predict_num)

        # calculate f-score.
        exact_measure = self.get_f_score(gold_num, predict_num, exact_correct_num, multi_elem_score=False)
        prop_measure = self.get_f_score(gold_num, predict_num, prop_correct_num, multi_elem_score=False)
        binary_measure = self.get_f_score(gold_num, predict_num, binary_correct_num, multi_elem_score=False)

        keep_rate = predict_num['pair'] / predict_num['init_pair'] * 100
        keep_rate_dict = {"P": keep_rate, "R": keep_rate, "F": keep_rate}
        exact_measure['keep_rate'], prop_measure['keep_rate'] = keep_rate_dict, keep_rate_dict
        binary_measure['keep_rate'] = keep_rate_dict

        # add polarity accuracy.
        exact_measure = self.get_polarity_acc(exact_measure, exact_correct_num['pair'], gold_num['pair'])
        prop_measure = self.get_polarity_acc(prop_measure, prop_correct_num['pair'], gold_num['pair'])
        binary_measure = self.get_polarity_acc(binary_measure, binary_correct_num['pair'], gold_num['pair'])

        # print result in file
        self.print_measure(exact_measure, measure_file, measure_type='exact')
        self.print_measure(prop_measure, measure_file, measure_type='prop')
        self.print_measure(binary_measure, measure_file, measure_type='binary')

        if self.is_optimize(self.optimize_exact_measure, exact_measure, "pair"):
            self.optimize_exact_measure = copy.deepcopy(exact_measure)
            self.optimize_prop_measure = copy.deepcopy(prop_measure)
            self.optimize_binary_measure = copy.deepcopy(binary_measure)

            if self.save_model:
                torch.save(model, model_path)

        if initialize[0]:
            self.polarity_hat = []
        elif initialize[1]:
            self.y_hat = []

    @staticmethod
    def get_effective_pair_num(tuple_pair_col):
        """
        :param tuple_pair_col:
        :return:
        """
        elem_length = len(tuple_pair_col[0]) if len(tuple_pair_col) != 0 else 5
        null_pair, pair_num = [(-1, -1)] * elem_length, 0
        for index in range(len(tuple_pair_col)):
            if tuple_pair_col[index] == null_pair:
                continue
            pair_num += 1
        return pair_num

    def add_pair_data(self, match_label):
        if torch.cuda.is_available():
            match_label = match_label.cpu().numpy().tolist()
        else:
            match_label = match_label.numpy().tolist()

        self.y_hat.append(match_label)

    def add_polarity_data(self, predict_polarity):
        if torch.cuda.is_available():
            predict_polarity = predict_polarity.cpu().numpy().tolist()
        else:
            predict_polarity = predict_polarity.numpy().tolist()

        self.polarity_hat.append(predict_polarity)

    @staticmethod
    def get_polarity_acc(measure, correct_num, gold_num):
        """
        :param measure:
        :param correct_num:
        :param gold_num:
        :return:
        """
        if gold_num == 0:
            acc = 0
        else:
            acc = correct_num / gold_num * 100

        measure['polarity_acc'] = {}
        for val in {"P", "R", "F"}:
            measure['polarity_acc'][val] = acc

        return measure

    def get_predict_truth_tuple_pair(self, candidate_tuple_pair_col):
        """
        :param candidate_tuple_pair_col:
        :return:
        """
        truth_tuple_pair_col = []

        # with polarity and is_pair.
        if len(self.y_hat) != 0 and len(self.polarity_hat) != 0:

            for index in range(len(candidate_tuple_pair_col)):
                cur_predicate_tuple_pair = []

                # drop none-pair and add polarity to pair.
                for k in range(len(self.y_hat[index])):
                    if self.y_hat[index][k] == 1:
                        cur_predicate_tuple_pair.append(
                            self.add_polarity_to_tuple_pair(candidate_tuple_pair_col[index][k], self.polarity_hat[index][k])
                        )

                truth_tuple_pair_col.append(cur_predicate_tuple_pair)

        elif len(self.polarity_hat) != 0:
            for index in range(len(candidate_tuple_pair_col)):
                cur_predicate_tuple_pair = []

                # drop none-pair and add polarity to pair.
                for k in range(len(self.polarity_hat[index])):
                    cur_predicate_tuple_pair.append(
                        self.add_polarity_to_tuple_pair(candidate_tuple_pair_col[index][k], self.polarity_hat[index][k])
                    )

                truth_tuple_pair_col.append(cur_predicate_tuple_pair)

        elif len(self.y_hat) != 0:
            for index in range(len(candidate_tuple_pair_col)):
                cur_predicate_tuple_pair = []

                # drop none-pair and add polarity to pair.
                for k in range(len(self.y_hat[index])):
                    if self.y_hat[index][k] == 1:
                        cur_predicate_tuple_pair.append(copy.deepcopy(candidate_tuple_pair_col[index][k]))

                truth_tuple_pair_col.append(cur_predicate_tuple_pair)

        assert len(self.y_hat) != 0 or len(self.polarity_hat) != 0, "[ERROR] Data Process Error!"

        return truth_tuple_pair_col

    @staticmethod
    def add_polarity_to_tuple_pair(tuple_pair, polarity):
        return copy.deepcopy(tuple_pair + [(int(polarity - 1), int(polarity - 1))])

    def print_tuple_pair(self, gold_tuple_pair, predict_tuple_pair, correct_num):
        """
        :param gold_tuple_pair:
        :param predict_tuple_pair:
        :param correct_num:
        :return:
        """
        write_str = ""
        for index in range(len(gold_tuple_pair)):
            write_str += self.tuple_pair_to_string(gold_tuple_pair[index])

        write_str += "----------------------------------\n"

        for index in range(len(predict_tuple_pair)):
            write_str += self.tuple_pair_to_string(predict_tuple_pair[index])

        for index in range(len(correct_num)):
            write_str += str(correct_num[index])

            if index != len(correct_num) - 1:
                write_str += " "
            else:
                write_str += "\n"

    @staticmethod
    def tuple_pair_to_string(tuple_pair):
        """
        :param tuple_pair:
        :return:
        """
        write_str = "["
        for index in range(len(tuple_pair)):
            write_str += "(" + str(tuple_pair[index][0]) + ", " + str(tuple_pair[index][1]) + ")"

            if index != len(tuple_pair) - 1:
                write_str += " , "
            else:
                write_str += "]\n"

        return write_str

    def get_pair_num(self, gold_pair_col, predict_pair_col, polarity=False):
        """
        :param gold_pair_col: [n, tuple_pair]
        :param predict_pair_col: [n, tuple_pair]
        :param polarity:
        :return:
        """
        exact_num = self.get_exact_pair_num(gold_pair_col, predict_pair_col, polarity)
        prop_num = self.get_cover_pair_num(gold_pair_col, predict_pair_col, "prop", polarity)
        binary_num = self.get_cover_pair_num(gold_pair_col, predict_pair_col, "binary", polarity)
        return exact_num, prop_num, binary_num

    @staticmethod
    def get_exact_pair_num(gold_col, predict_col, polarity=False):
        """
        :param gold_col: [gold_pair_num, tuple_pair]
        :param predict_col: [predict_pair_num, tuple_pair]
        :param polarity
        :return: correct_num.
        """
        correct_num, null_pair = 0.0, [(-1, -1)] * 5

        for gold_index in range(len(gold_col)):
            if gold_col[gold_index] == null_pair:
                continue

            for predict_index in range(len(predict_col)):
                if polarity and gold_col[gold_index] == predict_col[predict_index]:
                    correct_num += 1
                    break
                elif not polarity and gold_col[gold_index][: -1] == predict_col[predict_index]:
                    correct_num += 1
                    break

        return correct_num

    def get_cover_pair_num(self, gold_col, predict_col, measure_type="binary", polarity=False):
        """
        :param gold_col:
        :param predict_col:
        :param measure_type:
        :param polarity:
        :return:
        """
        correct_num, null_pair = 0.0, [(-1, -1)] * 5

        for gold_index in range(len(gold_col)):
            if gold_col[gold_index] == null_pair:
                continue

            for predict_index in range(len(predict_col)):
                is_pair, cover_prop = self.pair_is_cover(gold_col[gold_index], predict_col[predict_index], polarity)

                if is_pair:
                    correct_num = correct_num + 1 if measure_type == "binary" else correct_num + cover_prop
                    break

        return correct_num

    @staticmethod
    def pair_is_cover(gold_tuple_pair, predict_tuple_pair, polarity=False):
        """
        :param gold_tuple_pair: [(s_index, e_index)]
        :param predict_tuple_pair: [(s_index, e_index)]
        :param polarity: False denote without polarity, True denote with polarity
        :return:
        """
        null_elem = (-1, -1)
        gold_elem_length, cover_elem_length = 0, 0

        for index in range(4):
            if gold_tuple_pair[index] == null_elem and predict_tuple_pair[index] == null_elem:
                continue

            cur_gold_length = gold_tuple_pair[index][1] - gold_tuple_pair[index][0]
            cur_cover_length = shared_utils.cover_rate(
                gold_tuple_pair[index], predict_tuple_pair[index], proportion=False
            )

            if cur_cover_length > 0:
                gold_elem_length += cur_gold_length
                cover_elem_length += cur_cover_length

            else:
                return False, 0

        if polarity and gold_tuple_pair[-1] != predict_tuple_pair[-1]:
            return False, 0

        return True, cover_elem_length / gold_elem_length if gold_elem_length != 0 else 0

    def get_f_score(self, gold_num, predict_num, correct_num, multi_elem_score=False):
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

        result_dict = self.get_macro_measure(result_dict, base_elem_col, elem_name="macro")
        result_dict = self.get_micro_measure(
            result_dict, gold_num, predict_num, correct_num, base_elem_col, elem_name="micro"
        )

        return result_dict

    def get_macro_measure(self, result_dict, multi_key_col, elem_name="macro"):
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
        macro_precision = macro_precision / float(len(self.elem_col))
        macro_recall = macro_recall / float(len(self.elem_col))

        try:
            macro_f_score = 2 * macro_precision * macro_recall / (macro_precision + macro_recall)
        except ZeroDivisionError:
            macro_f_score = 0.0

        result_dict[elem_name] = {"P": macro_precision, "R": macro_recall, "F": macro_f_score}

        return result_dict

    @staticmethod
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

    def init_elem(self, key_col=None):
        if key_col is None:
            return {elem: 0.0 for elem in self.elem_col} if self.elem_col is not None else {}
        return {elem: 0.0 for elem in key_col}

    @staticmethod
    def is_optimize(optimize_measure, measure, multi_measure_type):
        """
        :param optimize_measure: {elem: {"P": num, "R": num, "F": num}} or {}
        :param measure: {elem: {"P": num, "R": num, "F": num}}
        :param multi_measure_type: "macro" or "micro"
        :return:
        """
        if len(optimize_measure) == 0 or optimize_measure[multi_measure_type]['F'] < measure[multi_measure_type]['F']:
            return True
        return False

    def print_measure(self, measure, measure_file, measure_type="exact"):
        """
        :param measure: {elem: {P: xx, R: xx, F: xx}.....}
        :param measure_file: a file path to write.
        :param measure_type: "exact", "binary" or "prop"
        :return:
        """
        assert measure_type in {"exact", "prop", "binary"}, "unknown measure type."

        with open(measure_file, "a") as f:
            self.standard_print(measure, measure_type, f)

    @staticmethod
    def standard_print(measure, measure_type, file_point):
        print("========================================", file=file_point)
        print("{} Measure".format(measure_type), file=file_point)
        print("========================================", file=file_point)
        for elem in measure.keys():
            if elem == "sent_acc":
                print("Comparative Sentence Label Accuracy is {:.2f}%".format(measure[elem]['F']), file=file_point)
                continue

            if elem == "elem_acc":
                print("Predicate Correspond Sentence Label Accuracy is {:.2f}%".format(measure[elem]['F']),
                      file=file_point)
                continue

            if elem == "polarity_acc":
                print("Polarity Label Accuracy is {:.2f}%".format(measure[elem]['F']),
                      file=file_point)
                continue

            print("{} Measure's {} Precision value is {:.2f}"
                  .format(measure_type, elem, measure[elem]['P']), file=file_point)

            print("{} Measure's {} Recall value is {:.2f}"
                  .format(measure_type, elem, measure[elem]['R']), file=file_point)

            print("{} Measure's {} F-Measure value is {:.2f}"
                  .format(measure_type, elem, measure[elem]['F']), file=file_point)


class BaseEvaluation(object):
    def __init__(self, elem_col=None, fold=0, save_model=False):
        """
        :param config: program config table.
        :param elem_col: ["entity_1", "entity_2", "aspect", "scale", "predicate"].
        :param ids_to_tags: {0: "O", 1: "B-entity_1"}.
        :param save_model: True denote save model by optimize exact measure.
        """
        self.elem_col = elem_col
        self.fold = fold

        self.save_model = save_model

        # store predict out.
        self.elem_hat = []
        self.result_hat = []
        self.predict_dict = {}

        # store optimize measure.
        self.optimize_exact_measure = {}
        self.optimize_prop_measure = {}
        self.optimize_binary_measure = {}

        # store average measure.
        self.avg_exact_measure = self.init_elem()
        self.avg_prop_measure = self.init_elem()
        self.avg_binary_measure = self.init_elem()

    ####################################################################################################################
    # each elem measure. (exact / proportional / binary)
    ####################################################################################################################

    def init_elem(self, key_col=None):
        if key_col is None:
            return {elem: 0.0 for elem in self.elem_col} if self.elem_col is not None else {}
        return {elem: 0.0 for elem in key_col}

    @staticmethod
    def is_optimize(optimize_measure, measure, multi_measure_type):
        """
        :param optimize_measure: {elem: {"P": num, "R": num, "F": num}} or {}
        :param measure: {elem: {"P": num, "R": num, "F": num}}
        :param multi_measure_type: "macro" or "micro"
        :return:
        """
        if len(optimize_measure) == 0 or optimize_measure[multi_measure_type]['F'] < measure[multi_measure_type]['F']:
            return True
        return False

    ####################################################################################################################
    # Evaluation Write Result txt Part.
    ####################################################################################################################
    def print_measure(self, measure, measure_file, measure_type="exact"):
        """
        :param measure: {elem: {P: xx, R: xx, F: xx}.....}
        :param measure_file: a file path to write.
        :param measure_type: "exact", "binary" or "prop"
        :return:
        """
        assert measure_type in {"exact", "prop", "binary"}, "unknown measure type."

        with open(measure_file, "a") as f:
            self.standard_print(measure, measure_type, f)

    def best_model(self, measure_file):
        """
        :param measure_file:
        :return:
        """
        with open(measure_file, "a") as f:
            print("========================================", file=f)
            print("fold: {} Best Model Measure".format(self.fold), file=f)
            print("========================================", file=f)

            self.standard_print(self.optimize_exact_measure, "Exact", f)
            self.standard_print(self.optimize_prop_measure, "Proportional", f)
            self.standard_print(self.optimize_binary_measure, "Binary", f)

            print("========================================", file=f)

    @staticmethod
    def standard_print(measure, measure_type, file_point):
        print("========================================", file=file_point)
        print("{} Measure".format(measure_type), file=file_point)
        print("========================================", file=file_point)
        for elem in measure.keys():
            if elem == "sent_acc":
                print("Comparative Sentence Label Accuracy is {:.2f}%".format(measure[elem]['F']), file=file_point)
                continue

            if elem == "elem_acc":
                print("Predicate Correspond Sentence Label Accuracy is {:.2f}%".format(measure[elem]['F']),
                      file=file_point)
                continue

            if elem == "polarity_acc":
                print("Polarity Label Accuracy is {:.2f}%".format(measure[elem]['F']),
                      file=file_point)
                continue

            print("{} Measure's {} Precision value is {:.2f}"
                  .format(measure_type, elem, measure[elem]['P']), file=file_point)

            print("{} Measure's {} Recall value is {:.2f}"
                  .format(measure_type, elem, measure[elem]['R']), file=file_point)

            print("{} Measure's {} F-Measure value is {:.2f}"
                  .format(measure_type, elem, measure[elem]['F']), file=file_point)

    ####################################################################################################################
    # Average Measure Process Part
    ####################################################################################################################
    def avg_model(self, write_file):
        """
        :param write_file:
        :return:
        """
        with open(write_file, "a") as f:
            print("========================================", file=f)
            print("fold: {} Best Model Measure".format(self.fold), file=f)
            print("========================================", file=f)

            self.standard_print(self.avg_exact_measure, "Exact", f)
            self.standard_print(self.avg_prop_measure, "Proportional", f)
            self.standard_print(self.avg_binary_measure, "Binary", f)

            print("========================================", file=f)

    @staticmethod
    def add_fold_measure(avg_measure, opt_measure, fold_num=5):
        """
        :param avg_measure: store all fold average measure.
        :param opt_measure: each fold's optimize measure.
        :param fold_num: fold number.
        :return:
        """
        if len(avg_measure) == 0:
            avg_measure = copy.deepcopy(opt_measure)

            for elem, eval_dict in opt_measure.items():
                if not isinstance(eval_dict, dict):
                    continue

                for eval_key, num in eval_dict.items():
                    avg_measure[elem][eval_key] = num / fold_num
        else:
            for elem, eval_dict in opt_measure.items():
                if not isinstance(eval_dict, dict):
                    continue

                for eval_key, num in eval_dict.items():
                    avg_measure[elem][eval_key] += num / fold_num

        return avg_measure

    def store_result_to_csv(self, model_name, csv_file_path):
        """
        :param model_name: [model name]
        :param csv_file_path: write csv file path.
        :return:
        """
        row_label = ["measure_type"]
        data = [["Exact"], ["Prop"], ["Binary"]]

        for elem in self.avg_exact_measure.keys():
            row_label.append(elem)
            data[0].append(round(self.avg_exact_measure[elem]['F'], 2))
            data[1].append(round(self.avg_prop_measure[elem]['F'], 2))
            data[2].append(round(self.avg_binary_measure[elem]['F'], 2))

        with open(csv_file_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(model_name)
            writer.writerow(row_label)
            writer.writerows(data)



