from data_utils import shared_utils


class LabelParser(object):
    def __init__(self, label_col, elem_col, intermittent=False):
        """
        :param label_col:
        :param elem_col: ["entity_1", "entity_2", "aspect", "result"]
        :param intermittent: True denote "result" using intermittent representation
        """
        self.label_col = label_col
        self.elem_col = elem_col
        self.intermittent = intermittent

    def parse_sequence_label(self, split_symbol="&", sent_col=None, file_type="cn"):
        """
        :param split_symbol:
        :param sent_col:
        :param file_type
        :return:
        """
        null_label = "[[];[];[];[];[]]"
        tuple_pair_col, elem_representation_col = [], []

        for index in range(len(self.label_col)):
            # For non-comparative sentences' label.
            if self.label_col[index][0] == null_label:
                tuple_pair_col.append([[(-1, -1)] * 5])
                elem_representation_col.append(self.init_label_representation())

            else:
                global_elem_col = self.init_label_representation()

                sequence_tuple_pair = []
                for pair_index in range(len(self.label_col[index])):
                    global_elem_col, cur_tuple_pair = self.parse_each_pair_label(
                        self.label_col[index][pair_index], global_elem_col, split_symbol, sent_col[index], file_type
                    )
                    sequence_tuple_pair.append(cur_tuple_pair)

                tuple_pair_col.append(sequence_tuple_pair)
                elem_representation_col.append(global_elem_col)

        return elem_representation_col, tuple_pair_col

    def parse_each_pair_label(self, sequence_label, global_elem_col, split_symbol, sent=None, file_type="cn"):
        """
        :param sequence_label:
        :param global_elem_col:
        :param split_symbol:
        :param sent:
        :param file_type:
        :return:
        """
        elem_representation = shared_utils.split_string(sequence_label[1:-1], ";")
        tuple_pair_representation, result_elem = [], []
        for elem_index, each_elem in enumerate(elem_representation):
            if elem_index == 3 and each_elem == "[]":
                print(elem_representation)
            if self.intermittent:
                seg_elem_col = shared_utils.split_string(each_elem[1: -1], " , ")
            else:
                seg_elem_col = [each_elem[1: -1]] if each_elem[1:-1] != "" else []

            elem_tuple = ()

            # not polarity
            if elem_index != len(elem_representation) - 1:
                for each_seg_elem in seg_elem_col:
                    number_char_col = shared_utils.split_string(each_seg_elem, " ")

                    if file_type == "cn":
                        s_index = int(shared_utils.split_string(number_char_col[0], split_symbol)[0])
                        e_index = int(shared_utils.split_string(number_char_col[-1], split_symbol)[0]) + 1
                    else:
                        s_index = int(shared_utils.split_string(number_char_col[0], split_symbol)[0]) - 1
                        e_index = int(shared_utils.split_string(number_char_col[-1], split_symbol)[0])

                    elem_tuple += (s_index, e_index)

                    if self.elem_col[elem_index] == "result":
                        result_elem += [s_index, e_index]

                    # [check sentence and label position]
                    # if sent is not None:
                    #     cur_elem_str = self.get_sub_elem(number_char_col, split_symbol)
                    #
                    #     if cur_elem_str != sent[s_index: e_index]:
                    #         print("----------------------------")
                    #         print(cur_elem_str)
                    #         print(sent[s_index: e_index])
                    #         print(s_index, e_index)
                    #         print(number_char_col)
                    #         print("----------------------------")

            else:
                polarity = int(seg_elem_col[0])
                elem_tuple += (polarity, polarity)

                # 针对英文中可能存在空的情况
                if len(result_elem) == 0:
                    result_elem = [-1, -1]

                result_elem.append(polarity)

            elem_tuple = (-1, -1) if len(elem_tuple) == 0 else elem_tuple
            tuple_pair_representation.append(elem_tuple)

            if elem_index < 3 and elem_tuple != (-1, -1):
                global_elem_col[self.elem_col[elem_index]].add(elem_tuple)

        global_elem_col["result"].add(tuple(result_elem))

        return global_elem_col, tuple_pair_representation

    @staticmethod
    def get_sub_elem(number_char_col, split_symbol):
        """
        :param number_char_col:
        :param split_symbol:
        :return:
        """
        elem_str = ""
        for num_char in number_char_col:
            elem_str += shared_utils.split_string(num_char, split_symbol)[1]

        return elem_str

    def init_label_representation(self):
        return {key: set() for key in self.elem_col}
