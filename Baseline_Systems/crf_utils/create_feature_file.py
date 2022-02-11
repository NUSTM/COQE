import copy
import json
from data_utils import shared_utils
from stanfordcorenlp import StanfordCoreNLP


class stanfordFeature(object):
    def __init__(self, sentences, stanford_path, lang="en"):
        """
        :param sentences: a list of sentence, [sentence1, sentence2......]
        :param stanford_path: nlp stanford core path
        :param lang: denote which language sentences need to process
        """
        self.sentences_col = sentences
        self.nlp = StanfordCoreNLP(stanford_path, lang=lang)

        # store the maximum length of sequence
        self.max_len = -1

    def get_tokenizer(self):
        """
        :return: a list of token by stanford tokenizer
        """
        input_tokens = []
        for i in range(len(self.sentences_col)):
            token_list = self.nlp.word_tokenize(self.sentences_col[i])
            input_tokens.append(token_list)

        return input_tokens

    def get_pos_feature(self):
        """
        :param: pos_dict:
        :param: pos_index:
        :return: a list of pos-tag, with id
        """
        pos_feature = []
        for index in range(len(self.sentences_col)):
            tag_list = self.nlp.pos_tag(self.sentences_col[index])
            pos_tag_list = list(list(zip(*tag_list))[1])

            pos_feature.append(pos_tag_list)

        return pos_feature

    def get_parse_feature(self, token_col):
        parse_col = []
        for index in range(len(self.sentences_col)):
            s_index, seq_parse = 0, []
            while s_index < len(token_col[index]):
                sentence_parse = self.nlp.parse("".join(token_col[index][s_index: ]).rstrip('\n'))
                sentence_parse = [x.lstrip(" ") for x in shared_utils.split_string(sentence_parse, "\r\n")]
                part_parse = self.parse_parser_tree("".join(sentence_parse))

                seq_parse.extend(part_parse)
                s_index = len(seq_parse)

            parse_col.append(seq_parse)

        return parse_col

    @staticmethod
    def parse_parser_tree(sentence_parse):
        index, s_index, symbol_stack, seq_parse = 0, -1, [], []
        while index < len(sentence_parse):
            if sentence_parse[index] == "(":
                if s_index != -1:
                    symbol_stack.append(sentence_parse[s_index: index])

                s_index = index + 1

            elif sentence_parse[index] == ")":
                if s_index != -1:
                    seq_parse.append(symbol_stack[-1])
                elif len(symbol_stack) > 0:
                    symbol_stack.pop()
                s_index = -1

            index += 1

        return seq_parse


# 获取比较候选词
def get_kw_vocab(path):
    kw_vocab = []
    with open(path, "r", encoding='gb18030', errors='ignore') as f:
        for line in f.readlines():
            if line == '\n':
                continue

            word_list = []
            if line.find("...") != -1:
                word_list = line.rstrip('\n').split('...')
            else:
                word_list.append(line.rstrip('\n'))

            kw_vocab.append(word_list)

    return kw_vocab


def create_sequence_position_feature(token_col, sequence_keyword):
    token_index, sequence_position = 0, ["LBF"] * len(sequence_keyword)

    segment_index_col, keyword_index_col, segment_symbol = [], [], {"。", "，", "？", "！", "?", "!", ",", "."}
    for index in range(len(token_col)):
        if token_col[index] in segment_symbol:
            segment_index_col.append(index)

        if sequence_keyword[index] == "YES":
            keyword_index_col.append(index)
    segment_index_col.append(len(token_col))

    kw_index, seg_index = 0, 0
    while kw_index < len(keyword_index_col):
        while segment_index_col[seg_index] < keyword_index_col[kw_index]:
            seg_index += 1

        if kw_index + 1 >= len(keyword_index_col):
            end_board = segment_index_col[seg_index]
        else:
            end_board = min(keyword_index_col[kw_index + 1], segment_index_col[seg_index])

        for t in range(keyword_index_col[kw_index] + 1, end_board):
            sequence_position[t] = "LAF"

        kw_index += 1

    return sequence_position


def create_sequence_keyword_feature(token_col, keyword_vocab):
    """
    :param token_col:
    :param keyword_vocab: a list of list
    :return:
    """

    sequence_keyword = ["NO"] * len(token_col)

    for keyword in keyword_vocab:
        kw_index, token_index, match_index_col = 0, 0, []
        while kw_index < len(keyword) and token_index < len(token_col):
            if keyword[kw_index] == token_col[token_index]:
                match_index_col.append(token_index)
                kw_index += 1

            token_index += 1

        if len(match_index_col) == len(keyword):
            for k in match_index_col:
                sequence_keyword[k] = "YES"

    sequence_position = create_sequence_position_feature(token_col, sequence_keyword)
    return sequence_keyword, sequence_position


def create_keyword_and_position_feature(token_col, keyword_path):
    keyword_vocab = get_kw_vocab(keyword_path)

    keyword_col, position_col = [], []
    for index in range(len(token_col)):
        sequence_keyword, sequence_position = create_sequence_keyword_feature(token_col[index], keyword_vocab)

        keyword_col.append(sequence_keyword)
        position_col.append(sequence_position)

    return keyword_col, position_col


def sequence_token_to_char(token_list):
    """
    :param token_list: a list of ddParser tokenizer result
    :param add_hidden: True denote add [CLS] and [SEP]
    :return: a matrix like: [token_length, char_length]
    """
    token_index, token_length = 0, len(token_list)

    char_index, char_length = 0, len("".join(token_list))

    token_to_char = [[0 for _ in range(char_length)] for _ in range(token_length)]

    while token_index < token_length and char_index < char_length:
        for k in range(len(token_list[token_index])):
            token_to_char[token_index][char_index + k] = 1

        char_index += len(token_list[token_index])
        token_index = token_index + 1

    assert token_index == token_length and char_index == char_length, "appear special token"

    return token_to_char


def create_token_to_char_feature(token_col):
    token_char_col = []
    for index in range(len(token_col)):
        token_char_col.append(sequence_token_to_char(token_col[index]))

    return token_char_col
