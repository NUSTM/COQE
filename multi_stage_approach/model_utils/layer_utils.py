import torch
import torch.nn as nn
from TorchCRF import CRF

from transformers import BertModel, BertLayer

########################################################################################################################
# Model Encoder Layer.
########################################################################################################################


# Bert layer only with BertModel
class BERTCell(nn.Module):
    def __init__(self, model_path):
        super(BERTCell, self).__init__()
        self.bert = BertModel.from_pretrained(model_path)
        self.hidden_size = self.bert.config.hidden_size
        self.hidden_dropout_prob = self.bert.config.hidden_dropout_prob

    def forward(self, input_ids, attn_mask):
        # encoded_layers shape: [layers, batch_size, tokens, hidden_size]
        # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output, pooled_output = self.bert(input_ids, attention_mask=attn_mask)[:2]

        return sequence_output, pooled_output


# bi-direction-LSTM Layer.
class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, device, batch_first=True, bidirectional=True):
        super(LSTMCell, self).__init__()
        # define hyper-parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layer = num_layers
        self.device = device

        self.hidden = None
        self.lstm = nn.LSTM(
            self.input_size, self.hidden_size, self.num_layer, batch_first=batch_first, bidirectional=bidirectional
        )

    def init_hidden(self, batch_size):

        # h0: [num_layer * num_directions, batch_size, hidden_size]
        # c0: [num_layer * num_directions, batch_size, hidden_size]
        return (torch.randn(self.num_layer * 2, batch_size, self.hidden_size).to(self.device),
                torch.randn(self.num_layer * 2, batch_size, self.hidden_size).to(self.device))

    def forward(self, x):
        self.hidden = self.init_hidden(x.size(0))

        print(self.hidden[0].size(), self.hidden[1].size())
        # output: [batch, seq_length, num_directions * hidden_size]
        output, self.hidden = self.lstm(x, self.hidden)

        return output, torch.cat([self.hidden[0][-1, :, :], self.hidden[0][-2, :, :]], dim=-1)


########################################################################################################################
# Graph Neural Network part
########################################################################################################################


########################################################################################################################
# Other Neural Network part.
########################################################################################################################


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size), requires_grad=False)
        self.bias = nn.Parameter(torch.zeros(hidden_size), requires_grad=False)
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


########################################################################################################################
# Decoder Neural Network part.
########################################################################################################################

class CRFCell(nn.Module):
    def __init__(self, num_classes, batch_first=True):
        super(CRFCell, self).__init__()
        self.crf = CRF(num_tags=num_classes, batch_first=batch_first)

    def forward(self, feature, mask, target):
        mask = mask.bool()

        # target is None means decode sequence
        if target is None:
            output = self.crf.decode(feature, mask)
            max_seq_len = feature.size(1)
            output = torch.as_tensor(
                [e + [-1] * (max_seq_len - len(e)) for e in output],
                dtype=torch.long,
                device=feature.device
            )
        # target is not None means get loss
        else:
            output = self.crf(feature, target, mask, reduction='mean')
            output = output.neg()
        return output

