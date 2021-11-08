import torch
import math
import copy
import torch.nn as nn
from torchcrf import CRF
import torch.nn.functional as F

from transformers.modeling_bert import BertModel, BertLayer

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


class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, device):
        super(GRUCell, self).__init__()
        # define hyper-parameters
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size

        # reset gate parameters.
        self.W_ir = nn.Linear(input_size, hidden_size, bias=True)
        self.W_hr = nn.Linear(hidden_size, hidden_size, bias=True)

        # update gate parameters.
        self.W_iz = nn.Linear(input_size, hidden_size, bias=True)
        self.W_hz = nn.Linear(hidden_size, hidden_size, bias=True)

        # init feature parameters.
        self.W_in = nn.Linear(input_size, hidden_size, bias=True)
        self.W_hn = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, x, h):
        # calculate two gate result.
        reset_gate = torch.sigmoid(self.W_ir(x) + self.W_hr(h))
        update_gate = torch.sigmoid(self.W_iz(x) + self.W_hz(h))

        # calculate middle result.
        n_t = torch.tanh(self.W_in(x) + reset_gate * self.W_hn(h))
        h_t = (torch.ones_like(update_gate) - update_gate) * n_t + update_gate * h

        return h_t


########################################################################################################################
# Graph Neural Network part
########################################################################################################################


def matrix_norm(A):
    """
    :param A: a matrix need be normed, 2D or 3D
    :return: a normed matrix
    """
    # the function using to norm 2D matrix
    def norm(adj_matrix):
        D = torch.pow(torch.sum(adj_matrix, dim=1), -0.5)
        D = torch.diag(D)
        A_hat = torch.mm(torch.mm(D, adj_matrix), D)
        return A_hat

    # 3D matrix need traverse batch size
    if A.dim() == 3:
        for i in range(A.size(0)):
            A[i] = norm(A[i])
    else:
        A = norm(A)

    return A


# GCN layer using multi-GCN layer
class GCNCell(nn.Module):
    def __init__(self, hidden_size, gcn_layer, dropout=0.5, all_layer=False, residual=False, device="cuda"):
        super(GCNCell, self).__init__()
        self.device = device
        self.all_layer = all_layer
        self.gcn_layers = gcn_layer
        self.residual = residual

        self.gcn_hidden_dim = [hidden_size] * (self.gcn_layers + 1)
        self.w = nn.ModuleList()

        self.gcn_drop_out = nn.Dropout(dropout)
        self.layer_norm = LayerNorm(hidden_size)

        for layer in range(self.gcn_layers):
            self.w.append(nn.Linear(self.gcn_hidden_dim[layer], self.gcn_hidden_dim[layer + 1]).to(device))

    def forward(self, hidden_states, adj_matrix, norm=True):
        # norm dependency matrix
        adj_matrix = self.Matrix_Norm(adj_matrix)

        gcn_out = []
        for layer in range(self.gcn_layers):
            feature = torch.bmm(adj_matrix, hidden_states)
            hidden_states = self.gcn_drop_out(F.relu(self.w[layer](feature)))
            gcn_out.append(hidden_states)

        return gcn_out if self.all_layer else gcn_out[-1]


class UpdateGCNCell(nn.Module):
    def __init__(self, hidden_size, gcn_layer, dropout=0.1, all_layer=False, residual=False, device="cuda"):
        super(UpdateGCNCell, self).__init__()
        self.device = device
        self.all_layer = all_layer
        self.gcn_layers = gcn_layer
        self.residual = residual

        self.gcn_hidden_dim = [hidden_size] * (self.gcn_layers + 1)
        self.w = nn.ModuleList()

        self.gcn_drop_out = nn.Dropout(dropout)
        self.layer_norm = LayerNorm(hidden_size)

        for layer in range(self.gcn_layers):
            self.w.append(nn.Linear(self.gcn_hidden_dim[layer], self.gcn_hidden_dim[layer + 1]).to(device))

    def forward(self, hidden_states, adj_matrix, norm=True):
        # print(hidden_states.size(), adj_matrix.size())
        gcn_out = []
        for layer in range(self.gcn_layers):
            feature = torch.bmm(adj_matrix, hidden_states)
            gcn_output = self.gcn_drop_out(F.relu(self.w[layer](feature)))

            if self.residual:
                hidden_states = self.layer_norm(hidden_states + gcn_output)
            else:
                hidden_states = self.layer_norm(gcn_output)

            gcn_out.append(hidden_states)

        return gcn_out if self.all_layer else gcn_out[-1]


# using decrease factor to reduce influence each layer.
class DecreaseGCNCell(nn.Module):
    def __init__(self, config, all_layer=False, gcn_layer=1):
        super(DecreaseGCNCell, self).__init__()
        self.config = config
        self.device = config.device
        self.all_layer = all_layer
        self.gcn_layers = gcn_layer
        # self.gcn_hidden_dim = config.gcn_hidden_dim
        self.gcn_hidden_dim = [768] * (self.gcn_layers + 1)
        self.w = nn.ModuleList()

        for layer in range(self.gcn_layers):
            self.w.append(nn.Linear(self.gcn_hidden_dim[layer], self.gcn_hidden_dim[layer + 1]).to(config.device))

    def Matrix_Decrese(self, A, factor):
        def decrease(adj_matrix, factor):
            result_matrix = copy.deepcopy(adj_matrix) * factor

            # using mask to change diagonal data
            mask = torch.diag(torch.ones(result_matrix.size(0))).bool().to(self.config.device)
            result_matrix = result_matrix.masked_fill(mask, 1.)
            return result_matrix

        if A.dim() == 3:
            for i in range(A.size(0)):
                A[i] = decrease(copy.deepcopy(A[i]), factor)
        else:
            A = decrease(copy.deepcopy(A), factor)

        return A

    def forward(self, bert_feature, dep_matrix, norm=True, decrease_factor=0.8):
        gcn_out = []
        for layer in range(self.gcn_layers):
            # norm dependency matrix
            norm_matrix = self.Matrix_Norm(copy.deepcopy(dep_matrix))

            feature = torch.bmm(norm_matrix, bert_feature)
            bert_feature = F.relu(self.w[layer](feature))

            # using decrease factor
            dep_matrix = self.Matrix_Decrese(copy.deepcopy(norm_matrix), decrease_factor)
            gcn_out.append(bert_feature)

        if self.all_layer:
            return gcn_out

        else:
            return gcn_out[-1]


# using bi-graph to get more information.
class BiGCNCell(nn.Module):
    def __init__(self, hidden_size, gcn_layer, dropout=0.5, all_layer=False, device="cuda"):
        super(BiGCNCell, self).__init__()
        self.device = device
        self.all_layer = all_layer
        self.gcn_layers = gcn_layer

        self.gcn_hidden_dim = [hidden_size] * (self.gcn_layers + 1)
        self.w_in = nn.ModuleList()
        self.w_out = nn.ModuleList()
        self.w_0 = nn.ModuleList()

        self.layer_norm = LayerNorm(hidden_size)

        for layer in range(self.gcn_layers):
            self.w_in.append(nn.Linear(self.gcn_hidden_dim[layer], self.gcn_hidden_dim[layer + 1]).to(device))
            self.w_out.append(nn.Linear(self.gcn_hidden_dim[layer], self.gcn_hidden_dim[layer + 1]).to(device))
            self.w_0.append(nn.Linear(2 * self.gcn_hidden_dim[layer], self.gcn_hidden_dim[layer + 1]).to(device))

    def forward(self, hidden_states, adj_matrix, norm=True):
        # norm dependency matrix
        out_graph = self.Matrix_Norm(adj_matrix)
        in_graph = out_graph.transpose(2, 1)

        gcn_out = []
        for layer in range(self.gcn_layers):
            out_feature = torch.bmm(out_graph, hidden_states)
            in_feature = torch.bmm(in_graph, hidden_states)

            out_feature = F.relu(self.w_out[layer](out_feature))
            in_feature = F.relu(self.w_in[layer](in_feature))

            concat_feature = torch.cat((out_feature, in_feature), dim=2)

            bert_feature = self.layer_norm(hidden_states + self.w_0[layer](F.relu(concat_feature)))

            gcn_out.append(bert_feature)

        if self.all_layer:
            return gcn_out

        else:
            return gcn_out[-1]


class GATCell(nn.Module):
    def __init__(self, in_feature, out_feature, gat_layer, all_layer=True):
        super(GATCell, self).__init__()
        self.gat_layer = gat_layer
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.alpha = 0.2

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.w = nn.ModuleList()
        self.a = nn.ModuleList()

        self.all_layer = all_layer

        for index in range(self.gat_layer):
            self.w.append(nn.Linear(in_feature, out_feature))
            self.a.append(nn.Linear(2 * out_feature, 1))

    def forward(self, input_feature, matrix):
        """
        :param input_feature: [Batch_size, seq_length, feature_dim]
        :param matrix: [Batch_size, seq_length, seq_length]
        :return:
        """
        hidden_states = input_feature
        batch_size, seq_length, feature_dim = input_feature.size()

        gat_out = []
        for index in range(self.gat_layer):
            h = self.w[index](hidden_states)

            hi = h.unsqueeze(2).repeat(1, 1, seq_length, 1)
            hj = h.unsqueeze(1).repeat(1, seq_length, 1, 1)

            a_input = torch.cat([hi, hj], dim=-1)
            e = self.leakyrelu(self.a[index](a_input)).view(batch_size, seq_length, seq_length)
            zero_vec = -1e12 * torch.ones_like(e)

            attention = torch.where(matrix > 0, e, zero_vec)
            attention = F.softmax(attention, dim=1)
            attention = F.dropout(attention, 0.1, training=self.training)

            hidden_states = torch.bmm(attention, h)

            gat_out.append(hidden_states)

        return gat_out if self.all_layer else gat_out[-1]


class GNNCell(nn.Module):
    def __init__(self, layer, seq_len, device):
        super(GNNCell, self).__init__()

        # define GNN W1 and W2
        self.W1 = nn.Parameter(torch.ones(seq_len, 300), requires_grad=True)
        self.W2 = nn.Parameter(torch.ones(seq_len, 300), requires_grad=True)

        # define GNN B1 and B2
        self.B1 = nn.Parameter(torch.ones(seq_len, 300), requires_grad=True)
        self.B2 = nn.Parameter(torch.ones(seq_len, 300), requires_grad=True)

        self.layer = layer

    def distance(self, head_embed, dep_embed):
        batch_size, seq_len = head_embed.size(0), head_embed.size(1)

        head_embed = head_embed.unsqueeze(2).expand(batch_size, seq_len, seq_len, -1)
        dep_embed = dep_embed.unsqueeze(1).expand(batch_size, seq_len, seq_len, -1)

        score_matrix = torch.sqrt(torch.sum(torch.pow((head_embed - dep_embed), 2), dim=3, keepdim=True))

        return score_matrix.view(batch_size, seq_len, seq_len)

    def forward(self, head_embed, dep_embed):
        for l in range(self.layer):
            score_matrix = self.distance(head_embed, dep_embed)

            h_update = torch.matmul(score_matrix, head_embed) + torch.matmul(score_matrix.transpose(-2, -1), dep_embed)
            d_update = torch.matmul(score_matrix.transpose(-2, -1), head_embed) + torch.matmul(score_matrix, dep_embed)

            head_embed = F.leaky_relu(self.W1 * h_update + self.B1 * head_embed)
            dep_embed = F.leaky_relu(self.W2 * d_update + self.B2 * dep_embed)

        score_matrix = self.distance(head_embed, dep_embed)

        return score_matrix


########################################################################################################################
# Other Neural Network part.
########################################################################################################################


class MLP_Cell(nn.Module):
    def __init__(self, layer, layer_size, device):
        """
        :param layer: the number of perception layer
        :param layer_size: [input_size, layer1_out_size, layer2_out_size, ......]
        """
        super(MLP_Cell, self).__init__()
        self.layer = layer
        self.device = device
        self.layer_size = layer_size
        self.w = nn.ModuleList()

        for l in range(layer):
            self.w.append(nn.Linear(layer_size[l], layer_size[l + 1]).to(self.device))

    def forward(self, x):
        assert x.size(-1) == self.layer_size[0], "input dim error or layer_size error"

        for l in range(self.layer):
            x = torch.tanh(self.w[l](x))

        return x


class CNN_Cell(nn.Module):
    def __init__(self, config, window_size, op_type="cat"):
        """
        :param config: config contain some CNN hyper-parameters
        :param window_size: a list of the kernel_size collection
        """
        super(CNN_Cell, self).__init__()
        self.cnn_layers = len(window_size)
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=config.bert_hidden_size,
                          out_channels=200,
                          kernel_size=k,
                          padding=(k - 1)//2),
                nn.BatchNorm1d(num_features=200),
                nn.ReLU()) for k in window_size])

    def forward(self, x, op_type="cat"):
        cnn_input = x.permute(0, 2, 1)
        output = [conv(cnn_input) for conv in self.convs]

        if op_type == "sum":
            output = torch.stack(output, dim=1)
            output = torch.sum(output, dim=1)

        elif op_type == "max":
            output = torch.stack(output, dim=1)
            output = torch.max(output, dim=1)

        elif op_type == "cat":
            output = torch.cat(output, dim=1)

        return output.permute(0, 2, 1)


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

