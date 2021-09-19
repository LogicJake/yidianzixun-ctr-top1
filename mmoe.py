import torch
import torch.nn as nn
import torch.nn.functional as F


def activation_layer(act_name):
    """Construct activation layers

    Args:
        act_name: str or nn.Module, name of activation function
        hidden_size: int, used for Dice activation
        dice_dim: int, used for Dice activation
    Return:
        act_layer: activation layer
    """
    if isinstance(act_name, str):
        if act_name.lower() == 'sigmoid':
            act_layer = nn.Sigmoid()
        elif act_name.lower() == 'relu':
            act_layer = nn.ReLU(inplace=True)
        elif act_name.lower() == 'prelu':
            act_layer = nn.PReLU()
    elif issubclass(act_name, nn.Module):
        act_layer = act_name()
    else:
        raise NotImplementedError

    return act_layer


class DNN(nn.Module):
    def __init__(self,
                 inputs_dim,
                 hidden_units,
                 activation='relu',
                 dropout_rate=0,
                 use_bn=False,
                 init_std=0.0001):
        super(DNN, self).__init__()
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
        self.use_bn = use_bn
        if len(hidden_units) == 0:
            raise ValueError("hidden_units is empty!!")
        hidden_units = [inputs_dim] + list(hidden_units)

        self.linears = nn.ModuleList([
            nn.Linear(hidden_units[i], hidden_units[i + 1])
            for i in range(len(hidden_units) - 1)
        ])

        if self.use_bn:
            self.bn = nn.ModuleList([
                nn.BatchNorm1d(hidden_units[i + 1])
                for i in range(len(hidden_units) - 1)
            ])

        self.activation_layers = nn.ModuleList([
            activation_layer(activation) for i in range(len(hidden_units) - 1)
        ])

        for name, tensor in self.linears.named_parameters():
            if 'weight' in name:
                nn.init.normal_(tensor, mean=0, std=init_std)

    def forward(self, inputs):
        deep_input = inputs

        for i in range(len(self.linears)):

            fc = self.linears[i](deep_input)

            if self.use_bn:
                fc = self.bn[i](fc)

            fc = self.activation_layers[i](fc)

            fc = self.dropout(fc)
            deep_input = fc
        return deep_input

class LHUC(nn.Module):
    def __init__(self,
                 inputs_dim,
                 hidden_units,
                 lhuc_size,
                 activation='relu',
                 dropout_rate=0,
                 use_bn=False,
                 init_std=0.0001):
        super(LHUC, self).__init__()
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
        self.use_bn = use_bn
        if len(hidden_units) == 0:
            raise ValueError("hidden_units is empty!!")
        hidden_units = [inputs_dim] + list(hidden_units)

        self.linears = nn.ModuleList([
            nn.Linear(hidden_units[i], hidden_units[i + 1])
            for i in range(len(hidden_units) - 1)
        ])

        self.lhucs = nn.ModuleList()
        for i in range(len(hidden_units) - 1):
            input_size = hidden_units[i]
            lhuc = nn.Sequential(
                  nn.Linear(lhuc_size, 256),
                  nn.ReLU(),
                  nn.Linear(256, input_size),
                  nn.Sigmoid()
                )
            self.lhucs.append(lhuc)

        if self.use_bn:
            self.bn = nn.ModuleList([
                nn.BatchNorm1d(hidden_units[i + 1])
                for i in range(len(hidden_units) - 1)
            ])

        self.activation_layers = nn.ModuleList([
            activation_layer(activation) for i in range(len(hidden_units) - 1)
        ])

        for name, tensor in self.linears.named_parameters():
            if 'weight' in name:
                nn.init.normal_(tensor, mean=0, std=init_std)

    def forward(self, inputs, lhuc_input):
        deep_input = inputs

        for i in range(len(self.linears)):
            lhuc_scale = self.lhucs[i](lhuc_input)

            fc = self.linears[i](deep_input * lhuc_scale * 2)

            if self.use_bn:
                fc = self.bn[i](fc)

            fc = self.activation_layers[i](fc)

            fc = self.dropout(fc)
            deep_input = fc
        return deep_input

    
class FM(nn.Module):
    """Factorization Machine models pairwise (order-2) feature interactions
     without linear term and bias.
      Input shape
        - 3D tensor with shape: ``(batch_size,field_size,embedding_size)``.
      Output shape
        - 2D tensor with shape: ``(batch_size, 1)``.
      References
        - [Factorization Machines](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)
    """
    def __init__(self):
        super(FM, self).__init__()

    def forward(self, inputs):
        fm_input = inputs

        square_of_sum = torch.pow(torch.sum(fm_input, dim=1, keepdim=True), 2)
        sum_of_square = torch.sum(fm_input * fm_input, dim=1, keepdim=True)
        cross_term = square_of_sum - sum_of_square
        cross_term = 0.5 * torch.sum(cross_term, dim=2, keepdim=False)

        return cross_term


# 离散特征统一 embedding 层
class EmbeddingLayer(nn.Module):
    def __init__(self, sparse_features, embedding_dim, weights):
        super(EmbeddingLayer, self).__init__()

        embedding_dict = nn.ModuleDict()

        for name, num_embeddings in sparse_features.items():
            embedding_dict[name] = nn.Embedding(num_embeddings, embedding_dim)

        for tensor in embedding_dict.values():
            nn.init.normal_(tensor.weight, mean=0, std=0.01)

        # 加载预训练权重
        if weights is not None:
            for k, weight in weights.items():
                embedding_dict[k].weight.data.copy_(torch.from_numpy(weight))
                embedding_dict[k].weight.requires_grad = True

        self.embedding_dict = embedding_dict

        # self.multihead_attn = nn.MultiheadAttention(embedding_dim, num_heads=1)

    def _sequence_mask(self, ids):
        mask = ids
        mask = mask > 0
        return mask

    def _sequence_pooling(self, emb, id, weight=None, mode='mean'):
        # True 正常
        if weight is not None:
            mask = weight
        else:
            mask = self._sequence_mask(id)  # [B, maxlen]
        length = torch.sum(mask, dim=-1, keepdim=True)  # [B, maxlen]

        if mode == 'mean':
            mask = mask.view(mask.shape[0], mask.shape[1], 1)  # [B, maxlen, 1]
            mask = torch.repeat_interleave(mask, emb.shape[2],
                                           dim=2)  # [B, maxlen, E]

            hist = emb * mask.float()
            hist = torch.sum(hist, dim=1, keepdim=False)

            eps = torch.FloatTensor([1e-8]).to(length.device)
            hist = torch.div(hist, length.type(torch.float32) + eps)

            return hist
        if mode == 'max':
            mask = mask.view(mask.shape[0], mask.shape[1], 1)  # [B, maxlen, 1]
            mask = torch.repeat_interleave(mask, emb.shape[2],
                                           dim=2)  # [B, maxlen, E]

            hist = emb * mask.float()
            hist, _ = torch.max(hist, dim=1, keepdim=False)

            return hist
        if mode == 'sum':
            mask = mask.view(mask.shape[0], mask.shape[1], 1)  # [B, maxlen, 1]
            mask = torch.repeat_interleave(mask, emb.shape[2],
                                           dim=2)  # [B, maxlen, E]

            hist = emb * mask.float()
            hist = torch.sum(hist, dim=1, keepdim=False)

            return hist
        if mode == 'lstm':
            mask = mask.view(mask.shape[0], mask.shape[1], 1)  # [B, maxlen, 1]
            mask = torch.repeat_interleave(mask, emb.shape[2],
                                           dim=2)  # [B, maxlen, E]
            hist, _ = self.word_lstm(emb)
            hist = hist * mask.float()
            hist = torch.sum(hist, dim=1, keepdim=False)

            return hist

        if mode == 'attention':
            emb = emb.transpose(0, 1)  # [maxlen, B, E]
            hist, _ = self.multihead_attn(
                emb, emb, emb, key_padding_mask=None)  # [maxlen, B, E]
            hist = hist.transpose(0, 1)  # [B, maxlen, E]

            mask = mask.view(mask.shape[0], mask.shape[1], 1)  # [B, maxlen, 1]
            mask = torch.repeat_interleave(mask, emb.shape[2],
                                           dim=2)  # [B, maxlen, E]
            hist = hist * mask.float()
            hist = torch.sum(hist, dim=1, keepdim=False)

            eps = torch.FloatTensor([1e-8]).to(length.device)
            hist = torch.div(hist, length.type(torch.float32) + eps)

            return hist

    def forward(self, id_list):
        emb_list = []
        for id, conf in id_list:
            if len(id.shape) == 1:
                name = conf
                emb = self.embedding_dict[name](id.long())
                emb = emb.view((emb.shape[0], 1, emb.shape[1]))
                emb_list.append(emb)
            else:
                name = conf['name']
                weight = None
                mode = 'mean'
                if 'weight' in conf:
                    weight = conf['weight']
                if 'mode' in conf:
                    mode = conf['mode']

                emb = self.embedding_dict[name](id.long())  # [B, maxlen, E]
                emb = self._sequence_pooling(emb, id, weight=weight, mode=mode)
                emb = emb.view((emb.shape[0], 1, emb.shape[1]))
                emb_list.append(emb)

        output = torch.cat(emb_list, 1)

        return output

    def global_embedding(self, id_list):
        emb_list = []
        for id, name in id_list:
            emb = self.embedding_dict[name](id.long())
            if len(emb.shape) == 2:
                emb = emb.view((emb.shape[0], 1, emb.shape[1]))
            else:
                emb = emb.view((emb.shape[0], 1, emb.shape[1], emb.shape[2]))
            emb_list.append(emb)

        output = torch.cat(emb_list, 1)
        global_emb = torch.mean(output, 1)
        return global_emb


class DocRec(nn.Module):
    def __init__(self,
                 sparse_features,
                 dense_features,
                 embedding_dim,
                 device='cpu',
                 weights=None):

        super(DocRec, self).__init__()

        self.embedding_layers = EmbeddingLayer(sparse_features, embedding_dim,
                                               weights)

        # DNN
        dnn_input_size = len(sparse_features) * embedding_dim + len(
            dense_features)
        lhuc_size = 3 * embedding_dim
        self.layer_norm = nn.LayerNorm(dnn_input_size)
        dnn_hidden_units = [1024, 512, 64]
        self.lhuc = LHUC(inputs_dim=dnn_input_size, hidden_units=dnn_hidden_units, lhuc_size=lhuc_size)
#         self.dnn = DNN(inputs_dim=dnn_input_size,
#                        hidden_units=dnn_hidden_units)
        self.dnn_linear = nn.Linear(dnn_hidden_units[-1], 1, bias=False)
        self.fm = FM()

        self.to(device)

    def forward(self, userid, docid, network, device_t, os, province, city, age, gender, category1st, category2nd, dense_features, keywords, click):

        id_list = [[userid, 'userid'], [docid, 'docid'], [network, 'network'],
                   [device_t, 'device'], [os, 'os'], [province, 'province'],
                   [city, 'city'], [age, 'age'], [gender, 'gender'],
                   [category1st, 'category1st'], [category2nd, 'category2nd'],
                   [keywords, {
                       'name': 'keyword'
                   }]]
        lhuc_id_list = [[device_t, 'device'], [province, 'province'], [city, 'city'], ]

        embeding_input = self.embedding_layers(id_list)
        lhuc_embeding_input = self.embedding_layers(lhuc_id_list)

        dnn_input = torch.cat(
            [torch.flatten(embeding_input, start_dim=1), dense_features], 1)
        lhuc_input = torch.flatten(lhuc_embeding_input, start_dim=1)

        # dnn_input = self.layer_norm(dnn_input)
        dnn_output = self.lhuc(dnn_input, lhuc_input)
#         dnn_output = self.dnn(dnn_input)
        dnn_logit = self.dnn_linear(dnn_output)

        fm_input = embeding_input
        fm_logit = self.fm(fm_input)

        logit = dnn_logit + fm_logit
        pred = torch.sigmoid(logit)
        pred = pred.squeeze()

        loss = F.binary_cross_entropy(pred, click)

        return pred, loss
