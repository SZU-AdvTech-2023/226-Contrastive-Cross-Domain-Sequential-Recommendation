import numpy as np
import torch
import torch.nn as nn
from model.GNN import GCNLayer


class Discriminator(torch.nn.Module):
    
    def __init__(self, n_in,n_out):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_in, n_out, 1)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, S, node, s_bias=None):
        score = self.f_k(node, S)
        if s_bias is not None:
            score += s_bias
        return score


# 解释： SASREC中的最后的全连接层，这里的实现是一个1*1的卷积层
class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs

# pls use the following self-made multihead attention layer
# in case your pytorch version is below 1.16 or for other reasons
# https://github.com/pmixer/TiSASRec.pytorch/blob/master/model.py

# 参照SASREC设计的多头注意力机制层
class ATTENTION(torch.nn.Module):
    def __init__(self, opt):
        super(ATTENTION, self).__init__()
        self.opt = opt
        self.emb_dropout = torch.nn.Dropout(p=self.opt["dropout"])
        self.pos_emb = torch.nn.Embedding(self.opt["maxlen"], self.opt["hidden_units"], padding_idx=0)  # TO IMPROVE
        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(self.opt["hidden_units"], eps=1e-8)

        for _ in range(self.opt["num_blocks"]):
            new_attn_layernorm = torch.nn.LayerNorm(self.opt["hidden_units"], eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.MultiheadAttention(self.opt["hidden_units"],
                                                            self.opt["num_heads"],
                                                            self.opt["dropout"])
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(self.opt["hidden_units"], eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(self.opt["hidden_units"], self.opt["dropout"])
            self.forward_layers.append(new_fwd_layer)

    def forward(self, seqs_data, seqs, position):
        # positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(position)
        seqs = self.emb_dropout(seqs)

        timeline_mask = torch.BoolTensor(seqs_data.cpu() == self.opt["itemnum"] - 1)
        if self.opt["cuda"]:
            timeline_mask = timeline_mask.cuda()
        seqs *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim

        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool))
        if self.opt["cuda"]:
            attention_mask = attention_mask.cuda()

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs,
                                            attn_mask=attention_mask)
                                            # key_padding_mask=timeline_mask
                                            # need_weights=False) this arg do not work?
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *=  ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)

        return log_feats


class PWLayer(nn.Module):
    """Single Parametric Whitening Layer
    """
    def __init__(self, input_size, output_size, dropout=0.0):
        super(PWLayer, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.bias = nn.Parameter(torch.zeros(input_size), requires_grad=True)
        self.lin = nn.Linear(input_size, output_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, x):
        return self.lin(self.dropout(x) - self.bias)

# 自定义的C2DSR模型
class C2DSR(torch.nn.Module):
    def __init__(self, opt, adj, adj_single):
        super(C2DSR, self).__init__()
        self.opt = opt
        # 初始化3个embedding层
        self.item_emb_X = torch.nn.Embedding(self.opt["itemnum"], self.opt["hidden_units"],
                                           padding_idx=self.opt["itemnum"] - 1)
        self.item_emb_Y = torch.nn.Embedding(self.opt["itemnum"], self.opt["hidden_units"],
                                           padding_idx=self.opt["itemnum"] - 1)
        self.item_emb = torch.nn.Embedding(self.opt["itemnum"], self.opt["hidden_units"],
                                           padding_idx=self.opt["itemnum"] - 1)
        
        if self.opt["load_text"]:
            pwlayer = PWLayer(input_size=768,output_size=self.opt["hidden_units"],dropout=self.opt["dropout"])
            source_text_embs = np.fromfile(self.opt["source_text"], dtype=np.float32).reshape(-1,768)
            target_text_embs = np.fromfile(self.opt["target_text"], dtype=np.float32).reshape(-1,768)
            # 两个领域的文本拼接
            concat_text_embs =  np.vstack((source_text_embs,target_text_embs))
            
            self.Linear_text_x = nn.Linear(self.opt["hidden_units"] * 2 , self.opt["hidden_units"])
            self.Linear_text_y = nn.Linear(self.opt["hidden_units"] * 2 , self.opt["hidden_units"])
            self.Linear_text = nn.Linear(self.opt["hidden_units"] * 2 , self.opt["hidden_units"])
            # 对两个领域的文本进行PW 参数白化
            # shape : (source+target_nums,768)
            pw_text_embs = pwlayer(torch.Tensor(concat_text_embs))
            
            # 由于C2DSR作者添加上了没用的一行，因此在将文本拼接到ID embedding中时需要保持相对应，。
            pw_text_embs = torch.cat((pw_text_embs,torch.zeros(1,self.opt["hidden_units"])),dim=0)
           
            self.item_emb_X.weight = torch.nn.Parameter(self.Linear_text_x(torch.cat((self.item_emb_X.weight, pw_text_embs),dim=1)))
            self.item_emb_Y.weight= torch.nn.Parameter(self.Linear_text_y(torch.cat((self.item_emb_Y.weight, pw_text_embs),dim=1)) )
            self.item_emb.weight = torch.nn.Parameter(self.Linear_text(torch.cat((self.item_emb.weight, pw_text_embs),dim=1)) )
        
        # 使用GNN层提取特征
        self.GNN_encoder_X = GCNLayer(opt)
        self.GNN_encoder_Y = GCNLayer(opt)
        self.GNN_encoder = GCNLayer(opt)
        self.adj = adj
        self.adj_single = adj_single

        self.D_X = Discriminator(self.opt["hidden_units"], self.opt["hidden_units"])
        self.D_Y = Discriminator(self.opt["hidden_units"], self.opt["hidden_units"])

        self.lin_X = nn.Linear(self.opt["hidden_units"], self.opt["source_item_num"])
        self.lin_Y = nn.Linear(self.opt["hidden_units"], self.opt["target_item_num"])
        self.lin_PAD = nn.Linear(self.opt["hidden_units"], 1)
        self.encoder = ATTENTION(opt)
        self.encoder_X = ATTENTION(opt)
        self.encoder_Y = ATTENTION(opt)

        self.source_item_index = torch.arange(0, self.opt["source_item_num"], 1)
        self.target_item_index = torch.arange(self.opt["source_item_num"], self.opt["source_item_num"]+self.opt["target_item_num"], 1)
        self.item_index = torch.arange(0, self.opt["itemnum"], 1)
        if self.opt["cuda"]:
            self.source_item_index = self.source_item_index.cuda()
            self.target_item_index = self.target_item_index.cuda()
            self.item_index = self.item_index.cuda()

    def my_index_select_embedding(self, memory, index):
        tmp = list(index.size()) + [-1]
        index = index.view(-1)
        ans = memory(index)
        ans = ans.view(tmp)
        return ans

    def my_index_select(self, memory, index):
        tmp = list(index.size()) + [-1]
        index = index.view(-1)
        ans = torch.index_select(memory, 0, index)
        ans = ans.view(tmp)
        return ans

    # 解释：
    def graph_convolution(self):
        fea = self.my_index_select_embedding(self.item_emb, self.item_index)
        fea_X = self.my_index_select_embedding(self.item_emb_X, self.item_index)
        fea_Y = self.my_index_select_embedding(self.item_emb_Y, self.item_index)

        self.cross_emb = self.GNN_encoder(fea, self.adj)
        self.single_emb_X = self.GNN_encoder_X(fea_X, self.adj_single)
        self.single_emb_Y = self.GNN_encoder_Y(fea_Y, self.adj_single)
     
    
    def forward(self, o_seqs, x_seqs, y_seqs, position, x_position, y_position):
        seqs = self.my_index_select(self.cross_emb, o_seqs) + self.item_emb(o_seqs)
        seqs *= self.item_emb.embedding_dim ** 0.5
        seqs_fea = self.encoder(o_seqs, seqs, position)

        seqs = self.my_index_select(self.single_emb_X, x_seqs) + self.item_emb_X(x_seqs)
        seqs *= self.item_emb.embedding_dim ** 0.5
        x_seqs_fea = self.encoder_X(x_seqs, seqs, x_position)

        seqs = self.my_index_select(self.single_emb_Y, y_seqs) + self.item_emb_Y(y_seqs)
        seqs *= self.item_emb.embedding_dim ** 0.5
        y_seqs_fea = self.encoder_Y(y_seqs, seqs, y_position)

        return seqs_fea, x_seqs_fea, y_seqs_fea

        # 负样本的前向传播
    def false_forward(self, false_seqs, position):
        seqs = self.my_index_select(self.cross_emb, false_seqs) + self.item_emb(false_seqs)
        seqs *= self.item_emb.embedding_dim ** 0.5
        false_seqs_fea = self.encoder(false_seqs, seqs, position)
        return false_seqs_fea