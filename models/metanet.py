import sys
sys.path.append('..')
import fewshot_re_kit
from fewshot_re_kit.network.embedding import Embedding
from fewshot_re_kit.network.encoder import Encoder
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np

def log_and_sign(inputs, k=7):
    eps = 1e-7
    log = torch.log(torch.abs(inputs) + eps) / k
    log[log < -1.0] = -1.0 
    sign = log * np.exp(k)
    sign[sign < -1.0] = -1.0
    sign[sign > 1.0] = 1.0
    return torch.cat([log, sign], 1)

class LearnerForAttention(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)
        self.conv_lstm = nn.LSTM(2, 20, batch_first=True)
        self.conv_fc = nn.Linear(20, 1) 
        self.fc_lstm = nn.LSTM(2, 20, batch_first=True)
        self.fc_fc = nn.Linear(20, 1) 

    def forward(self, inputs, is_conv):
        size = inputs.size()
        x = inputs.view((-1, 1))
        x = log_and_sign(x) # (-1, 2)

        #### NO BACKPROP
        x = Variable(x, requires_grad=False).unsqueeze(0) # (1, param_size, 2)
        #### 
         
        if is_conv:
            x, _ = self.conv_lstm(x) # (1, param_size, 1)
            x = x.squeeze() 
            x = self.conv_fc(x)
        else:
            x, _ = self.fc_lstm(x) # (1, param_size, 1)
            x = x.squeeze()
            x = self.fc_fc(x)
        return x.view(size)

class LearnerForBasic(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)
        self.conv_fc1 = nn.Linear(2, 20)
        self.conv_fc2 = nn.Linear(20, 20)
        self.conv_fc3 = nn.Linear(20, 1)
        self.fc_fc1 = nn.Linear(2, 20)
        self.fc_fc2 = nn.Linear(20, 20)
        self.fc_fc3 = nn.Linear(20, 1)


    def forward(self, inputs, is_conv):
        size = inputs.size()
        x = inputs.view((-1, 1))
        x = log_and_sign(x) # (-1, 2)

        #### NO BACKPROP
        x = Variable(x, requires_grad=False)
        ####
        
        if is_conv:
            x = F.relu(self.conv_fc1(x)) 
            x = F.relu(self.conv_fc2(x))
            x = self.conv_fc3(x)
        else:
            x = F.relu(self.fc_fc1(x)) 
            x = F.relu(self.fc_fc2(x))
            x = self.fc_fc3(x)
        return x.view(size)

class MetaNet(fewshot_re_kit.framework.FewShotREModel):
    
    def __init__(self, N, K, embedding, max_length, hidden_size=230):
        '''
        N: num of classes
        K: num of instances for each class
        word_vec_mat, max_length, hidden_size: same as sentence_encoder
        '''
        fewshot_re_kit.framework.FewShotREModel.__init__(self, None)
        self.max_length = max_length
        self.hidden_size = hidden_size
        self.N = N
        self.K = K
        
        # self.embedding = Embedding(word_vec_mat, max_length, word_embedding_dim=50, pos_embedding_dim=5)
        self.embedding = embedding

        self.basic_encoder = Encoder(max_length, word_embedding_dim=50, pos_embedding_dim=5, hidden_size=hidden_size)
        self.attention_encoder = Encoder(max_length, word_embedding_dim=50, pos_embedding_dim=5, hidden_size=hidden_size)

        self.basic_fast_conv_W = None
        self.attention_fast_conv_W = None

        self.basic_fc = nn.Linear(hidden_size, N, bias=False)
        self.attention_fc = nn.Linear(hidden_size, N, bias=False)

        self.basic_fast_fc_W = None
        self.attention_fast_fc_W = None

        self.learner_basic = LearnerForBasic()
        self.learner_attention = LearnerForAttention()

    def basic_emb(self, inputs, size, use_fast=False):
        x = self.embedding(inputs)
        output = self.basic_encoder(x)
        if use_fast:
            output += F.relu(F.conv1d(x.transpose(-1, -2), self.basic_fast_conv_W, padding=1)).max(-1)[0]
        return output.view(size)
    
    def attention_emb(self, inputs, size, use_fast=False):
        x = self.embedding(inputs)
        output = self.attention_encoder(x)
        if use_fast:
            output += F.relu(F.conv1d(x.transpose(-1, -2), self.attention_fast_conv_W, padding=1)).max(-1)[0]
        return output.view(size)

    def attention_score(self, s_att, q_att):
        '''
        s_att: (B, N, K, D)
        q_att: (B, NQ, D)
        '''
        s_att = s_att.view(s_att.size(0), s_att.size(1) * s_att.size(2), s_att.size(3)) # (B, N * K, D)
        s_att = s_att.unsqueeze(1) # (B, 1, N * K, D)
        q_att = q_att.unsqueeze(2) # (B, NQ, 1, D)
        cos = F.cosine_similarity(s_att, q_att, dim=-1) # (B, NQ, N * K)
        score = F.softmax(cos, -1) # (B, NQ, N * K)
        return score

    def forward(self, support, query, N, K, Q):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        '''

        # learn fast parameters for attention encoder
        s = self.attention_emb(support, (-1, N, K, self.hidden_size))
        logits = self.attention_fc(s) # (B, N, K, N)

        B = s.size(0)
        NQ = N * Q
        assert(B == 1)

        self.zero_grad()
        tmp_label = Variable(torch.tensor([[x] * K for x in range(N)] * B, dtype=torch.long).cuda())
        loss = self.cost(logits.view(-1, N), tmp_label.view(-1))
        loss.backward(retain_graph=True)
            
        grad_conv = self.attention_encoder.conv.weight.grad
        grad_fc = self.attention_fc.weight.grad

        self.attention_fast_conv_W = self.learner_attention(grad_conv, is_conv=True)
        self.attention_fast_fc_W = self.learner_attention(grad_fc, is_conv=False)
        
        # learn fast parameters for basic encoder (each class)
        s = self.basic_emb(support, (-1, N, K, self.hidden_size))
        logits = self.basic_fc(s) # (B, N, K, N)

        basic_fast_conv_params = []
        basic_fast_fc_params = []
        for i in range(N):
            for j in range(K):
                self.zero_grad()
                tmp_label = Variable(torch.tensor([i], dtype=torch.long).cuda())
                loss = self.cost(logits[:, i, j].view(-1, N), tmp_label.view(-1))
                loss.backward(retain_graph=True)

                grad_conv = self.basic_encoder.conv.weight.grad
                grad_fc = self.basic_fc.weight.grad

                basic_fast_conv_params.append(self.learner_basic(grad_conv, is_conv=True))
                basic_fast_fc_params.append(self.learner_basic(grad_fc, is_conv=False))
        basic_fast_conv_params = torch.stack(basic_fast_conv_params, 0) # (N * K, conv_weight_size)
        basic_fast_fc_params = torch.stack(basic_fast_fc_params, 0) # (N * K, fc_weight_size)

        # final
        self.zero_grad()
        s_att = self.attention_emb(support, (-1, N, K, self.hidden_size), use_fast=True)
        q_att = self.attention_emb(query, (-1, NQ, self.hidden_size), use_fast=True)
        score = self.attention_score(s_att, q_att).squeeze(0) # assume B = 1, (NQ, N * K)
        size_conv_param = basic_fast_conv_params.size()[1:]
        size_fc_param = basic_fast_fc_params.size()[1:]
        final_fast_conv_param = torch.matmul(score, basic_fast_conv_params.view(N * K, -1)) # (NQ, conv_weight_size)
        final_fast_fc_param = torch.matmul(score, basic_fast_fc_params.view(N * K, -1)) # (NQ, fc_weight_size)
        stack_logits = []
        for i in range(NQ):
            self.basic_fast_conv_W = final_fast_conv_param[i].view(size_conv_param)
            self.basic_fast_fc_W = final_fast_fc_param[i].view(size_fc_param)
            q = self.basic_emb({'word': query['word'][i:i+1], 'pos1': query['pos1'][i:i+1], 'pos2': query['pos2'][i:i+1], 'mask': query['mask'][i:i+1]}, (self.hidden_size), use_fast=True)
            logits = self.basic_fc(q) + F.linear(q, self.basic_fast_fc_W) 
            stack_logits.append(logits)
        logits = torch.stack(stack_logits, 0)

        _, pred = torch.max(logits.view(-1, N), 1)
        return logits, pred
