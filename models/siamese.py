import sys
sys.path.append('..')
import fewshot_re_kit
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F

class Siamese(fewshot_re_kit.framework.FewShotREModel):
    
    def __init__(self, sentence_encoder, hidden_size=230):
        fewshot_re_kit.framework.FewShotREModel.__init__(self, sentence_encoder)
        self.hidden_size = hidden_size
        self.fc = nn.Linear(hidden_size, 1)
        # self.fc2 = nn.Linear(hidden_size * 2, hidden_size)
        # self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        # self.fc4 = nn.Linear(hidden_size // 2, 1)
        self.drop = nn.Dropout()
        self.cost = nn.BCELoss(reduction="none")
        # self.cost = nn.MSELoss(reduction="none")
        self.relu = nn.ReLU()

    def loss(self, logits, label):
        '''
        logits: Logits with the size (..., class_num)
        label: Label with whatever size. 
        return: [Loss] (A single value)
        '''
        N = logits.size(-1)
        logits = logits.view(-1, N)
        _label = torch.zeros(logits.size()).cuda()
        _label.scatter_(1, label.view(-1, 1), 1)
        # print(logits[:10])
        # print(_label[:10])
        _loss = self.cost(logits, _label)
        _loss = (((_label + 1.0 / N) * _loss) * N)
        # print(_loss[:10])
        _loss = _loss.mean()

        return _loss

    def forward(self, support, query, N, K, total_Q):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances in the query set
        '''

        support = self.sentence_encoder(support) # (B * N * K, D), where D is the hidden size
        query = self.sentence_encoder(query) # (B * total_Q, D)
        # support = self.drop(support)
        # query = self.drop(query)
        support = support.view(-1, N * K, self.hidden_size) # (B, N * K, D)
        query = query.view(-1, total_Q, self.hidden_size) # (B, total_Q, D)

        B = support.size(0) # Batch size

        support = support.unsqueeze(1) # (B, 1, N * K, D)
        query = query.unsqueeze(2) # (B, total_Q, 1, D)

        # relation net
        '''
        support = support.expand(B, total_Q, N * K, self.hidden_size)
        query = query.expand(B, total_Q, N * K, self.hidden_size)
        z = torch.cat([support, query], -1) # (B, total_Q, N * K, 2D)
        z = self.fc2(z)
        z = self.relu(z)
        z = self.fc3(z)
        z = self.relu(z)
        z = self.fc4(z).squeeze(-1) # (B, total_Q, N * K)
        z = z.view(-1, total_Q, N, K) # (B, total_Q, N, K)
        z = torch.sigmoid(z)
        '''

        # simple dot
        z = support * query
        # z = self.drop(z)
        z = self.fc(z).squeeze(-1) # (B, total_Q, N * K)
        z = z.view(-1, total_Q, N, K) # (B, total_Q, N, K)
        z = torch.sigmoid(z)
        
        # average policy
        logits = z.mean(-1) # (B, total_Q, N)
        maxx, _ = logits.max(-1) # (B, total_Q)
        logits = torch.cat([logits, 1 - maxx.unsqueeze(2)], 2) # (B, total_Q, N + 1)
        # zero NA
        # logits = torch.cat([logits, torch.zeros((B, total_Q, 1)).cuda()], 2) # (B, total_Q, N + 1)
        _, pred = torch.max(logits.view(-1, N+1), 1)
        return logits, pred 
