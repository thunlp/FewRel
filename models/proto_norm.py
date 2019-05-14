import sys
sys.path.append('..')
import fewshot_re_kit
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F

def l2norm(X):
    norm = torch.pow(X, 2).sum(dim=-1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X

class ProtoNorm(fewshot_re_kit.framework.FewShotREModel):
    
    def __init__(self, sentence_encoder, hidden_size=230):
        fewshot_re_kit.framework.FewShotREModel.__init__(self, sentence_encoder)
        self.hidden_size = hidden_size
        # self.fc = nn.Linear(hidden_size, hidden_size)
        self.drop = nn.Dropout()

    def __dist__(self, x, y, dim):
        return (torch.pow(x - y, 2)).sum(dim)

    def __batch_dist__(self, S, Q):
        return self.__dist__(S.unsqueeze(1), Q.unsqueeze(2), 3)

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
        support = l2norm(support)
        query = l2norm(query)
        support = self.drop(support)
        query = self.drop(query)
        support = support.view(-1, N, K, self.hidden_size) # (B, N, K, D)
        query = query.view(-1, total_Q, self.hidden_size) # (B, total_Q, D)

        B = support.size(0) # Batch size
         
        # Prototypical Networks 
        # Ignore NA policy
        support = torch.mean(support, 2) # Calculate prototype for each class
        logits = -self.__batch_dist__(support, query) # (B, total_Q, N)
        minn, _ = logits.min(-1)
        logits = torch.cat([logits, minn.unsqueeze(2) - 1], 2) # (B, total_Q, N + 1)
        _, pred = torch.max(logits.view(-1, N+1), 1)
        return logits, pred
    
    
    
