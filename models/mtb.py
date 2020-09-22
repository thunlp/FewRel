import sys
sys.path.append('..')
import fewshot_re_kit
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F

class Mtb(fewshot_re_kit.framework.FewShotREModel):
    """
    Use the same few-shot model as the paper "Matching the Blanks: Distributional Similarity for Relation Learning".
    """
    
    def __init__(self, sentence_encoder, use_dropout=True, combiner="max"):
        fewshot_re_kit.framework.FewShotREModel.__init__(self, sentence_encoder)
        # self.fc = nn.Linear(hidden_size, hidden_size)
        self.drop = nn.Dropout()
        self.use_dropout = use_dropout
        self.layer_norm = torch.nn.LayerNorm(sentence_encoder.bert.config.hidden_size * (2 if sentence_encoder.cat_entity_rep else 1))
        self.combiner = combiner

    def __dist__(self, x, y, dim):
        return (x * y).sum(dim)

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
        hidden_size = support.size(-1)
        if self.use_dropout:
            support = self.drop(support)
            query = self.drop(query)
        support = self.layer_norm(support)
        query = self.layer_norm(query)
        support = support.view(-1, N, K, hidden_size).unsqueeze(1) # (B, 1, N, K, D)
        query = query.view(-1, total_Q, hidden_size).unsqueeze(2).unsqueeze(2) # (B, total_Q, 1, 1, D)

        logits = (support * query).sum(-1) # (B, total_Q, N, K)

        # aggregate result
        if self.combiner == "max":
            combined_logits, _ = logits.max(-1) # (B, total, N)
        elif self.combiner == "avg":
            combined_logits = logits.mean(-1) # (B, total, N)
        else:
            raise NotImplementedError
        _, pred = torch.max(combined_logits.view(-1, N), -1)

        return combined_logits, pred
    
    
