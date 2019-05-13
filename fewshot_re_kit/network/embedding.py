import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math

class Embedding(nn.Module):

    def __init__(self, word_vec_mat, max_length, word_embedding_dim=50, pos_embedding_dim=5):
        nn.Module.__init__(self)

        self.max_length = max_length
        self.word_embedding_dim = word_embedding_dim
        self.pos_embedding_dim = pos_embedding_dim
        
        # Word embedding
        # unk = torch.randn(1, word_embedding_dim) / math.sqrt(word_embedding_dim)
        # blk = torch.zeros(1, word_embedding_dim)
        word_vec_mat = torch.from_numpy(word_vec_mat)
        self.word_embedding = nn.Embedding(word_vec_mat.shape[0], self.word_embedding_dim, padding_idx=word_vec_mat.shape[0] - 1)
        self.word_embedding.weight.data.copy_(word_vec_mat)

        # Position Embedding
        self.pos1_embedding = nn.Embedding(2 * max_length, pos_embedding_dim, padding_idx=0)
        self.pos2_embedding = nn.Embedding(2 * max_length, pos_embedding_dim, padding_idx=0)

    def forward(self, inputs):
        word = inputs['word']
        pos1 = inputs['pos1']
        pos2 = inputs['pos2']
        
        x = torch.cat([self.word_embedding(word), 
                            self.pos1_embedding(pos1), 
                            self.pos2_embedding(pos2)], 2)
        return x


