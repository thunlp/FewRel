import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import os
from torch import optim
from . import network
# from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification
from pytorch_transformers import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification,RobertaModel, RobertaTokenizer,RobertaForSequenceClassification 


class CNNSentenceEncoder(nn.Module):

    def __init__(self, word_vec_mat, word2id, max_length, word_embedding_dim=50, 
            pos_embedding_dim=5, hidden_size=230):
        nn.Module.__init__(self)
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.embedding = network.embedding.Embedding(word_vec_mat, max_length, 
                word_embedding_dim, pos_embedding_dim)
        self.encoder = network.encoder.Encoder(max_length, word_embedding_dim, 
                pos_embedding_dim, hidden_size)
        self.word2id = word2id

    def forward(self, inputs):
        x = self.embedding(inputs)
        x = self.encoder(x)
        return x

    def tokenize(self, raw_tokens, pos_head, pos_tail):
        # token -> index
        indexed_tokens = []
        for token in raw_tokens:
            token = token.lower()
            if token in self.word2id:
                indexed_tokens.append(self.word2id[token])
            else:
                indexed_tokens.append(self.word2id['[UNK]'])
        
        # padding
        while len(indexed_tokens) < self.max_length:
            indexed_tokens.append(self.word2id['[PAD]'])
        indexed_tokens = indexed_tokens[:self.max_length]

        # pos
        pos1 = np.zeros((self.max_length), dtype=np.int32)
        pos2 = np.zeros((self.max_length), dtype=np.int32)
        pos1_in_index = min(self.max_length, pos_head[0])
        pos2_in_index = min(self.max_length, pos_tail[0])
        for i in range(self.max_length):
            pos1[i] = i - pos1_in_index + self.max_length
            pos2[i] = i - pos2_in_index + self.max_length

        # mask
        mask = np.zeros((self.max_length), dtype=np.int32)
        mask[:len(indexed_tokens)] = 1

        return indexed_tokens, pos1, pos2, mask


class BERTSentenceEncoder(nn.Module):

    def __init__(self, pretrain_path, max_length): 
        nn.Module.__init__(self)
        # self.bert = BertModel.from_pretrained(pretrain_path)
        self.bert = BertForSequenceClassification.from_pretrained(
                pretrain_path,
                num_labels=2)
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained(os.path.join(
            pretrain_path, 'bert_vocab.txt'))
        self.modelName = 'Bert'

    def forward(self, inputs):
        x = self.bert(inputs['word'], inputs['seg'], attention_mask=inputs['mask'])[0]
        return x
    
    def tokenize(self, raw_tokens, pos_head, pos_tail):
        # token -> index
        # tokens = ['[CLS]']
        tokens = []
        cur_pos = 0
        pos1_in_index = 0
        pos2_in_index = 0
        for token in raw_tokens:
            token = token.lower()
            if cur_pos == pos_head[0]:
                tokens.append('[unused0]')
                pos1_in_index = len(tokens)
            if cur_pos == pos_tail[0]:
                tokens.append('[unused1]')
                pos2_in_index = len(tokens)
            tokens += self.tokenizer.tokenize(token)
            if cur_pos == pos_head[-1]:
                tokens.append('[unused2]')
            if cur_pos == pos_tail[-1]:
                tokens.append('[unused3]')
            cur_pos += 1
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        
        # padding
        '''
        while len(indexed_tokens) < self.max_length:
            indexed_tokens.append(0)
        indexed_tokens = indexed_tokens[:self.max_length]

        # pos
        pos1 = np.zeros((self.max_length), dtype=np.int32)
        pos2 = np.zeros((self.max_length), dtype=np.int32)
        for i in range(self.max_length):
            pos1[i] = i - pos1_in_index + self.max_length
            pos2[i] = i - pos2_in_index + self.max_length

        # mask
        mask = np.zeros((self.max_length), dtype=np.int32)
        mask[:len(indexed_tokens)] = 1
        '''

        return indexed_tokens # , pos1, pos2, mask

class RobertaSentenceEncoder(nn.Module):

    def __init__(self, pretrain_path, max_length): 
        nn.Module.__init__(self)
        self.bert = RobertaForSequenceClassification.from_pretrained(pretrain_path, num_labels=2)
        #self.bert = RobertaModel.from_pretrained(pretrain_path)
        self.max_length = max_length
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.modelName = 'Roberta'

    def forward(self, inputs):
        x = self.bert(inputs['word'], attention_mask=inputs['mask'])[0]
        return x
    
    def tokenize(self, raw_tokens, pos_head, pos_tail):
        # token -> index
        '''
        tokens = ['[CLS]']
        cur_pos = 0
        pos1_in_index = 0
        pos2_in_index = 0
        for token in raw_tokens:
            token = token.lower()
            if cur_pos == pos_head[0]:
                tokens.append('[HEADSTART]')
                pos1_in_index = len(tokens)
            if cur_pos == pos_tail[0]:
                tokens.append('[TAILSTART]')
                pos2_in_index = len(tokens)
            tokens += self.tokenizer.tokenize(token)
            if cur_pos == pos_head[-1]:
                tokens.append('[HEADEND]')
            if cur_pos == pos_tail[-1]:
                tokens.append('[TAILEND]')
            cur_pos += 1
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        '''
        def getIns(bped,bpeTokens,tokens,L,R):
            resL=0
            tkL=" ".join(tokens[:L])
            bped_tkL=" ".join(self.tokenizer.tokenize(tkL))
            if bped.find(bped_tkL)==0:
                resL=len(bped_tkL.split())
            else:
                tkL+=" "
                bped_tkL=" ".join(self.tokenizer.tokenize(tkL))
                if bped.find(bped_tkL)==0:
                    resL=len(bped_tkL.split())
            resR=0
            tkR=" ".join(tokens[R:])
            bped_tkR=" ".join(self.tokenizer.tokenize(tkR))
            if bped.rfind(bped_tkR)+len(bped_tkR)==len(bped):
                resR=len(bpeTokens)-len(bped_tkR.split())
            else:
                tkR=" "+tkR
                bped_tkR=" ".join(self.tokenizer.tokenize(tkR))
                if bped.rfind(bped_tkR)+len(bped_tkR)==len(bped):
                    resR=len(bpeTokens)-len(bped_tkR.split())
            return resL, resR

        s=" ".join(raw_tokens)
        sst=self.tokenizer.tokenize(s)
        headL=pos_head[0]
        headR=pos_head[-1]
        hiL, hiR=getIns(" ".join(sst),sst,raw_tokens,headL,headR)
        tailL=pos_tail[0]
        tailR=pos_tail[-1]
        tiL, tiR=getIns(" ".join(sst),sst,raw_tokens,tailL,tailR)
        E1b='madeupword0000'
        E1e='madeupword0001'
        E2b='madeupword0002'
        E2e='madeupword0003'
        ins=[(hiL, E1b), (hiR, E1e), (tiL, E2b), (tiR, E2e)]
        ins=sorted(ins)
        pE1=0
        pE2=0
        pE1_=0
        pE2_=0
        for i in range(0,4):
            sst.insert(ins[i][0]+i,ins[i][1])
            if ins[i][1]==E1b:
                pE1=ins[i][0]+i
            elif ins[i][1]==E2b:
                pE2=ins[i][0]+i
            elif ins[i][1]==E1e:
                pE1_=ins[i][0]+i
            else:
                pE2_=ins[i][0]+i
        pos1_in_index=pE1+1
        pos2_in_index=pE2+1
        indexed_tokens=self.tokenizer.convert_tokens_to_ids(sst)
        #indexed_tokens=self.tokenizer.add_special_tokens_single_sentence(indexed_tokens)
        # padding
        #while len(indexed_tokens) < self.max_length:
        #    indexed_tokens.append(0)
        #indexed_tokens = indexed_tokens[:self.max_length]
        return indexed_tokens #, pos1, pos2, mask
