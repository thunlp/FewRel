import json
import os
import multiprocessing
import numpy as np
import random
import torch
from torch.autograd import Variable

class FileDataLoader:
    def next_batch(self, B, N, K, Q):
        '''
        B: batch size.
        N: the number of relations for each batch
        K: the number of support instances for each relation
        Q: the number of query instances for each relation
        return: support_set, query_set, query_label
        '''
        raise NotImplementedError

class JSONFileDataLoader(FileDataLoader):
    def _load_preprocessed_file(self):
        name_prefix = '.'.join(self.file_name.split('/')[-1].split('.')[:-1])
        word_vec_name_prefix = '.'.join(self.word_vec_file_name.split('/')[-1].split('.')[:-1])
        processed_data_dir = '_processed_data'
        if not os.path.isdir(processed_data_dir):
            return False
        word_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_word.npy')
        pos1_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_pos1.npy')
        pos2_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_pos2.npy')
        mask_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_mask.npy')
        length_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_length.npy')
        rel2scope_file_name = os.path.join(processed_data_dir, name_prefix + '_rel2scope.json')
        word_vec_mat_file_name = os.path.join(processed_data_dir, word_vec_name_prefix + '_mat.npy')
        word2id_file_name = os.path.join(processed_data_dir, word_vec_name_prefix + '_word2id.json')
        if not os.path.exists(word_npy_file_name) or \
           not os.path.exists(pos1_npy_file_name) or \
           not os.path.exists(pos2_npy_file_name) or \
           not os.path.exists(mask_npy_file_name) or \
           not os.path.exists(length_npy_file_name) or \
           not os.path.exists(rel2scope_file_name) or \
           not os.path.exists(word_vec_mat_file_name) or \
           not os.path.exists(word2id_file_name):
            return False
        print("Pre-processed files exist. Loading them...")
        self.data_word = np.load(word_npy_file_name)
        self.data_pos1 = np.load(pos1_npy_file_name)
        self.data_pos2 = np.load(pos2_npy_file_name)
        self.data_mask = np.load(mask_npy_file_name)
        self.data_length = np.load(length_npy_file_name)
        self.rel2scope = json.load(open(rel2scope_file_name))
        self.word_vec_mat = np.load(word_vec_mat_file_name)
        self.word2id = json.load(open(word2id_file_name))
        if self.data_word.shape[1] != self.max_length:
            print("Pre-processed files don't match current settings. Reprocessing...")
            return False
        print("Finish loading")
        return True

    def __init__(self, file_name, word_vec_file_name, max_length=40, case_sensitive=False, reprocess=False, cuda=True):
        '''
        file_name: Json file storing the data in the following format
            {
                "P155": # relation id
                    [
                        {
                            "h": ["song for a future generation", "Q7561099", [[16, 17, ...]]], # head entity [word, id, location]
                            "t": ["whammy kiss", "Q7990594", [[11, 12]]], # tail entity [word, id, location]
                            "token": ["Hot", "Dance", "Club", ...], # sentence
                        },
                        ...
                    ],
                "P177": 
                    [
                        ...
                    ]
                ...
            }
        word_vec_file_name: Json file storing word vectors in the following format
            [
                {'word': 'the', 'vec': [0.418, 0.24968, ...]},
                {'word': ',', 'vec': [0.013441, 0.23682, ...]},
                ...
            ]
        max_length: The length that all the sentences need to be extend to.
        case_sensitive: Whether the data processing is case-sensitive, default as False.
        reprocess: Do the pre-processing whether there exist pre-processed files, default as False.
        cuda: Use cuda or not, default as True.
        '''
        self.file_name = file_name
        self.word_vec_file_name = word_vec_file_name
        self.case_sensitive = case_sensitive
        self.max_length = max_length
        self.cuda = cuda

        if reprocess or not self._load_preprocessed_file(): # Try to load pre-processed files:
            # Check files
            if file_name is None or not os.path.isfile(file_name):
                raise Exception("[ERROR] Data file doesn't exist")
            if word_vec_file_name is None or not os.path.isfile(word_vec_file_name):
                raise Exception("[ERROR] Word vector file doesn't exist")

            # Load files
            print("Loading data file...")
            self.ori_data = json.load(open(self.file_name, "r"))
            print("Finish loading")
            print("Loading word vector file...")
            self.ori_word_vec = json.load(open(self.word_vec_file_name, "r"))
            print("Finish loading")
            
            # Eliminate case sensitive
            if not case_sensitive:
                print("Elimiating case sensitive problem...")
                for relation in self.ori_data:
                    for ins in self.ori_data[relation]:
                        for i in range(len(ins['tokens'])):
                            ins['tokens'][i] = ins['tokens'][i].lower()
                print("Finish eliminating")

      
            # Pre-process word vec
            self.word2id = {}
            self.word_vec_tot = len(self.ori_word_vec)
            UNK = self.word_vec_tot
            BLANK = self.word_vec_tot + 1
            self.word_vec_dim = len(self.ori_word_vec[0]['vec'])
            print("Got {} words of {} dims".format(self.word_vec_tot, self.word_vec_dim))
            print("Building word vector matrix and mapping...")
            self.word_vec_mat = np.zeros((self.word_vec_tot, self.word_vec_dim), dtype=np.float32)
            for cur_id, word in enumerate(self.ori_word_vec):
                w = word['word']
                if not case_sensitive:
                    w = w.lower()
                self.word2id[w] = cur_id
                self.word_vec_mat[cur_id, :] = word['vec']
                self.word_vec_mat[cur_id] = self.word_vec_mat[cur_id] / np.sqrt(np.sum(self.word_vec_mat[cur_id] ** 2))
            self.word2id['UNK'] = UNK
            self.word2id['BLANK'] = BLANK
            print("Finish building")

            # Pre-process data
            print("Pre-processing data...")
            self.instance_tot = 0
            for relation in self.ori_data:
                self.instance_tot += len(self.ori_data[relation])
            self.data_word = np.zeros((self.instance_tot, self.max_length), dtype=np.int32)
            self.data_pos1 = np.zeros((self.instance_tot, self.max_length), dtype=np.int32)
            self.data_pos2 = np.zeros((self.instance_tot, self.max_length), dtype=np.int32)
            self.data_mask = np.zeros((self.instance_tot, self.max_length), dtype=np.int32)
            self.data_length = np.zeros((self.instance_tot), dtype=np.int32)
            self.rel2scope = {} # left close right open
            i = 0
            for relation in self.ori_data:
                self.rel2scope[relation] = [i, i]
                for ins in self.ori_data[relation]:
                    head = ins['h'][0]
                    tail = ins['t'][0]
                    pos1 = ins['h'][2][0][0]
                    pos2 = ins['t'][2][0][0]
                    words = ins['tokens']
                    cur_ref_data_word = self.data_word[i]         
                    for j, word in enumerate(words):
                        if j < max_length:
                            if word in self.word2id:
                                cur_ref_data_word[j] = self.word2id[word]
                            else:
                                cur_ref_data_word[j] = UNK
                    for j in range(j + 1, max_length):
                        cur_ref_data_word[j] = BLANK
                    self.data_length[i] = len(words)
                    if len(words) > max_length:
                        self.data_length[i] = max_length
                    if pos1 >= max_length:
                        pos1 = max_length - 1
                    if pos2 >= max_length:
                        pos2 = max_length - 1
                    pos_min = min(pos1, pos2)
                    pos_max = max(pos1, pos2)
                    for j in range(max_length):
                        self.data_pos1[i][j] = j - pos1 + max_length
                        self.data_pos2[i][j] = j - pos2 + max_length
                        if j >= self.data_length[i]:
                            self.data_mask[i][j] = 0
                        elif j <= pos_min:
                            self.data_mask[i][j] = 1
                        elif j <= pos_max:
                            self.data_mask[i][j] = 2
                        else:
                            self.data_mask[i][j] = 3
                    i += 1
                self.rel2scope[relation][1] = i 

            print("Finish pre-processing")     

            print("Storing processed files...")
            name_prefix = '.'.join(file_name.split('/')[-1].split('.')[:-1])
            word_vec_name_prefix = '.'.join(word_vec_file_name.split('/')[-1].split('.')[:-1])
            processed_data_dir = '_processed_data'
            if not os.path.isdir(processed_data_dir):
                os.mkdir(processed_data_dir)
            np.save(os.path.join(processed_data_dir, name_prefix + '_word.npy'), self.data_word)
            np.save(os.path.join(processed_data_dir, name_prefix + '_pos1.npy'), self.data_pos1)
            np.save(os.path.join(processed_data_dir, name_prefix + '_pos2.npy'), self.data_pos2)
            np.save(os.path.join(processed_data_dir, name_prefix + '_mask.npy'), self.data_mask)
            np.save(os.path.join(processed_data_dir, name_prefix + '_length.npy'), self.data_length)
            json.dump(self.rel2scope, open(os.path.join(processed_data_dir, name_prefix + '_rel2scope.json'), 'w'))
            np.save(os.path.join(processed_data_dir, word_vec_name_prefix + '_mat.npy'), self.word_vec_mat)
            json.dump(self.word2id, open(os.path.join(processed_data_dir, word_vec_name_prefix + '_word2id.json'), 'w'))
            print("Finish storing")

    def next_one(self, N, K, Q):
        target_classes = random.sample(self.rel2scope.keys(), N)
        support_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
        query_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
        query_label = []

        for i, class_name in enumerate(target_classes):
            scope = self.rel2scope[class_name]
            indices = np.random.choice(list(range(scope[0], scope[1])), K + Q, False)
            word = self.data_word[indices]
            pos1 = self.data_pos1[indices]
            pos2 = self.data_pos2[indices]
            mask = self.data_mask[indices]
            support_word, query_word, _ = np.split(word, [K, K + Q])
            support_pos1, query_pos1, _ = np.split(pos1, [K, K + Q])
            support_pos2, query_pos2, _ = np.split(pos2, [K, K + Q])
            support_mask, query_mask, _ = np.split(mask, [K, K + Q])
            support_set['word'].append(support_word)
            support_set['pos1'].append(support_pos1)
            support_set['pos2'].append(support_pos2)
            support_set['mask'].append(support_mask)
            query_set['word'].append(query_word)
            query_set['pos1'].append(query_pos1)
            query_set['pos2'].append(query_pos2)
            query_set['mask'].append(query_mask)
            query_label += [i] * Q

        support_set['word'] = np.stack(support_set['word'], 0)
        support_set['pos1'] = np.stack(support_set['pos1'], 0)
        support_set['pos2'] = np.stack(support_set['pos2'], 0)
        support_set['mask'] = np.stack(support_set['mask'], 0)
        query_set['word'] = np.concatenate(query_set['word'], 0)
        query_set['pos1'] = np.concatenate(query_set['pos1'], 0)
        query_set['pos2'] = np.concatenate(query_set['pos2'], 0)
        query_set['mask'] = np.concatenate(query_set['mask'], 0)
        query_label = np.array(query_label)

        perm = np.random.permutation(N * Q)
        query_set['word'] = query_set['word'][perm]
        query_set['pos1'] = query_set['pos1'][perm]
        query_set['pos2'] = query_set['pos2'][perm]
        query_set['mask'] = query_set['mask'][perm]
        query_label = query_label[perm]

        return support_set, query_set, query_label

    def next_batch(self, B, N, K, Q):
        support = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
        query = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
        label = []
        for one_sample in range(B):
            current_support, current_query, current_label = self.next_one(N, K, Q)
            support['word'].append(current_support['word'])
            support['pos1'].append(current_support['pos1'])
            support['pos2'].append(current_support['pos2'])
            support['mask'].append(current_support['mask'])
            query['word'].append(current_query['word'])
            query['pos1'].append(current_query['pos1'])
            query['pos2'].append(current_query['pos2'])
            query['mask'].append(current_query['mask'])
            label.append(current_label)
        support['word'] = Variable(torch.from_numpy(np.stack(support['word'], 0)).long().view(-1, self.max_length))
        support['pos1'] = Variable(torch.from_numpy(np.stack(support['pos1'], 0)).long().view(-1, self.max_length)) 
        support['pos2'] = Variable(torch.from_numpy(np.stack(support['pos2'], 0)).long().view(-1, self.max_length)) 
        support['mask'] = Variable(torch.from_numpy(np.stack(support['mask'], 0)).long().view(-1, self.max_length)) 
        query['word'] = Variable(torch.from_numpy(np.stack(query['word'], 0)).long().view(-1, self.max_length)) 
        query['pos1'] = Variable(torch.from_numpy(np.stack(query['pos1'], 0)).long().view(-1, self.max_length)) 
        query['pos2'] = Variable(torch.from_numpy(np.stack(query['pos2'], 0)).long().view(-1, self.max_length)) 
        query['mask'] = Variable(torch.from_numpy(np.stack(query['mask'], 0)).long().view(-1, self.max_length)) 
        label = Variable(torch.from_numpy(np.stack(label, 0).astype(np.int64)).long())
        
        # To cuda
        if self.cuda:
            for key in support:
                support[key] = support[key].cuda()
            for key in query:
                query[key] = query[key].cuda()
            label = label.cuda()

        return support, query, label
