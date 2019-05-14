import models
from fewshot_re_kit.data_loader import get_loader
from fewshot_re_kit.framework import FewShotREFramework
from fewshot_re_kit.sentence_encoder import CNNSentenceEncoder, BERTSentenceEncoder
from models.proto import Proto
from models.proto_norm import ProtoNorm
from models.gnn import GNN
from models.snail import SNAIL
from models.metanet import MetaNet
from models.siamese import Siamese
import sys
from torch import optim
import numpy as np
import torch
import json
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default='train',
            help='train file')
    parser.add_argument('--val', default='val',
            help='val file')
    parser.add_argument('--test', default='test',
            help='test file')
    parser.add_argument('--pretrain', default="",
            help='pretrain path')
    parser.add_argument('--trainN', default=10, type=int,
            help='N in train')
    parser.add_argument('--N', default=5, type=int,
            help='N way')
    parser.add_argument('--K', default=5, type=int,
            help='K shot')
    parser.add_argument('--Q', default=5, type=int,
            help='Num of query per class')
    parser.add_argument('--batch_size', default=4, type=int,
            help='batch size')
    parser.add_argument('--train_epoch', default=30000, type=int,
            help='num of epochs training')
    parser.add_argument('--model', default='proto',
            help='model name')
    parser.add_argument('--encoder', default='cnn',
            help='encoder: cnn or bert')
    parser.add_argument('--max_length', default=120, type=int,
           help='max length')
    parser.add_argument('--val_step', default=2000, type=int,
           help='val after training for how many steps')
    parser.add_argument('--lr', default=1e-1, type=float,
           help='learning rate')
    parser.add_argument('--na_rate', default=0, type=int,
           help='NA rate (NA = Q * na_rate)')
    parser.add_argument('--weight_decay', default=1e-5, type=float,
           help='weight decay')
    parser.add_argument('--grad_iter', default=1, type=int,
           help='grad iter')
    parser.add_argument('--optim', default='sgd',
           help='sgd / adam')
    parser.add_argument('--hidden_size', default=230, type=int,
           help='hidden size')
    parser.add_argument('--test_ckpt', default='',
           help='test ckpt')

    parser.add_argument('--only_test', action='store_true') 

    opt = parser.parse_args()
    trainN = opt.trainN
    N = opt.N
    K = opt.K
    Q = opt.Q
    batch_size = opt.batch_size
    model_name = opt.model
    encoder_name = opt.encoder
    max_length = opt.max_length
    
    print("{}-way-{}-shot Few-Shot Relation Classification".format(N, K))
    print("model: {}".format(model_name))
    print("encoder: {}".format(encoder_name))
    print("max_length: {}".format(max_length))
    
    if encoder_name == 'cnn':
        sentence_encoder = CNNSentenceEncoder(
                np.load('./data/glove_mat.npy'),
                json.load(open('./data/glove_word2id.json')),
                max_length)
    elif encoder_name == 'bert':
        sentence_encoder = BERTSentenceEncoder(
                './data/bert-base-uncased',
                max_length)
    else:
        raise NotImplementedError

    train_data_loader = get_loader(opt.train, sentence_encoder,
            N=trainN, K=K, Q=Q, na_rate=opt.na_rate, batch_size=batch_size)
    val_data_loader = get_loader(opt.val, sentence_encoder,
            N=N, K=K, Q=Q, na_rate=opt.na_rate, batch_size=batch_size)
    test_data_loader = get_loader(opt.test, sentence_encoder,
            N=N, K=K, Q=Q, na_rate=opt.na_rate, batch_size=batch_size)
    
    if opt.optim == 'sgd':
        optimizer = optim.SGD
    elif opt.optim == 'adam':
        optimizer = optim.Adam
    else:
        raise NotImplementedError
    framework = FewShotREFramework(train_data_loader, 
            val_data_loader, test_data_loader)
        
    prefix = '-'.join([model_name, encoder_name, opt.train, opt.val, 
        str(N), str(K)])
    if opt.na_rate != 0:
        prefix += '-na{}'.format(opt.na_rate)
    
    if model_name == 'proto':
        model = Proto(sentence_encoder, hidden_size=opt.hidden_size)
    elif model_name == 'proto_norm':
        model = ProtoNorm(sentence_encoder, hidden_size=opt.hidden_size)
    elif model_name == 'gnn':
        model = GNN(sentence_encoder, N)
    elif model_name == 'snail':
        print("HINT: SNAIL works only in PyTorch 0.3.1")
        model = SNAIL(sentence_encoder, N, K)
    elif model_name == 'metanet':
        model = MetaNet(N, K, train_data_loader.word_vec_mat, max_length)
    elif model_name == 'siamese':
        model = Siamese(sentence_encoder, hidden_size=opt.hidden_size)
    else:
        raise NotImplementedError

    if torch.cuda.is_available():
        model.cuda()
    
    if not opt.only_test:
        bert_optim = False
        if encoder_name == 'bert':
            bert_optim = True
        framework.train(model, prefix, batch_size, trainN, N, K, Q,
                optimizer=optimizer, pretrain_model=opt.pretrain, 
                bert_optim=bert_optim, na_rate=opt.na_rate, val_step=opt.val_step)

    # if model_name == 'proto':
    #     model = Proto(sentence_encoder)
    #     if torch.cuda.is_available():
    #         model.cuda()
    #     if not opt.only_test:
    #         framework.train(model, prefix, 4, 10, N, K, 5)
    # elif model_name == 'gnn':
    #     model = GNN(sentence_encoder, N)
    #     if torch.cuda.is_available():
    #         model.cuda()
    #     if not opt.only_test:
    #         framework.train(model, prefix, 2, N, N, K, 1, 
    #             learning_rate=1e-3, weight_decay=0, optimizer=optim.Adam)
    # elif model_name == 'snail':
    #     print("HINT: SNAIL works only in PyTorch 0.3.1")
    #     model = SNAIL(sentence_encoder, N, K)
    #     if torch.cuda.is_available():
    #         model.cuda()
    #     if not opt.only_test:
    #         framework.train(model, prefix, 25, N, N, K, 1, 
    #             learning_rate=1e-2, weight_decay=0, optimizer=optim.SGD)
    # elif model_name == 'metanet':
    #     model = MetaNet(N, K, train_data_loader.word_vec_mat, max_length)
    #     if torch.cuda.is_available():
    #         model.cuda()
    #     if not opt.only_test:
    #         framework.train(model, prefix, 1, N, N, K, 1, 
    #             learning_rate=5e-3, weight_decay=0, optimizer=optim.Adam, 
    #             train_iter=300000)
    # else:
    #     raise NotImplementedError
    
    # test

    # debug:

    # if len(opt.pretrain) > 0:
    #     state_dict = torch.load(opt.pretrain)['state_dict']
    #     own_state = model.state_dict()
    #     for name, param in state_dict.items():
    #         if name not in own_state:
    #             continue
    #         own_state[name].copy_(param)

    test_ckpt = 'checkpoint/{}.pth.tar'.format(prefix)
    if len(opt.test_ckpt) > 0:
        test_ckpt = opt.test_ckpt

    acc = 0
    his_acc = []
    total_test_round = 5
    for i in range(total_test_round):
        cur_acc = framework.eval(model, 4, N, K, Q, 3000, na_rate=opt.na_rate,
                ckpt=test_ckpt)
        his_acc.append(cur_acc)
        acc += cur_acc
    acc /= total_test_round

    result_file = open('./result.txt', 'a+')
    result_file.write("test data: %12s | model: %45s | acc: %.6f\n"
            % (opt.test, prefix, acc))

    result_file = open('./result_detail.txt', 'a+')
    result_detail = {'test': opt.test, 'model': prefix, 'acc': acc, 
            'his': his_acc}
    result_file.write("%s\n" % (json.dumps(result_detail)))
    
if __name__ == "__main__":
    main()
