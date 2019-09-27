from fewshot_re_kit.data_loader import get_loader, get_loader_pair
from fewshot_re_kit.framework import FewShotREFramework
from fewshot_re_kit.sentence_encoder import CNNSentenceEncoder, BERTSentenceEncoder, BERTPAIRSentenceEncoder
import models
from models.proto import Proto
from models.gnn import GNN
from models.snail import SNAIL
from models.metanet import MetaNet
from models.siamese import Siamese
from models.pair import Pair
import sys
import torch
from torch import optim, nn
import numpy as np
import json
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default='train_wiki',
            help='train file')
    parser.add_argument('--val', default='val_wiki',
            help='val file')
    parser.add_argument('--test', default='test_wiki',
            help='test file')
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
    parser.add_argument('--train_iter', default=30000, type=int,
            help='num of epochs training')
    parser.add_argument('--val_iter', default=1000, type=int,
            help='num of epochs training')
    parser.add_argument('--val_step', default=2000, type=int,
           help='val after training how many iters')
    parser.add_argument('--model', default='proto',
            help='model name')
    parser.add_argument('--encoder', default='cnn',
            help='encoder: cnn or bert')
    parser.add_argument('--max_length', default=128, type=int,
           help='max length')
    parser.add_argument('--lr', default=1e-1, type=float,
           help='learning rate')
    parser.add_argument('--weight_decay', default=1e-5, type=float,
           help='weight decay')
    parser.add_argument('--dropout', default=0.0, type=float,
           help='dropout rate')
    parser.add_argument('--na_rate', default=0, type=int,
           help='NA rate (NA = Q * na_rate)')
    parser.add_argument('--grad_iter', default=1, type=int,
           help='accumulate gradient every x iterations')
    parser.add_argument('--optim', default='sgd',
           help='sgd / adam / bert_adam')
    parser.add_argument('--hidden_size', default=230, type=int,
           help='hidden size')
    parser.add_argument('--load_ckpt', default=None,
           help='load ckpt')
    parser.add_argument('--save_ckpt', default=None,
           help='save ckpt')
    parser.add_argument('--fp16', action='store_true',
           help='use nvidia apex fp16')
    parser.add_argument('--only_test', action='store_true',
           help='only test')
    parser.add_argument('--pair', action='store_true',
           help='use pair model')

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
        try:
            glove_mat = np.load('./pretrain/glove/glove_mat.npy')
            glove_word2id = json.load(open('./pretrain/glove/glove_word2id.json'))
        except:
            raise Exception("Cannot find glove files. Run glove/download_glove.sh to download glove files.")
        sentence_encoder = CNNSentenceEncoder(
                glove_mat,
                glove_word2id,
                max_length)
    elif encoder_name == 'bert':
        if opt.pair:
            sentence_encoder = BERTPAIRSentenceEncoder(
                    './pretrain/bert-base-uncased',
                    max_length)
        else:
            sentence_encoder = BERTSentenceEncoder(
                    './pretrain/bert-base-uncased',
                    max_length)
    else:
        raise NotImplementedError
    
    if opt.pair:
        train_data_loader = get_loader_pair(opt.train, sentence_encoder,
                N=trainN, K=K, Q=Q, na_rate=opt.na_rate, batch_size=batch_size)
        val_data_loader = get_loader_pair(opt.val, sentence_encoder,
                N=N, K=K, Q=Q, na_rate=opt.na_rate, batch_size=batch_size)
        test_data_loader = get_loader_pair(opt.test, sentence_encoder,
                N=N, K=K, Q=Q, na_rate=opt.na_rate, batch_size=batch_size)
    else:
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
    elif opt.optim == 'bert_adam':
        from pytorch_transformers import AdamW
        optimizer = AdamW
    else:
        raise NotImplementedError
    framework = FewShotREFramework(train_data_loader, val_data_loader, test_data_loader)
        
    prefix = '-'.join([model_name, encoder_name, opt.train, opt.val, str(N), str(K)])
    if opt.na_rate != 0:
        prefix += '-na{}'.format(opt.na_rate)
    
    if model_name == 'proto':
        model = Proto(sentence_encoder, hidden_size=opt.hidden_size)
    elif model_name == 'gnn':
        model = GNN(sentence_encoder, N)
    elif model_name == 'snail':
        print("HINT: SNAIL works only in PyTorch 0.3.1")
        model = SNAIL(sentence_encoder, N, K)
    elif model_name == 'metanet':
        model = MetaNet(N, K, sentence_encoder.embedding, max_length)
    elif model_name == 'siamese':
        model = Siamese(sentence_encoder, hidden_size=opt.hidden_size, dropout=opt.dropout)
    elif model_name == 'pair':
        model = Pair(sentence_encoder, hidden_size=opt.hidden_size)
    else:
        raise NotImplementedError
    
    if not os.path.exists('checkpoint'):
        os.mkdir('checkpoint')
    ckpt = 'checkpoint/{}.pth.tar'.format(prefix)
    if opt.save_ckpt:
        ckpt = opt.save_ckpt

    if torch.cuda.is_available():
        model.cuda()

    if not opt.only_test:
        if encoder_name == 'bert':
            bert_optim = True
        else:
            bert_optim = False

        framework.train(model, prefix, batch_size, trainN, N, K, Q,
                optimizer=optimizer, load_ckpt=opt.load_ckpt, save_ckpt=ckpt,
                na_rate=opt.na_rate, val_step=opt.val_step, fp16=opt.fp16, pair=opt.pair, 
                train_iter=opt.train_iter, val_iter=opt.val_iter, bert_optim=bert_optim)
    else:
        ckpt = opt.load_ckpt

    acc = 0
    his_acc = []
    total_test_round = 5
    for i in range(total_test_round):
        cur_acc = framework.eval(model, batch_size, N, K, Q, 3000, na_rate=opt.na_rate,
                ckpt=ckpt, pair=True)
        his_acc.append(cur_acc)
        acc += cur_acc
    acc /= total_test_round
    nhis_acc = np.array(his_acc)
    error = nhis_acc.std() * 1.96 / (nhis_acc.shape[0] ** 0.5)
    print("RESULT: %.2f\\pm%.2f" % (acc * 100, error * 100))

    result_file = open('./result.txt', 'a+')
    result_file.write("test data: %12s | model: %45s | acc: %.6f\n | error: %.6f\n"
            % (opt.test, prefix, acc, error))

    result_file = open('./result_detail.txt', 'a+')
    result_detail = {'test': opt.test, 'model': prefix, 'acc': acc, 
            'his': his_acc}
    result_file.write("%s\n" % (json.dumps(result_detail)))
    
if __name__ == "__main__":
    main()
