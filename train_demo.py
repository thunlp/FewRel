import models
from fewshot_re_kit.data_loader import JSONFileDataLoader
from fewshot_re_kit.framework import FewShotREFramework
from fewshot_re_kit.sentence_encoder import CNNSentenceEncoder 
from models.proto import Proto
from models.gnn import GNN
from models.snail import SNAIL
from models.metanet import MetaNet
import sys
from torch import optim
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default='train',
            help='train file')
    parser.add_argument('--val', default='val',
            help='val file')
    parser.add_argument('--test', default='test',
            help='test file')
    parser.add_argument('--N', default=5, type=int,
            help='N way')
    parser.add_argument('--K', default=5, type=int,
            help='K shot')
    parser.add_argument('--model', default='proto',
            help='model name')
    parser.add_argument('--encoder', default='cnn',
            help='encoder: cnn or bert')
    parser.add_argument_group('--max_length', default=120, type=int,
           help='model name')
   
    opt = parser.parse_args()
    N = opt.N
    K = opt.K
    model_name = opt.model_name
    max_length = opt.max_length
    
    print("{}-way-{}-shot Few-Shot Relation Classification".format(N, K))
    print("model: {}".format(model_name))
    print("max_length: {}".format(max_length))
    
    train_data_loader = JSONFileDataLoader('./data/{}.json'.format(opt.train),
            './data/glove.6B.50d.json', 
            max_length=max_length)
    val_data_loader = JSONFileDataLoader('./data/{}.json'.format(opt.val),
            './data/glove.6B.50d.json', 
            max_length=max_length)
    test_data_loader = JSONFileDataLoader('./data/{}.json'.format(opt.test), 
            './data/glove.6B.50d.json', 
            max_length=max_length)
    
    framework = FewShotREFramework(train_data_loader, 
            val_data_loader, test_data_loader)
    sentence_encoder = CNNSentenceEncoder(train_data_loader.word_vec_mat, 
            max_length)
    
    if model_name == 'proto':
        model = Proto(sentence_encoder)
        framework.train(model, model_name, 4, 10, N, K, 5)
    elif model_name == 'gnn':
        model = GNN(sentence_encoder, N)
        framework.train(model, model_name, 2, N, N, K, 1, 
                learning_rate=1e-3, weight_decay=0, optimizer=optim.Adam)
    elif model_name == 'snail':
        print("HINT: SNAIL works only in PyTorch 0.3.1")
        model = SNAIL(sentence_encoder, N, K)
        framework.train(model, model_name, 25, N, N, K, 1, 
                learning_rate=1e-2, weight_decay=0, optimizer=optim.SGD)
    elif model_name == 'metanet':
        model = MetaNet(N, K, train_data_loader.word_vec_mat, max_length)
        framework.train(model, model_name, 1, N, N, K, 1, 
                learning_rate=5e-3, weight_decay=0, optimizer=optim.Adam, 
                train_iter=300000)
    else:
        raise NotImplementedError
    
if __name__ == "__main__":
    main()
