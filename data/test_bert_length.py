from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import json
import os

tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased/bert_vocab.txt')

l = []
max_l = 0
ave_l = 0
total = 0

for root, dirs, files in os.walk('.'):
    for f in files:
        if f[-4:] == 'json' and 'glove' not in f and 'bert' not in f:
            data = json.load(open(f))
            for k, v in data.items():
                for item in v:
                    s = ' '.join(item['tokens'])
                    tokens = tokenizer.tokenize(s)
                    total += 1
                    ave_l += len(tokens) + 5
                    max_l = max(max_l, len(tokens) + 5)
                    l.append(len(tokens) + 5)

l.sort(reverse=True)
print(total)
print(l[:50])
print(float(ave_l) / total)
print(max_l)
