import json
import numpy as np
glove = json.load(open('./glove.6B.50d.json'))

word2id = {}
word_vec_mat = []

idx = 0
sum_vec = np.zeros((50), dtype=np.float32)
for item in glove:
    word = item['word']
    vec = np.array(item['vec'])
    sum_vec += vec
    word_vec_mat.append(vec)
    word2id[word] = idx
    idx += 1

sum_vec = sum_vec / float(idx)
word_vec_mat.append(sum_vec)
word2id['[UNK]'] = idx
idx += 1

word_vec_mat.append(np.zeros((50), dtype=np.float32))
word2id['[PAD]'] = idx
idx += 1

word_vec_mat = np.stack(word_vec_mat, 0)
json.dump(word2id, open('glove_word2id.json', 'w'))
np.save('glove_mat.npy', word_vec_mat)
