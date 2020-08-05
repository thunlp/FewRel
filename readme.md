# FewRel Dataset, Toolkits and Baseline Models

Our benchmark website: [https://thunlp.github.io/fewrel.html](https://thunlp.github.io/fewrel.html)

FewRel is a large-scale few-shot relation extraction dataset, which contains more than one hundred relations and tens of thousands of annotated instances cross different domains. Our dataset is presented in our EMNLP 2018 paper [FewRel: A Large-Scale Few-Shot Relation Classification Dataset with State-of-the-Art Evaluation](https://www.aclweb.org/anthology/D18-1514.pdf) and a following-up version is presented in our EMNLP 2019 paper [FewRel 2.0: Towards More Challenging Few-Shot Relation Classification](https://www.aclweb.org/anthology/D19-1649.pdf).

Based on our dataset and designed few-shot settings, we have two different benchmarks:

* FewRel 1.0: This is the first one to incorporate few-shot learning with relation extraction, where your model need to handle both the few-shot challenge and extracting entity relations from plain text. This benchmark provides a training dataset with 64 relations and a validation set with 16 relations. Once you submit your code to our [benchmark website](https://thunlp.github.io/1/fewrel1.html), it will be evaluated on a hidden test set with 20 relations. Each relation has 100 human-annotated instances. 

* FewRel 2.0: We found out that there are two long-neglected aspects in previous few-shot research: (1) How well models can transfer across different domains. (2) Can few-shot models detect instances belonging to none of the given few-shot classes. To dig deeper in these two aspects, we propose the 2.0 version of our dataset, with newly-added **domain adaptation (DA)** and **none-of-the-above (NOTA) detection** challenges. Find our more in our [paper](https://www.aclweb.org/anthology/D19-1649.pdf) and evaluation websites [FewRel 2.0 domain adaptation](https://thunlp.github.io/2/fewrel2_da.html) / [FewRel 2.0 none-of-the-above detection](https://thunlp.github.io/2/fewrel2_nota.html)

## Citing
If you used our data, toolkits or baseline models, please kindly cite our paper:
```
@inproceedings{han-etal-2018-fewrel,
    title = "{F}ew{R}el: A Large-Scale Supervised Few-Shot Relation Classification Dataset with State-of-the-Art Evaluation",
    author = "Han, Xu and Zhu, Hao and Yu, Pengfei and Wang, Ziyun and Yao, Yuan and Liu, Zhiyuan and Sun, Maosong",
    booktitle = "Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing",
    month = oct # "-" # nov,
    year = "2018",
    address = "Brussels, Belgium",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/D18-1514",
    doi = "10.18653/v1/D18-1514",
    pages = "4803--4809"
}

@inproceedings{gao-etal-2019-fewrel,
    title = "{F}ew{R}el 2.0: Towards More Challenging Few-Shot Relation Classification",
    author = "Gao, Tianyu and Han, Xu and Zhu, Hao and Liu, Zhiyuan and Li, Peng and Sun, Maosong and Zhou, Jie",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/D19-1649",
    doi = "10.18653/v1/D19-1649",
    pages = "6251--6256"
}
```

If you have questions about any part of the paper, submission, leaderboard, codes, data, please e-mail `gaotianyu1350@126.com`.

## Contributions

For FewRel 1.0, Hao Zhu first proposed this problem and proposed the way to build the dataset and the baseline system; Ziyuan Wang built and maintained the crowdsourcing website; Yuan Yao helped download the original data and conducted preprocess; 
Xu Han, Hao Zhu, Pengfei Yu and Ziyun Wang implemented baselines and wrote the paper together; Zhiyuan Liu provided thoughtful advice and funds through the whole project. The order of the first four authors are determined by dice rolling. 

## Dataset and Pretrain files

The dataset has already be contained in the github repo. However, due to the large size, glove files (pre-trained word embeddings) and BERT pretrain checkpoint are not included. Please download `pretrain.tar` from [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/58f57bda00eb40be8d10/?dl=1) and put it under the root. Then run `tar xvf pretrain.tar` to decompress it.

We also provide [pid2name.json](https://github.com/thunlp/FewRel/blob/master/data/pid2name.json) to show the Wikidata PID, name and description for each relation. 

**Note: We did not release the test dataset for both FewRel 1.0 and 2.0 for fair comparison. We recommend you to evaluate your models on the validation set first, and then submit it to our evaluation websites (which you can find above).** 

## Training a Model

To run our baseline models, use command

```bash
python train_demo.py
```

This will start the training and evaluating process of Prototypical Networks in a 5-way 5-shot setting. You can also use different args to start different process. Some of them are here:

* `train / val / test`: Specify the training / validation / test set. For example, if you use `train_wiki` for `train`, the program will load `data/train_wiki.json` for training. You should always use `train_wiki` for training and `val_wiki` (FewRel 1.0 and FewRel 2.0 NOTA challenge) or `val_pubmed` (FewRel 2.0 DA challenge) for validation.
* `trainN`: N in N-way K-shot. `trainN` is the specific N in training process.
* `N`: N in N-way K-shot.
* `K`: K in N-way K-shot.
* `Q`: Sample Q query instances for each relation.
* `model`: Which model to use. The default one is `proto`, standing for Prototypical Networks. Note that if you use the **PAIR** model from our paper [FewRel 2.0](https://www.aclweb.org/anthology/D19-1649.pdf), you should also use `--encoder bert --pair`.
* `encoder`: Which encoder to use. You can choose `cnn` or `bert`. 
* `na_rate`: NA rate for FewRel 2.0 none-of-the-above (NOTA) detection. Note that here `na_rate` specifies the rate between Q for NOTA and Q for positive. For example, `na_rate=0` means the normal setting, `na_rate=1,2,5` corresponds to NA rate = 15%, 30% and 50% in 5-way settings.

There are also many args for training (like `batch_size` and `lr`) and you can find more details in our codes.

## Inference

You can evaluate an existing checkpoint by

```bash
python train_demo.py --only_test --load_ckpt {CHECKPOINT_PATH} {OTHER_ARGS}
```

Here we provide a BERT-PAIR [checkpoint](https://thunlp.oss-cn-qingdao.aliyuncs.com/fewrel/pair-bert-train_wiki-val_wiki-5-1.pth.tar) (trained on FewRel 1.0 dataset, 5 way 1 shot).

## Reproduction

**BERT-PAIR for FewRel 1.0**

```bash
python train_demo.py \
    --trainN 5 --N 5 --K 1 --Q 1 \
    --model pair --encoder bert --pair --hidden_size 768 --val_step 1000 \
    --batch_size 4  --fp16 \
```

Note that `--fp16` requires Nvidia's [apex](https://github.com/NVIDIA/apex).

|                   | 5 way 1 shot | 5 way 5 shot | 10 way 1 shot | 10 way 5 shot |
|  ---------------  | -----------  | ------------- | ------------ | ------------- |
| Val               | 85.66 | 89.48 | 76.84 | 81.76 |
| Test              | 88.32 | 93.22 | 80.63 | 87.02 |

**BERT-PAIR for Domain Adaptation (FewRel 2.0)**

```bash
python train_demo.py \
    --trainN 5 --N 5 --K 1 --Q 1 \
    --model pair --encoder bert --pair --hidden_size 768 --val_step 1000 \
    --batch_size 4  --fp16 --val val_pubmed --test val_pubmed \
```

|                   | 5 way 1 shot | 5 way 5 shot | 10 way 1 shot | 10 way 5 shot |
|  ---------------  | -----------  | ------------- | ------------ | ------------- |
| Val               | 70.70 | 80.59 | 59.52 | 70.30 |
| Test              | 67.41	| 78.57 | 54.89	| 66.85 |

**BERT-PAIR for None-of-the-Above (FewRel 2.0)**

```bash
python train_demo.py \
    --trainN 5 --N 5 --K 1 --Q 1 \
    --model pair --encoder bert --pair --hidden_size 768 --val_step 1000 \
    --batch_size 4  --fp16 --na_rate 5 \
```

|                   | 5 way 1 shot (0% NOTA) | 5 way 1 shot (50% NOTA) | 5 way 5 shot (0% NOTA) | 5 way 5 shot (50% NOTA) |
|  ---------------  | -----------  | ------------- | ------------ | ------------- |
| Val               | 74.56        | 73.09         | 75.01        | 75.38         |
| Test              | 76.73        | 80.31         | 83.32        | 84.64         |

**Proto-CNN + Adversarial Training for Domain Adaptation (FewRel 2.0)**

```bash
python train_demo.py \
    --val val_pubmed --adv pubmed_unsupervised --trainN 10 --N {} --K {} \ 
    --model proto --encoder cnn --val_step 1000 \
```

|                   | 5 way 1 shot | 5 way 5 shot | 10 way 1 shot | 10 way 5 shot |
|  ---------------  | -----------  | ------------- | ------------ | ------------- |
| Val               | 48.73 | 64.38 | 34.82 | 50.39 |
| Test              | 42.21 | 58.71 | 28.91 | 44.35 |
