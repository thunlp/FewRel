<script type="text/javascript" charset="utf-8" src="
https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML,
https://vincenttam.github.io/javascripts/MathJaxLocal.js"></script>


# FewRel: Few-Shot Relation Classification Dataset

[Xu Han](https://thucsthanxu13.github.io)\*, [Hao Zhu](http://zhuhao.me)\*, Pengfei Yu\*, Ziyuan Wang\*, Yuan Yao, [Zhiyuan Liu](http://nlp.csai.tsinghua.edu.cn/~lzy/), Maosong Sun (* the first four authors contribute equally.)

EMNLP 2018 

[[paper]](https://github.com/thunlp/FewRel/blob/master/paper/fewrel.pdf) [[dataset]](#datasets) [[baseline code]](#baseline)


## Introduction

We present a Few-Shot Relation Classification Dataset FewRel, consisting of 70, 000 sentences on 100 relations derived from Wikipedia and annotated by crowdworkers. The relation of each sentence is first recognized by distant supervision methods, and then filtered by crowdworkers. We adapt the most recent state-of-the-art few-shot learning methods for relation classification and conduct thorough evaluation of these methods. Empirical results show that even the most competitive few-shot learning models struggle on this task, especially as compared with humans. We also show that a range of different reasoning skills are needed to solve our task. These results indicate that few-shot relation classification remains an open problem and still requires further research. Our detailed analysis points multiple directions for future research.



## Task


In few-shot relation classification, we intend to obtain a function $$F:(\mathcal{R}, \mathcal{S}, x)\mapsto y,$$ 
Here \\(\mathcal{R}=\{r_1, \ldots,r_m\}\\)defines the relations that the instances are classified into. 
\\(\mathcal{S}\\) is a support set including \\(n_i\\) instances for each relation \\(r_i\in\mathcal{R}\\).
$$
\begin{equation}
\begin{aligned}
\mathcal{S}= \\{(x_1^1, r_1), &(x_1^2, r_1), \ldots,(x_1^{n_1}, r_1),\\\\
&\ldots,\\\\
(x_m^1, r_m), &(x_m^2, r_m), \ldots,(x_m^{n_m}, r_m)
\\}, \\\\
r_1, & \ldots, r_m\ \in \mathcal{R} 
\end{aligned}
\end{equation}
$$
Each instance \\(x_i^j\\) is a sentence accompanied with a pair of entities. The query data \\(x\\) is an unlabeled instance to classify, and \\(y\in\mathcal{R}\\) is the prediction of \\(x\\) given by \\(F\\).

In recent research on few-shot learning, \\(N\\) way \\(K\\) shot setting is widely adopted. We follow this setting for the few-shot relation classification problem. To be exact, for \\(N\\) way \\(K\\) shot learning, there are 
$$
N=m=|\mathcal{R}|, K=n_1=\ldots=n_m
$$

Here we give an example (3 way 2 shot) in the form of the following table. This task requires the models to learn parameters based on the supporting set, and then predict the relation of the query instance.


<table style="width:80%;">
        <tr>
            <th colspan ="2" style="text-align:center;">Supporting Set</th>
        </tr>
		<tr >
            <th style="text-align:center;width:10%">#REL</th>
            <th style="text-align:center;width:40%">#INS</th>
        </tr>
        <tr>
            <th rowspan="2" style="text-align:center;vertical-align:middle;">I. capital_of</th>
            <th align="left"> (1) <font style="color:blue;font-weight:bold;font-style:italic;">London</font> is the capital of <font style="color:red;font-weight:bold;font-style:italic;">the U.K</font>. </th>
        </tr>
        <tr>
            <th align="left"> (2) <font style="color:blue;font-weight:bold;font-style:italic;">Washington</font> is the capital of <font style="color:red;font-weight:bold;font-style:italic;">the U.S.A</font>. </th>
        </tr>
        <tr align="left">
            <th rowspan="2" style="text-align:center;vertical-align:middle;"> II. member_of</th>
            <th> (1) <font style="color:blue;font-weight:bold;font-style:italic;">Newton</font> served as the president of <font style="color:red;font-weight:bold;font-style:italic;">the Royal Society</font>. </th>
        </tr>
        <tr align="left">
            <th> (2) <font style="color:blue;font-weight:bold;font-style:italic;">Leibniz</font> was a member of <font style="color:red;font-weight:bold;font-style:italic;">the Prussian Academy of Sciences</font>. </th>
        </tr>
        <tr align="left">
            <th rowspan="2" style="text-align:center;vertical-align:middle;"> III. birth_name</th>
            <th>  (1) <font style="color:blue;font-weight:bold;font-style:italic;">Samuel Langhorne Clemens</font>, better known by his pen name <font style="color:red;font-weight:bold;font-style:italic;">Mark Twain</font>, was an American writer. </th>
        </tr>
        <tr align="left">
                    <th> (2) <font style="color:blue;font-weight:bold;font-style:italic;">Alexei Maximovich Peshkov</font>, primarily known as <font style="color:red;font-weight:bold;font-style:italic;">Maxim Gorky</font>, was a Russian and Soviet writer. </th>            
        </tr>                
    </table>
    
<table  style="width:80%">
        <tr>
            <th colspan ="2" style="text-align:center;">Test Set</th>
        </tr>
        <tr>
            <th style="text-align:center;width:10%"> #ANS</th>
            <th style="text-align:center;width:40%"> #QUERY</th>
        </tr>
        <tr>
            <th style="text-align:center;vertical-align:middle;"> II. member_of</th>
            <th> &nbsp;&nbsp;&nbsp;&nbsp; <font style="color:blue;font-weight:bold;font-style:italic;">Euler</font> was elected a foreign member of <font style="color:red;font-weight:bold;font-style:italic;">the Royal Swedish Academy of Sciences</font>.
            </th>
        </tr>
</table>


## Datasets and Downloads


*	Download a copy of the datasets (distributed under the MIT license):


*	[[Training Set]] (https://github.com/thunlp/FewRel/tree/master/data)

*	[[Validation Set]] (https://github.com/thunlp/FewRel/tree/master/data)


*	Download a copy of the baseline codes (distributed under the MIT license):


*	[[Baselines]] (https://github.com/thunlp/FewRel/tree/master/baseline)

## Submit your models


*	Once you have a built a model that works to your expectations on the dev set, you submit it to get official scores on the dev and a hidden test set. To preserve the integrity of test results, we do not release the test set to the public. Instead, we require you to submit your model so that we can run it on the test set for you. Submission methods coming soon!


*	If you have questions about any part of the paper, submission, leaderboard, codes, data, please e-mail zhuhao15@mails.tsinghua.edu.cn.


## Citation


If you use the code, please cite the following paper:


```
@inproceedings{hao2018fewrel,
               title={FewRel:A Large-Scale Supervised Few-shot Relation Classification Dataset with State-of-the-Art Evaluation},
               author={Han, Xu and Zhu, Hao and Yu, Pengfei and Wang, Ziyun and Yao, Yuan and Liu, Zhiyuan and Sun, Maosong},
               booktitle={EMNLP},
               year={2018}
}
```



## Contributions


Hao Zhu first proposed this problem and proposed the way to build the dataset and the baseline system; Ziyuan Wang built and maintained the crowdsourcing website; Yuan Yao helped download the original data and conducted preprocess; 
Xu Han, Hao Zhu, Pengfei Yu and Ziyun Wang implemented baselines and wrote the paper together; Zhiyuan Liu provided thoughtful advice and funds through the whole project. The order of the first four authors are determined by dice rolling. 




