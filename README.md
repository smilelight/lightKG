# lightKG，lightsmile个人的知识图谱技术框架

## 前言

根据知识图谱发展报告2018相关介绍，框架主要设计为有以下五大功能：

- 知识表示学习， Knowledge Representation Learning
- 实体识别与链接， Entity Recognition and Linking
- 实体关系抽取， Entity Relation Extraction
- 事件检测与抽取， Event Detection and Extraction
- 知识存储与查询， Knowledge Storage and Query
- 知识推理， Knowledge Reasoning

因此将有六个主要的功能模块：krl（知识表示学习）、erl（实体识别与链接）、ere（实体关系抽取）、ede（实体检测与抽取）、ksq（知识存储与查询）、kr（知识推理）以及其他功能模块。

## 当前已实现的功能

### 知识表示学习

- 基于翻译模型(Trans系列)的知识表示学习， TransE

### 实体识别与链接

- 命名实体识别， ner

### 实体关系抽取

- 关系抽取， re

### 事件检测与抽取

- 语义角色标注， srl

### 知识存储与查询

### 知识推理

## 安装

本项目基于Pytorch1.0

```bash
pip install lightKG
```

建议使用国内源来安装，如使用以下命令：
```bash
pip install -i https://pypi.douban.com/simple/ lightKG
```

### 安装依赖

由于有些库如pytorch、torchtext并不在pypi源中或者里面只有比较老旧的版本，我们需要单独安装一些库。
#### 安装pytorch

具体安装参见[pytorch官网](https://pytorch.org/get-started/locally/)来根据平台、安装方式、Python版本、CUDA版本来选择适合自己的版本。

#### 安装torchtext

使用以下命令安装最新版本torchtext：
```bash
pip install https://github.com/pytorch/text/archive/master.zip
```

## 模型

- krl：TransE等
- re: TextCNN
- srl: BiLstm-CRF
- ner: BiLstm-CRF

## 训练数据说明

#### krl

csv格式
 
共三列，依次为`头实体`、`关系`、`尾实体`， 示例如下：
 
 ```bash
科学,包涵,自然、社会、思维等领域
科学,外文名,science
科学,拼音,kē xué
科学,中文名,科学
科学,解释,发现、积累的真理的运用与实践
语法学,外文名,syntactics
语法学,中文名,语法学
物理宇宙学,对象,大尺度结构和宇宙形成
物理宇宙学,时间,二十世纪
物理宇宙学,所属,天体物理学
 ```
 
 #### ner

BIO

训练数据示例如下：

```bash
清 B_Time
明 I_Time
是 O
人 B_Person
们 I_Person
祭 O
扫 O
先 B_Person
人 I_Person
， O
怀 O
念 O
追 O
思 O
的 O
日 B_Time
子 I_Time
。 O

正 O
如 O
宋 B_Time
代 I_Time
诗 B_Person
人 I_Person
```

#### srl

CONLL

训练数据示例如下，其中各列分别为`词`、`词性`、`是否语义谓词`、`角色`，每句仅有一个谓语动词为语义谓词，即每句中第三列仅有一行取值为1，其余都为0.

```bash
宋浩京  NR      0       O
转达    VV      0       O
了      AS      0       O
朝鲜    NR      0       O
领导人  NN      0       O
对      P       0       O
中国    NR      0       O
领导人  NN      0       O
的      DEG     0       O
亲切    JJ      0       O
问候    NN      0       O
，      PU      0       O
代表    VV      0       O
朝方    NN      0       O
对      P       0       O
中国    NR      0       B-ARG0
党政    NN      0       I-ARG0
领导人  NN      0       I-ARG0
和      CC      0       I-ARG0
人民    NN      0       E-ARG0
哀悼    VV      1       rel
金日成  NR      0       B-ARG1
主席    NN      0       I-ARG1
逝世    VV      0       E-ARG1
表示    VV      0       O
深切    JJ      0       O
谢意    NN      0       O
。      PU      0       O
```

#### re

训练数据示例如下，其中各列分别为`实体1`、`实体2`、`关系`、`句子`

```bash
钱钟书	辛笛	同门	与辛笛京沪唱和聽钱钟书与钱钟书是清华校友，钱钟书高辛笛两班。
元武	元华	unknown	于师傅在一次京剧表演中，选了元龙（洪金宝）、元楼（元奎）、元彪、成龙、元华、元武、元泰7人担任七小福的主角。
```

 ## 使用
 
 ### krl

#### 训练

```python
from lightkg.krl import KRL

train_path = '/home/lightsmile/NLP/corpus/kg/baike/train.sample.csv'
dev_path = '/home/lightsmile/NLP/corpus/kg/baike/test.sample.csv'
model_type = 'TransE'

krl = KRL()
krl.train(train_path, model_type=model_type, dev_path=train_path, save_path='./krl_{}_saves'.format(model_type))
```

#### 测试

```python
krl.load(save_path='./krl_{}_saves'.format(model_type), model_type=model_type)
krl.test(train_path)
```

#### 预测

##### 根据头实体、关系、尾实体，预测其概率

```python
print(krl.predict(head='编译器', rel='外文名', tail='Compiler'))
```

输出为：
```bash
0.998942494392395
```
##### 根据头实体和关系，预测训练集词表中topk(默认为3)个可能尾实体

```python
print(krl.predict_tail(head='编译器', rel='外文名'))
```

输出为：
```bash
[('Compiler', 0.998942494392395), ('20世纪50年代末', 0.3786872327327728), ('译码器', 0.3767447769641876)]
```
##### 根据头实体和尾实体，预测训练集词表中topk(默认为3)个可能关系

```python
print(krl.predict_rel(head='编译器', tail='Compiler'))
```

输出为：
```bash
[('外文名', 0.998942494392395), ('英译', 0.8240533471107483), ('拼音', 0.4082326292991638)]
```
##### 根据尾实体和关系，预测训练集词表中topk(默认为3)个可能头实体
```python
print(krl.predict_head(rel='外文名', tail='Compiler'))
```

输出为：
```bash
[('编译器', 0.998942494392395), ('译码器', 0.36795616149902344), ('计算机，单片机，编程语言', 0.36788302659988403)]
```

### ner

#### 训练

```python
from lightkg.erl import NER

# 创建NER对象
ner_model = NER()

train_path = '/home/lightsmile/NLP/corpus/ner/train.sample.txt'
dev_path = '/home/lightsmile/NLP/corpus/ner/test.sample.txt'
vec_path = '/home/lightsmile/NLP/embedding/char/token_vec_300.bin'

# 只需指定训练数据路径，预训练字向量可选，开发集路径可选，模型保存路径可选。
ner_model.train(train_path, vectors_path=vec_path, dev_path=dev_path, save_path='./ner_saves')
```

#### 测试

```python
# 加载模型，默认当前目录下的`saves`目录
ner_model.load('./ner_saves')
# 对train_path下的测试集进行读取测试
ner_model.test(train_path)
```

#### 预测

```python
from pprint import pprint

pprint(ner_model.predict('另一个很酷的事情是，通过框架我们可以停止并在稍后恢复训练。'))
```

预测结果：

```bash
[{'end': 15, 'entity': '我们', 'start': 14, 'type': 'Person'}]
```

### re

#### 训练

```python
from lightkg.ere import RE

re = RE()

train_path = '/home/lightsmile/Projects/NLP/ChineseNRE/data/people-relation/train.sample.txt'
dev_path = '/home/lightsmile/Projects/NLP/ChineseNRE/data/people-relation/test.sample.txt'
vec_path = '/home/lightsmile/NLP/embedding/word/sgns.zhihu.bigram-char'

re.train(train_path, dev_path=dev_path, vectors_path=vec_path, save_path='./re_saves')

```

#### 测试

```python
re.load('./re_saves')
re.test(dev_path)
```

#### 预测

```python
print(re.predict('钱钟书', '辛笛', '与辛笛京沪唱和聽钱钟书与钱钟书是清华校友，钱钟书高辛笛两班。'))
```

预测结果：

```python
(0.7306928038597107, '同门') # return格式为（预测概率，预测标签）
```

### srl

#### 训练

```python
from lightkg.ede import SRL

srl_model = SRL()

train_path = '/home/lightsmile/NLP/corpus/srl/train.sample.tsv'
dev_path = '/home/lightsmile/NLP/corpus/srl/test.sample.tsv'
vec_path = '/home/lightsmile/NLP/embedding/word/sgns.zhihu.bigram-char'


srl_model.train(train_path, vectors_path=vec_path, dev_path=dev_path, save_path='./srl_saves')
```

#### 测试

```python
srl_model.load('./srl_saves')

srl_model.test(dev_path)
```

#### 预测

```python
word_list = ['代表', '朝方', '对', '中国', '党政', '领导人', '和', '人民', '哀悼', '金日成', '主席', '逝世', '表示', '深切', '谢意', '。']
pos_list = ['VV', 'NN', 'P', 'NR', 'NN', 'NN', 'CC', 'NN', 'VV', 'NR', 'NN', 'VV', 'VV', 'JJ', 'NN', 'PU']
rel_list = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

print(srl_model.predict(word_list, pos_list, rel_list))
```

预测结果：

```bash
{'ARG0': '中国党政领导人和人民', 'rel': '哀悼', 'ARG1': '金日成主席逝世'}
```

## 项目组织结构

### 项目架构
- base
    - config.py
    - model.py
    - module.py
    - tool.py
- common
    - entity.py
    - relation.py
- ede
    - srl, 语义角色标注
- ere
    - re， 关系抽取
- erl
    - ner， 命名实体识别
- kr
- krl，知识表示学习
    - models
        - transE
    - utils
- ksq
- utils

### 架构说明

#### base目录
放一些基础的模块实现，其他的高层业务模型以及相关训练代码都从此module继承相应父类。

##### config
存放模型训练相关的超参数等配置信息

##### model
模型的实现抽象基类，包含`base.model.BaseConfig`和`base.model.BaseModel`，包含`load`、`save`等方法

##### module
业务模块的训练验证测试等实现抽象基类，包含`base.module.Module`，包含`train`、`load`、`_validate`、`test`等方法

##### tool
业务模块的数据处理抽象基类，包含`base.tool.Tool`，包含`get_dataset`、`get_vectors`、`get_vocab`、`get_iterator`、`get_score`等方法

#### common目录

##### entity

实体基类, 所有需要使用实体对象的使用此类或从此类继承子类

##### relation

关系基类, 所有需要使用关系对象的使用此类或从此类继承子类

#### util目录
放一些通用的方法

## todo

### 业务

### 工程

- [ ] 增加断点重训功能。
- [ ] 增加earlyStopping。
- [x] 重构项目结构，将相同冗余的地方合并起来，保持项目结构清晰
- [ ] 现在模型保存的路径和名字默认一致，会冲突，接下来每个模型都有自己的`name`。

### 功能

- [x] 增加关系抽取相关模型以及训练预测代码
- [x] 增加事件抽取相关模型以及训练预测代码
- [x] 增加命名实体识别相关模型以及预测训练代码
- [x] 增加基于翻译模型的知识表示学习相关模型以及训练预测代码

## 参考

### Deep Learning

- [What's the difference between “hidden” and “output” in PyTorch LSTM?](https://stackoverflow.com/questions/48302810/whats-the-difference-between-hidden-and-output-in-pytorch-lstm)
- [What's the difference between LSTM() and LSTMCell()?](https://stackoverflow.com/questions/48187283/whats-the-difference-between-lstm-and-lstmcell)
- [深度学习框架技术剖析[转]](https://aiuai.cn/aifarm904.html)

### NLP

- [基于表示学习的信息抽取方法浅析](https://www.jiqizhixin.com/articles/2016-11-15-5)
- [知识抽取-实体及关系抽取](http://www.shuang0420.com/2018/09/15/%E7%9F%A5%E8%AF%86%E6%8A%BD%E5%8F%96-%E5%AE%9E%E4%BD%93%E5%8F%8A%E5%85%B3%E7%B3%BB%E6%8A%BD%E5%8F%96/)
- [知识抽取-事件抽取](http://www.shuang0420.com/2018/10/15/%E7%9F%A5%E8%AF%86%E6%8A%BD%E5%8F%96-%E4%BA%8B%E4%BB%B6%E6%8A%BD%E5%8F%96/)

### 知识图谱

- [翻译模型(Trans系列)的知识表示学习](https://mp.weixin.qq.com/s/STflo3c8nyG6iHh9dEeKOQ)
- [知识图谱向量化表示](https://zhuanlan.zhihu.com/p/30320631)

### Pytorch教程

- [PyTorch 常用方法总结4：张量维度操作（拼接、维度扩展、压缩、转置、重复……）](https://zhuanlan.zhihu.com/p/31495102)
- [Pytorch中的RNN之pack_padded_sequence()和pad_packed_sequence()](https://www.cnblogs.com/sbj123456789/p/9834018.html)
- [pytorch学习笔记（二）：gradient](https://blog.csdn.net/u012436149/article/details/54645162)
- [torch.multinomial()理解](https://blog.csdn.net/monchin/article/details/79787621)
- [Pytorch 细节记录](https://www.cnblogs.com/king-lps/p/8570021.html)
- [What does flatten_parameters() do?](https://stackoverflow.com/questions/53231571/what-does-flatten-parameters-do)
- [关于Pytorch的二维tensor的gather和scatter_操作用法分析](https://www.cnblogs.com/HongjianChen/p/9450987.html)
- [Pytorch scatter_ 理解轴的含义](https://blog.csdn.net/qq_16234613/article/details/79827006)
- [‘model.eval()’ vs ‘with torch.no_grad()’](https://discuss.pytorch.org/t/model-eval-vs-with-torch-no-grad/19615)
- [到底什么是生成式对抗网络GAN？](https://www.msra.cn/zh-cn/news/features/gan-20170511)

### torchtext介绍

- [torchtext](https://github.com/pytorch/text)
- [A Tutorial on Torchtext](http://anie.me/On-Torchtext/)
- [Torchtext 详细介绍](https://zhuanlan.zhihu.com/p/37223078)
- [torchtext入门教程，轻松玩转文本数据处理](https://zhuanlan.zhihu.com/p/31139113)

### 其他工具模块

- [python的Tqdm模块](https://blog.csdn.net/langb2014/article/details/54798823)
- [pytorch-crf](https://github.com/kmkurn/pytorch-crf)

### 数据集

- [Chinese-Literature-NER-RE-Dataset](https://github.com/lancopku/Chinese-Literature-NER-RE-Dataset)
- [ChineseTextualInference](https://github.com/liuhuanyong/ChineseTextualInference)

### 表示学习

- [TransE-Knowledge-Graph-Embedding](https://github.com/Lapis-Hong/TransE-Knowledge-Graph-Embedding)
- [OpenKE-PyTorch](https://github.com/ShulinCao/OpenKE-PyTorch)
- [【语料】2500万中文三元组！](https://spaces.ac.cn/archives/4359)

### 命名实体识别

- [sequence_tagging](https://github.com/AdolHong/sequence_tagging)

### 关系抽取

- [ChineseNRE](https://github.com/buppt/ChineseNRE)
- [pytorch-pcnn](https://github.com/ShomyLiu/pytorch-pcnn)
- [关系抽取(分类)总结](http://shomy.top/2018/02/28/relation-extraction/)

### 事件抽取

这里暂时粗浅的将语义角色标注技术实现等同于事件抽取任务。

- [语义角色标注](http://wiki.jikexueyuan.com/project/deep-learning/wordSence-identify.html)
- [iobes_iob 与 iob_ranges 函数借鉴](https://github.com/glample/tagger/blob/master/utils.py)
- [BiRNN-SRL](https://github.com/zxplkyy/BiRNN-SRL)
- [chinese_semantic_role_labeling](https://github.com/Nrgeup/chinese_semantic_role_labeling)

### 其他
