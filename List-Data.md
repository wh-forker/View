# Data-list

+ 中文文本语料库整理（不定时更新2015-10-24）

## 搜狗语料
+ QA
	+ http://task.www.sogou.com/cips-sogou_qa/
+ Others

## QA

### MS MARCO
+ 相比SQuAD，MARCO的挑战难度更大，因为它需要测试者提交的模型具备理解复杂文档、回答复杂问题的能力。
+ 据了解，对于每一个问题，MARCO 提供多篇来自搜索结果的网页文档，系统需要通过阅读这些文档来回答用户提出的问题。但是，文档中是否含有答案，以及答案具体在哪一篇文档中，都需要系统自己来判断解决。更有趣的是，有一部分问题无法在文档中直接找到答案，需要阅读理解模型自己做出判断；MARCO 也不限制答案必须是文档中的片段，很多问题的答案必须经过多篇文档综合提炼得到。这对机器阅读理解提出了更高的要求，需要机器具备综合理解多文档信息、聚合生成问题答案的能力。

### NarrativeQA
+ Deepmind 最新阅读理解数据集 NarrativeQA ，让机器挑战更复杂阅读理解问题
+ https://www.leiphone.com/news/201712/mjCYZ8WTiREqja6L.html
+ https://github.com/deepmind/narrativeqa
+ DeepMind认为目前的阅读理解数据集均存在着一定的局限性，包括：数据集小、不自然、只需要一句话定位回答的必须信息，等等。因而 Deepmind 认为，在这些数据集上的测试可能都是一个不能真实反映机器阅读理解能力的伪命题。

+ The NarrativeQA Reading Comprehension Challenge
+ 由 DeepMind 发布的全新机器阅读理解数据集 NarrativeQA，其难度和复杂度都进行了全面升级。
+ 论文链接：https://www.paperweekly.site/papers/1397
+ 代码链接：https://github.com/deepmind/narrativeqa

### SQuAD
+ news
	+ 这个竞赛基于SQuAD问答数据集，考察两个指标：EM和F1。
	EM是指精确匹配，也就是模型给出的答案与标准答案一模一样；F1，是根据模型给出的答案和标准答案之间的重合度计算出来的，也就是结合了召回率和精确率。
	目前阿里、微软团队并列第一，其中EM得分微软（r-net+融合模型）更高，F1得分阿里（SLQA+融合模型）更高。但是他们在EM成绩上都击败了“人类表现”
+ EMNLP2016 SQuAD:100,000+ Questions for Machine Comprehension of Text
	+ https://arxiv.org/pdf/1606.05250.pdf
+ SQuAD，斯坦福在自然语言处理的野心
	+ http://blog.csdn.net/jdbc/article/details/52514050
+ 一共有107,785问题，以及配套的 536 篇文章
+ 数据集的具体构建如下：
	1. 文章是随机sample的wiki百科，一共有536篇wiki被选中。而每篇wiki，会被切成段落，最终生成了23215个自然段。之后就对这23215个自然段进行阅读理解，或者说自动问答。
	2. 之后斯坦福，利用众包的方式，进行了给定文章，提问题并给答案的人工标注。他们将这两万多个段落给不同人，要求对每个段落提五个问题。
	3. 让另一些人对提的这个问题用文中最短的片段给予答案，如果不会或者答案没有在文章中出现可以不给。之后经过他们的验证，人们所提的问题在问题类型分布上足够多样，并且有很多需要推理的问题，也就意味着这个集合十分有难度。如下图所示，作者列出了该数据集答案的类别分布，我们可以看到 日期，人名，地点，数字等都被囊括，且比例相当。
	4. 这个数据集的评测标准有两个：
	第一：F1
	第二：EM。
        EM是完全匹配的缩写，必须机器给出的和人给出的一样才算正确。哪怕有一个字母不一样，也会算错。而F1是将答案的短语切成词，和人的答案一起算recall，Precision和F1，即如果你match了一些词但不全对，仍然算分。
	5. 为了这个数据集，他们还做了一个baseline，是通过提特征，用LR算法将特征组合，最终达到了40.4的em和51的f1。而现在IBM和新加坡管理大学利用深度学习模型，均突破了这个算法。可以想见，在不远的将来会有更多人对阅读理解发起挑战，自然语言的英雄也必将诞生。甚至会有算法超过人的准确度。

### 对比
- 当前的公开数据集对比如下，MCTest，Algebra和Science是现在的三个公开的阅读理解数据集，
- 我们可以看到Squad在数量上远远超过这三个数据集，这使得在这个数据集上训练大规模复杂算法成为可能。
- 同时，相比于WikiQA和TrecQA这两个著名问答数据集，Squad也在数量上远远超过。
- 而CNN Mail和CBT虽然大，但是这两个数据集都是挖空猜词的数据集，并不是真正意义上的问答。


### 阅读理解与问答数据集 https://zhuanlan.zhihu.com/p/30308726
1. On Generating Characteristic-rich Question Sets for QA Evaluation
	文章发表在 EMNLP 2016，本文详细阐述了 GraphQuestions 这个数据集的构造方法，强调这个数据集是富含特性的（Characteristic-rich）。
	+ 数据集特点：
		1. 基于 Freebase，有 5166 个问题，涉及 148 个不同领域；
		2. 从知识图谱中产生 Minimal Graph Queries，再将 Query 自动转换成规范化的问题；
		3. 由于 2，Logical Form 不需要人工标注，也不存在无法用 Logical Form 表示的问题；
		4. 使用人工标注的办法对问题进行 paraphrasing，使得每个问题有多种表述方式（答案不变），主要是 Entity-level Paraphrasing，也有 sentence-level；
		5. Characteristic-rich 指数据集提供了问题在下列维度的信息，使得研究者可以对问答系统进行细粒度的分析, 找到研究工作的前进方向：关系复杂度（Structure Complexity），普遍程度（Commonness），函数（Function），多重释义（Paraphrasing），答案候选数（Answer Cardinality）。

	+ 论文链接：http://www.paperweekly.site/papers/906
	+ 数据集链接：https://github.com/ysu1989/GraphQuestions
2. LSDSem 2017 Shared Task: The Story Cloze Test
	+ Story Cloze Test：人工合成的完形填空数据集。
	+ 论文链接：http://www.paperweekly.site/papers/917
	+ 数据集链接：http://cs.rochester.edu/nlp/rocstories/
3. Dataset and Neural Recurrent Sequence Labeling Model for Open-Domain Factoid Question Answering
	+ 百度深度学习实验室创建的中文开放域事实型问答数据集。
	+ 论文链接：http://www.paperweekly.site/papers/914
	+ 数据集链接：http://idl.baidu.com/WebQA.html
4. Program Induction by Rationale Generation : Learning to Solve and Explain Algebraic Word Problems
	+ DeepMind 和牛津大学共同打造的代数问题数据集 AQuA（Algebra Question Answering）。
	+ 论文链接：http://www.paperweekly.site/papers/913
	+ 数据集链接：https://github.com/deepmind/AQuA
5. Frames: A Corpus for Adding Memory to Goal-Oriented Dialogue Systems
	+ Maluuba 放出的对话数据集。
	+ 论文链接：http://www.paperweekly.site/papers/407
	+ 数据集链接：http://datasets.maluuba.com/Frames
6. Teaching Machines to Read and Comprehend
	+ DeepMind Q&A Dataset 是一个经典的机器阅读理解数据集，分为两个部分：
		+ 1. CNN：~90k 美国有线电视新闻网（CNN）的新闻文章，~380k 问题；
		+ 2. Daily Mail：~197k DailyMail 新闻网的新闻文章（不是邮件正文），~879k 问题。
	+ 论文链接：http://www.paperweekly.site/papers/915
	+ 数据集链接：http://cs.nyu.edu/~kcho/DMQA/
7. Semantic Parsing on Freebase from Question-Answer Pairs
	+ 文章发表在 EMNLP-13，The Stanford NLP Group 是世界领先的 NLP 团队。
	+ 他们在这篇文章中引入了 WebQuestions 这个著名的问答数据集，WebQuestion 主要是借助 Google Suggestion 构造的
	+ 依靠 Freebase（一个大型知识图谱）中的实体来回答，属于事实型问答数据集（比起自然语言，容易评价结果优劣）
	+ 有 6642 个问答对
	+ 最初，他们构造这个数据集是为了做 Semantic Parsing，以及发布自己的系统 SEMPRE system。
	+ 论文链接：http://www.paperweekly.site/papers/827
	+ 数据集链接：http://t.cn/RWPdQQO
8. A Corpus and Evaluation Framework for Deeper Understanding of Commonsense Stories
	+ ROCStories dataset for story cloze test.
	+ 论文链接：http://www.paperweekly.site/papers/918
	+ 数据集链接：http://cs.rochester.edu/nlp/rocstories/
9. MoleculeNet: A Benchmark for Molecular Machine Learning
	+ 一个分子机器学习 benchmark，最喜欢看到这种将机器学习应用到传统学科领域了。
	+ 论文链接：http://www.paperweekly.site/papers/862
	+ 数据集链接：http://t.cn/RWPda8r

---
## Sequence Label
+ 国内可用免费语料库
  + http://www.cnblogs.com/mo-wang/p/4444858.html

+ 一个中文的标注语料库。可用于训练HMM模型。
	+ https://github.com/liwenzhu/corpusZh

+ 可以提供给题主两份相对较新的中文分词语料
	+ 第一份是SIGHAN的汉语处理评测的Bakeoff语料，从03年起首次进行评测，评测的内容针对汉语分词的准确性和合理性，形成Bakeoff 2005评测集，包含简、繁体中文的训练集和测试集，训练集有四个，单句量在1.5W~8W+。内容比较偏向于书面语。后面05 07年分别对中文命名实体识别和词性标注给出了评测。Bakeoff 2005中文分词熟语料传送门：Second International Chinese Word Segmentation Bakeoff
	+ 第二份语料来自GitHub作者liwenzhu，于14年发布于GitHub，总词汇量在7400W+，可以用于训练很多模型例如Max Entropy、CRF、HMM......，优点是这份语料在分词基础上还做了词性标注，至于准确性还有待考究。传送门：liwenzhu/corpusZh

## CLF(SA)
+ Kaggle Twitter Sentiment Analysis
