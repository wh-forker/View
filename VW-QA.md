# Question && Answer
- A Question-Focused Multi-Factor Attention Network for Question Answering
	- https://www.paperweekly.site/papers/1597
	- https://github.com/nusnlp/amanda
- PaperWeekly 第37期 | 论文盘点：检索式问答系统的语义匹配模型（神经网络篇）
	- https://zhuanlan.zhihu.com/p/26879507
# Knowledge base
- 知识图谱与知识表征学习系列
	- https://zhuanlan.zhihu.com/p/27664263

---

+ IRGAN: A Minimax Game for Unifying Generative and Discriminative Information Retrieval Models
	+ 在现代信息检索领域一直是两大学派之争的局面。一方面，经典思维流派是假设在文档和信息需求（由查询可知）之间存在着一个独立的随机生成过程。另一方面，现代思维流派则充分利用机器学习的优势，将文档和搜索词联合考虑为特征，并从大量训练数据中预测其相关性或排序顺序标签。
	+ 本篇 SIGIR2017 的满分论文则首次提出将两方面流派的数据模型通过一种对抗训练的方式统一在一起，使得两方面的模型能够相互提高，最终使得检索到的文档更加精准。文章的实验分别在网络搜索、推荐系统以及问答系统三个应用场景中实现并验证了结果的有效性。
---
# Dataset
+ ## NarrativeQA
    + Deepmind 最新阅读理解数据集 NarrativeQA ，让机器挑战更复杂阅读理解问题
        + https://www.leiphone.com/news/201712/mjCYZ8WTiREqja6L.html
        + https://github.com/deepmind/narrativeqa
        + DeepMind认为目前的阅读理解数据集均存在着一定的局限性，包括：数据集小、不自然、只需要一句话定位回答的必须信息，等等。因而 Deepmind 认为，在这些数据集上的测试可能都是一个不能真实反映机器阅读理解能力的伪命题。

    + The NarrativeQA Reading Comprehension Challenge
        + 由 DeepMind 发布的全新机器阅读理解数据集 NarrativeQA，其难度和复杂度都进行了全面升级。
        + 论文链接：https://www.paperweekly.site/papers/1397
        + 代码链接：https://github.com/deepmind/narrativeqa
+ ## SQuAD
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

	+ 对比
		- 当前的公开数据集对比如下，MCTest，Algebra和Science是现在的三个公开的阅读理解数据集，
		- 我们可以看到Squad在数量上远远超过这三个数据集，这使得在这个数据集上训练大规模复杂算法成为可能。
		- 同时，相比于WikiQA和TrecQA这两个著名问答数据集，Squad也在数量上远远超过。
		- 而CNN Mail和CBT虽然大，但是这两个数据集都是挖空猜词的数据集，并不是真正意义上的问答。

+ Others
---
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

+ 三分熟博士生の阅读理解与问答数据集 | 论文集精选 #03
	+ https://zhuanlan.zhihu.com/p/30308726
---


---
#Knowledge Base
+ 怎么利用知识图谱构建智能问答系统？
    + https://www.zhihu.com/question/30789770/answer/116138035
    + https://zhuanlan.zhihu.com/p/25735572
+ 揭开知识库问答KB-QA的面纱1·简介篇
	+ 什么是知识库（knowledge base, KB）
	+ 什么是知识库问答（knowledge base question answering, KB-QA）
	+ 知识库问答的主流方法
	+ 知识库问答的数据集