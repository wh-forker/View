# Papers
1. A deep reinforcement learning chatbot
+ https://arxiv.org/pdf/1801.06700.pdf

2. A through examination of the CNN Daily reading comprehension task

3. Ask the right question : active question reformulation with reinforcement learning

4. Gated attention readers for text comprehension

5. Hybird attention over attention reader

6. MaskGAN : better text generation via filling

7. Teach Mechine to read and comprehension


# Question && Answer
- A Question-Focused Multi-Factor Attention Network for Question Answering
	- https://www.paperweekly.site/papers/1597
	- https://github.com/nusnlp/amanda
- PaperWeekly 第37期 | 论文盘点：检索式问答系统的语义匹配模型（神经网络篇）
	- https://zhuanlan.zhihu.com/p/26879507
- Fast and Accurate Reading Comprehension by Combining Self-Attention and Convolution
	- 本文是 CMU 和 Google Brain 发表于 ICLR 2018 的文章，论文改变了以往机器阅读理解均使用 RNN 进行建模的习惯，使用卷积神经网络结合自注意力机制，完成机器阅读理解任务。
	- 其中作者假设，卷积神经网络可建模局部结构信息，而自注意力机制可建模全文互动（Interaction）关系，这两点就足以完成机器阅读理解任务。
	- 论文链接
	- https://www.paperweekly.site/papers/1759
- Attentive Recurrent Tensor Model for Community Question Answering
	- 社区问答有一个很主要的挑战就是句子间词汇与语义的鸿沟。本文使用了 phrase-level 和 token-level 两个层次的 attention 来对句子中的词赋予不同的权重，并参照 CNTN 模型用神经张量网络计算句子相似度的基础上，引入额外特征形成 3-way 交互张量相似度计算。
	- 围绕答案选择、最佳答案选择、答案触发三个任务，论文提出的模型 RTM 取得了多个 state-of-art 效果。
	- 论文链接 : https://www.paperweekly.site/papers/1741

# Dialog Systems
- Feudal Reinforcement Learning for Dialogue Management in Large Domains
	- 本文来自剑桥大学和 PolyAI，论文提出了一种新的强化学习方法来解决对话策略的优化问题
	- https://www.paperweekly.site/papers/1756

# Reading and Comprehension
- 百度 2018 机器阅读理解竞赛
- 搜狗 问答竞赛
- 机器阅读理解相关论文汇总（截止2017年底）
	- https://www.zybuluo.com/ShawnNg/note/622592
- 百度NLP团队登顶MARCO阅读理解测试排行榜
	- http://tech.qq.com/a/20180222/008569.htm
	- 使用了一种新的多候选文档联合建模表示方法，通过注意力机制使不同文档产生的答案之间能够产生交换信息，互相印证，从而更好的预测答案。据介绍，此次百度只凭借单模型（single model）就拿到了第一名，并没有提交更容易拿高分的多模型集成（ensemble）结果

# Knowledge base
- 「知识表示学习」专题论文推荐 | 每周论文清单
	- https://zhuanlan.zhihu.com/p/33606964
- 知识图谱与知识表征学习系列
	- https://zhuanlan.zhihu.com/p/27664263
- 怎么利用知识图谱构建智能问答系统？
    + https://www.zhihu.com/question/30789770/answer/116138035
    + https://zhuanlan.zhihu.com/p/25735572
+ 揭开知识库问答KB-QA的面纱1·简介篇
	+ 什么是知识库（knowledge base, KB）
	+ 什么是知识库问答（knowledge base question answering, KB-QA）
	+ 知识库问答的主流方法
	+ 知识库问答的数据集
- 经典论文解读 | 基于Freebase的问答研究
	- 本文给出了一种 end-to-end 的系统来自动将 NL 问题转换成 SPARQL 查询语言。
	- 作者综合了**实体识别**以及**距离监督**和 **learning-to-rank** 技术，使得 QA 系统的精度提高了不少，整个过程介绍比较详细，模型可靠接地气。
	- 本文要完成的任务是根据 **KB** 知识来回答自然语言问题，给出了一个叫 Aqqu 的系统，首先为问题生成一些备选 query，然后使用学习到的模型来对这些备选 query 进行排名，返回排名最高的 query
	- 论文 | More Accurate Question Answering on Freebase
	- 链接 | https://www.paperweekly.site/papers/1356
	- 源码 | https://github.com/ad-freiburg/aqqu
- Question Answering on Knowledge Bases and Text using Universal Schema and Memory Networks
	- 传统 QA 问题的解决方法是从知识库或者生文本中推测答案，本文将通用模式扩展到自然语言 QA 的应用当中，采用记忆网络来关注文本和 KB 相结合的大量事实。
	- 论文链接
		- https://www.paperweekly.site/papers/1734
	- 代码链接
		- https://github.com/rajarshd/TextKBQA


---

+ IRGAN: A Minimax Game for Unifying Generative and Discriminative Information Retrieval Models
	+ 在现代信息检索领域一直是两大学派之争的局面。一方面，经典思维流派是假设在文档和信息需求（由查询可知）之间存在着一个独立的随机生成过程。另一方面，现代思维流派则充分利用机器学习的优势，将文档和搜索词联合考虑为特征，并从大量训练数据中预测其相关性或排序顺序标签。
	+ 本篇 SIGIR2017 的满分论文则首次提出将两方面流派的数据模型通过一种对抗训练的方式统一在一起，使得两方面的模型能够相互提高，最终使得检索到的文档更加精准。文章的实验分别在网络搜索、推荐系统以及问答系统三个应用场景中实现并验证了结果的有效性。
---



## Reference links:
+ https://zhuanlan.zhihu.com/p/33124445
+ SQuAD
	+ https://rajpurkar.github.io/SQuAD-explorer/
+ NewsQA
+ cmrc
	+ https://github.com/ymcui/cmrc2017
	+ 第二届
		+ 本届中文机器阅读理解评测将开放首个人工标注的中文篇章片段抽取型阅读理解数据集
		+ 今年我们将聚焦基于篇章片段抽取的阅读理解(Span-Extraction Machine ReadingComprehension)，作为填空型阅读理解任务的进一步延伸。虽然在英文阅读理解研究上有例如斯坦福SQuAD、NewsQA等篇章片段抽取型阅读理解数据集，但目前相关中文资源仍然处于空白状态。本届中文机器阅读理解评测将开放首个人工标注的中文篇章片段抽取型阅读理解数据集，参赛选手需要对篇章、问题进行建模，并从篇章中抽取出连续片段作为答案。本次评测依然采取训练集、开发集公开，测试集隐藏的形式以保证评测的公平性。

+ ### SQuAD
	+ Hybrid AoA Reader (ensemble)
		+ Joint Laboratory of HIT and iFLYTEK Research
	+ r-net + 融合模型
		+ Microsoft Research Asia
	+ SLQA + 融合模型
		+ Alibaba iDST NLP

## projects
- ## DrQA
	- project
	- paper
		- Reading Wikipedia to Answer Open-Domain Questions

- ## LSTM 中文
	- https://github.com/S-H-Y-GitHub/QA
	- 本项目通过建立双向长短期记忆网络模型，实现了在多个句子中找到给定问题的答案所在的句子这一功能。在使用了互联网第三方资源的前提下，用training.data中的数据训练得到的模型对develop.data进行验证，MRR可达0.75以上
	- MRR
		- 是一个国际上通用的对搜索算法进行评价的机制，即第一个结果匹配，分数为1，第二个匹配分数为0.5，第n个匹配分数为1/n，如果没有匹配的句子分数为0。最终的分数为所有得分之和

- ## DeepQA
	- projects
		- https://github.com/Conchylicultor/DeepQA
		- work as server
		- 
		- demo
		```
		Q: Who is Laura ?
		A: My brother.
		Q: Say 'goodbye'
		A: Alright.
		Q: What is cooking ?
		A: A channel.
		```

	- improcements
		- In addition to trying larger/deeper model, there are a lot of small improvements which could be tested. Don't hesitate to send a pull request if you implement one of those. Here are some ideas:

		- For now, the predictions are deterministic (the network just take the most likely output) so when answering a question, the network will always gives the same answer. By adding a sampling mechanism, the network could give more diverse (and maybe more interesting) answers. The easiest way to do that is to sample the next predicted word from the SoftMax probability distribution. By combining that with the loop_function argument of tf.nn.seq2seq.rnn_decoder, it shouldn't be too difficult to add. After that, it should be possible to play with the SoftMax temperature to get more conservative or exotic predictions.
		- Adding attention could potentially improve the predictions, especially for longer sentences. It should be straightforward by replacing embedding_rnn_seq2seq by embedding_attention_seq2seq on model.py.
		- Having more data usually don't hurt. Training on a bigger corpus should be beneficial. Reddit comments dataset seems the biggest for now (and is too big for this program to support it). Another trick to artificially increase the dataset size when creating the corpus could be to split the sentences of each training sample (ex: from the sample Q:Sentence 1. Sentence 2. => A:Sentence X. Sentence Y. we could generate 3 new samples: Q:Sentence 1. Sentence 2. => A:Sentence X., Q:Sentence 2. => A:Sentence X. Sentence Y. and Q:Sentence 2. => A:Sentence X.. Warning: other combinations like Q:Sentence 1. => A:Sentence X. won't work because it would break the transition 2 => X which links the question to the answer)
		- The testing curve should really be monitored as done in my other music generation project. This would greatly help to see the impact of dropout on overfitting. For now it's just done empirically by manually checking the testing prediction at different training steps.
		- For now, the questions are independent from each other. To link questions together, a straightforward way would be to feed all previous questions and answer to the encoder before giving the answer. Some caching could be done on the final encoder stated to avoid recomputing it each time. To improve the accuracy, the network should be retrain on entire dialogues instead of just individual QA. Also when feeding the previous dialogue to the encoder, new tokens Q and A could be added so the encoder knows when the interlocutor is changing. I'm not sure though that the simple seq2seq model would be sufficient to capture long term dependencies between sentences. Adding a bucket system to group similar input lengths together could greatly improve training speed.
