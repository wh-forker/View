# PL-QA
+ Reference links:
	+ https://zhuanlan.zhihu.com/p/33124445
+ SQuAD
	+ https://rajpurkar.github.io/SQuAD-explorer/
+ NewsQA
+ cmrc
	+ https://github.com/ymcui/cmrc2017
	+ 第二届
		+ 本届中文机器阅读理解评测将开放首个人工标注的中文篇章片段抽取型阅读理解数据集
		+ 今年我们将聚焦基于篇章片段抽取的阅读理解(Span-Extraction Machine ReadingComprehension)，作为填空型阅读理解任务的进一步延伸。虽然在英文阅读理解研究上有例如斯坦福SQuAD、NewsQA等篇章片段抽取型阅读理解数据集，但目前相关中文资源仍然处于空白状态。本届中文机器阅读理解评测将开放首个人工标注的中文篇章片段抽取型阅读理解数据集，参赛选手需要对篇章、问题进行建模，并从篇章中抽取出连续片段作为答案。本次评测依然采取训练集、开发集公开，测试集隐藏的形式以保证评测的公平性。

## Dataset
+ NarrativeQA
	- DeepMind
+ SQuAD
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

- papers && projects
	- DrQA
		- project
		- paper
			- Reading Wikipedia to Answer Open-Domain Questions
	- 基于LSTM的中文问答系统
		- https://github.com/S-H-Y-GitHub/QA
		- 本项目通过建立双向长短期记忆网络模型，实现了在多个句子中找到给定问题的答案所在的句子这一功能。在使用了互联网第三方资源的前提下，用training.data中的数据训练得到的模型对develop.data进行验证，MRR可达0.75以上
		- MRR
			- 是一个国际上通用的对搜索算法进行评价的机制，即第一个结果匹配，分数为1，第二个匹配分数为0.5，第n个匹配分数为1/n，如果没有匹配的句子分数为0。最终的分数为所有得分之和
	- DeepQA
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
	+ improcements
		+ In addition to trying larger/deeper model, there are a lot of small improvements which could be tested. Don't hesitate to send a pull request if you implement one of those. Here are some ideas:

		+ For now, the predictions are deterministic (the network just take the most likely output) so when answering a question, the network will always gives the same answer. By adding a sampling mechanism, the network could give more diverse (and maybe more interesting) answers. The easiest way to do that is to sample the next predicted word from the SoftMax probability distribution. By combining that with the loop_function argument of tf.nn.seq2seq.rnn_decoder, it shouldn't be too difficult to add. After that, it should be possible to play with the SoftMax temperature to get more conservative or exotic predictions.
		+ Adding attention could potentially improve the predictions, especially for longer sentences. It should be straightforward by replacing embedding_rnn_seq2seq by embedding_attention_seq2seq on model.py.
Having more data usually don't hurt. Training on a bigger corpus should be beneficial. Reddit comments dataset seems the biggest for now (and is too big for this program to support it). Another trick to artificially increase the dataset size when creating the corpus could be to split the sentences of each training sample (ex: from the sample Q:Sentence 1. Sentence 2. => A:Sentence X. Sentence Y. we could generate 3 new samples: Q:Sentence 1. Sentence 2. => A:Sentence X., Q:Sentence 2. => A:Sentence X. Sentence Y. and Q:Sentence 2. => A:Sentence X.. Warning: other combinations like Q:Sentence 1. => A:Sentence X. won't work because it would break the transition 2 => X which links the question to the answer)
		+ The testing curve should really be monitored as done in my other music generation project. This would greatly help to see the impact of dropout on overfitting. For now it's just done empirically by manually checking the testing prediction at different training steps.
		+ For now, the questions are independent from each other. To link questions together, a straightforward way would be to feed all previous questions and answer to the encoder before giving the answer. Some caching could be done on the final encoder stated to avoid recomputing it each time. To improve the accuracy, the network should be retrain on entire dialogues instead of just individual QA. Also when feeding the previous dialogue to the encoder, new tokens <Q> and <A> could be added so the encoder knows when the interlocutor is changing. I'm not sure though that the simple seq2seq model would be sufficient to capture long term dependencies between sentences. Adding a bucket system to group similar input lengths together could greatly improve training speed.

