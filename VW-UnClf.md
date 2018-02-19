- DR-BiLSTM: Dependent Reading Bidirectional LSTM for Natural Language Inference
	- https://arxiv.org/abs/1802.05577
	- https://www.arxiv-vanity.com/papers/1802.05577/
- Deep Learning for Sentiment Analysis : A Survey
	- 近年来，深度学习有了突破性发展，NLP 领域里的情感分析任务逐渐引入了这种方法，并形成了很多业内最佳结果。本文中，来自领英与伊利诺伊大学芝加哥分校的研究人员对基于深度学习的情感分析研究进行了详细论述。
- Generating Wikipedia by Summarizing Long Sequences
	- 本文来自 Google Brain，通过长序列摘要生成维基百科。
- MaskGAN: Better Text Generation via Filling in the ______
	- 谷歌大脑提出使用生成对抗网络（GAN）来提高文本质量，它通过显式地训练生成器产生高质量文本，并且已经在图像生成领域取得很大成功。GAN 最初设计用于输出可微分的值，所以生成离散语言是具有挑战性的。作者认为验证复杂度本身不代表模型生成的文本质量。
	- 本文引入条件 actor-critic GAN，它可以基于上下文填充缺失的文本。本文从定性和定量的角度证明，相比最大似然训练的模型，这种方法生成了更真实的有条件和无条件文本样本。
- Investigating the Working of Text Classifiers
	- 文本分类问题，给一段文本指定一个类别，在主题分类和情感分析中都有应用到。它的难点在于如何在具有语义的文本中，对句子之间的内在联系（语义或句法）进行编码。这对文本情感分类很关键，因为比如像“对照”或者“因果”等关系，会直接决定整个文档的性质。
	- 本文并没有提出一套完整的解决方法，而是通过构建新的数据集（训练集和测试集尽可能不包含共同的关键词），验证上面的猜想。此外，作者还设计了一种 ANON 的正则方法，让网络不那么容易记住文档的关键词。
	
- A Hybrid Framework for Text Modeling with Convolutional RNN
	- 本文使用 RNN+CNN 的结构来完成 NLP 中的问答任务，其亮点在于使用 RNN 获取 question 和 answer 的上下文语义，CNN 在语义层面对二者进行操作。
	- http://www.sohu.com/a/165562211_651893
- Model Evaluation, Model Selection, and Algorithm Selection in Machine Learning
	- 本文回顾了用于模型评估、模型选择和算法选择任务中的不同技术，并参考理论和实证研究讨论了每种技术的主要优势和劣势。
- Training Neural Networks by Using Power Linear Units (PoLUs)
	- 本文设计了一种 PoLU 激活函数，在多个数据集上取得超过 ReLU 的性能。
- Learning Continuous User Representations through Hybrid Filtering with doc2vec
	- 本文把用行为使用 item 描述进行串连，构成文档，并使用 doc2vec 训练用户表示向量。
- NDDR-CNN: Layer-wise Feature Fusing in Multi-Task CNN by Neural Discriminative Dimensionality Reduction
	- 本文研究的问题是多任务学习。作者提出了一种对多个网络（对应多个任务）进行逐层特征融合的方法。
- An Attention-based Collaboration Framework for Multi-View Network Representation Learning
	- 本文是网络表示学习大牛 Jian Tang 的工作，论文利用 multi-view 来对网络进行表示学习。各个 view 之间通过共享邻居来保证所有节点表示在同一个空间中，同时，通过引入 attention 机制，可以学到不同节点在不同 view 的权重。
	

- 2017年度最值得读的AI论文 | NLP篇
	- http://chuansong.me/n/2168901252931
- 总结 | 2016年最值得读的自然语言处理领域Paper 
	- http://chuansong.me/n/1440373152360
- PaperWeekly 第二十七期 | VAE for NLP 
	- http://chuansong.me/n/1628708852431
- 数据开放 | PaperWeekly交流群对话数据 
	- http://chuansong.me/n/1653837452325
- 综述 | 知识图谱研究进展 
	- http://chuansong.me/n/1687512252413

- OpenAI-2018年强化学习领域7大最新研究方向全盘点
	- https://zhuanlan.zhihu.com/p/33630520

- TensorForce Bitcoin Trading Bot
	- 基于深度增强学习的比特币交易机器人
- CakeChat
	- 情感对话机器人
- SentiBridge
	- 中文实体情感知识库

- gradient-checkpointing
	- 神经网络训练省内存神器
	- 本项目是由 OpenAI 提供的内存平衡工具。前馈模型可以在仅增加20%计算时间的基础上，让 GPU 适应十倍大的模型。
	- 项目链接：https://github.com/openai/gradient-checkpointing
- Minigo
	- 基于AlphaGo Zero核心算法的围棋AI
	- 项目链接：https://github.com/tensorflow/minigo

- Synonyms
	- 最好的中文近义词工具包

- (***)「知识表示学习」专题论文推荐 | 每周论文清单
	- https://zhuanlan.zhihu.com/p/33606964

- Deep learning in production with Keras, Redis, Flask, and Apache
	- https://www.pyimagesearch.com/2018/02/05/deep-learning-production-keras-redis-flask-apache/
- MIT course 深度学习导论 by Alexander Amini, Ava Soleimany, Harini Suresh
	- https://www.bilibili.com/video/av19113488/
	- Sequence Modeling with Neural Networks、Convolutional Neural Networks、Deep Generative Modeling
- A Tutorial on Modeling and Inference in Undirected Graphical Models for Hyperspectral Image Analysis
	- https://arxiv.org/abs/1801.08268
	- https://github.com/UBGewali/tutorial-UGM-hyperspectral
---


- word embedding
  - word2vec
  - glove
  - fastText


- represent
  - word2vec(distributed representation)
  - memory neural networks
    - https://github.com/facebook/MemNN
    - https://arxiv.org/abs/1701.08718
- precision 和 accuracy
  -     TP：True Positive，即正确预测出的正样本个数
        FP：False Positive，即错误预测出的正样本个数（本来是负样本，被我们预测成了正样本）
        TN：True Negative，即正确预测出的负样本个数
        FN：False Negative，即错误预测出的负样本个数（本来是正样本，被我们预测成了负样本）
        
        Precision：TP÷(TP+FP)，分类器预测出的正样本中，真实正样本的比例
        Recall：TP÷(TP+FN)，在所有真实正样本中，分类器中能找到多少
        Accuracy：(TP+TN)÷(TP+NP+TN+FN)，分类器对整体的判断能力，即正确预测的比例
- 中文分词指标评价
  -     准确率(Precision)和召回率(Recall)
        Precision = 正确切分出的词的数目/切分出的词的总数
        Recall = 正确切分出的词的数目/应切分出的词的总数
        
        综合性能指标F-measure
        Fβ = (β2 + 1)*Precision*Recall/(β2*Precision + Recall)
        β为权重因子，如果将准确率和召回率同等看待，取β = 1，就得到最常用的F1-measure
        F1 = 2*Precisiton*Recall/(Precision+Recall)
        
        未登录词召回率(R_OOV)和词典中词的召回率(R_IV)
        R_OOV = 正确切分出的未登录词的数目/标准答案中未知词的总数
        R_IV = 正确切分出的已知词的数目/标准答案中已知词的总数
        
        
        
        
CU-NLP
aspect level情感分析相关工作

1. SemEval-2014 Task 4: Aspect Based Sentiment Analysis**
2. NRC-Canada-2014: Detecting Aspects and Sentiment in Customer Reviews**
3. DCU: Aspect-based Polarity Classification for SemEval Task 4**
4. Adaptive Recursive Neural Network for Target-dependent Twitter Sentiment Classification**
5. Aspect Specific Sentiment Analysis using Hierarchical Deep Learning**
6. PhraseRNN: Phrase Recursive Neural Network for Aspect-based Sentiment Analysis**
7. Effective LSTMs for Target-Dependent Sentiment Classification**

1：aspect level情感分析的系统介绍；

2、3：传统分类器方法实现aspect level的情感分析；

4、5、6、7：神经网络方法实现aspect level的情感分析。


---

  - http://aclweb.org/anthology/D16-1058
- 在中文命名实体识别中，现在比较好（准确率和召回率）的算法都有哪些？
  - https://www.zhihu.com/question/19994255


- Attention 
  
    The idea was proposed in the article by Z. Yang et al., "Hierarchical Attention Networks for Document Classification", 2016: http://www.aclweb.org/anthology/N16-1174.
  - http://www.cnblogs.com/robert-dlut/p/5952032.html
    - Attention机制最早是在视觉图像领域提出来的，应该是在九几年思想就提出来了，但是真正火起来应该算是google mind团队的这篇论文《Recurrent Models of Visual Attention》[14]，他们在RNN模型上使用了attention机制来进行图像分类。随后，Bahdanau等人在论文《Neural Machine Translation by Jointly Learning to Align and Translate》 [1]中，使用类似attention的机制在机器翻译任务上将翻译和对齐同时进行，他们的工作算是是第一个提出attention机制应用到NLP领域中。接着类似的基于attention机制的RNN模型扩展开始应用到各种NLP任务中。最近，如何在CNN中使用attention机制也成为了大家的研究热点。下图表示了attention研究进展的大概趋势。
