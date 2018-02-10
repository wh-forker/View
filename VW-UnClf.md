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
