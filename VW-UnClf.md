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


---

Uber 论文5连发宣告神经演化新时代，深度强化学习训练胜过 SGD 和策略梯度

- 介绍了他们在基因算法（genetic algorithm）、突变方法（mutation）和进化策略（evolution strategies）等神经演化思路方面的研究成果，同时也理论结合实验证明了神经演化可以取代 SGD 等现有主流方法用来训练深度强化学习模型，同时取得更好的表现
- http://www.gzhphb.com/article/104/1046105.html
- Jeff Dean 
  - 神经演化是一个非常有潜力的研究方向
  - 另一个方向是稀疏激活的网络
- 在深度学习领域，大家已经习惯了用随机梯度下降 SGD 来训练上百层的、包含几百万个连接的深度神经网络。虽然一开始没能严格地证明 SGD 可以让非凸函数收敛，但许多人都认为 SGD 能够高效地训练神经网络的重要原因是它计算梯度的效率很高
- 借助新开发出的技术，Uber AI 的研究人员已经可以让深度神经网络高效地进化。同时他们也惊讶地发现，一个非常简单的基因算法（genetic algorithm）就可以训练带有超过四百万个参数的卷积网络，让它能够直接看着游戏画面玩 Atari 游戏；这个网络可以在许多游戏里取得比现代深度强化学习算法（比如 DQN 和 A3C）或者进化策略（evolution strategies）更好的表现，同时由于算法有更强的并行能力，还可以运行得比这些常见方法更快
- Uber AI 的研究人员们进一步的研究表明，现代的一些基因算法改进方案，比如新颖性搜索算法（novelty search）不仅在基因算法的效果基础上得到提升，也可以在大规模深度神经网络上工作，甚至还可以改进探索效果、对抗带有欺骗性的问题（带有有挑战性的局部极小值的问题）；Q-learning（DQN）、策略梯度（A3C）、进化策略、基因算法之类的基于反馈最大化思路的算法在这种状况下的表现并不理想
- 基因算法可以在 Frostbite 游戏中玩到 10500 分；而 DQN、A3C 和进化策略的得分都不到 1000 分。
- 五篇新论文简介
  <1>
  《Deep Neuroevolution: Genetic Algorithms are a Competitive Alternative for Training Deep Neural Networks for Reinforcement Learning》
  深度神经进化：在强化学习中，基因算法是训练深度神经网络的有竞争力的替代方案
  论文地址：https://arxiv.org/abs/1712.06567 
  重点内容概要：
  用一个简单、传统、基于群落的基因算法 GA（genetic algorithm）就可以让深度神经网络进化，并且在有难度的强化学习任务中发挥良好表现。在 Atari 游戏中，基因算法的表现和进化策略 ES（evolution strategies）以及基于 Q-learning（DQN）和策略梯度的深度强化学习算法表现一样好。
  深度基因算法「Deep GA」可以成功让具有超过四百万个自由参数的网络进化，这也是有史以来用传统进化算法进化出的最大的神经网络。
  论文中展现出一个有意思的现象：如果想要优化模型表现，在某些情况下沿着梯度走并不是最佳选择
  新颖性搜索算法（Novelty Search）是一种探索算法，它适合处理反馈函数带有欺骗性、或者反馈函数稀疏的情况。把它和深度神经网络结合起来，就可以解决一般的反馈最大化算法（比如基因算法 GA 和进化策略 ES）无法起效的带有欺骗性的高维度问题。
  论文中也体现出，深度基因算法「Deep GA」具有比进化策略 ES、A3C、DQN 更好的并行性能，那么也就有比它们更快的运行速度。这也就带来了顶级的编码压缩能力，可以用几千个字节表示带有数百万个参数的深度神经网络。
  论文中还尝试了在 Atari 上做随机搜索实验。令人惊讶的是，在某些游戏中随机搜索的表现远远好于 DQN、A3C 和进化策略 ES，不过随机搜索的表现总还是不如基因算法 GA。
  <2>
  《Safe Mutations for Deep and Recurrent Neural Networks through Output Gradients》
  通过输出梯度在深度神经网络和循环神经网络中安全地进行突变
  论文地址： https://arxiv.org/abs/1712.06563 
  重点内容概要：
  借助梯度的安全突变 SM-G（Safe mutations through gradients）可以大幅度提升大规模、深度、循环网络中的突变的效果，方法是测量某些特定的连接权重发生改变时网络的敏感程度如何。
  计算输出关于权重的梯度，而不是像传统深度学习那样计算训练误差或者损失函数的梯度，这可以让随机的更新步骤也变得安全、带有探索性。
  以上两种安全突变的过程都不要增加新的尝试或者推演过程。
  实验结果：深度神经网络（超过 100 层）和大规模循环神经网络只通过借助梯度的安全突变 SM-G 的变体就可以高效地进化。
  <3>
  《On the Relationship Between the OpenAI Evolution Strategy and Stochastic Gradient Descent》
  对 OpenAI 的进化策略和随机梯度下降之间的关系的讨论
  论文地址：https://arxiv.org/abs/1712.06564
  重点内容概要：
  在 MNIST 数据集上的不同测试条件下，把进化策略 ES 近似计算出的梯度和随机梯度下降 SGD 精确计算出的梯度进行对比，以此为基础讨论了进化策略 ES 和 SGD 之间的关系。
  开发了快速的代理方法，可以预测不同群落大小下进化策略 ES 的预期表现
  介绍并展示了多种不同的方法用于加速以及提高进化策略 ES 的表现。
  受限扰动的进化策略 ES 在并行化的基础设施上可以大幅运行速度。
  把为 SGD 设计的 mini-batch 这种使用惯例替换为专门设计的进化策略 ES 方法：无 mini-batch 的进化策略 ES，它可以改进对梯度的估计。这种做法中会在算法的每次迭代中，把整个训练 batch 的一个随机子集分配给进化策略 ES 群落中的每一个成员。这种专门为进化策略 ES 设计的方法在同等计算量下可以提高进化策略 ES 的准确度，而且学习曲线即便和 SGD 相比都要顺滑得多。
  在测试中，无 mini-batch 的进化策略 ES 达到了 99% 准确率，这是进化方法在这项有监督学习任务中取得的最好表现。
  以上种种结果都可以表明在强化学习任务中进化策略 ES 比 SGD 更有优势。与有监督学习任务相比，强化学习任务中与环境交互、试错得到的关于模型表现目标的梯度信息的信息量要更少，而这样的环境就更适合进化策略 ES。
  <4>
  《ES Is More Than Just a Traditional Finite Difference Approximator》
  进化策略远不止是一个传统的带来有限个结果的近似方法
  论文地址：https://arxiv.org/abs/1712.06568
  重点内容概要：
  提出了进化策略 ES 和传统产生有限个结果的方法的一个重大区别，即进化策略 ES 优化的是数个解决方案的最优分布（而不是单独一个最优解决方案）。
  得到了一个有意思的结果：进化策略 ES 找到的解决方案对参数扰动有很好的健壮性。比如，作者们通过仿人类步行实验体现出，进化策略 ES 找到的解决方案要比基因算法 GA 和信赖域策略优化 TRPO 找到的类似解决方案对参数扰动的健壮性强得多。
  另一个有意思的结果：进化策略 ES 在传统方法容易困在局部极小值的问题中往往会有很好的表现，反过来说也是。作者们通过几个例子展示出了进化策略 ES 和传统的跟随梯度的方法之间的不同特性。
  <5>
  《Improving Exploration in Evolution Strategies for Deep Reinforcement Learning via a Population of Novelty-Seeking Agents》
  通过一个寻找新颖性的智能体群落，改进用于深度强化学习的进化策略的探索能力
  论文地址：https://arxiv.org/abs/1712.06560
  重点内容概要：
  对进化策略 ES 做了改进，让它可以更好地进行深度探索
  通过形成群落的探索智能体提高小尺度神经网络进化的探索的算法，尤其是新颖性搜索算法（novelty search）和质量多样性算法（quality diversity），可以和进化策略 ES 组合到一起，提高它在稀疏的或者欺骗性的深度强化学习任务中的表现，同时还能够保持同等的可拓展性。
  确认了组合之后得到的新算法新颖性搜索进化策略 NS-ES 和质量多样性进化策略 QD-ES 的变体 NSR-ES 可以避开进化策略 ES 会遇到的局部最优，并在多个不同的任务中取得更好的表现，包括从模拟机器人在欺骗性的陷阱附近走路，到玩高维的、输入图像输出动作的 Atari 游戏等多种任务。
  这一基于群落的探索算法新家庭现在已经加入了深度强化学习工具包。
