+ Adversarially Regularized Autoencoders
  本文探讨了用 GAN 作为 autoencoder 的 regularizer 的方法及其在文本生成和离散图像生成中的应用。
  此类方法可以同时得到一个文本生成模型和一个高质量的自编码器：编码空间受到 GAN loss 的约束后，相似离散结构的编码也会比较类似。
  方法稍加扩展则可以用作离散结构的 unaligned style transfer，并取得了这方面的 state-of-the-art results。
  论文链接：https://www.paperweekly.site/papers/1522
  代码链接：https://github.com/jakezhaojb/ARAE
