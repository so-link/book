# Graph Reid系列 结合谱聚类做特征变换
这是图森11月发布在arxiv上的一篇文章，应该是投了CVPR2019，这是Graph+reid系列第四篇文章，前三篇是reranking,[random walk + reid, SGGNN](https://zhuanlan.zhihu.com/p/47162428)，以下是读这篇文章的一点浅见，如有谬误欢迎指出一起讨论：

> Spectral Feature Transformation for Person Re-identification

# 整体结构

![](https://pic4.zhimg.com/v2-82bab3de90c948eeae564448767db34f_b.png)

与商汤的两篇类似，用ResNet提取特征，然后再对batch内所有image依据visual similarity建图（商汤的图和图森的这篇有点区别），然后从graph cut的角度出发，对某个person，希望找到一种cut能将与这个person相同的样本分到一起，不同的分到不同的group；接下来用谱聚类的思路解这个graph cut问题，对特征进行变换，使其包含group信息，然后将这样的特征继续用于CNN pipeline做分类。

商汤的两篇更注重从similarity这个结果在BP中调整图像特征向量，这种做法更像是将reid作为一个verification问题在做，而图森这篇注重的是对特征的变换。

有一点想吐槽的是，因为谱聚类中包含一步特征值分解提取特征向量，这一步的计算量很大而且难以求导，所以这个工作里把这一步特征值分解去掉了，直接通过SGD来求解。因为本质上都是通过矩阵分解来做优化，所以是等价的，但总感觉这样就不是那么谱聚类了。。只剩下了和谱聚类相关的graph cut这样的motivation.

## 谱聚类

谱聚类是一个很常用的聚类方法，这里不细讲原理，论文里其实也简单介绍了一点，可以参考[这篇博客](https://www.cnblogs.com/pinard/p/6221564.html)，文中用到的是谱聚类中的Ncut，通常包括几个步骤：

> 输入：样本集 $D=(x_1,x_2,...,x_n)$ 输出： 簇划分 $C(c_1,c_2,...c_m)$

* 为每个样本提取特征 $f_0$
* 计算各个样本之间的相似度矩阵 $W$ ，构建度数矩阵 $D$
* 计算拉普拉斯矩阵 $L=D-W$
* 构建标准化后的拉普拉斯矩阵 $L' = D^{−1/2}LD^{−1/2}$
* 找到 $L'$ 最小的 $k_1$ 个特征值（需要做特征值分解），构成的特征向量 $f$ ，当 $k_1$ 维数小于 $f_0$ 时，有降维的效果
* 将各自对应的特征向量 $f$ 组成的矩阵按行标准化，最终组成 $n×k_1$ 维的特征矩阵 $F$
* 以 $F$ 的每行为特征向量进行聚类

我们可以发现，在谱聚类的这个过程中，每个样本的特征向量从 $f_0$ 变成了 $f$ ，包含了group信息。

谱聚类在sklearn中也有对应的api：![](https://pic2.zhimg.com/v2-35c7063950fac60a4f82aabaecb12d49_b.png)

可以看到谱聚类适合Few clusters, even cluster size, non-flat geometry，所以如果batch中的类别很多的话，这个套路是不是就不一定work了呢…

> 我有一个大胆的想法，用sklearn里的这堆聚类方法换掉谱聚类，好好调参，好好讲故事，岂不是可以水好几篇CVPR？

## 谱特征变换

首先当然是把batch样本对应的邻接矩阵 $W$ 求出来：

 $w_{ij} = exp(\frac{x_i^Tx_j}{\sigma * ||x_i||_2||x_j||_2})$

然后再对 $W$ 矩阵做行归一化；

其中 $\frac{x_i^Tx_j}{||x_i||_2||x_j||_2}$ 是我们熟悉的余弦相似度， $\sigma$ 是一个温度参数（就是一个线性系数用来做bias进一步拟合数据集的），做过这方面实验的同学应该知道，余弦相似度虽然用来做ranking效果挺好的，但是这些相似度本身区分度不大，都是0.9x-0.8x，这样概率转移矩阵 $W$ 就没什么意义了。所以用了一个softmax做归一化，使得转移概率具有区分度。

一个很直接的想法是，在前向的过程中，对一个batch的数据直接加入上述的谱聚类过程，然而找特征值需要做特征值分解，开销很大而且不好求导，所以文中用了另一种方法来近似：

 $f = fW$

但从这个式子来看， $f_i = \sum_j^n f_i * W_{ij}$ 是依据转移概率，将 $f$ 和它的邻居们进行加权求和，可以视为做了一次传播，然后把 $f$ 输入到classifier，做person reid的分类，利用SGD来优化 $f$ 和 $W$ 。

文章里没有说为什么对 $f$ 的优化等价于对谱聚类中 $f$ 导出的拉普拉斯矩阵 $L$ 做的特征值分解，只是在section3.2做了一堆graph cut的推导之后，在最后一段说：This can be simply implemented by multiplying $T$ (也就是归一化后的 $W$ ) with origin feature $X$ .

这种等价性意味着，用SGD优化收敛后得到的feature等价于奇异值分解得到的feature。虽然两者都是由 $f$ 推出 $W$ ，然后用 $W$ 得到更好的 $f$ ，再用 $f$ 去做其他的任务（后边的person reid分类），输入输出和motivation是一致的，但直接这么说它俩等价感觉还是有点跳，记得NLP那边Google曾证明过word embedding是等价于某种分解，并且奇异值分解可以达到近似的效果，但这边的等价性我觉得还要另外好好推导一下。

## 训练

由于训练初始 $f$ 随机，求出来的 $W$ 是随机的，直接使用transform feature $fW$ 很容易训崩，所以文中同时把 $f$ 和 $fW$ 都放到classifier中去训练，使得训练过程稳定，并且保证 $f$ 和 $fW$ 处于同一空间

## Result

![](https://pic3.zhimg.com/v2-31f3763810b1edd52cadbad489130efe_b.png)

从Ablation study看，各个组件在有监督训练中当然是有提升了，但是把谱聚类放到post processing中，提升并没有reranking大（大家比较关注mAP对吧），其实我比较想看第一行baseline+reranking是不是比倒数第二行更好。。

## Summary

这是graph+reid系列工作中，直接优化feature而非优化similarity的，更像reid了。这个领域应该还有很多坑可以挖，坐等CVPR2019吃瓜。