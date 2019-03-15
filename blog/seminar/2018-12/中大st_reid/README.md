# AAAI2018 Spatial-Temporal Person Re-identification
这是中山大学赖剑煌老师团队发表在AAAI2018上的一篇论文，赖老师团队在reid领域耕耘很久，郑伟诗老师也在这个团队，有很多有影响力的工作。这篇论文在有监督的reid上结合了时空数据做多模态融合，在market和duke两个数据集上达到了超越所有纯视觉有监督方法的效果。

实际上时空+视觉的reid我们在CVPR2018就提出过[TFusion](https://zhuanlan.zhihu.com/p/34778414)，并且当时做的是一个无监督的版本，包含了无监督的时空模型估计，贝叶斯的多模态融合，以及learning to rank迁移学习。

在我们看来，aaai这篇是对TFusion中时空模型估计和多模态融合在有监督的条件下做了更细致的优化，从而达到了更好的效果。

> 以下是本人对这篇文章的一些理解，如有错漏欢迎指出探讨

## Overview

![](https://pic1.zhimg.com/v2-f759acaacef182427c08d4c9d7e2f800_b.png)

非常简单的架构，用ResNet50+PCB提取视觉特征和视觉相似度，在从数据集中提取时间和摄像头编号构造行人迁移的时空模型，算出时空相似度，然后对两个分数做一点精修再相乘，得到融合的分数。

## 视觉模型

ResNet50+PCB，很简单有效的模型，可以在Market上把rank1直接怼到91，可以用郑哲东开源的[person reid pytorch baseline](https://github.com/layumi/Person_reID_baseline_pytorch)复现。不多赘述。

## 时空模型

可以参考我们[TFusion解读](https://zhuanlan.zhihu.com/p/34778414)中关于时空数据的分析，这里的出发点也很相似，但是做了许多的优化。

## Spatial-temporal histogram

 $\hat{p}(y=1|k,c_i,c_j) = \frac{n_{c_i,c_j}^k}{\sum_l n_{c_i,c_j}^l}$

与TFusion类似，用某个时间区间内的历史样本数量（分子）除以总样本数（分母）得到一个时空概率，与TFusion不同的是，这里把统计出来的时空概率按迁移时间 $t$ 分成了很多个bin（100帧一个bin），每个bin里的样本时空概率是相同的，求解时空概率时，只需要看 $t$ 是落在哪个bin内，比如说是第 $k$ 个bin，就返回这个bin的极大似然估计结果。这种做法的测试时效率会比TFusion高一点。

> 代码里在统计历史样本数量的时候，还对一个人在某个摄像头的多次出现的时间做了平均，这样可以避免某个人在摄像头中出现时间太长引入的统计误差。

## Parzen Window smooth

 $p(y=1|k,c_i,c_j) = \frac{1}{Z}\sum_t \hat{p}(y=1|k,c_i,c_j) K(1-k)$

看着很复杂，但是记住一个目标就是smooth！当你要计算第 $k$ 个bin的分数时，还会统计周围几个bin的分数（看了代码，统计的是全部bin的分数，），用 $K(·)$ 这个函数求第 $l$ 个bin的平滑权重，这个 $K(·)$ 是一个高斯函数，也就是离得越远权重越小，实际上这种平滑TFusion代码里也有做，只是我们觉得太细枝末节了没有讲。

## Joint Metric

视觉分数和时空分数要怎么结合呢？我们在TFusion中也讲了，直接相乘是不行的，在这篇论文里也分析了，两个分数，一高一低相乘会低与两个中等的分数，这不太符合需求。所以这篇论文做了一个操作将分数限制在一个范围，并对最终分数做sigmoid激活，这样相乘效果就比较好了。

## Laplace smoothing

 $p_{\lambda(Y=d_k)}=\frac{m_k+\lambda}{M+D\lambda}$

 $\frac{m_k}{M}$ 是类别为 $d_k$ 的样本在所有样本中的比例作为一个先验。但是这里其实讲得很模糊，类别是啥？这是哪个分数的先验？如果这是做视觉分数的先验，训练集的类别跟测试集的类别是不重叠的，训练集的先验概率是不能用在测试集上的；如果是时空分数的先验，那就是简单的未平滑的时空概率了， $k$ 应该是指第 $k$ 个bin，说成第 $k$ 个category更难理解。对于这个未平滑的时空，分子加一个 $\lambda$ ，分母加一个 $D\lambda$ ，这样 $p_{\lambda(Y=d_k)}$ 就被限制在 $[\frac{1}{D}, \frac{m_k}{M}]$ 这个区间里，这样就可以防止出现比较小的时空分数。

> 话说我在代码里没找到这部分逻辑，如果有找到的同学欢迎留言。从代码中统计历史数据的逻辑看，也可能是指第 $k$ 个person id，为 $d_k$ 的样本在所有样本中的比例作为一个先验。但测试集没有 $d_k$ 的person id啊，搞不懂…
> 另外我觉得这种时空分数信息量比较小，在视觉比较强的时候效果会比较好，在视觉比较弱（比如无监督或者迁移学习场景下），这样的时空帮不上什么忙。

## Logistic function

正常的带超参数的sigmoid激活：

 $f(x;\lambda;\gamma)= \frac{1}{1+\lambda e^{-\gamma x}}$

于是最终的分数

 $p_{joint} = f(s;\lambda_0\gamma_0)f(p_{st};\lambda_1\gamma_1)$

激活是对视觉和时空都做的，并且两个分数各有两个超参数来做权衡（这样就能通过调参把分数调上去了）。

## 实验

实验数据集就是market和duke，没有常见的cuhk03，原因在TFusion中也说过，只有这两个数据集和grid数据集有时间数据。

实验效果还是挺好的，多模态方法好好调参完虐纯视觉方法。实验中也和SOTA方法做了对比，包括我们的TFusion，当时做了一个source和target都一样的实验，但是还是用无监督的时空估计方法，所以准确率还是aaai这篇比较高。然而他们说

> Therefore, TFusion-sup actually does not investigate how to estimate the spatial-temporal probability distribution and how to model the joint probability of the visual similarity and the spatial-temporal probability distribution.

这一段我们是不认的，仔细读我们的论文就知道我们是有做这两个事情的，并且是在无监督条件下。当然这篇能在有监督下把多模态融合和时空模型估计做到极致也是很有价值的。