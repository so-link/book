# 增强学习交易之DDR

论文[Deep Direct Reinforecement Learning for Financial Signal Representation and Trading](https://ieeexplore.ieee.org/document/7407387/)是Yue Deng等于17年3月发表在IEEE Transaction on Neural Networks and Learning System期刊上的(实际投稿时间是15年)。
这篇论文

## 1. 摘要
这篇论文是基于2001年发表的[Learning to Trade via Direct Reinforcement](https://ieeexplore.ieee.org/document/935097/?arnumber=935097)论文的方法DRL(direct rl)，加了
1. deep network来为市场状态提取更深层的特征表达；
2. fuzzy representation用来降低市场不确定性，模糊化表达市场的状态。
另外也给出了网络的训练方法task-aware BPTT。实验在三个商品期货数据上进行验证，也与DRL, LSTM等方法做了对比。

## 2. 介绍
增强学习是agent在环境中自我学习，寻找策略的过程，在金融交易里就是学习如何在观察市场状态环境后，做出一个能让未来收益最大化的交易动作，比如我在观察股票行情后，是决定买还是卖。这个动作是可以多样的，最简单的买卖，看涨看跌中立，也可以是多只金融产品的投资占比等。
RL在交易中有两个挑战：
1. **对市场环境状态的表达（特征）。**
2. **根据当前状态以及先前动作等做出决策。** 
第一点，金融市场往往是多变的，充满了大量噪声，波动，这就导致了价格曲线的不稳定性。目前有许多人工提取的特征，比如移动平均线，减少了噪声，反应了市场的总体趋势。但是这些特征有些依赖于专家，领域知识，不能完整或深层次地表达市场环境。为了解决这个问题，文章使用**AE，模糊表达**来对市场状态提取特征。
第二点，使用了RNN形式，**从当前状态和上一个动作到当前动作的直接映射**。

## 3. 算法DDR
### Direct Reinforcement Trading （DRL）
文章是基于DRL的，所以这一节会先介绍DRL：
定义：
价格 $p_1, p_2, ..., p_t, ...$
回报 $ z_t=p_t-p_{t-1} $
决策 $ \delta_t \in \{ long, neutral, short\} = \{1, 0, -1\} $ 其中long是看涨，neutral是中立，short是看跌。
收益 $ R_t=\delta_{t-1}z_t-c|\delta_t-\delta_{t-1}| $ 其中$\delta_{t-1}z_t$是执行决策$\delta_{t-1}$后得到的回报，c是交易费用，且仅当两次决策不一样时（毁约）才需要交费。
在周期1到T的累积收益函数 $ U_T\{R_1,...,R_T|\theta\} $，最直接的就是求和 $ \sum_{t=1}^T R_t $。其他复杂的函数比如加了风险调整的收益等也可以作为目标函数。

好了，现在的目标就是如何定义**策略**的结构和学习方法。
![drl.png](https://upload-images.jianshu.io/upload_images/11731515-751a53e0e1559ce6.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/300)

策略：$\delta_t=tanh[(w, f_t)+b+u\delta_{t-1}]$
其中$f_t$是特征向量，在DRL中$f_t=[z_{t-m+1},...,z_t]$，即过去m个回报作为特征。然后特征经过线性变换，在加上上一次的动作(构成循环)，经过tanh函数得到-1到1的值，作为当前动作。

在DRL中，**direct指的是直接从状态到动作映射**，而不是学习一个值函数V(或者动作值函数Q)。论文中的解释是这样的：
>In the conventional RL works, the value functions defined in the discrete space are directly iterated by dynamic programming. However, as indicated in [17] and [19], learning the value function directly is not plausible for the dynamic trading problem, because complicated market conditions are hard to be explained within some discrete states.

大致意思就是传统RL是针对离散状态空间，对于交易问题，很难用几个离散状态来表示复杂的市场状态。其实这种说法是不妥的，因为值函数是可以表示连续变量(无穷变量)的。就像游戏一样，输入的游戏画面就是大量的状态，仍然可以用DQN来学习Q函数。

### DNN 与 fuzzy 模糊表达
![fig2.png](https://upload-images.jianshu.io/upload_images/11731515-c648a99ab4c90ef1.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/600)

1. 在drl的基础上，用dnn来提取特征。
2. 市场的行情是有很多噪声，波动的，为了减少不确定性，使用了fuzzy learning. 就是对原始数据进行模糊表达，相当于预处理。
**模糊表达将每个数据表达为k个模糊成员组**(fuzzy  membership groups)，比如对于市场行情，可分为增长，下降，无趋势三组。对于每个组会有一个组函数vi() R->[0, 1]：
![eq7.png](https://upload-images.jianshu.io/upload_images/11731515-650331be20bf93bd.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/400)
可以看出是一个高斯函数。每一维会经过组函数得到3个组对应的值，值越大说明属于这个组的概率越大，越接近中心点
最终，这个优化的目标可以表示为
![eq8.png](https://upload-images.jianshu.io/upload_images/11731515-ea0711de45dbf6bf.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/400)


### 参数学习
介绍完架构以及优化函数后，如何得到模型参数呢？文章先初始化参数，然后再进行调优fine-tuning.

#### (1)初始化
1. fuzzy: 使用k-means将数据聚为三类，分别算出均值和方差作为高斯函数的参数。
2. dnn part: 这部分的初始化其实就是一个深度置信网络dbn。使用三层结构，定义loss为
公式
从x重构到x，训练完后把最后一层去掉，如隐藏层作为特征。重复n次。
3. DRL part: 固定前面参数，优化drl部分的参数。

#### (2)fine tuning （task-aware BPTT）
根据目标函数UT和链式规则对参数求导：
![gradient of function Ut](https://upload-images.jianshu.io/upload_images/11731515-9df44cb7bf2399ea.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/450)
第二个式子是一个递归的形式，我们将网络按时间展开，可以得到
![BPTT](https://upload-images.jianshu.io/upload_images/11731515-a964a999163f393e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/600)
展开后对每个时间段我们能求得一个梯度，然后将每个参数的所有梯度取平均。这是对delta_t求得的梯度，另外我们对每个Rt都要算一次梯度，然后求和（平均？），最后我们得到参数的梯度的更新值。
另外加了红线部分，是对Ut-1，Ut-2,.... 也进行求导。文中说明是为了解决DNN部分梯度消失的问题。这两个部分称为1) the previous time stack (lower order time delay) and 2) the reward function (learning task)  作者将这种方式的权重更新称为task-aware BPTT。
（另外一种对红线部分的理解是对Rt-1，Rt-2的求导）

#### 算法总结
最终，用伪代码表示为：
![algorithm](https://upload-images.jianshu.io/upload_images/11731515-91de7efe03be4f6b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/500)

## 4. 实验
### 实验设置
（1）实验选取了三个期货合约: 股指期货IF, 白银期货AG，白糖期货SU，用的是每分钟的数据(属于T+0):
![prices](https://upload-images.jianshu.io/upload_images/11731515-066468161b9c81df.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/800)
期货的一些信息：
![summary_of_contracts.png](https://upload-images.jianshu.io/upload_images/11731515-85b4fba0d8a1f60f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/600)
CNY/pnt是内在价值，每增长或下降一个point所能得到的回报；TC，c是交易费用（考虑其他风险因素，实际上高于真实的5倍）。
（2）模型的参数
输入是过去45min的历史价格加上过去3h, 5h, 1 day, 3 day, 10day的变化值，一共50个输入，得到的fuzzy层就是150；
DNN层设置是128，128，128，20.

### 训练
训练集：前15000点用作训练模型；当模型在测试/预测了5000个点后，会用最近的15000点作为新的训练集，重新训练一次模型，让模型能拟合最近数据。
验证集：12000用作训练，3000用作验证，防止过拟合
训练过程：early stopping；learning rate；100epoch
文中用task-aware BPTT和BPTT两种方法分别训练，进行对比，结果如下图：
![fig5.png](https://upload-images.jianshu.io/upload_images/11731515-f4bb6b1bb5a9e463.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/700)
可以看出用了task-aware效果提升了。

### 实验结果
文中将提出的DDR，FDDR与DRL，SCOT进行了对比。
前面提到的累积回报函数记为TP；另一个常用可替换的收益函数是夏普率SR（单位风险得到的收益）：
![SR.png](https://upload-images.jianshu.io/upload_images/11731515-08cb34b56710a716.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/300)
论文对两个目标分别做了实验。
![fig6.png](https://upload-images.jianshu.io/upload_images/11731515-93cd2c830937ae7d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
P&L是profit & loss。第一行是期货价格，第二行是以TP作为收益函数的，第三行是以SR作为收益函数。结论就是DDR，FDDR好于DRL，SCOT；不同函数的区别就不是很明显。
最后的收益用表格展示就是：
![table2.PNG](https://upload-images.jianshu.io/upload_images/11731515-4744be3d31c50b42.PNG?imageMogr2/auto-orient/strip%7CimageView2/2/w/900)

### 与基于预测的DL方法对比
对比的DL有CDNN，RNN和LSTM。预测模型就是一个softmax方式的三分类。
metrics: PR(profitable rate)，TT(trading times).
![table3.png](https://upload-images.jianshu.io/upload_images/11731515-f1f1f1b3f363a7c6.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/600)
收益率都是很小的，比一半多一点，但是最终还是有收益的。交易次数是前三个方法多，FDDR只有他们的十分之一左右。如果不考虑cost，那么前三个的收益是要高于FDDR的；考虑cost时，其他方法因为交易次数多使得手续费高，收益就降低了许多。
这种结果的原因是FDDR是考虑了上一个动作和交易费用的，所以交易次数就少了。但是可以看出LSTM等方法的潜力，如果他们也把费用考虑在内的话。

### S&P 500验证
S&P500是标准普尔500股票指数(美国)，数据从1990到2015年，以天为间隔共6500条记录。其中2000条用于训练，每个100训练一次。
因为指数很受经济危机影响，所以将其他国家的股票指数变化也加入作为特征，包括英国，香港，中国等。将多特征的FDDR记为multi-FDDR。实验结果为：
![fig7.png](https://upload-images.jianshu.io/upload_images/11731515-78a0c851b372deea.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/800)
在2010后multi-FDDR才超过FDDR，作者的解释是在2010年后，越来越多的算法交易公司参与到市场中，导致价格是多相关的。

### 不同参数的影响(鲁棒性)
测试的参数有DNN层数l，节点数N，展开的时间段(time stacks) τ。
![table4.png](https://upload-images.jianshu.io/upload_images/11731515-e4908b7ebdc6c334.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/800)
可以看出层数越深，效果变好；节点数和时间段的影响就不是很大。

## 5. 总结
文章在DRL的基础之上，用了模糊表达和多层神经网络来提取特征。DRl有别于建模值函数，是一种从状态到动作的直接映射，且考虑了上一动作和交易费用在内，取得还不错的结果。
事实上，论文有些地方没有讲明白，比如动作输出是-1到1的连续值，而真正交易动作是-1，0，1，作者没有给出每个的范围(可能在DRL论文有吧)。第二点就是BPTT中的红线部分含义了，如果有人也读了论文可以一起讨论讨论。

## 6. 想法
1. 状态表达方式；
2. 对过去历史的建立的模型，像LSTM这种长短时记忆时序模型等；
3. 上一动作；
4. 模糊表达这个也是有点意思的，不过文章中的方式我是不是很认可的，感觉就是一个高斯激活函数；
5. 训练方法，用值函数形式；
6. 多尺度：文中也用了几小时，几天前数据，也算是一种多尺度吧；
7. 多模态：结合多种其他市场行情；像交通预测中的股票embedding，得到不同股票间语义信息，比如相同行业间相关性是比较高的；股民情感分析；新闻文本数据，事件；....
8. GAN生成序列用于加强训练；
........暂时想到这么多，还是有挺多可以研究的地方。当然了，这些是要一步一步做起的。

**————原作西瓜小王子，转载请注明。**