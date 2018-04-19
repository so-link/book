# 增强学习交易之DDR

论文[Deep Direct Reinforecement Learning for Financial Signal Representation and Trading](https://ieeexplore.ieee.org/document/7407387/)是Yue Deng等于17年3月发表在IEEE Transaction on Neural Networks and Learning System期刊上的(实际投稿时间是15年)。
这篇论文

## 摘要
这篇论文是基于2001年发表的[Learning to Trade via Direct Reinforcement](https://ieeexplore.ieee.org/document/935097/?arnumber=935097)论文的方法DRL(direct rl)，加了
1. deep network来为市场状态提取更深层的特征表达；
2. fuzzy representation用来降低市场不确定性，模糊化表达市场的状态。
另外也给出了网络的训练方法task-aware BPTT。实验在三个商品期货数据上进行验证，也与DRL, LSTM等方法做了对比。

## 介绍
增强学习是agent在环境中自我学习，寻找策略的过程，在金融交易里就是学习如何在观察市场状态环境后，做出一个能让未来收益最大化的交易动作，比如我在观察股票行情后，是决定买还是卖。这个动作是可以多样的，最简单的买卖，看涨看跌中立，也可以是多只金融产品的投资占比等。
RL在交易中有两个挑战：
1. **对市场环境状态的表达（特征）。**
2. **根据当前状态以及先前动作等做出决策。** 
第一点，金融市场往往是多变的，充满了大量噪声，波动，这就导致了价格曲线的不稳定性。目前有许多人工提取的特征，比如移动平均线，减少了噪声，反应了市场的总体趋势。但是这些特征有些依赖于专家，领域知识，不能完整或深层次地表达市场环境。为了解决这个问题，文章使用**MLP，模糊表达**来对市场状态提取特征。
第二点，使用了RNN形式，**从当前状态和上一个动作到当前动作的映射**。

## DDR
### Direct Reinforcement Trading （DRL）
文章是基于DRL的，所以这一节会先介绍DRL：
定义：
价格 $$p_1, p_2, ..., p_t, ...$$
回报 $$ z_t=p_t-p_{t-1} $$
决策 $$ \delta_t \in \{ long, neutral, short\} = \{1, 0, -1\} $$ 其中long是看涨，neutral是中立，short是看跌。
收益 $$ R_t=\delta_{t-1}z_t-c|\delta_t-\delta_{t-1}| $$ 其中$$\delta_{t-1}z_t$$是执行决策$$\delta_{t-1}$$后得到的回报，c是交易费用，且仅当两次决策不一样时（毁约）才需要交费。
在周期1到T的累积收益函数 $$ U_T\{R_1,...,R_T|\theta\} $$，最直接的就是求和 $$ \sum_{t=1}^T R_t $$。其他复杂的函数比如加了风险调整的收益等也可以作为目标函数。

好了，现在的目标就是如何定义**策略**的结构和学习方法。
<img src="https://upload-images.jianshu.io/upload_images/11731515-751a53e0e1559ce6.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240"  width="250" height="200"/>
策略：$$\delta_t=tanh[(w, f_t)+b+u\delta_{t-1}]$$
其中$$f_t$$是特征向量，在DRL中$$f_t=[z_{t-m+1},...,z_t]$$，即过去m个回报作为特征。然后特征经过线性变换，在加上上一次的动作(构成循环)，经过tanh函数得到-1到1的值，作为当前动作。

在DRL中，**direct指的是直接从状态到动作映射**，而不是学习一个值函数V(或者动作值函数Q)。论文中的解释是这样的：
>In the conventional RL works, the value functions defined in the discrete space are directly iterated by dynamic programming. However, as indicated in [17] and [19], learning the value function directly is not plausible for the dynamic trading problem, because complicated market conditions are hard to be explained within some discrete states.

大致意思就是传统RL是针对离散状态空间，对于交易问题，很难用几个离散状态来表示复杂的市场状态。其实这种说法是不妥的，因为值函数是可以表示连续变量(无穷变量)的。就像游戏一样，输入的游戏画面就是大量的状态，仍然可以用DQN来学习Q函数。

### DNN for DRL
