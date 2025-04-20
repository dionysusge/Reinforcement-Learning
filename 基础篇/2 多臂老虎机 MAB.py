# Multi-armed bandit
# 一、问题介绍
## 问题简介
    # 可以看作简化版的强化学习，因为其不存在状态信息，只有动作和奖励，算是最简单的“和环境交互中的学习”

    # 多臂老虎机问题
    # 假设老虎机有K个拉杆，拉动每一个拉杆都对应一个关于奖励的概率分布。每拉动一次拉杆会从奖励堆里按照概率随机获得一个奖励

    # 目标是在每根拉杆奖励概率分布未知的情况下，从头开始尝试，在拉动T次后，获得尽可能高的累计奖励
    # 由于奖励的概率分布未知，我们需要在“探索拉杆的获奖概率”和“根据经验选择获奖最多的拉杆”中进行权衡，以探索采取怎样的策略额能使获得的累计奖励最高

## 形式化的描述
    # 多臂老虎机问题可以表示为一个元组（A，R）
    # 其中A为动作集合，表示拉动某个拉杆，有多少根就有多少个动作a_i
    # R表示奖励的概率分布，拉动每一个拉杆的动作都对应一个奖励概率分布
    # 那么目标即为最大化T步内累计的奖励，a_i表示在第t时间步拉动某一拉杆的动作，r_i表示a_i获得的奖励

## 累计懊悔
    # 对于每个动作a，定义其奖励的期望E（R）
    # 于是肯定存在一根拉杆，其奖励的期望最大，记为E（R）_max
    # 引入“懊悔”：拉动当前杆的动作a获得的奖励与最大奖励期望的差值
    # 累计懊悔：操作T次后的懊悔总量
    # 那么MAB问题的目标即为，最大化累计奖励，也即最小化累计懊悔

## 估计期望奖励
    # 由于不知道奖励的概率分布，必须从每次实验结果里更新
    # 采用增量式更新的方法代替所有数求和再求平均，时间复杂度为o（1）
    # 下面编码实现一个拉杆数为10的多臂老虎机

# 导入需要使用的库,其中numpy是支持数组和矩阵运算的科学计算库,而matplotlib是绘图库
import numpy as np
import matplotlib.pyplot as plt


class BernoulliBandit:
    """ 伯努利多臂老虎机,输入K表示拉杆个数 """
    def __init__(self, K):
        self.probs = np.random.uniform(size=K)  # 随机生成K个0～1范围内的数,作为拉动每根拉杆的获奖概率
        self.best_idx = np.argmax(self.probs)  # 获奖概率最大的拉杆
        self.best_prob = self.probs[self.best_idx]  # 最大的获奖概率
        self.K = K

    def step(self, k):
        # 当玩家选择了k号拉杆后,根据拉动该老虎机的k号拉杆获得奖励的概率返回1（获奖）或0（未获奖）
        if np.random.rand() < self.probs[k]:
            return 1
        else:
            return 0


np.random.seed(1)  # 设定随机种子,使实验具有可重复性
K = 10
bandit_10_arm = BernoulliBandit(K)
print("随机生成了一个%d臂伯努利老虎机" % K)
print("获奖概率最大的拉杆为%d号,其获奖概率为%.4f" %
      (bandit_10_arm.best_idx, bandit_10_arm.best_prob))

# 下面使用Solver类实现多臂老虎机的求解，需要实现以下函数功能
    # 根据策略选择动作
    # 根据动作获取奖励
    # 更新期望奖励的估值

class Solver:
    """ 多臂老虎机算法基本框架 """
    def __init__(self, bandit):
        self.bandit = bandit
        self.counts = np.zeros(self.bandit.K)  # # 初始化一个长度为 K 的数组，用于记录每根拉杆的尝试次数，初始值都为 0
        self.regret = 0.  # 初始化当前步的累积懊悔为 0
        self.actions = []  # 维护一个列表,记录每一步选择的动作
        self.regrets = []  # 维护一个列表,记录每一步的累积懊悔

    def update_regret(self, k):
        # 计算累积懊悔并保存,k为本次动作选择的拉杆的编号
        self.regret += self.bandit.best_prob - self.bandit.probs[k]
        self.regrets.append(self.regret)

    def run_one_step(self):
        # 返回当前动作选择哪一根拉杆,由每个具体的策略实现
        raise NotImplementedError

    def run(self, num_steps):
        # 运行一定次数,num_steps为总运行次数
        for _ in range(num_steps):
            k = self.run_one_step()
            self.counts[k] += 1
            self.actions.append(k)
            self.update_regret(k)

## 上面的solver类只给了run_one_step函数的接口，具体的策略需要继承solver类，实现run_one_step函数

# 二、探索与利用的平衡
    # 在上面的算法框架里，还没有策略说要采取哪个动作
    # 下面将学习如何设计一个策略，需要解决“探索”和“利用”的平衡问题
    # 探索是指尝试拉动更多可能的拉杆，这根拉杆不一定获得最大的奖励，但是可以摸清楚拉杆的一些情况
    # 利用是指拉动目前已知情况下，奖励期望最大的那根拉杆。但是目前的最优未必是全局最优

    # 所以在多臂老虎机问题，设计策略时就需要平衡这两者，使得奖励最大化

    # 一个常用的思路是，在开始时做比较多的探索，在对每根拉杆都有比较准确的估计后，再进行利用
    # 下面介绍一些这个思路下的经典算法

## 三、常见算法介绍

# 1、epsilon-贪心算法
    # 完全贪婪算法在每一时刻采取奖励期望最大的动作，就成了纯粹的利用，没有探索
    # epsilon-greedy算法，其实只是简单的添加了噪声epsilon，下面用e简写一下
    # 每次动作以1-e的概率选择奖励期望最大的拉杆，以e的概率随机选择一根
    # 当探索次数不断增加，对每个动作的奖励估计会越来越准确，探索次数可以减少，所以可以令e随时间衰减，但不会在有限的步骤衰减到0，否则仍是局部最优
    # 下面编码实现

class EpsilonGreedy(Solver):
    """ epsilon贪婪算法,继承Solver类 """
    def __init__(self, bandit, epsilon=0.01, init_prob=1.0):
        super(EpsilonGreedy, self).__init__(bandit)
        self.epsilon = epsilon
        # 初始化拉动所有拉杆的期望奖励估值
        self.estimates = np.array([init_prob] * self.bandit.K)

    def run_one_step(self):
        if np.random.random() < self.epsilon:
            k = np.random.randint(0, self.bandit.K)  # 随机选择一根拉杆
        else:
            k = np.argmax(self.estimates)  # 选择期望奖励估值最大的拉杆
        r = self.bandit.step(k)  # 得到本次动作的奖励
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])
        return k

# 通过可视化直观展示一下每一时间步的累计函数

def plot_results(solvers, solver_names):
    """生成累积懊悔随时间变化的图像。输入solvers是一个列表,列表中的每个元素是一种特定的策略。
    而solver_names也是一个列表,存储每个策略的名称"""
    for idx, solver in enumerate(solvers):
        time_list = range(len(solver.regrets))
        plt.plot(time_list, solver.regrets, label=solver_names[idx])
    plt.xlabel('Time steps')
    plt.ylabel('Cumulative regrets')
    plt.title('%d-armed bandit' % solvers[0].bandit.K)
    plt.legend()
    plt.show()


np.random.seed(1)
epsilon_greedy_solver = EpsilonGreedy(bandit_10_arm, epsilon=0.01)
epsilon_greedy_solver.run(50000)
print('epsilon-贪婪算法的累积懊悔为：', epsilon_greedy_solver.regret)
plot_results([epsilon_greedy_solver], ["EpsilonGreedy"])


# 上面实验结果可视化后发现，累计懊悔和时间成线性关系
# 采用不同的e值看一下：
np.random.seed(0)
epsilons = [1e-4, 0.01, 0.1, 0.25, 0.5]
epsilon_greedy_solver_list = [
    EpsilonGreedy(bandit_10_arm, epsilon=e) for e in epsilons
]
epsilon_greedy_solver_names = ["epsilon={}".format(e) for e in epsilons]
for solver in epsilon_greedy_solver_list:
    solver.run(5000)

plot_results(epsilon_greedy_solver_list, epsilon_greedy_solver_names)

# 可以发现，e值越小，也即越贪婪，那么累计的懊悔值就越小，但仍是线性的
# 下面采用e随时间减小的算法，具体的衰减形式采用反比例衰减（不会为0），其实可以预想到，斜率逐渐变小，那图形大概也知道结果了

class DecayingEpsilonGreedy(Solver):
    """ epsilon值随时间衰减的epsilon-贪婪算法,继承Solver类 """
    def __init__(self, bandit, init_prob=1.0):
        super(DecayingEpsilonGreedy, self).__init__(bandit)
        self.estimates = np.array([init_prob] * self.bandit.K)
        self.total_count = 0

    def run_one_step(self):
        self.total_count += 1
        if np.random.random() < 1 / self.total_count:  # epsilon值随时间衰减
            k = np.random.randint(0, self.bandit.K)
        else:
            k = np.argmax(self.estimates)

        r = self.bandit.step(k)
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])

        return k


np.random.seed(1)
decaying_epsilon_greedy_solver = DecayingEpsilonGreedy(bandit_10_arm)
decaying_epsilon_greedy_solver.run(5000)
print('epsilon值衰减的贪婪算法的累积懊悔为：', decaying_epsilon_greedy_solver.regret)
plot_results([decaying_epsilon_greedy_solver], ["DecayingEpsilonGreedy"])