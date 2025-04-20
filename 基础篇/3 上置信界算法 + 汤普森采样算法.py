import numpy as np
import matplotlib.pyplot as plt


class BernoulliBandit:
    """ 伯努利多臂老虎机,输入K表示拉杆个数 """
    def __init__(self, K):
        self.probs = np.random.uniform(size=K)  # 随机生成K个0～1的数,作为拉动每根拉杆的获奖
        # 概率
        self.best_idx = np.argmax(self.probs)  # 获奖概率最大的拉杆
        self.best_prob = self.probs[self.best_idx]  # 最大的获奖概率
        self.K = K

    def step(self, k):
        # 当玩家选择了k号拉杆后,根据拉动该老虎机的k号拉杆获得奖励的概率返回1（获奖）或0（未
        # 获奖）
        if np.random.rand() < self.probs[k]:
            return 1
        else:
            return 0


np.random.seed(1)  # 设定随机种子,使实验具有可重复性
K = 10
bandit_10_arm = BernoulliBandit(K)


class Solver:
    """ 多臂老虎机算法基本框架 """
    def __init__(self, bandit):
        self.bandit = bandit
        self.counts = np.zeros(self.bandit.K)  # 每根拉杆的尝试次数
        self.regret = 0.  # 当前步的累积懊悔
        self.actions = []  # 维护一个列表,记录每一步的动作
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
# upper confidence bound, UCB
# 上置信界算法介绍
    # 引入一个不确定性的度量U（a），一根拉杆的不确定性越大，就越具有探索价值
    # 该算法利用了“霍夫丁不等式”，该不等式刻画了某个变量的期望大于经验期望 + u的概率，不大于e^(-2nu^2)
    # P{E|X| >= avg(x_n) + u} <= e^(-2nu^2)

    # 上置信界算法会选择奖励期望上界最大的动作。给定概率P，可以求出不确定性度量U（a）
    # 直观的说，UCB算法在每次选择拉杆前，会计算每根拉杆的期望奖励上界
    # 每次拉动拉杆的期望奖励只有一个较小的概率P超过这个上界
    # 接着选择奖励期望上界最大的拉杆

class UCB(Solver):
    """ UCB算法,继承Solver类 """
    def __init__(self, bandit, coef, init_prob=1.0):
        super(UCB, self).__init__(bandit)
        self.total_count = 0
        self.estimates = np.array([init_prob] * self.bandit.K)
        self.coef = coef

    def run_one_step(self):
        self.total_count += 1
        ucb = self.estimates + self.coef * np.sqrt(
            np.log(self.total_count) / (2 * (self.counts + 1)))  # 计算上置信界
        k = np.argmax(ucb)  # 选出上置信界最大的拉杆
        r = self.bandit.step(k)
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])
        return k


np.random.seed(1)
coef = 1  # 控制不确定性比重的系数
UCB_solver = UCB(bandit_10_arm, coef)
UCB_solver.run(5000)
print('上置信界算法的累积懊悔为：', UCB_solver.regret)
plot_results([UCB_solver], ["UCB"])


# Thompson sampling
# 汤普森采样算法介绍
    # 先假设拉动每根拉杆的奖励服从一个特定的概率分布
    # 根据拉动每根拉杆的奖励期望来进行选择，但是计算所有的期望奖励，训练代价比较高，采用采样的方法
    # 对每个动作a的奖励概率分布进行一轮采样，得到一组奖励样本，再选择样本中奖励最大的动作
    # 是一种计算所有拉杆的最高奖励概率的蒙特卡洛采样方法
    # 蒙特卡洛采样方法介绍：https://zhuanlan.zhihu.com/p/338103692

    # 怎样获得每个动作的奖励分布并在过程中更新？
    # 使用beta分布对每个动作的奖励概率进行建模
    # 具体来说，若某拉杆被选择了k次，其中m_1次奖励为 1， m_2次奖励为 0，则该拉杆的奖励服从参数为(m_1 + 1, m_2 + 1)的 Beta 分布
    # Beta分布参考链接：https://blog.csdn.net/a358463121/article/details/52562940
    # 详细的：https://zhuanlan.zhihu.com/p/149964631
    # 其实关键公式，就是Γ（a）*Γ（b） / Γ（a+b）

    # 算法介绍
class ThompsonSampling(Solver):
    """ 汤普森采样算法,继承Solver类 """
    def __init__(self, bandit):
        super(ThompsonSampling, self).__init__(bandit)
        self._a = np.ones(self.bandit.K)  # 列表,表示每根拉杆奖励为1的次数
        self._b = np.ones(self.bandit.K)  # 列表,表示每根拉杆奖励为0的次数

    def run_one_step(self):
        samples = np.random.beta(self._a, self._b)  # 按照Beta分布采样一组奖励样本
        k = np.argmax(samples)  # 选出采样奖励最大的拉杆
        r = self.bandit.step(k)

        self._a[k] += r  # 更新Beta分布的第一个参数
        self._b[k] += (1 - r)  # 更新Beta分布的第二个参数
        return k


np.random.seed(1)
thompson_sampling_solver = ThompsonSampling(bandit_10_arm)
thompson_sampling_solver.run(5000)
print('汤普森采样算法的累积懊悔为：', thompson_sampling_solver.regret)
plot_results([thompson_sampling_solver], ["ThompsonSampling"])



# 小结
# e贪婪算法的累计懊悔是随时间线性增长的，通过e衰减可以实现斜率的逐渐减小
# 而汤普森采样、上置信界算法都是随时间对数增长的

# 多臂老虎机和强化学习的一大区别是，其与环境的交互不会改变环境，每次交互的结果和以往的动作无关。所以为：无状态的强化学习