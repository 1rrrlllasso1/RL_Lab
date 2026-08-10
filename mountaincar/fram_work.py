"""
fram_work.py —— MountainCar-v0（爬坡小车）强化学习测试主框架

【怎么用】
  1. 在下方 __main__ 中改 ALGORITHM 选择算法（三选一）
  2. 直接运行本文件，训练完成后自动弹出奖励曲线图

【环境说明】
  MountainCar-v0：小车被困在山谷里，引擎动力不足，
  必须来回荡（先往左冲积累动能，再往右冲）才能借惯性跃上山顶。
  - 状态(2维): [小车位置, 小车速度]
  - 动作(3个): 0 = 向左推, 1 = 不动, 2 = 向右推
  - 奖励: 每步 -1（希望尽快到达山顶），到达山顶位置 ≥ 0.5 时本轮结束
  - 最大步数: 200（gym 自动截断）

【难点说明】
  MountainCar 是"稀疏奖励"环境：只有到达山顶才知道成功，
  中途全是 -1。所以需要比 CartPole 多很多的轮数，
  且 γ 不宜太大（0.95 即可，0.99 会过度看重远期导致学得慢）。
"""

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

from utils import build_bins, discretize
from Q_learning import QLearningAgent
from sarsa import SarsaAgent
from gradient import GradientAgent


# ==================== 环境配置区 ====================
ENV_NAME = "MountainCar-v0"

# 状态取值范围（MountainCar 的观测空间本身有界，直接用）
OBS_BOUNDS = [
    (-1.2, 0.6),    # 小车位置 position
    (-0.07, 0.07),  # 小车速度 velocity
]

# 分箱数量：位置 20 箱 + 速度 20 箱 = 400 个状态
STATE_BINS = [20, 20]

# 动作空间大小
N_ACTIONS = 3

# 纯离散环境，动作编号直接用
ACTION_LEVELS = None


def get_agent(algorithm: str, state_shape):
    """按字符串创建对应的算法对象（三选一）"""
    if algorithm == "q_learning":
        return QLearningAgent(state_shape, N_ACTIONS, gamma=0.95, epsilon_decay=0.995)
    elif algorithm == "sarsa":
        return SarsaAgent(state_shape, N_ACTIONS, gamma=0.95, epsilon_decay=0.995)
    elif algorithm == "gradient":
        return GradientAgent(state_shape, N_ACTIONS, alpha=0.005)
    else:
        raise ValueError(f"未知算法: {algorithm}，可选 q_learning / sarsa / gradient")


def train(env, agent, algorithm: str, episodes: int, bins):
    """
    主训练循环：跑完 episodes 轮，每轮从 reset 到 done。

    参数:
        env:        gymnasium 环境对象
        agent:      算法对象
        algorithm:  算法名
        episodes:   训练总轮数
        bins:       build_bins() 生成的离散化边界

    返回:
        rewards: 每轮总奖励列表
    """
    rewards = []

    for episode in range(1, episodes + 1):
        # ---------- 每轮开始 ----------
        obs, _ = env.reset()
        state = discretize(obs, bins)
        total_reward = 0.0
        terminated, truncated = False, False
        trajectory = []                        # 仅 gradient 用

        # ---------- 单轮循环 ----------
        while not terminated and not truncated:
            action = agent.choose_action(state)

            # 纯离散环境：动作编号直接传给 step
            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state = discretize(next_obs, bins)
            total_reward += reward

            # 按算法更新（逻辑同 CartPole 版本，详见 cartpole/fram_work.py）
            if algorithm == "q_learning":
                agent.update(state, action, reward, next_state, terminated)
            elif algorithm == "sarsa":
                next_action = agent.choose_action(next_state)
                agent.update(state, action, reward, next_state, next_action, terminated)
            elif algorithm == "gradient":
                trajectory.append((state, action, reward))

            state = next_state

        # ---------- 每轮结束 ----------
        if algorithm == "gradient":
            agent.update_episode(trajectory)
        else:
            agent.decay_epsilon()

        rewards.append(total_reward)

        # 打印进度
        if episode % 100 == 0 or episode == 1:
            avg = np.mean(rewards[-100:])
            print(f"[{algorithm}] 第 {episode:4d}/{episodes} 轮  本轮奖励={total_reward:6.1f}  近100轮平均={avg:6.1f}")

    return rewards


def plot_rewards(rewards: list):
    """绘制"奖励-轮数"折线图"""
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(rewards) + 1), rewards, color="steelblue", alpha=0.4, label="每轮奖励")
    window = 50
    if len(rewards) >= window:
        smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
        plt.plot(range(window, len(rewards) + 1), smoothed, color="crimson", linewidth=2, label=f"{window}轮滑动平均")
    plt.xlabel("轮数 (Episode)")
    plt.ylabel("总奖励 (Total Reward)")
    plt.title(f"MountainCar-v0 训练奖励变化图")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()


# ==================== 手动可改参数区 ====================
if __name__ == "__main__":
    # 【手动修改】选择算法: "q_learning" / "sarsa" / "gradient"
    ALGORITHM = "q_learning"

    # 【手动修改】训练总轮数（稀疏奖励环境，建议 1000 轮起步）
    EPISODES = 1500

    # 创建环境
    env = gym.make(ENV_NAME)

    # 离散化边界 + 算法对象
    bins = build_bins(OBS_BOUNDS, STATE_BINS)
    agent = get_agent(ALGORITHM, tuple(STATE_BINS))

    # 开始训练
    print(f"{'='*50}")
    print(f"  环境: {ENV_NAME}    算法: {ALGORITHM}    轮数: {EPISODES}")
    print(f"{'='*50}")
    rewards = train(env, agent, ALGORITHM, EPISODES, bins)

    # 训练结束
    print(f"\n=== 训练完成 ===")
    print(f"平均奖励: {np.mean(rewards):.2f}  最高奖励: {max(rewards)}")
    plot_rewards(rewards)
