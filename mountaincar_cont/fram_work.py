"""
fram_work.py —— MountainCarContinuous-v0（连续版爬坡小车）强化学习测试主框架

【怎么用】
  1. 在下方 __main__ 中改 ALGORITHM 选择算法（三选一）
  2. 直接运行本文件，训练完成后自动弹出奖励曲线图

【环境说明】
  MountainCarContinuous-v0：MountainCar-v0 的连续动作版本，
  小车被困在山谷里，必须来回荡借惯性跃上山顶。
  - 状态(2维): [小车位置, 小车速度]
  - 动作: 连续推力 ∈ [-1, 1]（正=向右推，负=向左推）
  - 奖励: 每步 -1，到达山顶位置 ≥ 0.45 时 +100 并结束
  - 最大步数: 999（gym 自动截断）

【关键处理：连续动作 → 离散档位】
  把连续推力 [-1, 1] 离散成 3 档:
    编号 0: -1.0（向左）   1: 0.0（不推）   2: +1.0（向右）
  与 MountainCar-v0 的三个离散动作正好对应，方便对比两种版本。
"""

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

from utils import build_bins, discretize
from Q_learning import QLearningAgent
from sarsa import SarsaAgent
from gradient import GradientAgent


# ==================== 环境配置区 ====================
ENV_NAME = "MountainCarContinuous-v0"

# 状态取值范围（与 MountainCar-v0 相同）
OBS_BOUNDS = [
    (-1.2, 0.6),    # 小车位置 position
    (-0.07, 0.07),  # 小车速度 velocity
]

# 分箱数量：位置 20 箱 + 速度 20 箱
STATE_BINS = [20, 20]

# 动作档位数量（连续推力离散成 3 档）
N_ACTIONS = 3

# 连续动作环境的动作档位：编号 0~2 → 真实推力值
ACTION_LEVELS = [-1.0, 0.0, 1.0]


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

            # 连续动作环境：编号 → 真实推力值（要求 shape=(1,) 的数组）
            action_vec = np.array([ACTION_LEVELS[action]], dtype=np.float32)
            next_obs, reward, terminated, truncated, _ = env.step(action_vec)
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
    plt.title(f"MountainCarContinuous-v0 训练奖励变化图")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()


# ==================== 手动可改参数区 ====================
if __name__ == "__main__":
    # 【手动修改】选择算法: "q_learning" / "sarsa" / "gradient"
    ALGORITHM = "gradient"

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
    print(f"  动作档位: {ACTION_LEVELS}")
    print(f"{'='*50}")
    rewards = train(env, agent, ALGORITHM, EPISODES, bins)

    # 训练结束
    print(f"\n=== 训练完成 ===")
    print(f"平均奖励: {np.mean(rewards):.2f}  最高奖励: {max(rewards)}")
    plot_rewards(rewards)
