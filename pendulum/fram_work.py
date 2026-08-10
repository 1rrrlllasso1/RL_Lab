"""
fram_work.py —— Pendulum-v1（倒立单摆）强化学习测试主框架

【怎么用】
  1. 在下方 __main__ 中改 ALGORITHM 选择算法（三选一）
  2. 直接运行本文件，训练完成后自动弹出奖励曲线图

【环境说明】
  Pendulum-v1：单摆从任意角度开始，目标是稳定在竖直向上位置。
  - 状态(3维): [cosθ, sinθ, 角速度 θ̇]
  - 动作: 连续值 torque ∈ [-2, 2]（施力矩）
  - 奖励: 每步 −(θ² + 0.1·θ̇² + 0.001·a²)，越接近竖直向上奖励越高（接近0）
  - 每轮固定 200 步（gym 自动截断，没有提前终止）

【关键处理：连续动作 → 离散档位】
  Q-Learning / SARSA / 表格 REINFORCE 都只能处理"有限个离散动作"，
  所以把连续扭矩 [-2, 2] 离散成 5 档:
    编号 0: -2.0   1: -1.0   2: 0.0   3: +1.0   4: +2.0
  动作编号在算法内被查表学习，执行时通过 ACTION_LEVELS 还原成真实扭矩。
"""

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

from utils import build_bins, discretize
from Q_learning import QLearningAgent
from sarsa import SarsaAgent
from gradient import GradientAgent


# ==================== 环境配置区 ====================
ENV_NAME = "Pendulum-v1"

# 状态取值范围（前两维是 cos/sin 本身有界，角速度 ±8）
OBS_BOUNDS = [
    (-1.0, 1.0),    # cosθ
    (-1.0, 1.0),    # sinθ
    (-8.0, 8.0),    # 角速度 θ̇
]

# 分箱数量：3 维每维 8 箱
STATE_BINS = [8, 8, 8]

# 动作档位数量（连续扭矩离散成 5 档）
N_ACTIONS = 5

# 连续动作环境的动作档位：编号 0~4 → 真实扭矩值
ACTION_LEVELS = [-2.0, -1.0, 0.0, 1.0, 2.0]


def get_agent(algorithm: str, state_shape):
    """按字符串创建对应的算法对象（三选一）"""
    if algorithm == "q_learning":
        return QLearningAgent(state_shape, N_ACTIONS, gamma=0.95, epsilon_decay=0.998)
    elif algorithm == "sarsa":
        return SarsaAgent(state_shape, N_ACTIONS, gamma=0.95, epsilon_decay=0.998)
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

            # 连续动作环境：编号 → 真实扭矩值（Pendulum 要求 shape=(1,) 的数组）
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
        if episode % 50 == 0 or episode == 1:
            avg = np.mean(rewards[-50:])
            print(f"[{algorithm}] 第 {episode:4d}/{episodes} 轮  本轮奖励={total_reward:7.2f}  近50轮平均={avg:7.2f}")

    return rewards


def plot_rewards(rewards: list):
    """绘制"奖励-轮数"折线图（Pendulum 奖励是负的，越高越接近最优）"""
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(rewards) + 1), rewards, color="steelblue", alpha=0.4, label="每轮奖励")
    window = 20
    if len(rewards) >= window:
        smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
        plt.plot(range(window, len(rewards) + 1), smoothed, color="crimson", linewidth=2, label=f"{window}轮滑动平均")
    plt.xlabel("轮数 (Episode)")
    plt.ylabel("总奖励 (Total Reward)")
    plt.title(f"Pendulum-v1 训练奖励变化图（越高越好）")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()


# ==================== 手动可改参数区 ====================
if __name__ == "__main__":
    # 【手动修改】选择算法: "q_learning" / "sarsa" / "gradient"
    ALGORITHM = "gradient"

    # 【手动修改】训练总轮数（每轮固定 200 步，500 轮即可见趋势）
    EPISODES = 2400

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
