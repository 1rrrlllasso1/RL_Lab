"""
fram_work.py —— Acrobot-v1（双节摆）强化学习测试主框架

【怎么用】
  1. 在下方 __main__ 中改 ALGORITHM 选择算法（三选一）
  2. 直接运行本文件，训练完成后自动弹出奖励曲线图

【环境说明】
  Acrobot-v1：两根杆铰接（像双节棍），只有中间关节能施加扭矩，
  目标是把杆头荡到目标高度以上（鞭打效应：先荡起来再甩上去）。
  - 状态(6维): [cosθ1, sinθ1, cosθ2, sinθ2, 关节1角速度, 关节2角速度]
  - 动作(3个): 0 = 关节 -1 扭矩, 1 = 0 扭矩, 2 = +1 扭矩
  - 奖励: 每步 -1，杆头高度 -cosθ1 − cos(θ1+θ2) > 1.0 时成功结束
  - 最大步数: 500（gym 自动截断）

【分箱说明】
  6 维状态如果每维分太多箱，Q 表会指数爆炸（6维×8箱 = 26万个状态）。
  这里每维 6 箱 → 6^6 ≈ 4.7 万个状态，训练速度和效果均衡。
"""

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

from utils import build_bins, discretize
from Q_learning import QLearningAgent
from sarsa import SarsaAgent
from gradient import GradientAgent


# ==================== 环境配置区 ====================
ENV_NAME = "Acrobot-v1"

# 状态取值范围
OBS_BOUNDS = [
    (-1.0, 1.0),     # cosθ1
    (-1.0, 1.0),     # sinθ1
    (-1.0, 1.0),     # cosθ2
    (-1.0, 1.0),     # sinθ2
    (-15.0, 15.0),   # 关节1角速度（经验范围，实际约 ±12.6）
    (-15.0, 15.0),   # 关节2角速度
]

# 分箱数量：6 维每维 6 箱（避免 Q 表爆炸）
STATE_BINS = [6, 6, 6, 6, 6, 6]

# 动作空间大小
N_ACTIONS = 3

# 纯离散环境，动作编号直接用
ACTION_LEVELS = None


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
    plt.title(f"Acrobot-v1 训练奖励变化图")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()


# ==================== 手动可改参数区 ====================
if __name__ == "__main__":
    # 【手动修改】选择算法: "q_learning" / "sarsa" / "gradient"
    ALGORITHM = "q_learning"

    # 【手动修改】训练总轮数（Acrobot 较难，建议 1000 轮起步）
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
