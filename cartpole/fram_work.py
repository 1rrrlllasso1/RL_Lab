"""
fram_work.py —— CartPole-v1（倒立摆）强化学习测试主框架

【怎么用】
  1. 在下方 __main__ 中改 ALGORITHM 选择算法（三选一）
  2. 直接运行本文件，训练完成后自动弹出奖励曲线图

【环境说明】
  CartPole-v1：小车在轨道上左右移动，杆子立在车上，
  目标是通过推小车让杆子尽量久地保持竖直。
  - 状态(4维): [小车位置, 小车速度, 杆子角度, 杆子角速度]
  - 动作(2个): 0 = 向左推, 1 = 向右推
  - 奖励: 每存活一步 +1，最多 500 步

【算法说明】
  - q_learning: Q 表 + ε-greedy，离策略，乐观更新（用 max）
  - sarsa:      Q 表 + ε-greedy，在策略，务实更新（用实际动作）
  - gradient:   策略梯度 REINFORCE，softmax 偏好表，回合制更新
"""

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

from utils import build_bins, discretize
from Q_learning import QLearningAgent
from sarsa import SarsaAgent
from gradient import GradientAgent


# ==================== 环境配置区 ====================
ENV_NAME = "CartPole-v1"

# 每个状态维度的取值范围 [min, max]
# (环境观测空间是无穷的，这里按经验值裁剪，超出会自动归入边界箱，不会越界)
OBS_BOUNDS = [
    (-4.8, 4.8),    # 小车位置 x
    (-10.0, 10.0),  # 小车速度 x_dot
    (-0.5, 0.5),    # 杆子角度 theta（阈值约 ±0.21，放宽到 ±0.5）
    (-10.0, 10.0),  # 杆子角速度 theta_dot
]

# 每个状态维度分箱数量（离散化粒度，越大越精细但 Q 表越大）
STATE_BINS = [8, 8, 8, 8]

# 动作空间大小（CartPole 只有 2 个动作）
N_ACTIONS = 2

# 连续动作环境的动作档位（CartPole 是纯离散环境，填 None 表示动作编号直接用）
ACTION_LEVELS = None


def get_agent(algorithm: str, state_shape):
    """
    按字符串创建对应的算法对象。

    参数:
        algorithm:    "q_learning" / "sarsa" / "gradient" 三选一
        state_shape:  Q 表/偏好表各维箱数元组，如 (8, 8, 8, 8)

    返回:
        对应算法的 agent 对象
    """
    if algorithm == "q_learning":
        return QLearningAgent(state_shape, N_ACTIONS, gamma=0.99, epsilon_decay=0.995)
    elif algorithm == "sarsa":
        return SarsaAgent(state_shape, N_ACTIONS, gamma=0.99, epsilon_decay=0.995)
    elif algorithm == "gradient":
        return GradientAgent(state_shape, N_ACTIONS, alpha=0.005)
    else:
        raise ValueError(f"未知算法: {algorithm}，可选 q_learning / sarsa / gradient")


def train(env, agent, algorithm: str, episodes: int, bins):
    """
    主训练循环：跑完 episodes 轮，每轮从 reset 到 done。

    参数:
        env:        gymnasium 环境对象
        agent:      算法对象（三种算法的接口略不同，循环内按 algorithm 分支调用）
        algorithm:  算法名
        episodes:   训练总轮数
        bins:       build_bins() 生成的离散化边界

    返回:
        rewards: 每轮总奖励列表
    """
    rewards = []

    for episode in range(1, episodes + 1):
        # ---------- 每轮开始：重置环境，得到初始状态 ----------
        obs, _ = env.reset()
        state = discretize(obs, bins)          # 连续观测 → 离散索引
        total_reward = 0.0
        terminated, truncated = False, False
        trajectory = []                        # 仅 gradient 用，记录整轮轨迹

        # ---------- 单轮循环：直到本轮结束 ----------
        while not terminated and not truncated:
            # 1. 按策略选动作
            action = agent.choose_action(state)

            # 2. 执行动作（纯离散环境直接传编号；连续动作环境转成档位值）
            if ACTION_LEVELS is not None:
                action_vec = np.array([ACTION_LEVELS[action]], dtype=np.float32)
            else:
                action_vec = action
            next_obs, reward, terminated, truncated, _ = env.step(action_vec)
            next_state = discretize(next_obs, bins)
            total_reward += reward

            # 3. 按算法更新策略（三种算法学习方式不同）
            if algorithm == "q_learning":
                # 离策略：用 max_a' Q(S',a') 更新，立即执行
                agent.update(state, action, reward, next_state, terminated)
            elif algorithm == "sarsa":
                # 在策略：需要先选好下一个动作 A' 再更新
                next_action = agent.choose_action(next_state)
                agent.update(state, action, reward, next_state, next_action, terminated)
            elif algorithm == "gradient":
                # 策略梯度：先记录轨迹，等本轮结束后统一更新
                trajectory.append((state, action, reward))

            state = next_state

        # ---------- 每轮结束 ----------
        if algorithm == "gradient":
            agent.update_episode(trajectory)   # 回合制更新偏好表
        else:
            agent.decay_epsilon()              # ε-greedy 类算法降低探索率

        rewards.append(total_reward)

        # 打印进度（每 20 轮一次 + 最近 50 轮平均，便于观察收敛趋势）
        if episode % 20 == 0 or episode == 1:
            avg = np.mean(rewards[-50:])
            print(f"[{algorithm}] 第 {episode:4d}/{episodes} 轮  本轮奖励={total_reward:6.1f}  近50轮平均={avg:6.1f}")

    return rewards


def plot_rewards(rewards: list):
    """绘制"奖励-轮数"折线图，直观看到学习效果"""
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(rewards) + 1), rewards, color="steelblue", alpha=0.4, label="每轮奖励")
    # 平滑曲线（每 20 轮平均），更容易看出趋势
    window = 20
    if len(rewards) >= window:
        smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
        plt.plot(range(window, len(rewards) + 1), smoothed, color="crimson", linewidth=2, label=f"{window}轮滑动平均")
    plt.xlabel("轮数 (Episode)")
    plt.ylabel("总奖励 (Total Reward)")
    plt.title(f"CartPole-v1 训练奖励变化图")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()


# ==================== 手动可改参数区 ====================
if __name__ == "__main__":
    # 【手动修改】选择算法: "q_learning" / "sarsa" / "gradient"
    ALGORITHM = "q_learning"

    # 【手动修改】训练总轮数（表格法 CartPole 建议 1000 轮，能看出明显学习趋势）
    EPISODES = 1000

    # 创建环境（训练阶段不需要渲染窗口，render_mode 用默认 None 更快）
    env = gym.make(ENV_NAME)

    # 离散化边界 + 算法对象
    bins = build_bins(OBS_BOUNDS, STATE_BINS)
    agent = get_agent(ALGORITHM, tuple(STATE_BINS))

    # 开始训练
    print(f"{'='*50}")
    print(f"  环境: {ENV_NAME}    算法: {ALGORITHM}    轮数: {EPISODES}")
    print(f"{'='*50}")
    rewards = train(env, agent, ALGORITHM, EPISODES, bins)

    # 训练结束：输出统计 + 画图
    print(f"\n=== 训练完成 ===")
    print(f"平均奖励: {np.mean(rewards):.2f}  最高奖励: {max(rewards)}")
    plot_rewards(rewards)
