"""
fram_work —— 小车杆强化学习项目的核心框架

功能：
  1. 初始化 CartPole 环境
  2. 运行 N 轮循环（episode），每轮进行一次完整测试
  3. 每轮记录总奖励
  4. 绘制 "奖励-轮数" 折线图
  5. 根据 state 参数选择策略：0=随机, 1=SARSA, 2=Q-learning, 3=Policy Gradient

手动可改参数（在本文件末尾 __main__ 中修改）：
  - EPISODES：训练/测试的总轮数
  - state：   策略选择（0=随机, 1=SARSA, 2=Q-learning, 3=Policy Gradient）

当使用随机策略时，每轮仅做随机动作，不学习。
当使用 SARSA / Q-learning 策略时，智能体在每步更新 Q 表，
  理论上随着轮数增加，奖励会逐渐上升。
使用 Policy Gradient 时，智能体在每轮结束后更新偏好 H 表。
"""

import gymnasium as gym
import matplotlib.pyplot as plt
import time
from sarsa import SARSA
from Q_learning import QLearning
from gradient import PolicyGradient


def run_cartpole(num_episodes: int, agent=None):
    """
    运行小车杆环境的主循环

    参数:
        num_episodes: 运行的总轮数（每轮 = 一次完整游戏，直到 done）
        agent:        None = 使用随机策略
                      SARSA 实例或其他算法类实例 = 使用该策略决策

    返回:
        rewards: 列表，每轮的总奖励
    """
    # ========== 1. 创建环境 ==========
    env = gym.make("CartPole-v1", render_mode="None")

    # 记录每轮的总奖励
    rewards = []

    # ========== 2. 主循环 ==========
    for episode in range(1, num_episodes + 1):
        # ===== 最后一轮：切换到 human 模式，关闭探索 =====
        is_last_episode = (episode == num_episodes)
        if is_last_episode:
            env.close()
            env = gym.make("CartPole-v1", render_mode="human")
            if agent is not None:
                agent.set_epsilon(0.0)  # 完全利用已学策略

        # 重置环境，开始新的一轮
        obs, _ = env.reset()

        total_reward = 0.0
        terminated = False
        truncated = False

        # =========================================================
        #  策略分支：agent=None → 随机策略；agent≠None → 算法策略
        # =========================================================
        if agent is None:
            # ---------- 随机策略 ----------
            while not terminated and not truncated:
                action = env.action_space.sample()
                if is_last_episode:
                    time.sleep(0.04)
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward

        else:
            # ---------- 算法策略（SARSA / Q-learning） ----------
            from sarsa import SARSA
            is_sarsa = isinstance(agent, SARSA)

            state = agent.obs_to_state(obs)
            action = agent.choose_action(state)

            while not terminated and not truncated:
                if is_last_episode:
                    time.sleep(0.04)

                obs_next, reward, terminated, truncated, _ = env.step(action)

                total_reward += reward
                done = terminated or truncated

                if not done:
                    state_next = agent.obs_to_state(obs_next)
                    # 行为策略：用 ε-greedy 选下一个动作（SARSA 和 Q-learning 都需要）
                    action_next = agent.choose_action(state_next)
                else:
                    state_next = None
                    action_next = None

                if is_sarsa:
                    # SARSA: 更新用 Q(S',A')，需要 next_action
                    agent.update(state, action, reward,
                                 state_next, action_next, done)
                else:
                    # Q-learning: 更新用 max_a Q(S',a)，不需要 next_action
                    agent.update(state, action, reward,
                                 state_next, done)

                state = state_next
                action = action_next

        # 记录本轮总奖励
        rewards.append(total_reward)
        if episode % 100 == 0:
            print(f"[第 {episode:3d} 轮]  总奖励 = {total_reward}")


#-------------------------------------------------------------------------------
    
#-------------------------------------------------------------------------------

    # ========== 3. 关闭环境 ==========
    env.close()

    return rewards


def plot_rewards(rewards: list, block_size: int = 100):
    """
    绘制每 block_size 轮的平均奖励折线图

    参数:
        rewards:   每轮的总奖励列表
        block_size: 每组包含的轮数（默认 100）
    """
    # 将 rewards 按 block_size 分组，计算每组的平均奖励
    avg_rewards = []
    for i in range(0, len(rewards), block_size):
        block = rewards[i:i + block_size]
        avg_rewards.append(sum(block) / len(block))

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(avg_rewards) + 1), avg_rewards,
             marker="o", linestyle="-", color="b")
    plt.xlabel(f"n* {block_size} ")
    plt.ylabel(f"n* {block_size} average reward")
    plt.title("CartPole 训练奖励变化图")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()


# ==================== 手动可改参数 ====================
if __name__ == "__main__":
    # ===== 【手动修改】以下两个参数 =====

    # 训练总轮数（SARSA 建议 500+ 轮才能看到明显学习效果）
    EPISODES = 100000

    # 策略选择：0 = 随机策略, 1 = SARSA 策略, 2 = Q-learning 策略, 3 = Policy Gradient 策略
    state = 2

    # ==================================

    # 根据 state 创建智能体
    if state == 0:
        agent = None
        strategy_name = "随机策略"
    elif state == 1:
        agent = SARSA(n_bins=40)
        strategy_name = "SARSA"
    elif state == 2:
        agent = QLearning(n_bins=40,gamma=0.999)
        strategy_name = "Q-learning"
    elif state == 3:
        agent = PolicyGradient(n_bins=40)
        strategy_name = "Policy Gradient"
    else:
        raise ValueError(f"未知的策略编号 state={state}，请使用 0(随机), 1(SARSA), 2(Q-learning) 或 3(Policy Gradient)")

    # 运行主循环
    print(f"开始运行 CartPole，共 {EPISODES} 轮，当前策略: {strategy_name}\n")
    rewards = run_cartpole(EPISODES, agent)

    # 绘制折线图
    plot_rewards(rewards)

    # 打印统计信息
    print(f"\n=== 统计 ===")
    print(f"策略: {strategy_name}")
    print(f"平均奖励: {sum(rewards) / len(rewards):.2f}")
    print(f"最高奖励: {max(rewards)}")
    print(f"最低奖励: {min(rewards)}")
