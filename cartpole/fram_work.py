"""
fram_work —— 小车杆强化学习项目的核心框架

功能：
  1. 初始化 CartPole 环境
  2. 运行 N 轮循环（episode），每轮进行一次完整测试
  3. 每轮记录总奖励
  4. 绘制 "奖励-轮数" 折线图
  5. 当前使用随机策略（后续将替换为算法类的决策）

手动可改参数（在本文件末尾 __main__ 中修改）：
  - EPISODES：训练/测试的总轮数
"""

import gymnasium as gym
import matplotlib.pyplot as plt


def run_cartpole(num_episodes: int):
    """
    运行小车杆环境的主循环

    参数:
        num_episodes: 运行的总轮数（每轮 = 一次完整游戏，直到 done）

    返回:
        rewards: 列表，每轮的总奖励
    """
    # ========== 1. 创建环境 ==========
    # render_mode=None      不显示动画窗口（适合快速跑实验）
    # render_mode="human"   会弹出一个窗口显示小车杆动画（需安装 pygame）
    env = gym.make("CartPole-v1", render_mode="None")


    # 记录每轮的总奖励
    rewards = []

    # ========== 2. 主循环 ==========
    for episode in range(1, num_episodes + 1):
        # 重置环境，开始新的一轮
        # obs = [cart_pos, cart_vel, pole_angle, pole_angular_vel]
        obs, _ = env.reset()

        total_reward = 0.0      # 本轮累计奖励
        terminated = False      # 杆子倒下、小车超出边界、到达终点时变为true
        truncated = False       # 是否达到阶段限制（CartPole-v1 最大步数 = 500）
                                # 注：这两个变量都为 True 时，表示本轮结束

        # 单轮循环：每一步选择一个动作，推进环境
        while not terminated and not truncated:
            # ===== 当前使用随机策略 =====
            # action: 0 = 向左推, 1 = 向右推
            action = env.action_space.sample()

            # 执行动作，获取下一状态
            obs, reward, terminated, truncated, _ = env.step(action)

            # 累计奖励（CartPole-v1 每步存活 +1）
            total_reward += reward

        # 记录本轮总奖励
        rewards.append(total_reward)
        print(f"[第 {episode:3d} 轮]  总奖励 = {total_reward}")

    # ========== 3. 关闭环境 ==========
    env.close()

    return rewards


def plot_rewards(rewards: list):
    """
    绘制奖励随轮数变化的折线图

    参数:
        rewards: 每轮的总奖励列表
    """
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(rewards) + 1), rewards, marker="o", linestyle="-", color="b")
    plt.xlabel("轮数 (Episode)")
    plt.ylabel("总奖励 (Total Reward)")
    plt.title("CartPole 训练奖励变化图")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()


# ==================== 手动可改参数 ====================
if __name__ == "__main__":
    # 【手动修改】训练总轮数
    EPISODES = 20

    # 运行主循环
    print(f"开始运行 CartPole，共 {EPISODES} 轮，当前使用随机策略\n")
    rewards = run_cartpole(EPISODES)

    # 绘制折线图
    plot_rewards(rewards)

    # 打印统计信息
    print(f"\n=== 统计 ===")
    print(f"平均奖励: {sum(rewards) / len(rewards):.2f}")
    print(f"最高奖励: {max(rewards)}")
    print(f"最低奖励: {min(rewards)}")
