"""
fram_work —— 小车杆强化学习项目的核心框架

功能：
  1. 初始化 CartPole 环境
  2. 运行 N 轮循环（episode），每轮进行一次完整测试
  3. 每轮记录总奖励
  4. 绘制 "奖励-轮数" 折线图
  5. 根据 state 参数选择策略：0=随机, 1=SARSA

手动可改参数（在本文件末尾 __main__ 中修改）：
  - EPISODES：训练/测试的总轮数
  - state：   策略选择（0=随机, 1=SARSA）

当使用随机策略时，每轮仅做随机动作，不学习。
当使用 SARSA 策略时，智能体在每步更新 Q 表，
  理论上随着轮数增加，奖励会逐渐上升。
"""

import gymnasium as gym
import matplotlib.pyplot as plt
from sarsa import SARSA


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
        terminated = False      # 杆子倒下、小车超出边界、到达终点时变为 true
        truncated = False       # 是否达到步数限制（CartPole-v1 最大步数 = 500）
                                # 两者都为 True 时本轮结束

        # =========================================================
        #  策略分支：agent=None → 随机策略；agent≠None → 算法策略
        # =========================================================
        if agent is None:
            # ---------- 随机策略（每步随机选动作，不学习） ----------
            while not terminated and not truncated:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward

        else:
            # ---------- 算法策略（如 SARSA，每步选择+更新） ----------
            # 注：SARSA 是 on-policy 算法，需要先选好第一步的动作
            #     才能进入循环（因为更新时需要 (S,A,R,S',A') 四元组）
            state = agent.obs_to_state(obs)
            action = agent.choose_action(state)

            while not terminated and not truncated:
                # 执行动作，获取结果
                obs_next, reward, terminated, truncated, _ = env.step(action)

                total_reward += reward
                done = terminated or truncated

                if not done:
                    # 用当前策略选下一个动作 A'（这就是 SARSA 中第二个 A）
                    state_next = agent.obs_to_state(obs_next)
                    action_next = agent.choose_action(state_next)
                else:
                    # 本轮结束，没有下一步了
                    state_next = None
                    action_next = None

                # SARSA 更新：Q(S,A) ← Q(S,A) + α[R + γQ(S',A') - Q(S,A)]
                agent.update(state, action, reward,
                             state_next, action_next, done)

                # 前进一步：S ← S', A ← A'
                state = state_next
                action = action_next

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
    # ===== 【手动修改】以下两个参数 =====

    # 训练总轮数（SARSA 建议 500+ 轮才能看到明显学习效果）
    EPISODES = 500

    # 策略选择：0 = 随机策略, 1 = SARSA 策略
    state = 1

    # ==================================

    # 根据 state 创建智能体
    if state == 0:
        agent = None
        strategy_name = "随机策略"
    elif state == 1:
        agent = SARSA()
        strategy_name = "SARSA"
    else:
        raise ValueError(f"未知的策略编号 state={state}，请使用 0(随机) 或 1(SARSA)")

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
