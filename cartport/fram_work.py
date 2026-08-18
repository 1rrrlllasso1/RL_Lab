"""
fram_work —— 小车杆强化学习项目的核心框架

功能：
  1. 初始化 CartPole 环境
  2. 运行 N 轮循环（episode），每轮进行一次完整测试
  3. 每轮记录总奖励
  4. 绘制 "奖励-轮数" 折线图
  5. 根据 state 参数选择策略：0=随机, 1=SARSA, 2=Q-learning, 3=Policy Gradient, 4=DQN, 5=PPO

手动可改参数（在本文件末尾 __main__ 中修改）：
  - EPISODES：训练/测试的总轮数
  - state：   策略选择（0=随机, 1=SARSA, 2=Q-learning, 3=Policy Gradient, 4=DQN, 5=PPO）
  - PLOT_BLOCK：画图参数，横坐标 = 每多少回合的平均奖励（每个算法可单独设置）

绘图函数 plot_rewards() 已移到 utils.py，本文件仅导入调用。

当使用随机策略时，每轮仅做随机动作，不学习。
当使用 SARSA / Q-learning 策略时，智能体在每步更新 Q 表，
  理论上随着轮数增加，奖励会逐渐上升。
使用 Policy Gradient 时，智能体在每轮结束后更新偏好 H 表。
使用 DQN 时，智能体通过神经网络拟合 Q 函数，利用经验回放训练网络。
使用 PPO 时，智能体用 Actor-Critic 神经网络收集一段轨迹后统一更新策略。
"""

# ===== 修复 Windows 上 PyTorch + NumPy(MKL) 的 OpenMP DLL 冲突 =====
# 必须在所有其他 import 之前设置，否则 sarsa/gradient 等模块导入 numpy
# 时就会加载 libiomp5md.dll，等到导入 torch 时就会冲突崩溃
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
# ===== 修复 Windows 终端中文乱码 =====
# VS Code 终端按 UTF-8 解码，而 Python 在 Windows 上默认用 GBK(cp936)
# 输出，导致所有中文 print 变成乱码、看起来像"没有反馈"。
# 这里强制 stdout/stderr 用 UTF-8，保证在 VS Code 里中文正常显示。
# 注意：部分运行环境（如 VS Code 的 Python Interactive 窗口 / Jupyter）
# 里 stdout 不是真正的控制台流，没有 reconfigure 方法。这行代码位于所有
# print 之前，一旦抛异常脚本会在输出任何内容前直接退出（"无反馈直接退出"），
# 因此必须容错处理：能改就改，改不了就跳过，绝不阻断主流程。
for _s in (sys.stdout, sys.stderr):
    if _s is not None:
        try:
            _s.reconfigure(encoding="utf-8")
        except Exception:
            pass

import gymnasium as gym
import time
from sarsa import SARSA
from Q_learning import QLearning
from gradient import PolicyGradient
from dqn import DQN
from ppo import PPO
from utils import plot_rewards


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
    env = gym.make("CartPole-v1", render_mode=None)

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

                state_next = agent.obs_to_state(obs_next) if not done else None

                if is_sarsa:
                    # SARSA: 更新用 Q(S',A')，必须先选好 next_action 再更新
                    action_next = agent.choose_action(state_next) if not done else None
                    agent.update(state, action, reward,
                                 state_next, action_next, done)
                else:
                    # Q-learning / PolicyGradient / DQN / PPO：
                    # 先更新再选下一个动作。若先 choose_action(s')，会覆盖 PPO 暂存的
                    # _last_logprob，导致缓冲区里 (s,a) 配到 logπ(a'|s')，重要性比率算错。
                    agent.update(state, action, reward,
                                 state_next, done)
                    action_next = agent.choose_action(state_next) if not done else None

                state = state_next
                action = action_next

        # 记录本轮总奖励
        rewards.append(total_reward)

        # DQN / PPO 每轮都打印日志；其他算法按间隔打印
        from dqn import DQN
        from ppo import PPO
        is_dqn = isinstance(agent, DQN)
        is_ppo = isinstance(agent, PPO)
        if is_dqn or is_ppo:
            print(f"[第 {episode:3d} 轮]  总奖励 = {total_reward}", flush=True)
        elif episode % 100 == 0:
            print(f"[第 {episode:3d} 轮]  总奖励 = {total_reward}", flush=True)
        elif episode <= 10 or episode % 50 == 0:
            print(f"[第 {episode:3d} 轮]  总奖励 = {total_reward}", flush=True)


#-------------------------------------------------------------------------------
    
#-------------------------------------------------------------------------------

    # ========== 3. 关闭环境 ==========
    env.close()

    return rewards


# ==================== 手动可改参数 ====================
if __name__ == "__main__":
    # ===== 【手动修改】以下参数 =====

    # 训练总轮数（SARSA 建议 500+ 轮才能看到明显学习效果）
    EPISODES = 1500

    # 策略选择：0 = 随机策略, 1 = SARSA, 2 = Q-learning, 3 = Policy Gradient, 4 = DQN, 5 = PPO
    state = 5

    # 画图参数：横坐标 = 每多少回合的平均奖励（每个算法可单独设置，按键值 state 索引）
    # 调小能看清每轮的细节（如 1 = 每回合一个点），调大能看清整体的学习趋势
    PLOT_BLOCK = {
        0: 100,   # 随机策略
        1: 100,   # SARSA
        2: 100,   # Q-learning
        3: 100,   # Policy Gradient
        4: 1,     # DQN
        5: 30,   # PPO
    }

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
    elif state == 4:
        agent = DQN()
        strategy_name = "DQN"
    elif state == 5:
        agent = PPO()
        strategy_name = "PPO"
    else:
        raise ValueError(f"未知的策略编号 state={state}，请使用 0(随机), 1(SARSA), 2(Q-learning), 3(Policy Gradient), 4(DQN) 或 5(PPO)")

    # 运行主循环
    print(f"开始运行 CartPole，共 {EPISODES} 轮，当前策略: {strategy_name}\n", flush=True)
    rewards = run_cartpole(EPISODES, agent)

    # 绘制折线图：横坐标 = 每 PLOT_BLOCK[state] 回合的平均奖励（每个算法可手动调整）
    plot_rewards(rewards, block_size=PLOT_BLOCK[state])

    # 打印统计信息
    print(f"\n=== 统计 ===")
    print(f"策略: {strategy_name}")
    print(f"平均奖励: {sum(rewards) / len(rewards):.2f}")
    print(f"最高奖励: {max(rewards)}")
    print(f"最低奖励: {min(rewards)}")
