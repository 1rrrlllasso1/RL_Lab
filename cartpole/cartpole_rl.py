"""
=====================================================================
   CartPole 强化学习训练框架（Gymnasium）
   ======================================
   目标：在 CartPole-v1 环境中，用三种不同 RL 算法训练小车保持杆子
        直立。项目流程为：随机初始化策略 → 训练 → 清空策略 → 绘图。

   三种算法（用户可在 Config 中或运行时选择）：
     1. Policy Gradient / REINFORCE  — 策略梯度方法（神经网络参数化）
     2. SARSA                        — 在策略 TD 控制（表格方法）
     3. Q-Learning                   — 离策略 TD 控制（表格方法）
=====================================================================
"""

# ── 标准库 ──────────────────────────────────────────────
import sys      # sys.argv：接收命令行参数（算法名、回合数）
import os       # os.path：用于构造图片保存路径

# ── 第三方库 ────────────────────────────────────────────
import gymnasium as gym          # OpenAI Gymnasium：提供 CartPole-v1 环境
import numpy as np               # NumPy：矩阵运算、随机数、数学工具
import matplotlib.pyplot as plt  # Matplotlib：绘制训练奖励曲线图

# ── matplotlib 中文显示配置 ────────────────────────────
# 尝试一系列中文字体，按优先级排列；axes.unicode_minus=False 修复负号显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ── CartPole-v1 四个状态维度的取值范围 ──────────────────
# 每个元素为 (最小值, 最大值)，用于离散化前将状态值裁剪到合理范围
STATE_RANGES = [
    (-2.4, 2.4),     # 维度0：小车在轨道上的位置（超过 ±2.4 游戏结束）
    (-3.0, 3.0),     # 维度1：小车移动速度（无硬性限制，这里取经验范围）
    (-0.209, 0.209), # 维度2：杆子与垂直线的夹角（弧度），超过 ±12° 游戏结束
    (-2.0, 2.0),     # 维度3：杆子顶端角速度（无硬性限制，经验范围）
]


# ============================================================
#  第1部分：Config — 所有可调超参数的集中管理
#  每次训练前修改此类的属性即可改变算法和行为
# ============================================================

class Config:
    """集中存放所有可调超参数。直接在 __init__ 中修改默认值。"""

    def __init__(self):
        # ── 算法选择 ──────────────────────────────────────
        # 可选值：'policy_gradient' | 'sarsa' | 'q_learning'
        self.algorithm: str = 'policy_gradient'

        # ── 训练循环 ──────────────────────────────────────
        self.n_episodes: int = 500   # 总共训练多少个回合
        self.max_steps: int = 500    # 每回合最多执行多少步（CartPole-v1 默认 500）

        # ── SARSA 和 Q-Learning 的公共学习参数 ─────────────
        # learning_rate：Q 表更新的步长 α，太大震荡、太小收敛慢
        # gamma：折扣因子 γ，越接近 1 越重视远期回报
        self.learning_rate: float = 0.1   # 学习率 α（仅 SARSA / Q-Learning 使用）
        self.gamma: float = 0.99          # 折扣因子 γ（所有算法共用）
        self.epsilon: float = 1.0         # 初始探索率 ε（1.0 = 完全随机探索）
        self.epsilon_min: float = 0.01    # 探索率的下限（保持最低探索量）
        self.epsilon_decay: float = 0.995 # 每回合 ε 乘以该值，逐步减少探索

        # ── 状态离散化参数 ─────────────────────────────────
        # 将连续状态分成有限个区间（箱子），四个维度各自的箱子数
        # 格式：(小车位置箱数, 小车速度箱数, 杆角度箱数, 杆角速度箱数)
        # 箱子总数越多，Q 表/H 表越大，精度越高但收敛越慢
        self.n_bins = (6, 6, 12, 8)

        # ── 策略梯度（REINFORCE）专用参数 ──────────────────
        self.pg_lr: float = 0.1         # 策略梯度学习率（梯度上升步长，通常比 TD 方法大）
        self.use_baseline: bool = True  # True：用均值回报做基线（减小方差，加速收敛）
        self.hidden_size: int = 128     # 策略网络隐藏层神经元数量（越大表达能力越强）

        # ── 随机种子 ──────────────────────────────────────
        self.seed: int = 42             # 环境重置时的种子，设固定值可复现结果


# ============================================================
#  第2部分：环境工具 — 创建环境、状态离散化
# ============================================================

def make_env(render_human: bool = False):
    """
    创建一个新的 CartPole-v1 环境实例。
    每次训练循环调用一次，避免跨回合的状态残留。
    参数 render_human: 设为 True 则弹出图形窗口展示小车运动
    """
    return gym.make('CartPole-v1', render_mode='human' if render_human else None)


# ── 离散化函数：将连续状态 → 离散整数组 ────────────────

def _build_bins(n_bins: tuple) -> list:
    """
    根据 n_bins 中每个维度的箱子数，生成等距分箱边界。
    例如 n_bins[0]=6，则在 [-2.4, 2.4] 间均匀插入 5 个边界点，
    去掉首尾后得到 5 个内部边界：[-1.6, -0.8, 0.0, 0.8, 1.6]。
    np.digitize 用这 5 个边界将连续值映射到 [0, 1, 2, 3, 4, 5] 共 6 个箱。
    返回：列表，每个元素是一个一维 NumPy 数组（该维度的分箱边界）
    """
    bins = []
    for i, n in enumerate(n_bins):
        lo, hi = STATE_RANGES[i]          # 该维度的 (最小值, 最大值)
        # 在 [lo, hi] 间均匀生成 n+1 个点，去掉最小点和最大点
        # 剩下 n-1 个内部边界，np.digitize 用到这些边界
        bins.append(np.linspace(lo, hi, n + 1)[1:-1])
    return bins


def discretize(state: np.ndarray, bins: list) -> tuple:
    """
    将 4 维连续状态向量映射为离散元组索引，供 Q 表 / H 表查询。
    步骤：
      1. 将每个维度的值裁剪到 STATE_RANGES 范围内
      2. 用 np.digitize 将裁剪后的值映射到箱子编号
      3. 返回四元组如 (3, 2, 7, 4)
    参数：
      state: Gymnasium 返回的长度为 4 的连续状态数组
      bins: 由 _build_bins() 生成的边界列表
    返回：
      四元组，每个元素在 [0, n_bins[i]-1] 范围内
    """
    idx = []
    for i in range(len(state)):
        lo, hi = STATE_RANGES[i]          # 该维度的合法范围
        # 裁剪：将超出范围的值拉回到边界（避免极值导致索引越界）
        clipped = np.clip(state[i], lo, hi)
        # digitize：返回 clipped 落在 bins[i] 中哪个区间
        d = np.digitize(clipped, bins[i])
        # 防止 d 等于 len(bins[i])（即落在最后一个边界右边）
        d = min(d, len(bins[i]))
        idx.append(d)
    return tuple(idx)


# ============================================================
#  第3部分：Q-Learning 智能体
#  算法：Q(S,A) ← Q(S,A) + α[R + γ·maxₐQ(S',a) - Q(S,A)]
#  特点：离策略（off-policy），使用下一状态的最大 Q 值更新当前 Q
# ============================================================

class QLearningAgent:
    """Q-Learning 智能体：用 Q 表 + ε-贪婪策略 + 离策略 TD 更新"""

    def __init__(self, cfg: Config):
        # 保存配置引用
        self.cfg = cfg
        # 构建分箱边界（状态离散化用）
        self.bins = _build_bins(cfg.n_bins)
        # Q 表：形状为 (n_bins[0], n_bins[1], n_bins[2], n_bins[3], 2) 的全零数组
        # 前 4 维是离散化后的状态索引，最后一维是动作（0=左推，1=右推）
        self.q_table = np.zeros(cfg.n_bins + (2,))
        # 超参复制到实例变量，方便使用
        self.lr = cfg.learning_rate     # 学习率 α
        self.gamma = cfg.gamma          # 折扣因子 γ
        self.epsilon = cfg.epsilon      # 当前探索率（会逐回合衰减）
        self.epsilon_min = cfg.epsilon_min
        self.epsilon_decay = cfg.epsilon_decay

    def choose_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        ε-贪婪策略选择动作：
          - 以概率 ε 随机选（探索）
          - 以概率 1-ε 选 Q 值最大的动作（利用）
        参数 training=False 时强制利用（评估模式）
        """
        s = discretize(state, self.bins)          # 连续状态 → 离散索引
        if training and np.random.random() < self.epsilon:
            return np.random.randint(2)            # 探索：随机返回 0 或 1
        return int(np.argmax(self.q_table[s]))     # 利用：取当前状态 Q 值最大的动作

    def update(self, state, action, reward, next_state, done):
        """
        Q-Learning 更新公式（离策略）：
          Q(S,A) ← Q(S,A) + α [ R + γ·maxₐ Q(S',a) - Q(S,A) ]
        done=True 时目标值仅为 R（因为 S' 是终止状态，Q(S')=0）
        """
        s = discretize(state, self.bins)           # 当前状态索引
        ns = discretize(next_state, self.bins)     # 下一状态索引

        # 计算 TD 目标：reward + γ * maxₐ Q(next_state, a)
        # 如果游戏结束（done=True），则目标值就是 reward（没有下一状态了）
        td_target = reward + (0 if done else self.gamma * np.max(self.q_table[ns]))

        # TD 误差 = 目标值 - 当前 Q 值
        td_error = td_target - self.q_table[s + (action,)]

        # Q 表更新：向目标值靠近一步（步长为 α）
        self.q_table[s + (action,)] += self.lr * td_error

    def decay_epsilon(self):
        """每回合结束后衰减探索率：ε ← max(ε_min, ε * ε_decay)"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# ============================================================
#  第4部分：SARSA 智能体
#  算法：Q(S,A) ← Q(S,A) + α[R + γ·Q(S',A') - Q(S,A)]
#  特点：在策略（on-policy），使用实际执行的下一个动作的 Q 值
# ============================================================

class SarsaAgent:
    """SARSA 智能体：用 Q 表 + ε-贪婪策略 + 在策略 TD 更新"""

    def __init__(self, cfg: Config):
        # 与 QLearningAgent 结构相同，因为都用 Q 表
        self.cfg = cfg
        self.bins = _build_bins(cfg.n_bins)
        self.q_table = np.zeros(cfg.n_bins + (2,))
        self.lr = cfg.learning_rate
        self.gamma = cfg.gamma
        self.epsilon = cfg.epsilon
        self.epsilon_min = cfg.epsilon_min
        self.epsilon_decay = cfg.epsilon_decay

    def choose_action(self, state: np.ndarray, training: bool = True) -> int:
        """与 Q-Learning 完全相同的 ε-贪婪策略"""
        s = discretize(state, self.bins)
        if training and np.random.random() < self.epsilon:
            return np.random.randint(2)
        return int(np.argmax(self.q_table[s]))

    def update(self, state, action, reward, next_state, next_action, done):
        """
        SARSA 更新公式（在策略）：
          Q(S,A) ← Q(S,A) + α [ R + γ·Q(S',A') - Q(S,A) ]
        关键区别：用的是实际执行的 next_action 的 Q 值，而非最大 Q 值。
        因此调用方需要预先选好 next_action 再传入。
        done=True 时同样 Q(S',A')=0。
        """
        s = discretize(state, self.bins)            # 当前状态索引
        ns = discretize(next_state, self.bins)      # 下一状态索引

        # TD 目标：reward + γ * Q(next_state, next_action)
        td_target = reward + (0 if done else self.gamma * self.q_table[ns + (next_action,)])

        # TD 误差并更新
        td_error = td_target - self.q_table[s + (action,)]
        self.q_table[s + (action,)] += self.lr * td_error

    def decay_epsilon(self):
        """与 Q-Learning 相同的探索率衰减"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# ============================================================
#  第5部分：Policy Gradient (REINFORCE) 智能体
#  算法：θ ← θ + α G_t ∇_θ ln π_θ(A_t|S_t)
#  特点：
#    - 用神经网络参数化策略 π_θ(a|s)
#    - 每回合结束后，用整条轨迹计算折扣回报再更新
#    - 天然探索（按 softmax 概率采样），无需 ε-贪婪
#    - 训练结束后调用 clear_policy() 清空策略
# ============================================================

class PolicyGradientAgent:
    """
    基于 REINFORCE 的策略梯度智能体。
    策略由一个单隐层神经网络参数化：输入 4 维连续状态，输出 2 个动作的 softmax 概率。

    更新公式（策略梯度定理 + 数值稳定 softmax 梯度）：
      对已执行动作 A_t：  ∇_{H(s)} ln π(A_t|s) = 1 - π(A_t|s)   ← 偏好增加
      对未执行动作 a≠A_t： ∇_{H(s)} ln π(a|s)   = -π(a|s)        ← 偏好降低
      其中 H(s) 是网络输出的 logits，π = softmax(H(s))

    完整的梯度计算通过链式法则反向传播到网络各层参数 θ。
    """

    def __init__(self, cfg: Config):
        """
        初始化策略网络参数和超参数。
        网络权重随机初始化 → 初始策略接近均匀随机（满足"随机设定策略"的要求）。
        """
        self.cfg = cfg
        self.state_dim = 4                  # 环境状态维度（CartPole 为 4）
        self.action_dim = 2                 # 动作维度（左推 / 右推）
        self.hidden = cfg.hidden_size       # 隐藏层神经元数
        self.lr = cfg.pg_lr                 # 策略梯度学习率 α
        self.gamma = cfg.gamma              # 折扣因子 γ
        self.use_baseline = cfg.use_baseline  # 是否使用回报基线

        # ── 网络参数（神经网络参数化的策略 θ）──
        # W1 形状 (hidden, state_dim)：输入层到隐藏层的权重矩阵
        self.W1 = np.random.randn(self.hidden, self.state_dim) * 0.1
        # b1 形状 (hidden,)：隐藏层偏置向量
        self.b1 = np.zeros(self.hidden)
        # W2 形状 (action_dim, hidden)：隐藏层到输出层的权重矩阵
        self.W2 = np.random.randn(self.action_dim, self.hidden) * 0.1
        # b2 形状 (action_dim,)：输出层偏置向量
        self.b2 = np.zeros(self.action_dim)

        # ── 轨迹缓存（每个回合重新收集）──
        self._reset_memory()

    def _reset_memory(self):
        """清空轨迹缓存。每回合开始前和更新后调用。"""
        self.states = []    # 存储每个时间步的状态（连续向量）
        self.actions = []   # 存储每个时间步执行的动作（0 或 1）
        self.rewards = []   # 存储每个时间步获得的奖励

    def clear_policy(self):
        """
        清空策略 = 重新随机初始化网络参数 → 策略变回均匀随机。
        训练完成后调用，满足"清空小车的策略"的需求。
        """
        self.W1 = np.random.randn(self.hidden, self.state_dim) * 0.1
        self.b1 = np.zeros(self.hidden)
        self.W2 = np.random.randn(self.action_dim, self.hidden) * 0.1
        self.b2 = np.zeros(self.action_dim)
        self._reset_memory()  # 同时清空轨迹缓存

    def _forward(self, state: np.ndarray) -> tuple:
        """
        神经网络前向传播。
        输入：state — 长度为 4 的连续状态向量
        返回：(logits, probs)
          - logits：网络原始输出（softmax 之前的值，形状 (2,)）
          - probs：  softmax 归一化后的动作概率（形状 (2,)，和为 1）

        网络结构：
          输入 (4) → 线性层 W1+b1 → tanh 激活 → 线性层 W2+b2 → softmax → 输出 (2)
        """
        # 隐藏层：线性变换 + tanh 非线性激活
        h = np.tanh(self.W1 @ state + self.b1)   # @ 是矩阵乘法，结果形状 (hidden,)

        # 输出层：线性变换得到 logits
        logits = self.W2 @ h + self.b2            # 结果形状 (action_dim,)

        # 数值稳定 softmax：先减去最大值防止 exp 溢出
        logits_shifted = logits - np.max(logits)
        exp = np.exp(logits_shifted)
        probs = exp / np.sum(exp)                 # 归一化到概率分布

        return logits, probs

    def choose_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        根据策略 π_θ 选择动作：
          - 训练模式：按 softmax 概率分布采样（随机探索）
          - 评估模式：取概率最大的动作（确定性）
        训练模式下自动将 (state, action) 存入轨迹缓存。
        """
        _, probs = self._forward(state)           # 获取当前策略 π(·|s)

        if training:
            # 按概率分布采样：高概率动作更常被选中，但低概率动作仍有可能
            action = np.random.choice(self.action_dim, p=probs)

            # 存入轨迹（用于后续 REINFORCE 更新）
            self.states.append(state.copy())      # 保存状态副本（避免引用覆盖）
            self.actions.append(action)            # 保存动作
        else:
            # 评估模式：直接取概率最大的动作（确定性策略）
            action = int(np.argmax(probs))

        return action

    def store_reward(self, reward: float):
        """将当前步获得的奖励存入轨迹缓存。"""
        self.rewards.append(reward)

    def update(self):
        """
        REINFORCE 算法核心：用本回合完整轨迹计算策略梯度并更新网络参数。
        实现用户给出的公式 θ_{t+1} ← θ_t + α G_t ∇_θ ln π(A_t|S_t, θ)

        计算步骤：
          1. 从后向前计算折扣回报 G_t = Σ γ^{k-t} r_k
          2. （可选）标准化回报以减小方差
          3. 对每个时间步计算 ∇_θ ln π(a|s) 并累加梯度
          4. 梯度上升更新网络参数 θ ← θ + α · G_t · ∇_θ ln π
          5. 清空轨迹缓存
        """
        T = len(self.rewards)
        if T == 0:
            return  # 空轨迹不做任何更新

        # ── 步骤1：计算折扣回报 G_t（从后向前累计） ───
        # G_t = r_t + γ * r_{t+1} + γ² * r_{t+2} + ...
        returns = np.zeros(T)
        G = 0.0
        for t in reversed(range(T)):
            G = self.rewards[t] + self.gamma * G
            returns[t] = G

        # ── 步骤2：标准化回报（零均值单位方差） ───────
        # 标准化能大幅减小梯度方差，让学习更稳定
        returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)

        # ── 步骤3和4：对每个时间步计算梯度并更新 ─────
        for t in range(T):
            state = self.states[t]      # 当前状态 s_t
            action = self.actions[t]     # 执行的动作 a_t
            delta = returns[t]           # 标准化后的 G_t

            # 前向传播（重新计算以获取中间值，供反向传播用）
            # 注意：这里需要重新前向传播因为参数 θ 在上一步已更新
            h = np.tanh(self.W1 @ state + self.b1)
            logits = self.W2 @ h + self.b2

            # softmax 求概率
            probs = np.exp(logits - np.max(logits))
            probs = probs / np.sum(probs)

            # ── 计算 ∇_θ ln π(a|s) 的解析梯度 ──────────
            # softmax + 交叉熵的梯度 = one_hot(a) - π(·|s)
            # d(ln π(a|s)) / d(logits) = -probs，
            # 并且对 d(logits[a]) 额外 +1
            dlogits = -probs.copy()      # 对所有动作：-π(a|s)
            dlogits[action] += 1.0       # 对执行的动作为：(1 - π(a|s))

            # ── 输出层 W2, b2 的梯度 ──────────────────
            # dL/dW2 = dlogits ⊗ h   （外积：形状 (2, hidden)）
            dW2 = np.outer(dlogits, h)
            db2 = dlogits

            # ── 反向传播到隐藏层 ──────────────────────
            # dL/dh = W2^T @ dlogits
            dh = self.W2.T @ dlogits
            # 乘上 tanh 的导数：tanh'(x) = 1 - tanh²(x)
            dh = dh * (1.0 - h ** 2)

            # ── 隐藏层 W1, b1 的梯度 ─────────────────
            dW1 = np.outer(dh, state)
            db1 = dh

            # ── 梯度上升更新：θ ← θ + α · G_t · ∇_θ ln π ──
            # 注意是加（梯度上升）因为我们想最大化期望回报
            self.W1 += self.lr * delta * dW1
            self.b1 += self.lr * delta * db1
            self.W2 += self.lr * delta * dW2
            self.b2 += self.lr * delta * db2

        # ── 步骤5：清空轨迹，为下一回合做准备 ────────
        self._reset_memory()


# ============================================================
#  第6部分：训练循环 — 三个算法的各自训练主循环
#  每个训练函数：
#    1. 创建对应智能体（此时策略随机初始化）
#    2. 循环 n_episodes 回合
#    3. 每回合：重置环境 → 重复 选动作-执行-更新 直到终止
#    4. 记录每回合总奖励
#    5. 函数返回后智能体对象出作用域 → Python GC 自动回收 → 策略清空
# ============================================================

def train_policy_gradient(cfg: Config) -> list:
    """
    策略梯度 (REINFORCE) 训练主循环。
    参数 cfg: 训练配置
    返回 rewards_per_episode: 列表，每回合总奖励，长度 = cfg.n_episodes
    """
    # 创建策略梯度智能体 → 网络权重随机初始化 → 初始策略是均匀随机
    # 满足"随机设定一个策略"的需求
    agent = PolicyGradientAgent(cfg)
    rewards_per_episode = []  # 记录每回合总奖励

    for ep in range(1, cfg.n_episodes + 1):
        # ── 每个回合开始：创建新环境实例，重置到初始状态 ──
        env = make_env()
        state, _ = env.reset(seed=cfg.seed + ep)  # 用 seed+ep 保证不同回合不同初始状态
        total_reward = 0  # 本回合累计奖励
        done = False

        # ── 回合内循环：直到杆子倒下（或达到 max_steps） ──
        while not done:
            # 策略网络输出概率 → 采样一个动作
            action = agent.choose_action(state, training=True)
            # 执行动作，环境返回下一状态、奖励、终止标志等
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated     # 任一标志为真都表示本回合结束

            # 存储奖励（供回合末的 REINFORCE 更新使用）
            agent.store_reward(reward)

            # 推进到下一状态
            state = next_state
            total_reward += reward

        # ── 回合结束：REINFORCE 更新策略网络参数 ──
        agent.update()

        # 记录本回合总奖励
        rewards_per_episode.append(total_reward)
        env.close()  # 关闭环境实例，释放资源

        # ── 日志输出（每 50 回合或第 1 回合） ──
        if ep % 50 == 0 or ep == 1:
            avg = np.mean(rewards_per_episode[-50:]) if ep >= 50 else np.mean(rewards_per_episode)
            print(f"  Ep {ep:4d}/{cfg.n_episodes}  |  "
                  f"当前奖励={total_reward:5.0f}  |  最近平均={avg:.1f}")

    # 训练结束，agent 离开函数作用域后 Python 自动回收
    # 相当于"清空小车策略"
    return rewards_per_episode


def train_q_learning(cfg: Config) -> list:
    """
    Q-Learning 训练主循环。
    参数和返回值同 train_policy_gradient。
    """
    # 创建 Q-Learning 智能体 → Q 表全零 → 初始策略（argmax 零表 = 均匀随机）
    agent = QLearningAgent(cfg)
    rewards_per_episode = []

    for ep in range(1, cfg.n_episodes + 1):
        env = make_env()
        state, _ = env.reset(seed=cfg.seed + ep)
        total_reward = 0
        done = False

        while not done:
            action = agent.choose_action(state, training=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            # Q-Learning 更新：只用 state/action/reward/next_state，不需要 next_action
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        # ε 衰减（探索率逐回合降低）
        agent.decay_epsilon()
        rewards_per_episode.append(total_reward)
        env.close()

        if ep % 50 == 0 or ep == 1:
            avg = np.mean(rewards_per_episode[-50:]) if ep >= 50 else np.mean(rewards_per_episode)
            print(f"  Ep {ep:4d}/{cfg.n_episodes}  |  ε={agent.epsilon:.3f}  |  "
                  f"当前奖励={total_reward:5.0f}  |  最近平均={avg:.1f}")

    return rewards_per_episode


def train_sarsa(cfg: Config) -> list:
    """
    SARSA 训练主循环。
    与 Q-Learning 的关键区别：
      - 需要在执行前预先选择下一个动作 A'（因为 SARSA 需要 Q(S',A')）
      - 第一次选择 A 后，在循环内：执行 A → 选 A' → 用 A' 更新 → A ← A'
    """
    agent = SarsaAgent(cfg)
    rewards_per_episode = []

    for ep in range(1, cfg.n_episodes + 1):
        env = make_env()
        state, _ = env.reset(seed=cfg.seed + ep)

        # SARSA 需要在回合开始就选好第一个动作 A
        action = agent.choose_action(state, training=True)

        total_reward = 0
        done = False

        while not done:
            # 执行动作 A，得到下一状态 S' 和奖励 R
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # 在更新之前预先选好下一个动作 A'（SARSA 的关键步骤）
            next_action = agent.choose_action(next_state, training=True)

            # SARSA 更新：使用实际执行的 next_action
            agent.update(state, action, reward, next_state, next_action, done)

            # 推进到下一状态和动作（S ← S', A ← A'）
            state, action = next_state, next_action
            total_reward += reward

        agent.decay_epsilon()
        rewards_per_episode.append(total_reward)
        env.close()

        if ep % 50 == 0 or ep == 1:
            avg = np.mean(rewards_per_episode[-50:]) if ep >= 50 else np.mean(rewards_per_episode)
            print(f"  Ep {ep:4d}/{cfg.n_episodes}  |  ε={agent.epsilon:.3f}  |  "
                  f"当前奖励={total_reward:5.0f}  |  最近平均={avg:.1f}")

    return rewards_per_episode


# ============================================================
#  第7部分：绘图工具 — 绘制"奖励 vs 回合数"曲线
#  完成后展示：每回合奖励散点 + 滑动平均 + 最终均值线 + 参数信息
# ============================================================

def moving_average(data: np.ndarray, window: int = 20) -> np.ndarray:
    """
    计算一维数组的滑动平均（卷积方式）。
    用于平滑奖励曲线，让趋势更清晰可见。
    若数据长度 < window，返回全 NaN 数组。
    """
    if len(data) < window:
        return np.full_like(data, np.nan)
    # 用卷积实现滑动平均：与 [1/w, 1/w, ..., 1/w] 做卷积
    return np.convolve(data, np.ones(window) / window, mode='same')


def plot_results(rewards: list, cfg: Config, save_path: str = None):
    """
    绘制每回合奖励的折线图（包含滑动平均和关键指标标注）。
    参数：
      rewards:   train_* 返回的每回合奖励列表
      cfg:       训练配置（用于标注参数信息）
      save_path: 图片保存路径，None 则不保存
    """
    # 转换为 NumPy 数组便于计算
    rewards = np.array(rewards)
    episodes = np.arange(1, len(rewards) + 1)  # x 轴：回合编号（从 1 开始）

    # 计算滑动平均（窗口取 20 和 len/5 中较小的值，避免过平滑）
    window = min(20, len(rewards) // 5 or 1)
    smooth = moving_average(rewards, window=window)

    # ── 创建画布 ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))

    # 1) 原始每回合奖励（半透明灰色点+细线，表示原始数据分布）
    ax.plot(episodes, rewards, alpha=0.25, color='steelblue',
            linewidth=0.8, label='每回合奖励')

    # 2) 滑动平均曲线（红色粗线，表示学习趋势）
    mask = ~np.isnan(smooth)  # 排除 NaN 点
    ax.plot(episodes[mask], smooth[mask],
            color='tomato', linewidth=2.0,
            label=f'滑动平均 (窗口={window})')

    # 3) 最后 100 回合均值横线（绿色虚线，表示最终收敛水平）
    final_avg = np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
    ax.axhline(y=final_avg, color='seagreen', linestyle='--', linewidth=1.2,
               label=f'最终100回合均值 = {final_avg:.1f}')

    # ── 总奖励统计 ────────────────────────────────────
    total = np.sum(rewards)

    # ── 算法名称映射（中文显示友好） ──────────────────
    algo_names = {
        'q_learning': 'Q-Learning',
        'sarsa': 'SARSA',
        'policy_gradient': '策略梯度 (REINFORCE)',
    }
    algo_label = algo_names.get(cfg.algorithm, cfg.algorithm)

    # ── 坐标轴标签和标题 ──────────────────────────────
    ax.set_xlabel('回合数 (Episode)', fontsize=12)
    ax.set_ylabel('奖励 (Reward)', fontsize=12)
    ax.set_title(
        f'{algo_label} — 训练曲线  |  总奖励 = {total:.0f}  |  最终均值 = {final_avg:.1f}',
        fontsize=13, fontweight='bold'
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)  # 浅色网格辅助阅读

    # ── 在右下角标注关键超参数（方便对比不同实验） ────
    if cfg.algorithm == 'policy_gradient':
        param_text = (
            f"γ={cfg.gamma}, α_pg={cfg.pg_lr}\n"
            f"baseline={'On' if cfg.use_baseline else 'Off'}\n"
            f"hidden={cfg.hidden_size}"
        )
    else:
        param_text = (
            f"γ={cfg.gamma}, α={cfg.learning_rate}\n"
            f"ε_start={cfg.epsilon}, ε_min={cfg.epsilon_min}, ε_decay={cfg.epsilon_decay}\n"
            f"bins={cfg.n_bins}"
        )
    ax.text(
        0.98, 0.05, param_text,
        transform=ax.transAxes,
        fontsize=9, verticalalignment='bottom', horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.7)
    )

    plt.tight_layout()

    # ── 保存图片 ──────────────────────────────────────
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  训练曲线图已保存至: {save_path}")

    # ── 显示图形窗口 ──────────────────────────────────
    plt.show()


# ============================================================
#  第8部分：交互式主菜单（终端 UI）
#  提供算法选择、参数修改、训练开始等交互功能
# ============================================================

def print_header():
    """打印程序标题横幅"""
    print("""
+====================================================+
|       CartPole 强化学习训练框架                      |
|       三种算法  |  可视化训练曲线  |  参数可调        |
+====================================================+
    """)


def print_algo_menu():
    """打印算法选择菜单（1=Q-Learning, 2=SARSA, 3=REINFORCE, 0=退出）"""
    print("\n请选择训练算法（输入编号）：")
    print("  1) Q-Learning              — 离策略 TD 控制")
    print("  2) SARSA                   — 在策略 TD 控制")
    print("  3) 策略梯度 (REINFORCE)    — 策略梯度方法")
    print("  0) 退出")


def print_params_menu(cfg: Config):
    """
    打印当前配置参数（根据算法不同显示不同参数列表）。
    末尾提示：m=返回主菜单，s=开始训练
    """
    print("\n当前训练参数：")
    print(f"  ① 算法: {cfg.algorithm}")
    print(f"  ② 训练回合数 (n_episodes): {cfg.n_episodes}")

    if cfg.algorithm == 'policy_gradient':
        # 策略梯度专有参数列表
        print(f"  ③ 策略梯度学习率 (pg_lr): {cfg.pg_lr}")
        print(f"  ④ 折扣因子 (γ): {cfg.gamma}")
        print(f"  ⑤ 使用基线 (use_baseline): {cfg.use_baseline}")
        print(f"  ⑥ 隐藏层神经元数 (hidden_size): {cfg.hidden_size}")
    else:
        # SARSA / Q-Learning 共用参数列表
        print(f"  ③ 学习率 (α): {cfg.learning_rate}")
        print(f"  ④ 折扣因子 (γ): {cfg.gamma}")
        print(f"  ⑤ 初始探索率 (ε): {cfg.epsilon}")
        print(f"  ⑥ 探索率衰减 (ε_decay): {cfg.epsilon_decay}")
        print(f"  ⑦ 最小探索率 (ε_min): {cfg.epsilon_min}")
        print(f"  ⑧ 离散化格子数 (位置,速度,角度,角速度): {cfg.n_bins}")

    print("  m) 返回主菜单  |  s) 开始训练")


def interactive_main():
    """
    交互式主入口函数。
    流程：
      选择算法 → 查看/修改参数 → 开始训练 → 显示曲线 → 返回算法选择
    通过终端输入驱动，无需任何 GUI 库。
    """
    # 创建默认配置实例
    cfg = Config()

    # 编号 → 算法名映射
    algo_map = {'1': 'q_learning', '2': 'sarsa', '3': 'policy_gradient'}

    # ── 主循环（算法选择） ───────────────────────────
    while True:
        print_header()
        print_algo_menu()
        choice = input(">> 请输入编号: ").strip()

        if choice == '0':
            print("[再见！]")
            break

        if choice not in algo_map:
            print("[X] 无效输入，请重新选择。")
            continue

        # 记录用户选择的算法
        cfg.algorithm = algo_map[choice]
        print(f"\n[OK] 已选择: {cfg.algorithm}")

        # ── 参数配置子循环 ──────────────────────────
        while True:
            print("\n" + "-" * 55)
            print_params_menu(cfg)
            cmd = input("\n>> 输入编号修改参数，或 s/m: ").strip().lower()

            if cmd == 'm':
                break  # 返回算法选择主菜单

            elif cmd == 's':
                # ═══════════════════════════════════════
                #  开始训练
                # ═══════════════════════════════════════
                print("\n[开始训练...]\n")
                print(f"算法: {cfg.algorithm}  |  回合数: {cfg.n_episodes}")
                print("=" * 55)

                # 根据算法选择对应的训练函数
                trainer_map = {
                    'q_learning': train_q_learning,
                    'sarsa': train_sarsa,
                    'policy_gradient': train_policy_gradient,
                }
                trainer = trainer_map[cfg.algorithm]

                # 执行训练，获得每回合奖励列表
                rewards = trainer(cfg)

                # ── 训练完成，显示统计信息 ──────────
                total_reward = np.sum(rewards)
                avg_last_100 = np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
                print("\n" + "=" * 55)
                print(f"[完成] 训练完成！策略已自动清空（智能体已释放）。")
                print(f"   总收益（所有回合奖励之和）:     {total_reward:.0f}")
                print(f"   最后 100 回合平均奖励:          {avg_last_100:.2f}")
                print(f"   最大单回合奖励:                 {np.max(rewards):.0f}")
                print(f"   最小单回合奖励:                 {np.min(rewards):.0f}")
                print("=" * 55)

                # ── 绘图：奖励关于回合数的函数折线图 ──
                plot_results(
                    rewards, cfg,
                    save_path=os.path.join(
                        os.path.dirname(__file__) or '.',
                        f"training_curve_{cfg.algorithm}.png"
                    )
                )
                break  # 训练结束，返回算法选择主菜单

            # ──────────────────────────────────────────
            #  参数修改（cmd 为数字编号）
            # ──────────────────────────────────────────
            elif cmd == '1':
                # 修改算法
                print(f"  当前算法: {cfg.algorithm}")
                new_algo = input(
                    "  新算法 (1/q_learning, 2/sarsa, 3/policy_gradient) [回车不变]: "
                ).strip()
                if new_algo in algo_map:
                    cfg.algorithm = algo_map[new_algo]
                    print(f"  -> 算法已切换为: {cfg.algorithm}")

            elif cmd == '2':
                # 修改训练回合数
                try:
                    val = input(f"  训练回合数 (当前={cfg.n_episodes}): ").strip()
                    if val:
                        cfg.n_episodes = int(val)
                except ValueError:
                    print("  输入无效，保持原值。")

            elif cmd == '3':
                # 修改学习率（策略梯度用 pg_lr，其他用 learning_rate）
                if cfg.algorithm == 'policy_gradient':
                    key = 'pg_lr'
                    label = '策略梯度学习率'
                    cur = cfg.pg_lr
                else:
                    key = 'learning_rate'
                    label = '学习率 alpha'
                    cur = cfg.learning_rate
                try:
                    val = input(f"  {label} (当前={cur}): ").strip()
                    if val:
                        setattr(cfg, key, float(val))
                except ValueError:
                    print("  输入无效，保持原值。")

            elif cmd == '4':
                # 修改折扣因子 gamma
                try:
                    val = input(f"  折扣因子 gamma (当前={cfg.gamma}): ").strip()
                    if val:
                        cfg.gamma = float(val)
                except ValueError:
                    print("  输入无效，保持原值。")

            elif cmd == '5':
                # 修改 use_baseline（PG）或 epsilon（SARSA/Q-Learning）
                if cfg.algorithm == 'policy_gradient':
                    val = input(f"  使用基线 (当前={cfg.use_baseline}, y/n): ").strip().lower()
                    if val == 'y':
                        cfg.use_baseline = True
                    elif val == 'n':
                        cfg.use_baseline = False
                else:
                    try:
                        val = input(f"  初始探索率 epsilon (当前={cfg.epsilon}): ").strip()
                        if val:
                            cfg.epsilon = float(val)
                    except ValueError:
                        print("  输入无效，保持原值。")

            elif cmd == '6':
                # 修改隐藏层大小（PG）或 epsilon_decay（SARSA/Q-Learning）
                if cfg.algorithm == 'policy_gradient':
                    try:
                        val = input(f"  隐藏层神经元数 (当前={cfg.hidden_size}): ").strip()
                        if val:
                            cfg.hidden_size = int(val)
                    except ValueError:
                        print("  输入无效，保持原值。")
                else:
                    try:
                        val = input(f"  探索率衰减 (当前={cfg.epsilon_decay}): ").strip()
                        if val:
                            cfg.epsilon_decay = float(val)
                    except ValueError:
                        print("  输入无效，保持原值。")

            elif cmd == '7' and cfg.algorithm not in ('policy_gradient',):
                # 修改最小值 epsilon_min（仅 SARSA/Q-Learning）
                try:
                    val = input(f"  最小探索率 (当前={cfg.epsilon_min}): ").strip()
                    if val:
                        cfg.epsilon_min = float(val)
                except ValueError:
                    print("  输入无效，保持原值。")

            elif cmd == '8' and cfg.algorithm not in ('policy_gradient',):
                # 修改离散化格子数（仅 SARSA/Q-Learning）
                try:
                    raw = input(f"  离散化格数 (位置,速度,角度,角速度) 当前={cfg.n_bins}: ").strip()
                    if raw:
                        cfg.n_bins = tuple(int(x.strip()) for x in raw.split(','))
                except (ValueError, TypeError):
                    print("  格式错误，保持原值。")

            else:
                print("  无效选项。")

    print("程序结束。")


# ============================================================
#  第9部分：快捷入口 — 命令行一键运行，跳过交互菜单
#  用法：
#     python cartpole_rl.py q_learning 500
#     python cartpole_rl.py sarsa 300
#     python cartpole_rl.py policy_gradient 200
# ============================================================

def run_demo(algorithm: str = 'policy_gradient', n_episodes: int = 300):
    """
    非交互式快捷运行函数。
    直接创建 Config、选择算法、训练、绘图，无需菜单选择。
    参数:
      algorithm:  'q_learning' | 'sarsa' | 'policy_gradient'
      n_episodes: 训练回合数
    返回:
      rewards_per_episode 列表，或 None（算法不存在时）
    """
    # 创建配置并设置算法和回合数
    cfg = Config()
    cfg.algorithm = algorithm
    cfg.n_episodes = n_episodes

    # 算法名 → 训练函数的映射
    trainer_map = {
        'q_learning': train_q_learning,
        'sarsa': train_sarsa,
        'policy_gradient': train_policy_gradient,
    }

    print(f"\n[快捷运行] {algorithm}  |  {n_episodes} 回合\n")

    # 查找训练函数并执行
    trainer = trainer_map.get(algorithm)
    if trainer is None:
        print("[X] 无效算法")
        return None

    # 执行训练
    rewards = trainer(cfg)

    # 输出训练统计
    total_reward = np.sum(rewards)
    avg_last_100 = np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
    print(f"\n[完成] 总收益: {total_reward:.0f}  |  最后100回合均值: {avg_last_100:.2f}")
    print(f"   策略已自动清空。")

    # 绘制奖励曲线并显示
    plot_results(rewards, cfg)
    return rewards


# ============================================================
#  程序入口
#  运行方式：
#    1. 无参数 → 交互式菜单（interactive_main）
#    2. 有参数 → 命令行模式（run_demo）
#       python cartpole_rl.py <算法> [回合数]
#       例：python cartpole_rl.py policy_gradient 500
# ============================================================

if __name__ == '__main__':
    # ── 第一步：检查三方库是否已安装 ────────────────
    try:
        import gymnasium
        import numpy
        import matplotlib
    except ImportError as e:
        print(f"[X] 缺少依赖: {e.name}")
        print("请安装: pip install gymnasium numpy matplotlib")
        sys.exit(1)

    # ── 第二步：根据参数决定运行模式 ────────────────
    if len(sys.argv) > 1:
        # 命令行参数模式：python cartpole_rl.py <算法> [回合数]
        algo = sys.argv[1].lower()
        # 支持多种简写
        algo_map = {
            'q': 'q_learning',
            'q_learning': 'q_learning',
            'sarsa': 'sarsa',
            'pg': 'policy_gradient',
            'policy_gradient': 'policy_gradient',
        }
        if algo in algo_map:
            episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 300
            run_demo(algo_map[algo], episodes)
        else:
            print("用法: python cartpole_rl.py <算法> [回合数]")
            print("  算法可选: q_learning, sarsa, policy_gradient")
    else:
        # 无参数 → 交互式菜单模式
        interactive_main()
