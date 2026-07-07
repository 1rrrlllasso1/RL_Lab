
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