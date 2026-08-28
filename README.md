# RL_Lab
研究小组的强化学习算法测试项目，基于 **Gymnasium**。

## 项目结构

每个环境一个文件夹，文件夹内包含：

| 文件 | 作用 |
|---|---|
| `fram_work.py` | 主框架：初始化环境 → N 轮训练 → 奖励曲线图（改 `ALGORITHM` 切换算法） |
| `Q_learning.py` | Q-Learning 算法类（off-policy，`Q ← Q + α[R + γ·max Q' − Q]`） |
| `sarsa.py` | SARSA 算法类（on-policy，`Q ← Q + α[R + γ·Q' − Q]`） |
| `gradient.py` | 策略梯度 REINFORCE（softmax 偏好表，`H ← H + α(G−R̄)(1−π)`） |
| `utils.py` | 公共工具：连续状态离散化（分箱） |

## 环境列表

| 文件夹 | 环境 | 状态维度 | 动作 | 说明 |
|---|---|---|---|---|
| `cartpole/` | CartPole-v1 | 4 维 | 2 个离散 | 平衡小车上的倒立摆 |
| `mountaincar/` | MountainCar-v0 | 2 维 | 3 个离散 | 小车借惯性冲上陡坡 |
| `acrobot/` | Acrobot-v1 | 6 维 | 3 个离散 | 双节摆荡到目标高度 |
| `pendulum/` | Pendulum-v1 | 3 维 | 连续→5 档 | 单摆稳定在竖直向上 |
| `mountaincar_cont/` | MountainCarContinuous-v0 | 2 维 | 连续→3 档 | 连续版爬坡小车 |

## 使用方法

```bash
cd cartpole            # 进入想测试的环境文件夹
python fram_work.py    # 训练并出图
conda env create -f environment.yml # 创建conda环境
```

修改 `fram_work.py` 末尾的 `ALGORITHM` 和 `EPISODES` 即可切换算法与训练轮数。

## 设计说明

- 三种算法都是表格型实现，需将连续状态离散化（`utils.py` 统一处理）
- 连续动作环境（pendulum、mountaincar_cont）把动作离散成固定档位
- 代码风格遵循：多注释、简单可读优先、只实现必要功能
