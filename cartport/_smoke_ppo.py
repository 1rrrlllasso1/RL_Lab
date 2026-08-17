"""PPO 冒烟测试:验证 fram_work 循环里 _last_logprob 与执行动作 (s,a) 的配对是否正确。

方法:把 actor 换成固定 logits 的网络,则 logπ(a) 只由动作 a 决定、
与状态无关。因此 update() 暂存 _last_logprob 时,如果它对应的是本次
已执行的动作,就一定等于 log_softmax(const)[a];否则就是配错了。
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import torch.nn as nn

import gymnasium as gym

# 强制环境不弹窗渲染(避免最后一轮 human 模式弹出 pygame 窗口)
_orig_make = gym.make
gym.make = lambda id, **kw: _orig_make(id, render_mode=None)

from fram_work import run_cartpole
from ppo import PPO


class ConstLogits(nn.Module):
    """固定 logits:logπ(a) 只取决于动作 a,便于校验配对是否正确"""
    def __init__(self):
        super().__init__()
        self.const = nn.Parameter(torch.tensor([[1.0, -1.0]]))
    def forward(self, x):
        return self.const.expand(x.shape[0], -1)


agent = PPO()
agent.actor = ConstLogits()

# 常数 logits 下两个动作的期望 log 概率
expected = torch.log_softmax(torch.tensor([[1.0, -1.0]]), dim=-1)[0]

pairs = []          # update() 调用时记录的 (action, _last_logprob)
learn_calls = [0]

_orig_update = agent.update
def spy_update(obs, action, reward, next_obs, done):
    pairs.append((int(action), agent._last_logprob))
    return _orig_update(obs, action, reward, next_obs, done)
agent.update = spy_update

_orig_learn = agent._learn
def spy_learn():
    learn_calls[0] += 1
    return _orig_learn()
agent._learn = spy_learn

rewards = run_cartpole(120, agent)

total_steps = len(pairs)
print(f"\n共运行 {len(rewards)} 轮,累计 {total_steps} 步,PPO 触发 {learn_calls[0]} 次更新")
print(f"平均奖励: {np.mean(rewards):.2f}  前 5 轮奖励: {[int(r) for r in rewards[:5]]}")

# 校验 1:缓冲区至少填满一次,_learn 至少执行一次(否则测试没真正训练到)
assert learn_calls[0] >= 1, "缓冲区未填满,_learn 未触发,冒烟测试无效"

# 校验 2:每条 (action, logprob) 必须匹配常数 logits 下的期望值
mismatch = 0
for action, logp in pairs:
    if abs(logp - expected[action].item()) > 1e-4:
        mismatch += 1
print(f"配对校验: {len(pairs) - mismatch}/{len(pairs)} 条正确")
assert mismatch == 0, f"{mismatch} 条 (action, logprob) 配对错误 —— logprob 配错了动作!"

# 校验 3:奖励非负、数值正常
assert all(r >= 0 for r in rewards), "存在负奖励,异常"

print("冒烟测试通过:_learn 正常触发,(s,a) 与 logprob 配对正确,循环无异常")
