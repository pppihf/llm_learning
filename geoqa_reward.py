# GeoQA 选择题奖励函数 —— 官方 plugin 格式
# 用法:
#   --external_plugins /path/to/geoqa_reward.py
#   --reward_funcs geoqa_accuracy geoqa_format

import re
from typing import List

from swift.rewards import ORM, orms


class GeoQAAccuracy(ORM):
    """选择题准确率奖励: 答案匹配得 1.0，不匹配得 0.0"""

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        rewards = []
        for completion, sol in zip(completions, solution):
            # 优先从 <answer> 标签中提取
            answer_match = re.search(r'<answer>\s*([A-Da-d])\s*</answer>', completion, re.DOTALL)
            if not answer_match:
                # 回退: 尝试匹配 "选X" 或 "答案是X" 或 "故选X" 等
                answer_match = re.search(r'(?:选|答案[是为]?|选择)\s*[：:]?\s*([A-Da-d])', completion)
            if not answer_match:
                # 回退: 尝试匹配最后出现的 (X) 格式
                all_matches = re.findall(r'\(([A-Da-d])\)', completion)
                pred = all_matches[-1].upper() if all_matches else ""
            else:
                pred = answer_match.group(1).upper()

            correct = sol.strip().upper()
            rewards.append(1.0 if pred == correct else 0.0)
        return rewards


class GeoQAFormat(ORM):
    """格式奖励: 使用 <think>...</think><answer>X</answer> 格式得分"""

    def __call__(self, completions, **kwargs) -> List[float]:
        rewards = []
        for completion in completions:
            has_think = bool(re.search(r'<think>.*?</think>', completion, re.DOTALL))
            has_answer = bool(re.search(r'<answer>\s*[A-Da-d]\s*</answer>', completion, re.DOTALL))

            if has_think and has_answer:
                reward = 1.0
            elif has_answer:
                reward = 0.5
            else:
                reward = 0.0
            rewards.append(reward)
        return rewards


# 注册到 orms —— 必须，swift 通过名称查找
orms['geoqa_accuracy'] = GeoQAAccuracy
orms['geoqa_format'] = GeoQAFormat
