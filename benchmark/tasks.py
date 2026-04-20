"""
Task definitions for Phy-Bench: two-tier evaluation protocol.

Tier 1 — Prediction Task (能否"用"规律):
  Given the first N frames of an episode (optionally with a rule description),
  predict the next M frames. Score on physics consistency and visual quality.

Tier 2 — Induction Task (能否"发现"规律):
  Given K episodes from the same world (video only, no rule description),
  produce a natural language description of the world's physics rule.
  Score on coverage of key facts and absence of wrong conclusions.

Both tiers have a "zero-shot" variant (no rule description provided) and a
"rule-conditioned" variant (rule description provided), allowing us to measure
how much explicit rule knowledge helps.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum


class TaskVariant(str, Enum):
    ZERO_SHOT = "zero_shot"          # No rule description provided
    RULE_CONDITIONED = "rule_conditioned"  # Rule description provided as context


@dataclass
class PredictionSample:
    """
    A single sample for the Prediction Task.

    The model receives:
      - context_frames: list of frame indices to use as input context
      - rule_description (optional): the world's rule in natural language
      - initial_conditions: the initial state of all objects

    The model must produce a predicted trajectory for frames in target_frames.
    Ground truth is available in the episode's states.jsonl.
    """
    world_id: str
    episode_id: str
    context_frames: List[int]       # e.g. [0, 1, ..., 29]  (first 1 second at 30fps)
    target_frames: List[int]        # e.g. [30, 31, ..., 89] (next 2 seconds)
    variant: TaskVariant
    rule_description: Optional[str] = None  # Only set for RULE_CONDITIONED variant

    # The following are populated from the episode data at evaluation time
    context_video_path: str = ""
    episode_data_root: str = ""


@dataclass
class InductionSample:
    """
    A single sample for the Induction Task.

    The model receives:
      - context_episodes: K episode IDs from the same world
      - (no rule description)

    The model must produce a natural language description of the world's rule.
    Ground truth is the world's rule_description.standard (or concise/verbose).
    """
    world_id: str
    context_episodes: List[str]     # K episode IDs from train split
    num_context_episodes: int       # K (typically 3–5)
    probe_episode_id: str           # Episode used for follow-up generalization test

    episode_data_root: str = ""


@dataclass
class GeneralizationProbe:
    """
    Follow-up probe attached to an InductionSample.
    After the model produces a rule description, we ask it counterfactual questions
    based on a held-out episode to test whether the induced rule actually generalizes.
    """
    world_id: str
    episode_id: str
    questions: List[Dict[str, Any]]   # List of {question, expected_answer, answer_type}


# ---------------------------------------------------------------------------
# Counterfactual / probing question templates per world
# ---------------------------------------------------------------------------

GENERALIZATION_PROBES: Dict[str, List[Dict[str, Any]]] = {
    "anti_gravity_bounce": [
        {
            "question": "如果把一个球放在地板上松手，它会往哪个方向运动？",
            "expected_answer": "向上运动（向天花板方向）",
            "answer_type": "direction",
            "key_concepts": ["向上", "反重力", "天花板"],
        },
        {
            "question": "与正常世界相比，这个世界的球在碰撞后会更快还是更慢？",
            "expected_answer": "更快（每次碰撞速度增大）",
            "answer_type": "comparison",
            "key_concepts": ["更快", "速度增大", "碰撞加速"],
        },
    ],
    "black_hole": [
        {
            "question": "一个距离中心很远的静止球会怎样运动？",
            "expected_answer": "被吸引向中心加速运动",
            "answer_type": "trajectory_description",
            "key_concepts": ["向中心", "加速", "吸引"],
        },
        {
            "question": "一个球非常靠近屏幕中心时会发生什么？",
            "expected_answer": "被弹飞（进入核心后受斥力）",
            "answer_type": "event_description",
            "key_concepts": ["弹飞", "斥力", "核心区域"],
        },
    ],
    "time_dilation": [
        {
            "question": "一个以相同速度进入左侧区域和右侧区域的球，哪边看起来移动更快？",
            "expected_answer": "右侧移动更快",
            "answer_type": "comparison",
            "key_concepts": ["右侧", "更快", "慢时区/快时区"],
        },
        {
            "question": "一个球从右侧快速越过中线进入左侧，它的视觉速度会如何变化？",
            "expected_answer": "突然变慢（约变为原来的1/4）",
            "answer_type": "event_description",
            "key_concepts": ["变慢", "突然", "跨越边界"],
        },
    ],
    "mass_oscillation": [
        {
            "question": "为什么这个世界的球看起来时快时慢，而不是匀速运动？",
            "expected_answer": "因为球的质量在周期性变化，导致加速度随之变化",
            "answer_type": "explanation",
            "key_concepts": ["质量变化", "加速度变化", "周期"],
        },
        {
            "question": "球在什么时候运动最快？",
            "expected_answer": "质量最小的时候",
            "answer_type": "condition",
            "key_concepts": ["质量最小", "加速度最大"],
        },
    ],
    "bounce_accelerator": [
        {
            "question": "一个球碰了3次墙后，它的速度大约是初始速度的几倍？",
            "expected_answer": "8倍（2^3）",
            "answer_type": "quantitative",
            "key_concepts": ["8倍", "指数增长", "2的幂"],
        },
        {
            "question": "为什么最终所有球都会飞出屏幕？",
            "expected_answer": "因为每次碰墙速度翻倍，经过多次碰撞后速度过大无法被屏幕容纳",
            "answer_type": "explanation",
            "key_concepts": ["速度翻倍", "碰撞次数", "飞出"],
        },
    ],
    "gravity_merry_go_round": [
        {
            "question": "在第8秒时，球主要往哪个方向移动？",
            "expected_answer": "向右（第5-10秒重力向右）",
            "answer_type": "direction",
            "key_concepts": ["向右", "5秒", "第二个方向"],
        },
        {
            "question": "如果在第12秒放一个静止的球，它会朝哪个方向落？",
            "expected_answer": "向上（第10-15秒重力向上）",
            "answer_type": "direction",
            "key_concepts": ["向上", "反重力阶段"],
        },
    ],
    "memory_metal_ball": [
        {
            "question": "球在某一帧突然出现在一个新位置，但没有改变速度，可能发生了什么？",
            "expected_answer": "位置重置（被传送回5秒前的位置）",
            "answer_type": "event_description",
            "key_concepts": ["传送", "位置重置", "5秒历史"],
        },
        {
            "question": "传送发生后，球的运动方向会改变吗？",
            "expected_answer": "不会，速度方向保持不变",
            "answer_type": "yes_no_explanation",
            "key_concepts": ["不改变", "速度不变"],
        },
    ],
    "collision_split": [
        {
            "question": "两个球碰撞后，屏幕上的球数量会如何变化？",
            "expected_answer": "增加（从2个变成4个）",
            "answer_type": "quantitative",
            "key_concepts": ["增加", "分裂", "数量翻倍"],
        },
        {
            "question": "为什么随着时间推移，球越来越小？",
            "expected_answer": "因为每次碰撞分裂出的子球半径是母球的一半",
            "answer_type": "explanation",
            "key_concepts": ["分裂", "半径减半", "级联"],
        },
    ],
    "spring_floor": [
        {
            "question": "从更高处落下的球，弹起来会更高还是更低？",
            "expected_answer": "更高（更大冲击力→更大弹力）",
            "answer_type": "comparison",
            "key_concepts": ["更高", "冲击力", "弹力正相关"],
        },
        {
            "question": "地板在球落上去的瞬间会有什么变化？",
            "expected_answer": "向下变形/凹陷",
            "answer_type": "event_description",
            "key_concepts": ["变形", "下沉", "弹性"],
        },
    ],
    "anti_newton_pendulum": [
        {
            "question": "两个球碰撞后，它们会朝同一方向还是相反方向运动？",
            "expected_answer": "朝同一方向运动",
            "answer_type": "direction",
            "key_concepts": ["同一方向", "反常碰撞"],
        },
        {
            "question": "经过多次碰撞，球的分布趋势是什么？",
            "expected_answer": "越来越集中（聚集到一处）",
            "answer_type": "trend",
            "key_concepts": ["聚集", "越来越集中"],
        },
    ],
}


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

PREDICTION_PROMPT_ZERO_SHOT = """你是一个物理世界模型。你将看到一段视频（前{context_sec:.1f}秒），\
请预测接下来{target_sec:.1f}秒内每个物体的运动轨迹。

视频中共有{num_objects}个物体。请以JSON格式输出每帧每个物体的位置和速度，格式如下：
{{
  "predictions": [
    {{
      "frame": <帧号>,
      "timestamp": <秒>,
      "objects": [
        {{"id": <物体ID>, "position_x": <x>, "position_y": <y>, "velocity_x": <vx>, "velocity_y": <vy>}}
      ]
    }}
  ]
}}
"""

PREDICTION_PROMPT_RULE_CONDITIONED = """你是一个物理世界模型。你将看到一段视频（前{context_sec:.1f}秒）。

这个世界的物理规律如下：
{rule_description}

请基于上述规律，预测接下来{target_sec:.1f}秒内每个物体的运动轨迹。
视频中共有{num_objects}个物体。请以JSON格式输出每帧每个物体的位置和速度，格式同上。
"""

INDUCTION_PROMPT = """你是一个科学家，正在研究一个陌生的物理世界。\
你将观看{num_episodes}段来自同一个世界的视频，然后总结这个世界的物理规律。

请注意：
1. 这个世界的物理规律与真实世界不同，可能存在反常识的行为。
2. 请仔细观察多段视频中共同出现的规律性现象。
3. 忽略随机的初始条件差异（如物体的起始位置、颜色等），关注运动规律本身。

请用清晰的中文回答以下问题：
1. 【核心规律】用1-2句话描述这个世界最核心的物理规律。
2. 【详细说明】用3-5句话详细解释这个规律是如何体现的。
3. 【与真实世界的区别】这个规律与真实物理世界的哪条定律不同？
"""

GENERALIZATION_PROMPT = """你刚才总结了一个物理世界的规律：
{induced_rule}

现在请基于这个规律，回答以下关于新场景的问题：
{question}

请直接给出答案，并简要解释理由。
"""
