"""
Evaluation protocol for Phy-Bench.

Two-tier evaluation:

Tier 1 — Prediction Score (PhysScore):
  Measures whether the predicted trajectory obeys this world's physics rules.
  Sub-metrics:
    - trajectory_mse: Mean squared error between predicted and ground-truth positions
    - physics_consistency: Rule-specific checks (see world_configs.PhysicsCheckCriteria)
    - visual_quality: FID/SSIM (only if the model outputs video frames)

Tier 2 — Induction Score (InductScore):
  Measures whether the model's natural language rule description covers the key facts.
  Sub-metrics:
    - fact_coverage: Fraction of required_facts mentioned (LLM-judged)
    - bonus_coverage: Fraction of bonus_facts mentioned
    - confusion_penalty: Penalty for each common_confusion mentioned
    - generalization_accuracy: Accuracy on follow-up counterfactual questions
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import json
import math


# ---------------------------------------------------------------------------
# Prediction evaluation
# ---------------------------------------------------------------------------

@dataclass
class PredictionResult:
    world_id: str
    episode_id: str
    variant: str                        # "zero_shot" or "rule_conditioned"
    trajectory_mse: float               # Mean position MSE over target frames
    physics_checks: Dict[str, float]    # {check_name: pass_rate (0-1)}
    physics_consistency: float          # Mean of physics_checks values
    total_score: float                  # Weighted combination


def compute_trajectory_mse(
    predicted: List[Dict[str, Any]],  # [{frame, objects: [{id, position_x, position_y}]}]
    ground_truth: List[Dict[str, Any]],
) -> float:
    """Compute mean squared position error across all frames and objects."""
    total_sq_error = 0.0
    count = 0
    gt_by_frame = {s["frame"]: s for s in ground_truth}
    for pred_frame in predicted:
        frame_idx = pred_frame["frame"]
        if frame_idx not in gt_by_frame:
            continue
        gt_frame = gt_by_frame[frame_idx]
        gt_objs = {o["id"]: o for o in gt_frame["objects"]}
        for pred_obj in pred_frame["objects"]:
            obj_id = pred_obj["id"]
            if obj_id not in gt_objs:
                continue
            dx = pred_obj["position_x"] - gt_objs[obj_id]["position_x"]
            dy = pred_obj["position_y"] - gt_objs[obj_id]["position_y"]
            total_sq_error += dx * dx + dy * dy
            count += 1
    return total_sq_error / count if count > 0 else float("inf")


def check_upward_acceleration(states: List[Dict]) -> float:
    """Physics check: objects should accelerate upward (anti_gravity_bounce)."""
    pass_count = 0
    total = 0
    for i in range(1, len(states)):
        for obj in states[i]["objects"]:
            obj_id = obj["id"]
            prev_objs = {o["id"]: o for o in states[i-1]["objects"]}
            if obj_id not in prev_objs:
                continue
            dvy = obj["velocity_y"] - prev_objs[obj_id]["velocity_y"]
            # In anti-gravity world, vy should increase (upward acceleration, y increases = upward in screen coords)
            # Depends on coordinate system convention used in simulation
            pass_count += 1 if dvy < 0 else 0  # negative y = upward in typical screen coords
            total += 1
    return pass_count / total if total > 0 else 0.0


def check_centripetal_force(states: List[Dict], center: Tuple[float, float] = (400, 300)) -> float:
    """Physics check: acceleration should point toward center (black_hole, outside core)."""
    pass_count = 0
    total = 0
    core_radius = 50
    dt = states[1]["timestamp"] - states[0]["timestamp"] if len(states) > 1 else 1/30
    for i in range(1, len(states) - 1):
        for obj in states[i]["objects"]:
            obj_id = obj["id"]
            prev_objs = {o["id"]: o for o in states[i-1]["objects"]}
            next_objs = {o["id"]: o for o in states[i+1]["objects"]}
            if obj_id not in prev_objs or obj_id not in next_objs:
                continue
            px, py = obj["position_x"], obj["position_y"]
            dx_center = center[0] - px
            dy_center = center[1] - py
            dist = math.sqrt(dx_center**2 + dy_center**2)
            if dist < core_radius:
                continue  # Skip core region
            # Approximate acceleration from finite differences
            ax = (next_objs[obj_id]["velocity_x"] - prev_objs[obj_id]["velocity_x"]) / (2 * dt)
            ay = (next_objs[obj_id]["velocity_y"] - prev_objs[obj_id]["velocity_y"]) / (2 * dt)
            # Check dot product with direction-to-center
            dot = ax * dx_center + ay * dy_center
            pass_count += 1 if dot > 0 else 0
            total += 1
    return pass_count / total if total > 0 else 0.0


def check_speed_doubles_on_bounce(states: List[Dict]) -> float:
    """Physics check: speed doubles at each wall collision (bounce_accelerator)."""
    pass_count = 0
    total = 0
    for state in states:
        for event in state.get("events", []):
            if event.get("type") != "collision":
                continue
            # Requires pre/post velocity in event data
            if "velocity_before" not in event or "velocity_after" not in event:
                continue
            v_before = math.sqrt(event["velocity_before"]["x"]**2 + event["velocity_before"]["y"]**2)
            v_after = math.sqrt(event["velocity_after"]["x"]**2 + event["velocity_after"]["y"]**2)
            ratio = v_after / v_before if v_before > 0 else 0
            pass_count += 1 if abs(ratio - 2.0) < 0.3 else 0
            total += 1
    return pass_count / total if total > 0 else 0.0


def check_gravity_rotation(states: List[Dict]) -> float:
    """
    Physics check for gravity_merry_go_round: gravity direction rotates every 5 seconds.
    Simplified check: gravity vector should change direction at least once.
    """
    if not states:
        return 0.0

    # Collect unique gravity directions
    directions = []
    for state in states:
        world_state = state.get("world_state", {})
        gx = world_state.get("gravity_x", 0)
        gy = world_state.get("gravity_y", 0)
        # Normalize to direction tuple (round to reduce noise)
        dir_tuple = (round(gx, 3), round(gy, 3))
        if dir_tuple not in directions:
            directions.append(dir_tuple)

    # If gravity changes direction at least once, pass
    # (expect at least 2 distinct directions)
    if len(directions) >= 2:
        return 1.0
    else:
        return 0.0


# Map world_id -> list of (check_name, check_function)
PHYSICS_CHECK_FUNCTIONS: Dict[str, List[Tuple[str, Any]]] = {
    "anti_gravity_bounce": [
        ("upward_acceleration", check_upward_acceleration),
    ],
    "black_hole": [
        ("centripetal_force", check_centripetal_force),
    ],
    "bounce_accelerator": [
        ("speed_doubles_on_bounce", check_speed_doubles_on_bounce),
    ],
    "gravity_merry_go_round": [
        ("gravity_rotation", check_gravity_rotation),
    ],
    # Additional check functions for other worlds can be added here
}


def evaluate_prediction(
    world_id: str,
    predicted_states: List[Dict],
    ground_truth_states: List[Dict],
    variant: str,
    episode_id: str,
) -> PredictionResult:
    """Run all checks and return a PredictionResult."""
    # Trajectory MSE
    mse = compute_trajectory_mse(predicted_states, ground_truth_states)

    # Physics consistency checks
    checks = {}
    for check_name, check_fn in PHYSICS_CHECK_FUNCTIONS.get(world_id, []):
        checks[check_name] = check_fn(predicted_states)

    physics_consistency = sum(checks.values()) / len(checks) if checks else 0.0

    # Normalize MSE to a 0-1 score (lower is better; cap at some max MSE)
    max_mse = 10000.0
    mse_score = max(0.0, 1.0 - mse / max_mse)

    total_score = 0.5 * mse_score + 0.5 * physics_consistency

    return PredictionResult(
        world_id=world_id,
        episode_id=episode_id,
        variant=variant,
        trajectory_mse=mse,
        physics_checks=checks,
        physics_consistency=physics_consistency,
        total_score=total_score,
    )


# ---------------------------------------------------------------------------
# Induction evaluation
# ---------------------------------------------------------------------------

@dataclass
class InductionResult:
    world_id: str
    variant: str = "zero_shot"
    fact_coverage: float = 0.0          # [0, 1]: fraction of required_facts covered
    bonus_coverage: float = 0.0         # [0, 1]: fraction of bonus_facts covered
    confusion_penalty: float = 0.0      # [0, 1]: fraction of confusions present (subtract)
    generalization_accuracy: float = 0.0  # [0, 1]: accuracy on counterfactual questions
    total_score: float = 0.0
    llm_judgment: Optional[Dict] = None  # Raw LLM response for traceability


def score_induction_with_llm_judgment(
    world_id: str,
    induced_rule: str,
    llm_judgment: Dict[str, Any],
) -> InductionResult:
    """
    Score an induced rule description given a structured LLM judgment.

    Expected llm_judgment format (from judge LLM):
    {
      "required_facts_covered": [true/false, ...],   # one per required_fact
      "bonus_facts_covered": [true/false, ...],       # one per bonus_fact
      "confusions_present": [true/false, ...],        # one per common_confusion
      "generalization_answers": [
        {"question_id": 0, "correct": true/false},
        ...
      ]
    }
    """
    from benchmark.world_configs import get_world
    cfg = get_world(world_id)
    checks = cfg.induction_checks

    req = llm_judgment.get("required_facts_covered", [])
    fact_coverage = sum(req) / len(req) if req else 0.0

    bon = llm_judgment.get("bonus_facts_covered", [])
    bonus_coverage = sum(bon) / len(bon) if bon else 0.0

    conf = llm_judgment.get("confusions_present", [])
    confusion_penalty = sum(conf) / len(conf) if conf else 0.0

    gen = llm_judgment.get("generalization_answers", [])
    generalization_accuracy = sum(1 for a in gen if a.get("correct")) / len(gen) if gen else 0.0

    # Weighted total: fact_coverage is most important
    total_score = (
        0.40 * fact_coverage
        + 0.15 * bonus_coverage
        - 0.15 * confusion_penalty
        + 0.30 * generalization_accuracy
    )
    total_score = max(0.0, min(1.0, total_score))

    return InductionResult(
        world_id=world_id,
        fact_coverage=fact_coverage,
        bonus_coverage=bonus_coverage,
        confusion_penalty=confusion_penalty,
        generalization_accuracy=generalization_accuracy,
        total_score=total_score,
        llm_judgment=llm_judgment,
    )


# ---------------------------------------------------------------------------
# Judge prompt builder (for calling an LLM as judge)
# ---------------------------------------------------------------------------

def build_judge_prompt(world_id: str, induced_rule: str) -> str:
    """Build the prompt for the LLM judge to score an induced rule."""
    from benchmark.world_configs import get_world
    cfg = get_world(world_id)
    checks = cfg.induction_checks

    required_facts_json = json.dumps(checks.required_facts, ensure_ascii=False, indent=2)
    bonus_facts_json = json.dumps(checks.bonus_facts, ensure_ascii=False, indent=2)
    confusions_json = json.dumps(checks.common_confusions, ensure_ascii=False, indent=2)

    return f"""你是一个物理规律评估专家。请判断以下模型生成的"物理世界规律描述"是否正确覆盖了关键内容。

## 模型生成的规律描述
{induced_rule}

## 评估标准

### 必须覆盖的事实（required_facts）
{required_facts_json}

### 加分项（bonus_facts）
{bonus_facts_json}

### 常见错误认知（common_confusions，出现应扣分）
{confusions_json}

## 输出要求
请以JSON格式输出：
{{
  "required_facts_covered": [<true/false for each required fact, in order>],
  "bonus_facts_covered": [<true/false for each bonus fact, in order>],
  "confusions_present": [<true/false for each confusion, in order>],
  "brief_reasoning": "<1-2句话总结判断理由>"
}}
"""


# ---------------------------------------------------------------------------
# Aggregate benchmark results
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkSummary:
    """Summary of results across all worlds and variants."""
    prediction_scores: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # {world_id: {"zero_shot": score, "rule_conditioned": score}}

    induction_scores: Dict[str, float] = field(default_factory=dict)
    # {world_id: induction_total_score}

    overall_prediction_zero_shot: float = 0.0
    overall_prediction_rule_conditioned: float = 0.0
    overall_induction: float = 0.0

    def to_dict(self) -> dict:
        return {
            "prediction_scores": self.prediction_scores,
            "induction_scores": self.induction_scores,
            "overall_prediction_zero_shot": self.overall_prediction_zero_shot,
            "overall_prediction_rule_conditioned": self.overall_prediction_rule_conditioned,
            "overall_induction": self.overall_induction,
        }

    def print_table(self):
        """Print a formatted summary table."""
        print(f"\n{'='*70}")
        print(f"{'World':<30} {'Pred(0-shot)':>12} {'Pred(rule)':>11} {'Induction':>10}")
        print(f"{'-'*70}")
        for world_id in sorted(self.prediction_scores.keys()):
            scores = self.prediction_scores.get(world_id, {})
            zs = scores.get("zero_shot", float("nan"))
            rc = scores.get("rule_conditioned", float("nan"))
            ind = self.induction_scores.get(world_id, float("nan"))
            print(f"{world_id:<30} {zs:>12.3f} {rc:>11.3f} {ind:>10.3f}")
        print(f"{'-'*70}")
        print(f"{'Overall':<30} {self.overall_prediction_zero_shot:>12.3f} "
              f"{self.overall_prediction_rule_conditioned:>11.3f} "
              f"{self.overall_induction:>10.3f}")
        print(f"{'='*70}\n")
