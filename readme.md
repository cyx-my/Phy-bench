# Phy-Bench: Multi-World Physical Law Benchmark

A benchmark for evaluating **world models' ability to discover and apply physical laws** across multiple simulated worlds with distinct, non-Newtonian physics rules.

## Research Goal

We evaluate two tiers of ability:

| Tier | Question | Task |
|------|----------|------|
| **Prediction** | Can the model *apply* a rule? | Given first N frames (± rule description), predict next M frames |
| **Induction** | Can the model *discover* a rule? | Given K example videos, produce a natural language rule description |

This targets a gap in existing benchmarks: most prior work tests whether models *follow* known physics, but not whether they can *abstract* rules from data.

## The 10 Physics Worlds

Each world has one core physics rule that deviates from Newtonian physics:

| # | World ID | Name | Core Rule |
|---|----------|------|-----------|
| 1 | `anti_gravity_bounce` | 反重力弹跳 | 重力向上；每次弹跳速度翻倍 |
| 2 | `black_hole` | 黑洞吸尘器 | 中心引力 ∝ 1/r²；核心区域变斥力 |
| 3 | `time_dilation` | 时间减速带 | 左半屏时间步长为右半屏的1/4 |
| 4 | `mass_oscillation` | 质量震荡球 | 质量以~2s周期正弦震荡 |
| 5 | `bounce_accelerator` | 反弹加速器 | 每次碰壁速度×2（弹性系数=2） |
| 6 | `gravity_merry_go_round` | 重力旋转木马 | 重力方向每5s顺时针转90° |
| 7 | `memory_metal_ball` | 记忆金属球 | 每5s将物体传送回5s前的位置 |
| 8 | `collision_split` | 碰撞分裂术 | 碰撞时双方各分裂为两个半径/质量减半的子球 |
| 9 | `spring_floor` | 弹簧地板 | 地板可变形；弹起高度∝冲击速度 |
| 10 | `anti_newton_pendulum` | 反牛顿摆 | 碰撞后双方朝同向运动（反动量守恒） |

## Project Structure

```
Phy-bench/
├── benchmark/
│   ├── world_configs.py      # World definitions, rule descriptions, check criteria
│   ├── data_schema.py        # Data format spec and I/O helpers
│   ├── tasks.py              # Task definitions, prompt templates, probing questions
│   ├── evaluator.py          # Prediction + induction scoring protocol
│   └── dataset_builder.py   # Build structured dataset from simulation scripts
├── worlds/                   # Simulation scripts (one per world)
│   ├── black_hole.py         ✓ implemented
│   ├── gravity_merry_go_round.py  ✓ implemented
│   └── ...                   (8 more to implement)
├── data/                     # Generated dataset (gitignored)
│   └── worlds/<world_id>/
│       ├── world_meta.json
│       ├── rule_description.json
│       └── episodes/ep_NNN/
│           ├── video.mp4
│           ├── states.jsonl  # Per-frame physics state
│           └── episode_meta.json
├── research.md               # Research survey and design rationale
├── requirements.txt
└── readme.md
```

## Dataset Format

### Episode data (`states.jsonl`)

Each line is a JSON object for one frame:
```json
{
  "frame": 42,
  "timestamp": 1.4,
  "objects": [
    {"id": 0, "position_x": 320.5, "position_y": 240.1, "velocity_x": -50.2, "velocity_y": 120.0}
  ],
  "world_state": {"gravity_x": 0, "gravity_y": -500},
  "events": [{"type": "collision", "obj_ids": [0, 1]}]
}
```

### Rule descriptions (`rule_description.json`)

Three verbosity levels per world:
```json
{
  "world_id": "black_hole",
  "concise": "...",    // 1 sentence
  "standard": "...",   // 3-4 sentences, used as model input in rule-conditioned variant
  "verbose": "..."     // full paragraph with noise, tests robustness
}
```

## Evaluation

### Tier 1 — Prediction Score

Given context frames (first 1s) → predict target frames (next 2s):

- **trajectory_mse**: Mean position error vs. ground truth
- **physics_consistency**: Rule-specific checks (e.g., does speed double on bounce?)
- **total_score**: 0.5 × (1 − mse_norm) + 0.5 × physics_consistency

Two variants: `zero_shot` (no rule given) vs `rule_conditioned` (rule description given).

### Tier 2 — Induction Score

Given K=3 training episodes → produce natural language rule description:

- **fact_coverage** (40%): Fraction of required facts covered (LLM-judged)
- **bonus_coverage** (15%): Fraction of bonus facts covered
- **confusion_penalty** (−15%): Penalize common wrong conclusions
- **generalization_accuracy** (30%): Accuracy on follow-up counterfactual questions

## Quick Start

```bash
pip install -r requirements.txt

# Generate a small test dataset (2 episodes per world)
python -m benchmark.dataset_builder --episodes-per-world 2 --data-root data_test/

# Generate full dataset (20 episodes per world)
python -m benchmark.dataset_builder --episodes-per-world 20 --data-root data/

# Run a specific world
python worlds/black_hole.py
python worlds/gravity_merry_go_round.py
```

## Implementing a New World Script

Each script in `worlds/` should expose:

```python
def run_episode(output_dir: str, seed: int, fps: int = 30) -> dict:
    """
    Run simulation, write:
      - output_dir/video.mp4
      - output_dir/states.jsonl  (via benchmark.data_schema.StatesWriter)
    Return dict with keys: num_objects, initial_conditions, ...
    """
    ...
```

See `worlds/black_hole.py` for a reference implementation.

## Dependencies

```
pygame==2.5.2
pymunk==6.5.0
opencv-python==4.9.0.80
numpy==1.24.4
```
