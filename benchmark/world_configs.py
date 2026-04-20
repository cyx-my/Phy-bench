"""
World configurations for Phy-Bench.

Each world defines a distinct physics rule that deviates from Newtonian physics.
Worlds are the atomic unit of the benchmark — each has a unique rule, natural language
descriptions at multiple verbosity levels, and structured check criteria for evaluation.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class RuleDescription:
    """Natural language descriptions of a world's physics rule at different verbosity levels."""
    concise: str          # One sentence, clean statement of the rule
    standard: str         # 2-4 sentences, enough context for a model to understand
    verbose: str          # Full paragraph with noisy/redundant phrasing (tests robustness)


@dataclass
class PhysicsCheckCriteria:
    """
    Structured criteria for automatically checking if a trajectory obeys this world's rules.
    Each criterion is a dict with keys:
      - 'name': human-readable criterion name
      - 'description': what to check
      - 'type': 'trajectory' | 'event' | 'qualitative'
      - 'variables': list of state variables involved
      - 'expected_behavior': what the correct behavior looks like
    """
    criteria: List[Dict[str, Any]]
    key_variables: List[str]   # State variables that must be tracked per frame
    violation_signals: List[str]  # Qualitative signals that indicate rule violation


@dataclass
class InductionCheckItems:
    """
    Decomposed check items for scoring a model's induced rule description.
    Each item is a key fact that the description must contain to be considered correct.
    """
    required_facts: List[str]      # Must all be covered for full credit
    bonus_facts: List[str]         # Extra credit for precise quantitative details
    common_confusions: List[str]   # Known wrong conclusions to penalize


@dataclass
class WorldConfig:
    world_id: str
    name: str
    script: str                      # Simulation script filename
    output_video: str                # Default output video filename
    rule_description: RuleDescription
    physics_checks: PhysicsCheckCriteria
    induction_checks: InductionCheckItems
    default_params: Dict[str, Any]   # Key simulation parameters
    episode_duration_sec: float      # Duration of each episode in seconds
    num_objects_range: tuple          # (min, max) number of objects per episode


# ---------------------------------------------------------------------------
# World definitions
# ---------------------------------------------------------------------------

WORLDS: Dict[str, WorldConfig] = {}

def _register(w: WorldConfig):
    WORLDS[w.world_id] = w
    return w


# 1. 反重力弹跳 (Anti-gravity bounce)
_register(WorldConfig(
    world_id="anti_gravity_bounce",
    name="反重力弹跳",
    script="worlds/anti_gravity_bounce.py",
    output_video="anti_gravity_bounce.mp4",
    rule_description=RuleDescription(
        concise="物体向上加速下落，每次弹跳后速度增大而非减小。",
        standard=(
            "在这个世界中，重力方向与正常世界相反——物体向上加速。"
            "物体从地面出发时受向上的恒定加速度，接触【天花板】后弹回。"
            "每次弹跳时弹性系数大于1，因此碰撞后速度增大，弹跳越来越高。"
        ),
        verbose=(
            "你观察到的是一个与日常经验完全相反的世界。这里的重力不是把东西往下拉，"
            "而是往上推。换句话说，如果你把一个球放在桌上然后松手，它会飞向天花板，"
            "而不是掉到地板上。不仅如此，每次碰撞之后球的速度不会减小——恰恰相反，"
            "它会稍微增大一点点，就好像碰撞给了球额外的能量一样。所以你会看到球"
            "越弹越高、越来越快，直到飞出边界。地板和天花板的角色在这里是互换的。"
        ),
    ),
    physics_checks=PhysicsCheckCriteria(
        criteria=[
            {
                "name": "向上加速",
                "description": "物体的y坐标应随时间增大（向上运动），速度y分量随时间增大",
                "type": "trajectory",
                "variables": ["position_y", "velocity_y"],
                "expected_behavior": "velocity_y should increase (upward) at rate ~gravity_strength per second"
            },
            {
                "name": "弹跳后速度增大",
                "description": "每次碰撞天花板后，反弹速度的绝对值大于碰撞前",
                "type": "event",
                "variables": ["velocity_y"],
                "expected_behavior": "|velocity_after_bounce| > |velocity_before_bounce|"
            },
        ],
        key_variables=["position_x", "position_y", "velocity_x", "velocity_y", "timestamp"],
        violation_signals=["object falls down", "bouncing height decreases over time"],
    ),
    induction_checks=InductionCheckItems(
        required_facts=[
            "重力方向向上（或描述物体向上加速）",
            "弹跳后速度增大（而非减小）",
        ],
        bonus_facts=[
            "弹性系数大于1",
            "物体最终会因速度过大飞出边界",
        ],
        common_confusions=[
            "认为是正常重力但弹性异常高",
            "认为有额外向上的推力而非重力反向",
        ],
    ),
    default_params={"gravity": -500, "elasticity": 1.2, "num_balls": 5},
    episode_duration_sec=15.0,
    num_objects_range=(3, 8),
))


# 2. 黑洞吸尘器 (Black hole attractor)
_register(WorldConfig(
    world_id="black_hole",
    name="黑洞吸尘器",
    script="worlds/black_hole.py",
    output_video="black_hole.mp4",
    rule_description=RuleDescription(
        concise="所有物体受到屏幕中心强烈的引力吸引，进入核心区域后被弹飞。",
        standard=(
            "屏幕中心存在一个【黑洞】，对所有物体施加与距离平方成反比的向心引力。"
            "物体越靠近中心，受力越大、加速越快。"
            "当物体进入中心半径50像素的核心区域时，引力转变为斥力，将物体弹飞。"
            "整个过程中没有全局重力，物体仅受黑洞力和边界弹力的影响。"
        ),
        verbose=(
            "这个世界里没有普通意义上的重力——没有什么东西会让你往下掉。"
            "但屏幕的正中心有一个神秘的奇点，像黑洞一样把所有东西往自己身上吸。"
            "距离越近，这个吸引力就越强，大概是按照距离的平方成反比增长的。"
            "有意思的是，一旦物体太靠近这个奇点，进入一个很小的核心区域，"
            "吸引力突然反转，变成排斥力，把物体猛地弹出去。"
            "所以你永远看不到物体真正【消失】进黑洞——它们会在中心附近来回被吸进弹出。"
        ),
    ),
    physics_checks=PhysicsCheckCriteria(
        criteria=[
            {
                "name": "向心加速",
                "description": "物体加速度方向应始终指向屏幕中心（核心区域外）",
                "type": "trajectory",
                "variables": ["position_x", "position_y", "acceleration_x", "acceleration_y"],
                "expected_behavior": "dot(acceleration, direction_to_center) > 0 when outside core"
            },
            {
                "name": "距离平方反比律",
                "description": "加速度大小约与物体到中心距离的平方成反比",
                "type": "trajectory",
                "variables": ["distance_to_center", "acceleration_magnitude"],
                "expected_behavior": "acceleration_magnitude * distance^2 ≈ constant"
            },
            {
                "name": "核心区域斥力",
                "description": "进入核心区域(r<50)后，物体被向外弹飞",
                "type": "event",
                "variables": ["distance_to_center", "velocity_radial"],
                "expected_behavior": "when distance < 50: radial velocity flips outward"
            },
        ],
        key_variables=["position_x", "position_y", "velocity_x", "velocity_y",
                        "distance_to_center", "timestamp"],
        violation_signals=[
            "objects move away from center without external collision",
            "force does not increase near center",
            "objects disappear into center",
        ],
    ),
    induction_checks=InductionCheckItems(
        required_facts=[
            "存在中心吸引力（黑洞/向心力）",
            "越靠近中心力越强",
            "进入核心区域后被弹飞（斥力区域）",
        ],
        bonus_facts=[
            "力与距离平方成反比",
            "没有全局重力",
        ],
        common_confusions=[
            "认为是普通重力但方向变化",
            "没有注意到核心斥力区域",
        ],
    ),
    default_params={"attraction_constant": 50000, "core_radius": 50, "repulsion": 5000},
    episode_duration_sec=15.0,
    num_objects_range=(5, 10),
))


# 3. 时间减速带 (Time dilation zone)
_register(WorldConfig(
    world_id="time_dilation",
    name="时间减速带",
    script="worlds/time_dilation.py",
    output_video="time_dilation.mp4",
    rule_description=RuleDescription(
        concise="屏幕左半侧时间流速是右半侧的1/4，物体跨越分界线后运动速度突变。",
        standard=(
            "屏幕被竖直中线分为两个区域：左侧是【慢时区】，右侧是【快时区】。"
            "在慢时区，每个物理时间步对应的模拟时间只有快时区的1/4，"
            "因此同一物体在慢时区的视觉运动速度只有快时区的1/4。"
            "物体从快时区进入慢时区时，视觉上突然变慢；从慢时区进入快时区时，突然变快。"
        ),
        verbose=(
            "想象屏幕中间有一道无形的门。门的左边时间流速极慢，就像电影里的慢动作一样。"
            "门的右边时间是正常速度，甚至比正常还快一点。"
            "同一个球，在右半边飞得飞快，一旦越过那条线进入左半边，它的运动立刻慢了下来，"
            "看起来就像被什么东西阻住了一样，但实际上它没有受到任何力——只是时间变慢了。"
            "如果球再越过线回到右半边，它立刻又快起来了。这两个区域的时间步长比大约是4:1。"
        ),
    ),
    physics_checks=PhysicsCheckCriteria(
        criteria=[
            {
                "name": "左慢右快",
                "description": "同一物体在左侧的帧间位移应约为右侧的1/4",
                "type": "trajectory",
                "variables": ["position_x", "position_y", "region"],
                "expected_behavior": "displacement_per_frame_left ≈ 0.25 * displacement_per_frame_right"
            },
            {
                "name": "跨越边界速度突变",
                "description": "物体经过x=400时视觉速度发生跳变",
                "type": "event",
                "variables": ["position_x", "velocity_visual"],
                "expected_behavior": "speed ratio ~4x when crossing x=400"
            },
        ],
        key_variables=["position_x", "position_y", "velocity_x", "velocity_y",
                        "region", "effective_timestep", "timestamp"],
        violation_signals=[
            "uniform speed across both regions",
            "speed changes without crossing boundary",
        ],
    ),
    induction_checks=InductionCheckItems(
        required_facts=[
            "屏幕分为两个区域（左/右或某种分隔）",
            "两个区域中物体的运动速度不同",
            "慢区速度约为快区的1/4（或快区更快）",
        ],
        bonus_facts=[
            "速度差异来自时间步长不同（而非力/摩擦）",
            "跨越边界时速度突变而非渐变",
        ],
        common_confusions=[
            "认为是摩擦力或阻力导致速度差",
            "认为是重力方向不同",
        ],
    ),
    default_params={"slow_zone_x_max": 400, "time_ratio": 4.0},
    episode_duration_sec=15.0,
    num_objects_range=(4, 8),
))


# 4. 质量震荡球 (Mass oscillation)
_register(WorldConfig(
    world_id="mass_oscillation",
    name="质量震荡球",
    script="worlds/mass_oscillation.py",
    output_video="mass_oscillation.mp4",
    rule_description=RuleDescription(
        concise="物体质量以固定频率周期性震荡，导致相同力下加速度忽大忽小。",
        standard=(
            "每个物体的质量不是固定值，而是按照正弦波在最小值和最大值之间周期性变化。"
            "质量越小时，同样重力下加速度越大，物体运动越快；质量越大时加速度越小，运动越慢。"
            "因此物体呈现出时快时慢、节律性的运动模式，周期约为2秒。"
        ),
        verbose=(
            "这个世界里的物体有一个奇特的特性：它们的质量不是恒定的，而是不停地变来变去。"
            "每隔大约一秒，物体就会从很轻变到很重，然后再变回很轻，就像心跳一样有节奏。"
            "当物体很轻的时候，重力对它影响很大，它加速很快，跑得很快；"
            "当物体变重的时候，同样的重力对它影响变小，它就慢慢悠悠的。"
            "所以你会看到每个球的运动速度忽快忽慢，有一个明显的节奏感。"
            "注意：这个变化是内在的，不是因为碰撞或者外力，而是质量本身在振荡。"
        ),
    ),
    physics_checks=PhysicsCheckCriteria(
        criteria=[
            {
                "name": "速度周期变化",
                "description": "物体加速度大小应呈现周期性变化",
                "type": "trajectory",
                "variables": ["velocity_magnitude", "timestamp"],
                "expected_behavior": "acceleration oscillates with period ~2s"
            },
            {
                "name": "质量-加速度反比",
                "description": "在质量最小时加速度最大，质量最大时加速度最小",
                "type": "trajectory",
                "variables": ["mass", "acceleration_magnitude"],
                "expected_behavior": "acceleration ∝ 1/mass under constant force"
            },
        ],
        key_variables=["position_x", "position_y", "velocity_x", "velocity_y",
                        "mass", "acceleration_x", "acceleration_y", "timestamp"],
        violation_signals=[
            "constant acceleration throughout",
            "no periodic speed variation",
        ],
    ),
    induction_checks=InductionCheckItems(
        required_facts=[
            "物体质量周期性变化（而非恒定）",
            "质量变化导致加速度/速度周期性变化",
        ],
        bonus_facts=[
            "变化符合正弦/周期规律",
            "周期约为2秒",
        ],
        common_confusions=[
            "认为是外力周期性施加",
            "认为是弹性/摩擦力变化",
        ],
    ),
    default_params={"mass_min": 0.5, "mass_max": 5.0, "oscillation_period": 2.0},
    episode_duration_sec=15.0,
    num_objects_range=(3, 6),
))


# 5. 反弹加速器 (Bounce accelerator)
_register(WorldConfig(
    world_id="bounce_accelerator",
    name="反弹加速器",
    script="worlds/bounce_accelerator.py",
    output_video="bounce_accelerator.mp4",
    rule_description=RuleDescription(
        concise="每次碰撞边界后，物体速度翻倍，最终飞出屏幕。",
        standard=(
            "物体在正常重力下运动，但弹性系数为2：每次与边界发生碰撞时，"
            "法向速度不是减小而是翻倍。"
            "因此每碰一次墙，物体就飞快一倍，经过几次碰撞后速度极大，飞出屏幕边界。"
        ),
        verbose=(
            "通常弹球碰到墙会损失一些能量，弹起来没有原来那么高。"
            "但在这个奇怪的世界里，弹球每碰一次墙，反弹速度就比碰之前翻一倍。"
            "第一次碰墙，速度变成原来的2倍；再碰一次，变成4倍；再碰一次，变成8倍。"
            "这就意味着，球会越来越快，直到快到屏幕都装不下它，飞出去为止。"
            "重力是正常的，只有碰墙这件事是能量加倍的。"
        ),
    ),
    physics_checks=PhysicsCheckCriteria(
        criteria=[
            {
                "name": "碰后速度翻倍",
                "description": "每次与边界碰撞后，法向速度分量绝对值翻倍",
                "type": "event",
                "variables": ["velocity_normal", "collision_event"],
                "expected_behavior": "|velocity_after| = 2 * |velocity_before| at each wall collision"
            },
            {
                "name": "速度指数增长",
                "description": "碰撞次数与速度大小呈指数关系（每碰一次×2）",
                "type": "trajectory",
                "variables": ["velocity_magnitude", "collision_count"],
                "expected_behavior": "speed ≈ initial_speed * 2^collision_count"
            },
        ],
        key_variables=["position_x", "position_y", "velocity_x", "velocity_y",
                        "collision_count", "timestamp"],
        violation_signals=[
            "speed decreases after collision",
            "speed stays constant",
            "speed increases but not by factor 2",
        ],
    ),
    induction_checks=InductionCheckItems(
        required_facts=[
            "碰撞后速度增大（而非减小）",
            "增大方式是翻倍（×2，每次碰撞）",
        ],
        bonus_facts=[
            "速度增长是指数级的",
            "最终物体会飞出边界",
        ],
        common_confusions=[
            "认为有外部施力使物体加速",
            "认为速度增加是线性的",
        ],
    ),
    default_params={"gravity": 500, "elasticity": 2.0},
    episode_duration_sec=10.0,
    num_objects_range=(2, 5),
))


# 6. 重力旋转木马 (Gravity merry-go-round)
_register(WorldConfig(
    world_id="gravity_merry_go_round",
    name="重力旋转木马",
    script="worlds/gravity_merry_go_round.py",
    output_video="gravity_merry_go_round.mp4",
    rule_description=RuleDescription(
        concise="重力方向每5秒旋转90度，依次指向下、右、上、左，循环往复。",
        standard=(
            "全局重力方向不是固定向下，而是每隔5秒顺时针旋转90度。"
            "第0-5秒重力向下，第5-10秒向右，第10-15秒向上，第15-20秒向左，然后循环。"
            "屏幕中心有一个箭头指示当前重力方向，帮助观察者确认规律。"
        ),
        verbose=(
            "在这个世界里，重力不安分——它每隔5秒就转个方向。"
            "开始的时候重力是正常的向下，就像我们平时一样。"
            "5秒过去，重力突然转向右边，所有东西都往右边掉。"
            "再过5秒，重力转向上面，所有东西往上飞。"
            "再过5秒，转到左边，东西往左滑。"
            "然后回到向下，如此循环。"
            "你会在屏幕中间看到一个红色的箭头，它始终指向当前重力的方向，"
            "可以用来帮助你判断现在是哪个重力阶段。"
        ),
    ),
    physics_checks=PhysicsCheckCriteria(
        criteria=[
            {
                "name": "重力方向周期切换",
                "description": "物体的主加速度方向应每5秒切换一次，顺序为下→右→上→左",
                "type": "trajectory",
                "variables": ["velocity_x", "velocity_y", "timestamp"],
                "expected_behavior": "net_acceleration_direction changes by 90° every 5 seconds"
            },
            {
                "name": "四方向覆盖",
                "description": "在30秒视频内，重力至少经历完整的4个方向各一次",
                "type": "trajectory",
                "variables": ["gravity_direction", "timestamp"],
                "expected_behavior": "all 4 directions appear, each lasting 5s"
            },
        ],
        key_variables=["position_x", "position_y", "velocity_x", "velocity_y",
                        "gravity_x", "gravity_y", "timestamp"],
        violation_signals=[
            "gravity stays in one direction throughout",
            "objects don't change drift direction every 5 seconds",
        ],
    ),
    induction_checks=InductionCheckItems(
        required_facts=[
            "重力方向会周期性改变",
            "每次改变旋转90度",
            "切换间隔约为5秒",
        ],
        bonus_facts=[
            "方向顺序为下→右→上→左（顺时针）",
            "循环往复",
        ],
        common_confusions=[
            "认为是随机方向，没有注意到规律",
            "认为间隔时间不固定",
        ],
    ),
    default_params={"gravity_strength": 500, "gravity_change_interval": 5.0},
    episode_duration_sec=30.0,
    num_objects_range=(5, 10),
))


# 7. 记忆金属球 (Memory metal ball)
_register(WorldConfig(
    world_id="memory_metal_ball",
    name="记忆金属球",
    script="worlds/memory_metal_ball.py",
    output_video="memory_metal_ball.mp4",
    rule_description=RuleDescription(
        concise="物体每隔5秒被强制传送回5秒前所在的位置，速度保持不变。",
        standard=(
            "物体具有【位置记忆】：系统记录每个物体过去5秒内的位置历史。"
            "每隔5秒，物体被瞬间传送到5秒前的位置，但速度和方向保持当前值不变。"
            "传送后物体继续以当前速度运动，直到下一次5秒【重置】。"
        ),
        verbose=(
            "这里的球有一种奇怪的记忆能力。每隔5秒，不管球现在在哪儿，"
            "它都会瞬间跳回到5秒之前自己所在的位置，就好像被某种力量拉回去了一样。"
            "但注意：只有位置重置了，速度没有变。"
            "所以球跳回去之后，还是以跳之前的速度和方向继续飞，"
            "只是起始点变了。这就造成了一种很奇特的视觉效果：球在一条轨迹上飞，"
            "突然消失又出现在另一个地方，然后继续按原来的方向飞。"
        ),
    ),
    physics_checks=PhysicsCheckCriteria(
        criteria=[
            {
                "name": "位置周期重置",
                "description": "每隔5秒，物体的位置发生瞬间跳变，跳到5秒前的位置",
                "type": "event",
                "variables": ["position_x", "position_y", "timestamp"],
                "expected_behavior": "position teleports every 5s to position from 5s ago"
            },
            {
                "name": "速度不受重置影响",
                "description": "传送前后速度向量保持不变",
                "type": "event",
                "variables": ["velocity_x", "velocity_y", "teleport_event"],
                "expected_behavior": "velocity unchanged across teleport events"
            },
        ],
        key_variables=["position_x", "position_y", "velocity_x", "velocity_y",
                        "teleport_event", "timestamp"],
        violation_signals=[
            "no sudden position jumps",
            "velocity changes at teleport moment",
        ],
    ),
    induction_checks=InductionCheckItems(
        required_facts=[
            "物体会周期性地瞬间改变位置（传送/跳回）",
            "位置重置到某个历史位置（约5秒前）",
        ],
        bonus_facts=[
            "速度/方向在传送时不改变",
            "传送周期约为5秒",
        ],
        common_confusions=[
            "认为是弹簧力或磁力拉回原点",
            "认为速度也被重置了",
        ],
    ),
    default_params={"memory_duration": 5.0, "reset_interval": 5.0},
    episode_duration_sec=25.0,
    num_objects_range=(3, 6),
))


# 8. 碰撞分裂术 (Collision split)
_register(WorldConfig(
    world_id="collision_split",
    name="碰撞分裂术",
    script="worlds/collision_split.py",
    output_video="collision_split.mp4",
    rule_description=RuleDescription(
        concise="两球碰撞时，每个球分裂为两个半径和质量各为一半的子球。",
        standard=(
            "当两个球发生碰撞时，碰撞的双方各自分裂成两个相同的子球，"
            "子球半径和质量各为母球的一半，方向随机散开。"
            "子球继续遵守正常物理规律运动，并且也能继续分裂（直到半径过小为止）。"
            "因此屏幕上的球的数量在每次碰撞后增加，整体呈增长趋势。"
        ),
        verbose=(
            "这个世界里的球非常脆——一旦两个球碰到一起，它们不会互相弹开，"
            "而是各自裂成两半。每个球碰完之后，原来那个球消失，取而代之的是两个更小的球，"
            "半径和质量都是原来的一半。这两个小球以随机方向飞散出去。"
            "如果这些小球再碰到别的球，它们也会继续裂，变成更小的球。"
            "当然，球太小了之后就不再分裂，免得无限下去。"
            "所以随着时间推移，屏幕上的球会越来越多、越来越小。"
        ),
    ),
    physics_checks=PhysicsCheckCriteria(
        criteria=[
            {
                "name": "碰撞后数量增加",
                "description": "每次碰撞事件后，球的数量增加（2球→4球）",
                "type": "event",
                "variables": ["num_balls", "collision_event"],
                "expected_behavior": "num_balls increases by 2 per collision event"
            },
            {
                "name": "子球半径为母球一半",
                "description": "碰撞后产生的子球半径约为碰撞前的一半",
                "type": "event",
                "variables": ["radius", "collision_event"],
                "expected_behavior": "child_radius ≈ 0.5 * parent_radius"
            },
        ],
        key_variables=["num_balls", "ball_radii", "position_x", "position_y",
                        "collision_event", "timestamp"],
        violation_signals=[
            "number of balls stays constant",
            "balls bounce normally without splitting",
        ],
    ),
    induction_checks=InductionCheckItems(
        required_facts=[
            "碰撞时球会分裂（而非弹开）",
            "分裂产生两个更小的球",
        ],
        bonus_facts=[
            "子球半径/质量为母球一半",
            "分裂可以持续发生（级联）",
            "球数量随时间增长",
        ],
        common_confusions=[
            "认为是正常弹性碰撞",
            "认为分裂后速度翻倍",
        ],
    ),
    default_params={"gravity": 300, "split_min_radius": 5, "initial_radius": 20},
    episode_duration_sec=15.0,
    num_objects_range=(3, 5),
))


# 9. 弹簧地板 (Spring floor)
_register(WorldConfig(
    world_id="spring_floor",
    name="弹簧地板",
    script="worlds/spring_floor.py",
    output_video="spring_floor.mp4",
    rule_description=RuleDescription(
        concise="地板是弹性体，受到物体压力后下沉，并以弹簧力将物体弹起。",
        standard=(
            "屏幕底部是一个弹性地板，当物体落在上面时，地板会根据物体重量和速度向下变形。"
            "变形后的地板产生向上的弹簧恢复力，将物体弹起。"
            "地板的变形程度决定弹起高度：落得越重（速度越大），弹起越高。"
            "地板有一定阻尼，不会无限振荡。"
        ),
        verbose=(
            "这里的地板不是硬邦邦的，而是像蹦床一样软。"
            "当一个球从高处掉下来砸到地板，地板会在接触点被压下去，形成一个凹陷。"
            "凹陷越深，地板的弹力就越大，球就被弹得越高。"
            "从侧面看，你会看到地板形状在球落下时发生变形，然后慢慢恢复平整，"
            "同时把球往上推。如果球很轻或者落得很慢，弹起来就不高；"
            "如果球很重或者落得很快，就能弹得很高。地板本身有一点阻尼，所以不会一直颤抖。"
        ),
    ),
    physics_checks=PhysicsCheckCriteria(
        criteria=[
            {
                "name": "落地时地板变形",
                "description": "物体与地板碰撞时，地板接触点发生可见的向下位移",
                "type": "event",
                "variables": ["floor_y_deformation", "contact_event"],
                "expected_behavior": "floor deforms downward on contact, magnitude ∝ impact force"
            },
            {
                "name": "弹起高度与冲击成正比",
                "description": "物体弹起高度与落地时速度（或下落高度）正相关",
                "type": "trajectory",
                "variables": ["rebound_height", "impact_velocity"],
                "expected_behavior": "rebound_height ∝ impact_velocity (spring behavior)"
            },
        ],
        key_variables=["position_x", "position_y", "velocity_y", "floor_deformation",
                        "contact_force", "timestamp"],
        violation_signals=[
            "floor does not deform on contact",
            "rebound height is constant regardless of impact speed",
        ],
    ),
    induction_checks=InductionCheckItems(
        required_facts=[
            "地板是弹性/可变形的（而非刚性）",
            "地板被压缩后产生向上的弹力",
        ],
        bonus_facts=[
            "弹起高度与冲击力/速度正相关",
            "地板有阻尼（不会永远振荡）",
        ],
        common_confusions=[
            "认为只是高弹性系数的普通地板",
            "忽略地板变形，只描述弹跳行为",
        ],
    ),
    default_params={"spring_constant": 3000, "damping": 0.8, "floor_mass": 10},
    episode_duration_sec=15.0,
    num_objects_range=(2, 5),
))


# 10. 反牛顿摆 (Anti-Newton pendulum)
_register(WorldConfig(
    world_id="anti_newton_pendulum",
    name="反牛顿摆",
    script="worlds/anti_newton_pendulum.py",
    output_video="anti_newton_pendulum.mp4",
    rule_description=RuleDescription(
        concise="两球碰撞后朝同一方向运动，而非向相反方向弹开。",
        standard=(
            "与牛顿第三定律相反：当两个球碰撞时，碰撞后双方都沿碰撞前运动方向更快的那个方向运动。"
            "动量不守恒——碰撞后两球共同加速，朝原来较快球的方向飞去。"
            "这会导致所有球最终都聚集到边界角落，而不是分散运动。"
        ),
        verbose=(
            "通常两个球碰撞后会弹开，一个往左一个往右。"
            "但在这个奇异的世界里，两个球碰撞之后不会弹开——"
            "它们会立刻朝同一个方向飞去，就像一个球【感染】了另一个球的运动意图一样。"
            "具体来说，碰后两球都会以较大的速度朝原先速度较大的那个球的方向运动。"
            "这意味着动量和能量都不守恒。"
            "随着时间推移，你会发现球越来越集中，最终都挤到屏幕的某个角落里。"
        ),
    ),
    physics_checks=PhysicsCheckCriteria(
        criteria=[
            {
                "name": "碰后同向运动",
                "description": "两球碰撞后，双方的速度方向变为相同（而非相反）",
                "type": "event",
                "variables": ["velocity_x", "velocity_y", "collision_event", "ball_id"],
                "expected_behavior": "after collision: sign(v1_post) == sign(v2_post) for both axes"
            },
            {
                "name": "球逐渐聚集",
                "description": "随着碰撞积累，球的位置趋向聚集而非分散",
                "type": "trajectory",
                "variables": ["position_x", "position_y"],
                "expected_behavior": "inter-ball distances decrease over time"
            },
        ],
        key_variables=["position_x", "position_y", "velocity_x", "velocity_y",
                        "collision_event", "timestamp"],
        violation_signals=[
            "balls move apart after collision",
            "velocity directions opposite after collision",
        ],
    ),
    induction_checks=InductionCheckItems(
        required_facts=[
            "碰撞后两球朝同一方向运动（而非相反）",
            "违反正常碰撞/动量守恒规律",
        ],
        bonus_facts=[
            "碰后速度由碰前较快者决定",
            "最终球聚集到一处",
        ],
        common_confusions=[
            "认为是磁力吸引",
            "认为是高摩擦力效果",
        ],
    ),
    default_params={"gravity": 300, "same_direction_factor": 1.5},
    episode_duration_sec=15.0,
    num_objects_range=(4, 8),
))


def get_world(world_id: str) -> WorldConfig:
    if world_id not in WORLDS:
        raise ValueError(f"Unknown world_id: {world_id!r}. Available: {list(WORLDS.keys())}")
    return WORLDS[world_id]


def list_worlds() -> List[str]:
    return list(WORLDS.keys())
