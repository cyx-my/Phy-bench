"""
MCQ question bank for all Phy-Bench worlds.

Each world has 6-8 multiple-choice questions testing core facts,
bonus details, generalization ability, and resistance to confusions.
"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class MCQ:
    question: str                   # Question text (Chinese)
    options: Dict[str, str]         # {"A": "...", "B": "...", "C": "...", "D": "..."}
    correct_answer: str             # "A", "B", "C", or "D"
    source: str                     # Category: required_fact / bonus_fact / generalization / confusion / cross_world
    world_id: str                   # Which world this belongs to


# ---------------------------------------------------------------------------
# 1. anti_gravity_bounce
# ---------------------------------------------------------------------------

_anti_gravity_bounce: List[MCQ] = [
    MCQ(
        question="在这个世界中，物体的运动主要受什么影响？",
        options={
            "A": "重力方向向上，物体向上加速，碰到天花板后弹回",
            "B": "重力方向向下，所有物体正常下落",
            "C": "没有重力，物体做匀速直线运动",
            "D": "重力方向每5秒旋转90度",
        },
        correct_answer="A",
        source="required_fact",
        world_id="anti_gravity_bounce",
    ),
    MCQ(
        question="物体每次弹跳后的速度如何变化？",
        options={
            "A": "速度减小（能量损失）",
            "B": "速度增大（碰撞后速度更快）",
            "C": "速度保持不变",
            "D": "速度变为原来的两倍然后保持不变",
        },
        correct_answer="B",
        source="required_fact",
        world_id="anti_gravity_bounce",
    ),
    MCQ(
        question="这个世界的弹性系数有什么特点？",
        options={
            "A": "弹性系数等于1，完全弹性碰撞",
            "B": "弹性系数小于1，非弹性碰撞",
            "C": "弹性系数大于1，碰撞后能量增加",
            "D": "没有弹性系数，所有碰撞都完全吸收能量",
        },
        correct_answer="C",
        source="bonus_fact",
        world_id="anti_gravity_bounce",
    ),
    MCQ(
        question="长期来看，这些物体会怎样？",
        options={
            "A": "最终停下来静止在某处",
            "B": "最终因速度过大飞出屏幕边界",
            "C": "所有物体聚集到屏幕中心",
            "D": "物体质量周期性变化导致运动忽快忽慢",
        },
        correct_answer="B",
        source="bonus_fact",
        world_id="anti_gravity_bounce",
    ),
    MCQ(
        question="下面哪种说法是错误的？",
        options={
            "A": "物体向上加速是因为重力方向反向",
            "B": "每次弹跳后速度变大",
            "C": "这个世界的规律可以用正常重力加高弹性系数来解释",
            "D": "物体最终会飞出屏幕",
        },
        correct_answer="C",
        source="confusion",
        world_id="anti_gravity_bounce",
    ),
    MCQ(
        question="如果把一个球放在地板上松手，它会往哪个方向运动？",
        options={
            "A": "向上运动（向天花板方向）",
            "B": "向下运动（落向地板）",
            "C": "静止不动",
            "D": "水平方向运动",
        },
        correct_answer="A",
        source="generalization",
        world_id="anti_gravity_bounce",
    ),
    MCQ(
        question="与正常世界相比，这个世界的球在碰撞后会更快还是更慢？",
        options={
            "A": "更快（每次碰撞速度增大）",
            "B": "更慢（每次碰撞速度减小）",
            "C": "一样快（速度不变）",
            "D": "更快但只是第一次碰撞后，之后不变",
        },
        correct_answer="A",
        source="generalization",
        world_id="anti_gravity_bounce",
    ),
    MCQ(
        question="这个世界的运动规律与以下哪个最相似？",
        options={
            "A": "物体受到指向中心的吸引力，越近越快",
            "B": "物体在弹性蹦床上弹跳，越跳越高",
            "C": "重力方向反向且碰撞加速",
            "D": "碰撞后两球朝同一方向运动",
        },
        correct_answer="C",
        source="cross_world",
        world_id="anti_gravity_bounce",
    ),
]

# ---------------------------------------------------------------------------
# 2. black_hole
# ---------------------------------------------------------------------------

_black_hole: List[MCQ] = [
    MCQ(
        question="在这个世界中，物体的运动主要受什么影响？",
        options={
            "A": "屏幕中心有一个引力源，吸引所有物体向中心运动",
            "B": "全局重力方向向下，所有物体自然下落",
            "C": "物体之间相互吸引，与距离无关",
            "D": "屏幕四周的墙壁产生向心力",
        },
        correct_answer="A",
        source="required_fact",
        world_id="black_hole",
    ),
    MCQ(
        question="物体受到的引力大小与距离有什么关系？",
        options={
            "A": "距离越近引力越大，与距离平方成反比",
            "B": "引力大小恒定，与距离无关",
            "C": "距离越近引力越小",
            "D": "引力只与物体质量有关，与距离无关",
        },
        correct_answer="A",
        source="required_fact",
        world_id="black_hole",
    ),
    MCQ(
        question="当一个物体非常靠近屏幕中心时会发生什么？",
        options={
            "A": "物体被吸入中心并消失",
            "B": "物体被弹飞（受斥力作用）",
            "C": "物体停留在中心不动",
            "D": "物体速度减慢并绕中心旋转",
        },
        correct_answer="B",
        source="required_fact",
        world_id="black_hole",
    ),
    MCQ(
        question="除了中心引力和边界弹力外，物体还受什么力？",
        options={
            "A": "全局向下的重力",
            "B": "空气阻力",
            "C": "没有其他力",
            "D": "物体间的磁力",
        },
        correct_answer="C",
        source="bonus_fact",
        world_id="black_hole",
    ),
    MCQ(
        question="这个世界的规律与以下哪种误解最不同？",
        options={
            "A": "认为是普通重力但方向在变化",
            "B": "认为存在中心吸引力且核心区域有斥力",
            "C": "认为是摩擦力导致物体减速",
            "D": "认为是弹力使物体弹跳",
        },
        correct_answer="A",
        source="confusion",
        world_id="black_hole",
    ),
    MCQ(
        question="一个距离中心很远的静止球会怎样运动？",
        options={
            "A": "保持静止不动",
            "B": "被吸引向中心加速运动",
            "C": "沿直线匀速远离中心",
            "D": "绕中心做圆周运动",
        },
        correct_answer="B",
        source="generalization",
        world_id="black_hole",
    ),
    MCQ(
        question="一个球非常靠近屏幕中心（核心区域）时会发生什么？",
        options={
            "A": "被永久困在中心",
            "B": "被弹飞出去",
            "C": "分裂成两个更小的球",
            "D": "瞬间传送到5秒前的位置",
        },
        correct_answer="B",
        source="generalization",
        world_id="black_hole",
    ),
]

# ---------------------------------------------------------------------------
# 3. time_dilation
# ---------------------------------------------------------------------------

_time_dilation: List[MCQ] = [
    MCQ(
        question="屏幕左右两侧的运动有什么不同？",
        options={
            "A": "左右两侧运动速度不同，左侧慢、右侧快",
            "B": "左右两侧重力方向不同",
            "C": "左右两侧颜色不同",
            "D": "左右两侧物体数量不同",
        },
        correct_answer="A",
        source="required_fact",
        world_id="time_dilation",
    ),
    MCQ(
        question="慢区和快区的速度比例大约是多少？",
        options={
            "A": "慢区速度约为快区的1/2",
            "B": "慢区速度约为快区的1/4",
            "C": "慢区速度约为快区的1/10",
            "D": "慢区和快区速度相同",
        },
        correct_answer="B",
        source="required_fact",
        world_id="time_dilation",
    ),
    MCQ(
        question="速度差异的根本原因是什么？",
        options={
            "A": "左侧摩擦力更大",
            "B": "左右两侧的时间步长不同",
            "C": "左侧重力方向与右侧不同",
            "D": "左侧空气更稠密",
        },
        correct_answer="B",
        source="bonus_fact",
        world_id="time_dilation",
    ),
    MCQ(
        question="物体跨越左右分界线时速度如何变化？",
        options={
            "A": "速度渐变，平滑过渡",
            "B": "速度突变，瞬间变化约4倍",
            "C": "速度不变",
            "D": "速度先变慢再变快",
        },
        correct_answer="B",
        source="bonus_fact",
        world_id="time_dilation",
    ),
    MCQ(
        question="下面哪种说法是对这个世界的错误理解？",
        options={
            "A": "认为是摩擦力或阻力导致速度差",
            "B": "认为两个区域时间流速不同",
            "C": "认为跨越边界时速度突变",
            "D": "认为左侧慢、右侧快",
        },
        correct_answer="A",
        source="confusion",
        world_id="time_dilation",
    ),
    MCQ(
        question="一个以相同速度进入左侧区域和右侧区域的球，哪边看起来移动更快？",
        options={
            "A": "左侧更快",
            "B": "右侧更快",
            "C": "一样快",
            "D": "取决于球的质量",
        },
        correct_answer="B",
        source="generalization",
        world_id="time_dilation",
    ),
    MCQ(
        question="一个球从右侧快速越过中线进入左侧，它的视觉速度会如何变化？",
        options={
            "A": "突然变快",
            "B": "突然变慢",
            "C": "保持不变",
            "D": "先变快后变慢",
        },
        correct_answer="B",
        source="generalization",
        world_id="time_dilation",
    ),
]

# ---------------------------------------------------------------------------
# 4. mass_oscillation
# ---------------------------------------------------------------------------

_mass_oscillation: List[MCQ] = [
    MCQ(
        question="这个世界中物体的哪个属性在周期性变化？",
        options={
            "A": "物体的颜色",
            "B": "物体的质量",
            "C": "物体的形状",
            "D": "物体的数量",
        },
        correct_answer="B",
        source="required_fact",
        world_id="mass_oscillation",
    ),
    MCQ(
        question="质量变化对运动有什么影响？",
        options={
            "A": "质量变化导致加速度/速度周期性变化",
            "B": "质量变化不影响运动",
            "C": "质量变化只影响颜色",
            "D": "质量变化只影响碰撞",
        },
        correct_answer="A",
        source="required_fact",
        world_id="mass_oscillation",
    ),
    MCQ(
        question="质量变化的周期大约是多少？",
        options={
            "A": "约0.5秒",
            "B": "约2秒",
            "C": "约5秒",
            "D": "约10秒",
        },
        correct_answer="B",
        source="bonus_fact",
        world_id="mass_oscillation",
    ),
    MCQ(
        question="物体什么时候运动最快？",
        options={
            "A": "质量最大的时候",
            "B": "质量最小的时候",
            "C": "质量变化最快的时候",
            "D": "速度与质量无关，一直匀速",
        },
        correct_answer="B",
        source="bonus_fact",
        world_id="mass_oscillation",
    ),
    MCQ(
        question="下面哪种是对这个世界的错误解释？",
        options={
            "A": "认为是外力在周期性施加",
            "B": "认为物体的质量在周期性变化",
            "C": "认为质量变化导致加速度变化",
            "D": "认为运动速度会周期性变化",
        },
        correct_answer="A",
        source="confusion",
        world_id="mass_oscillation",
    ),
    MCQ(
        question="为什么这个世界的球看起来时快时慢，而不是匀速运动？",
        options={
            "A": "因为球受到间歇性的推力",
            "B": "因为球的质量在周期性变化，导致加速度随之变化",
            "C": "因为球在摩擦力和光滑表面之间切换",
            "D": "因为重力方向在周期性变化",
        },
        correct_answer="B",
        source="generalization",
        world_id="mass_oscillation",
    ),
    MCQ(
        question="这个世界的规律与以下哪个最不同？",
        options={
            "A": "物体的质量会变化",
            "B": "物体受中心引力且越近越快",
            "C": "运动速度会周期性变化",
            "D": "质量小时运动更快",
        },
        correct_answer="B",
        source="cross_world",
        world_id="mass_oscillation",
    ),
]

# ---------------------------------------------------------------------------
# 5. bounce_accelerator
# ---------------------------------------------------------------------------

_bounce_accelerator: List[MCQ] = [
    MCQ(
        question="物体与边界碰撞后速度如何变化？",
        options={
            "A": "速度减小（能量损失）",
            "B": "速度翻倍",
            "C": "速度不变",
            "D": "速度变为原来的1.5倍",
        },
        correct_answer="B",
        source="required_fact",
        world_id="bounce_accelerator",
    ),
    MCQ(
        question="与正常世界相比，碰撞有什么不同？",
        options={
            "A": "碰撞后速度增大而非减小",
            "B": "碰撞后速度减小得更快",
            "C": "碰撞后速度不变",
            "D": "碰撞后物体分裂",
        },
        correct_answer="A",
        source="required_fact",
        world_id="bounce_accelerator",
    ),
    MCQ(
        question="经过多次碰撞后，速度的增长模式是什么？",
        options={
            "A": "线性增长（每次加固定值）",
            "B": "指数增长（每次翻倍）",
            "C": "对数增长（增速越来越慢）",
            "D": "没有规律",
        },
        correct_answer="B",
        source="bonus_fact",
        world_id="bounce_accelerator",
    ),
    MCQ(
        question="长期来看物体会怎样？",
        options={
            "A": "最终停下来",
            "B": "最终因速度过大飞出屏幕边界",
            "C": "聚集到屏幕中心",
            "D": "分裂成更多小球",
        },
        correct_answer="B",
        source="bonus_fact",
        world_id="bounce_accelerator",
    ),
    MCQ(
        question="下面哪种是对这个世界的错误解释？",
        options={
            "A": "认为有外部施力使物体加速",
            "B": "认为每次碰撞后速度翻倍",
            "C": "认为速度增长是指数级的",
            "D": "认为物体会飞出边界",
        },
        correct_answer="A",
        source="confusion",
        world_id="bounce_accelerator",
    ),
    MCQ(
        question="一个球碰了3次墙后，它的速度大约是初始速度的几倍？",
        options={
            "A": "3倍",
            "B": "6倍",
            "C": "8倍",
            "D": "9倍",
        },
        correct_answer="C",
        source="generalization",
        world_id="bounce_accelerator",
    ),
    MCQ(
        question="为什么最终所有球都会飞出屏幕？",
        options={
            "A": "因为重力越来越大",
            "B": "因为每次碰墙速度翻倍，多次碰撞后速度过大",
            "C": "因为球在不断膨胀",
            "D": "因为屏幕在缩小",
        },
        correct_answer="B",
        source="generalization",
        world_id="bounce_accelerator",
    ),
]

# ---------------------------------------------------------------------------
# 6. gravity_merry_go_round
# ---------------------------------------------------------------------------

_gravity_merry_go_round: List[MCQ] = [
    MCQ(
        question="这个世界的重力方向有什么特点？",
        options={
            "A": "重力方向始终向下",
            "B": "重力方向周期性改变",
            "C": "没有重力",
            "D": "重力方向随机变化",
        },
        correct_answer="B",
        source="required_fact",
        world_id="gravity_merry_go_round",
    ),
    MCQ(
        question="重力方向每次改变多少度？",
        options={
            "A": "45度",
            "B": "90度",
            "C": "180度",
            "D": "360度",
        },
        correct_answer="B",
        source="required_fact",
        world_id="gravity_merry_go_round",
    ),
    MCQ(
        question="重力方向大约每隔多久改变一次？",
        options={
            "A": "1秒",
            "B": "5秒",
            "C": "10秒",
            "D": "15秒",
        },
        correct_answer="B",
        source="required_fact",
        world_id="gravity_merry_go_round",
    ),
    MCQ(
        question="重力方向的顺序是什么？",
        options={
            "A": "上→下→左→右",
            "B": "下→右→上→左（顺时针）",
            "C": "左→右→上→下",
            "D": "随机顺序",
        },
        correct_answer="B",
        source="bonus_fact",
        world_id="gravity_merry_go_round",
    ),
    MCQ(
        question="下面哪种是对这个世界的错误理解？",
        options={
            "A": "认为重力方向是随机变化的",
            "B": "认为重力方向会周期性改变",
            "C": "认为每次改变旋转90度",
            "D": "认为切换间隔约为5秒",
        },
        correct_answer="A",
        source="confusion",
        world_id="gravity_merry_go_round",
    ),
    MCQ(
        question="在第8秒时，球主要往哪个方向移动？",
        options={
            "A": "向下",
            "B": "向右",
            "C": "向上",
            "D": "向左",
        },
        correct_answer="B",
        source="generalization",
        world_id="gravity_merry_go_round",
    ),
    MCQ(
        question="如果在第12秒放一个静止的球，它会朝哪个方向落？",
        options={
            "A": "向下",
            "B": "向右",
            "C": "向上",
            "D": "向左",
        },
        correct_answer="C",
        source="generalization",
        world_id="gravity_merry_go_round",
    ),
]

# ---------------------------------------------------------------------------
# 7. memory_metal_ball
# ---------------------------------------------------------------------------

_memory_metal_ball: List[MCQ] = [
    MCQ(
        question="这个世界的物体有什么特殊能力？",
        options={
            "A": "物体能变色",
            "B": "物体能瞬间改变位置（传送/跳回）",
            "C": "物体能分裂成多个",
            "D": "物体能隐身",
        },
        correct_answer="B",
        source="required_fact",
        world_id="memory_metal_ball",
    ),
    MCQ(
        question="物体被传送到什么位置？",
        options={
            "A": "传送到屏幕中心",
            "B": "传送到5秒前所在的位置",
            "C": "传送到初始位置",
            "D": "传送到随机位置",
        },
        correct_answer="B",
        source="required_fact",
        world_id="memory_metal_ball",
    ),
    MCQ(
        question="传送发生后，物体的速度会改变吗？",
        options={
            "A": "速度不变，方向和大小都保持传送前的值",
            "B": "速度变为零",
            "C": "速度大小不变但方向反转",
            "D": "速度重置为初始速度",
        },
        correct_answer="A",
        source="bonus_fact",
        world_id="memory_metal_ball",
    ),
    MCQ(
        question="传送大约每隔多久发生一次？",
        options={
            "A": "1秒",
            "B": "5秒",
            "C": "10秒",
            "D": "随机间隔",
        },
        correct_answer="B",
        source="bonus_fact",
        world_id="memory_metal_ball",
    ),
    MCQ(
        question="下面哪种是对这个世界的错误解释？",
        options={
            "A": "认为是弹簧力或磁力将物体拉回原点",
            "B": "认为物体被传送到5秒前的位置",
            "C": "认为传送后速度保持不变",
            "D": "认为传送有固定周期",
        },
        correct_answer="A",
        source="confusion",
        world_id="memory_metal_ball",
    ),
    MCQ(
        question="球在某一帧突然出现在一个新位置，但没有改变速度，可能发生了什么？",
        options={
            "A": "被弹飞了",
            "B": "被传送回5秒前的位置",
            "C": "分裂成了两个",
            "D": "被吸引到中心",
        },
        correct_answer="B",
        source="generalization",
        world_id="memory_metal_ball",
    ),
    MCQ(
        question="传送发生后，球的运动方向会改变吗？",
        options={
            "A": "会改变，方向反转",
            "B": "不会改变，速度方向保持不变",
            "C": "会改变，变为朝向中心",
            "D": "会改变，变为随机方向",
        },
        correct_answer="B",
        source="generalization",
        world_id="memory_metal_ball",
    ),
]

# ---------------------------------------------------------------------------
# 8. collision_split
# ---------------------------------------------------------------------------

_collision_split: List[MCQ] = [
    MCQ(
        question="两球碰撞后会发生什么？",
        options={
            "A": "两球正常弹开",
            "B": "每个球分裂成两个更小的球",
            "C": "两球融合成一个更大的球",
            "D": "两球消失",
        },
        correct_answer="B",
        source="required_fact",
        world_id="collision_split",
    ),
    MCQ(
        question="分裂产生的子球与母球相比有什么变化？",
        options={
            "A": "子球更大",
            "B": "子球半径和质量各为母球的一半",
            "C": "子球与母球完全相同",
            "D": "子球质量不变但半径减半",
        },
        correct_answer="B",
        source="required_fact",
        world_id="collision_split",
    ),
    MCQ(
        question="随着时间推移，屏幕上的球数量如何变化？",
        options={
            "A": "保持不变",
            "B": "逐渐减少",
            "C": "逐渐增多",
            "D": "先增多后减少",
        },
        correct_answer="C",
        source="bonus_fact",
        world_id="collision_split",
    ),
    MCQ(
        question="为什么随着时间推移，球越来越小？",
        options={
            "A": "因为球在蒸发",
            "B": "因为每次碰撞分裂出的子球半径是母球的一半",
            "C": "因为球在互相摩擦",
            "D": "因为重力在增大",
        },
        correct_answer="B",
        source="bonus_fact",
        world_id="collision_split",
    ),
    MCQ(
        question="下面哪种是对这个世界的错误解释？",
        options={
            "A": "认为是正常的弹性碰撞",
            "B": "认为碰撞时球会分裂",
            "C": "认为分裂后产生两个更小的球",
            "D": "认为球的数量会增加",
        },
        correct_answer="A",
        source="confusion",
        world_id="collision_split",
    ),
    MCQ(
        question="两个球碰撞后，屏幕上的球数量会如何变化？",
        options={
            "A": "减少（2个变1个）",
            "B": "不变（还是2个）",
            "C": "增加（从2个变成4个）",
            "D": "增加（从2个变成3个）",
        },
        correct_answer="C",
        source="generalization",
        world_id="collision_split",
    ),
    MCQ(
        question="这个世界的规律与以下哪个最不同？",
        options={
            "A": "碰撞后产生更多物体",
            "B": "子球尺寸递减",
            "C": "碰撞后两球朝同一方向运动",
            "D": "碰撞是分裂的原因",
        },
        correct_answer="C",
        source="cross_world",
        world_id="collision_split",
    ),
]

# ---------------------------------------------------------------------------
# 9. spring_floor
# ---------------------------------------------------------------------------

_spring_floor: List[MCQ] = [
    MCQ(
        question="这个世界的底板有什么特性？",
        options={
            "A": "底板是刚性的，完全不会变形",
            "B": "底板是弹性/可变形的，受到压力后变形",
            "C": "底板不存在，物体直接穿过",
            "D": "底板是磁性的，吸引物体",
        },
        correct_answer="B",
        source="required_fact",
        world_id="spring_floor",
    ),
    MCQ(
        question="底板变形后会产生什么效果？",
        options={
            "A": "产生向下的吸力",
            "B": "产生向上的弹力，将物体弹起",
            "C": "产生侧向的推力",
            "D": "产生热量",
        },
        correct_answer="B",
        source="required_fact",
        world_id="spring_floor",
    ),
    MCQ(
        question="从更高处落下的球，弹起来会怎样？",
        options={
            "A": "弹起更低（能量损失更多）",
            "B": "弹起更高（更大冲击力产生更大弹力）",
            "C": "弹起高度与下落高度无关",
            "D": "不会弹起",
        },
        correct_answer="B",
        source="bonus_fact",
        world_id="spring_floor",
    ),
    MCQ(
        question="底板会永远振荡下去吗？",
        options={
            "A": "会，永远振荡",
            "B": "不会，底板有阻尼会逐渐恢复平整",
            "C": "会，且振幅越来越大",
            "D": "不会，底板立刻恢复平整",
        },
        correct_answer="B",
        source="bonus_fact",
        world_id="spring_floor",
    ),
    MCQ(
        question="下面哪种是对这个世界的错误理解？",
        options={
            "A": "认为只是弹性系数很高的普通地板",
            "B": "认为地板会变形并产生弹力",
            "C": "认为弹起高度与冲击力正相关",
            "D": "认为地板有阻尼",
        },
        correct_answer="A",
        source="confusion",
        world_id="spring_floor",
    ),
    MCQ(
        question="地板在球落上去的瞬间会有什么变化？",
        options={
            "A": "向上凸起",
            "B": "向下变形/凹陷",
            "C": "左右移动",
            "D": "没有任何变化",
        },
        correct_answer="B",
        source="generalization",
        world_id="spring_floor",
    ),
    MCQ(
        question="这个世界的底板与普通硬地板的根本区别是什么？",
        options={
            "A": "底板的弹性系数更高",
            "B": "底板会变形储能再释放",
            "C": "底板温度更高",
            "D": "底板在移动",
        },
        correct_answer="B",
        source="cross_world",
        world_id="spring_floor",
    ),
]

# ---------------------------------------------------------------------------
# 10. anti_newton_pendulum
# ---------------------------------------------------------------------------

_anti_newton_pendulum: List[MCQ] = [
    MCQ(
        question="两球碰撞后朝什么方向运动？",
        options={
            "A": "朝相反方向运动（正常弹开）",
            "B": "朝同一方向运动",
            "C": "静止不动",
            "D": "朝随机方向运动",
        },
        correct_answer="B",
        source="required_fact",
        world_id="anti_newton_pendulum",
    ),
    MCQ(
        question="这个世界的碰撞规律违反了哪个物理定律？",
        options={
            "A": "牛顿第一定律（惯性定律）",
            "B": "牛顿第三定律（作用力与反作用力）/ 动量守恒",
            "C": "能量守恒定律",
            "D": "万有引力定律",
        },
        correct_answer="B",
        source="required_fact",
        world_id="anti_newton_pendulum",
    ),
    MCQ(
        question="碰撞后两球的速度由什么决定？",
        options={
            "A": "由碰前速度较快的那个球的速度决定",
            "B": "由碰前速度较慢的那个球的速度决定",
            "C": "两球速度的平均值",
            "D": "随机决定",
        },
        correct_answer="A",
        source="bonus_fact",
        world_id="anti_newton_pendulum",
    ),
    MCQ(
        question="经过多次碰撞，球的分布趋势是什么？",
        options={
            "A": "越来越分散",
            "B": "越来越集中（聚集到一处）",
            "C": "均匀分布在整个屏幕",
            "D": "呈环形分布",
        },
        correct_answer="B",
        source="bonus_fact",
        world_id="anti_newton_pendulum",
    ),
    MCQ(
        question="下面哪种是对这个世界的错误解释？",
        options={
            "A": "认为是磁力吸引导致同向运动",
            "B": "认为碰撞后两球朝同一方向运动",
            "C": "认为违反动量守恒规律",
            "D": "认为最终球聚集到一处",
        },
        correct_answer="A",
        source="confusion",
        world_id="anti_newton_pendulum",
    ),
    MCQ(
        question="两个球碰撞后，它们会朝同一方向还是相反方向运动？",
        options={
            "A": "相反方向",
            "B": "同一方向",
            "C": "静止不动",
            "D": "方向不确定",
        },
        correct_answer="B",
        source="generalization",
        world_id="anti_newton_pendulum",
    ),
    MCQ(
        question="经过多次碰撞，球的分布会越来越怎样？",
        options={
            "A": "越来越均匀",
            "B": "越来越分散",
            "C": "越来越集中",
            "D": "呈圆形分布",
        },
        correct_answer="C",
        source="generalization",
        world_id="anti_newton_pendulum",
    ),
    MCQ(
        question="这个世界的碰撞规律与哪个世界有相似之处？",
        options={
            "A": "黑洞世界（中心吸引）",
            "B": "碰撞分裂世界（碰撞产生变化）",
            "C": "反重力弹跳世界（向上加速）",
            "D": "时间减速带世界（左右速度不同）",
        },
        correct_answer="B",
        source="cross_world",
        world_id="anti_newton_pendulum",
    ),
]


# ---------------------------------------------------------------------------
# Aggregate: map world_id -> list of MCQs
# ---------------------------------------------------------------------------

WORLD_MCQS: Dict[str, List[MCQ]] = {
    "anti_gravity_bounce": _anti_gravity_bounce,
    "black_hole": _black_hole,
    "time_dilation": _time_dilation,
    "mass_oscillation": _mass_oscillation,
    "bounce_accelerator": _bounce_accelerator,
    "gravity_merry_go_round": _gravity_merry_go_round,
    "memory_metal_ball": _memory_metal_ball,
    "collision_split": _collision_split,
    "spring_floor": _spring_floor,
    "anti_newton_pendulum": _anti_newton_pendulum,
}
