import pygame
import pymunk
import pymunk.pygame_util
import random
import cv2
import numpy as np
import math
import os
import json
from typing import Dict, Any, List, Tuple

# 导入 benchmark 的数据schema
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmark.data_schema import StatesWriter, FrameState


def run_episode(output_dir: str, seed: int, fps: int = 30) -> Dict[str, Any]:
    """
    运行黑洞世界模拟，生成视频和状态记录。

    Args:
        output_dir: 输出目录，将在此目录下创建 video.mp4 和 states.jsonl
        seed: 随机种子
        fps: 视频帧率

    Returns:
        包含元数据的字典，如初始条件、物体数量等。
    """
    # 设置随机种子
    random.seed(seed)

    # 初始化pygame
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    clock = pygame.time.Clock()
    draw_options = pymunk.pygame_util.DrawOptions(screen)

    # 视频保存设置
    video_path = os.path.join(output_dir, "video.mp4")
    os.makedirs(output_dir, exist_ok=True)
    video_writer = cv2.VideoWriter(
        video_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (800, 600)
    )

    # 创建物理空间
    space = pymunk.Space()
    space.gravity = (0, 0)   # 无重力，使用黑洞吸引力

    # 创建边界墙
    walls = [
        pymunk.Segment(space.static_body, (50, 50), (750, 50), 5),   # 上墙
        pymunk.Segment(space.static_body, (50, 550), (750, 550), 5), # 下墙
        pymunk.Segment(space.static_body, (50, 50), (50, 550), 5),   # 左墙
        pymunk.Segment(space.static_body, (750, 50), (750, 550), 5), # 右墙
    ]
    for wall in walls:
        wall.elasticity = 0.8
        wall.friction = 0.5
        space.add(wall)

    # 黑洞中心
    black_hole_center = (400, 300)
    black_hole_radius = 50  # 黑洞半径，进入此区域会被弹飞

    # 创建多个小球
    balls = []
    initial_conditions = []
    num_balls = random.randint(5, 10)  # 从配置中读取范围，暂定5-10个
    for i in range(num_balls):
        mass = 1
        radius = random.randint(10, 20)
        inertia = pymunk.moment_for_circle(mass, 0, radius)
        body = pymunk.Body(mass, inertia)
        body.position = (random.randint(100, 700), random.randint(100, 500))
        body.velocity = (random.uniform(-100, 100), random.uniform(-100, 100))

        shape = pymunk.Circle(body, radius)
        shape.elasticity = 0.8
        shape.friction = 0.3
        shape.color = (random.randint(50, 200), random.randint(50, 200), random.randint(50, 200), 255)

        space.add(body, shape)
        balls.append((body, shape, radius))

        # 记录初始条件
        initial_conditions.append({
            "id": i,
            "position": (float(body.position.x), float(body.position.y)),
            "velocity": (float(body.velocity.x), float(body.velocity.y)),
            "radius": radius,
            "mass": mass
        })

    # 自定义力函数：黑洞吸引力
    def apply_black_hole_force():
        for body, shape, radius in balls:
            # 计算物体到黑洞中心的向量
            dx = black_hole_center[0] - body.position.x
            dy = black_hole_center[1] - body.position.y
            distance = math.sqrt(dx*dx + dy*dy)

            if distance > 0:
                # 如果距离大于黑洞半径，施加向心力
                if distance > black_hole_radius:
                    # 吸引力与距离成反比（越近力越大）
                    force_strength = 50000 / (distance * distance)
                    force_x = force_strength * dx / distance
                    force_y = force_strength * dy / distance
                    body.apply_force_at_world_point((force_x, force_y), body.position)
                else:
                    # 进入黑洞区域，被弹飞（斥力）
                    repulse_strength = 5000
                    force_x = repulse_strength * dx / distance
                    force_y = repulse_strength * dy / distance
                    body.apply_force_at_world_point((force_x, force_y), body.position)

    # 游戏循环
    running = True
    frame_count = 0
    max_frames = fps * 15  # 录制15秒的视频

    # 打开状态记录文件
    states_path = os.path.join(output_dir, "states.jsonl")
    states_file = open(states_path, "w")

    # 实际循环
    while running and frame_count < max_frames:
        # 处理退出事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # 应用黑洞力
        apply_black_hole_force()

        # 物理步进
        space.step(1/60.0)

        # 收集当前帧状态
        timestamp = frame_count / fps
        objects_state = []
        for i, (body, shape, radius) in enumerate(balls):
            obj_state = {
                "id": i,
                "position_x": float(body.position.x),
                "position_y": float(body.position.y),
                "velocity_x": float(body.velocity.x),
                "velocity_y": float(body.velocity.y),
                "radius": radius,
                "mass": 1.0,
            }
            objects_state.append(obj_state)

        # 世界状态
        world_state = {
            "black_hole_center_x": black_hole_center[0],
            "black_hole_center_y": black_hole_center[1],
            "black_hole_radius": black_hole_radius,
            "gravity_x": 0.0,
            "gravity_y": 0.0,
        }

        # 事件检测（暂为空）
        events = []

        # 写入状态行
        frame_state = {
            "frame": frame_count,
            "timestamp": timestamp,
            "objects": objects_state,
            "world_state": world_state,
            "events": events,
        }
        states_file.write(json.dumps(frame_state) + "\n")

        # 绘制
        screen.fill((0, 0, 0))  # 黑色背景

        # 绘制黑洞区域
        pygame.draw.circle(screen, (100, 0, 150), black_hole_center, black_hole_radius, 2)
        pygame.draw.circle(screen, (50, 0, 100), black_hole_center, 5)

        # 绘制物理物体
        space.debug_draw(draw_options)

        pygame.display.flip()

        # 保存当前帧为视频
        frame = pygame.surfarray.array3d(screen)
        frame = frame.swapaxes(0, 1)  # 调整维度
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(frame)

        frame_count += 1
        clock.tick(fps)

    # 释放视频写入器
    video_writer.release()
    pygame.quit()

    # 返回元数据
    return {
        "num_objects": num_balls,
        "initial_conditions": initial_conditions,
        "black_hole_center": black_hole_center,
        "black_hole_radius": black_hole_radius,
        "duration_sec": frame_count / fps,
        "fps": fps,
        "width": 800,
        "height": 600,
    }


if __name__ == "__main__":
    # 测试运行
    import tempfile
    import shutil

    tmpdir = tempfile.mkdtemp()
    print(f"临时目录: {tmpdir}")

    try:
        meta = run_episode(tmpdir, seed=42, fps=30)
        print(f"生成完成，元数据: {json.dumps(meta, indent=2, default=str)}")

        # 检查文件
        video_file = os.path.join(tmpdir, "video.mp4")
        if os.path.exists(video_file):
            print(f"视频文件已生成: {video_file}")
        else:
            print("警告: 视频文件未生成")

    finally:
        # 清理临时目录
        shutil.rmtree(tmpdir)
        print("临时目录已清理")