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
    运行重力旋转木马世界模拟，生成视频和状态记录。

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
    gravity_strength = 500
    current_gravity_angle = 0  # 0度表示向下
    space.gravity = (0, gravity_strength)  # 初始向下

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

    # 创建多个小球
    balls = []
    initial_conditions = []
    num_balls = random.randint(5, 10)  # 根据配置范围，5-10个
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

    # 重力方向旋转设置
    gravity_directions = [
        (0, gravity_strength),      # 向下
        (gravity_strength, 0),      # 向右
        (0, -gravity_strength),     # 向上
        (-gravity_strength, 0),     # 向左
    ]
    current_direction_index = 0
    last_gravity_change_time = 0
    gravity_change_interval = 5  # 每5秒改变一次

    def update_gravity(current_time):
        nonlocal current_direction_index, last_gravity_change_time
        if current_time - last_gravity_change_time >= gravity_change_interval:
            last_gravity_change_time = current_time
            current_direction_index = (current_direction_index + 1) % 4
            space.gravity = gravity_directions[current_direction_index]

    # 游戏循环
    running = True
    frame_count = 0
    max_frames = fps * 30  # 录制30秒的视频，展示多个重力周期

    # 打开状态记录文件
    states_path = os.path.join(output_dir, "states.jsonl")
    states_file = open(states_path, "w")

    # 实际循环
    while running and frame_count < max_frames:
        # 处理退出事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # 更新重力方向
        current_time = frame_count / fps
        update_gravity(current_time)

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
            "gravity_x": float(space.gravity[0]),
            "gravity_y": float(space.gravity[1]),
            "gravity_direction_index": current_direction_index,
            "gravity_change_interval": gravity_change_interval,
            "last_gravity_change_time": last_gravity_change_time,
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
        screen.fill((255, 255, 255))  # 白色背景

        # 绘制重力方向指示器（无文字）
        gravity_x, gravity_y = space.gravity
        center_x, center_y = 400, 300
        indicator_length = 100
        indicator_end_x = center_x + gravity_x / gravity_strength * indicator_length
        indicator_end_y = center_y + gravity_y / gravity_strength * indicator_length

        pygame.draw.circle(screen, (200, 0, 0), (int(center_x), int(center_y)), 10)
        pygame.draw.line(screen, (200, 0, 0), (center_x, center_y),
                        (indicator_end_x, indicator_end_y), 5)
        pygame.draw.circle(screen, (200, 0, 0), (int(indicator_end_x), int(indicator_end_y)), 8)

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

    # 关闭状态文件
    states_file.close()

    # 返回元数据
    return {
        "num_objects": num_balls,
        "initial_conditions": initial_conditions,
        "gravity_strength": gravity_strength,
        "gravity_change_interval": gravity_change_interval,
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