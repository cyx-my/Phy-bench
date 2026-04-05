import pygame
import pymunk
import pymunk.pygame_util
import random
import cv2
import numpy as np
import math

# 初始化
pygame.init()
screen = pygame.display.set_mode((800, 600))
clock = pygame.time.Clock()
draw_options = pymunk.pygame_util.DrawOptions(screen)

# 视频保存设置
fps = 30
video_writer = cv2.VideoWriter(
    'gravity_merry_go_round.mp4', # 输出文件名
    cv2.VideoWriter_fourcc(*'mp4v'), # 编码格式
    fps,                        # 帧率
    (800, 600)                  # 画面大小
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
for _ in range(8):
    mass = 1
    radius = random.randint(15, 25)
    inertia = pymunk.moment_for_circle(mass, 0, radius)
    body = pymunk.Body(mass, inertia)
    body.position = (random.randint(100, 700), random.randint(100, 500))
    body.velocity = (random.uniform(-100, 100), random.uniform(-100, 100))

    shape = pymunk.Circle(body, radius)
    shape.elasticity = 0.8
    shape.friction = 0.3
    shape.color = (random.randint(50, 200), random.randint(50, 200), random.randint(50, 200), 255)

    space.add(body, shape)
    balls.append((body, shape))

# 重力方向旋转函数
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
    global current_direction_index, last_gravity_change_time
    if current_time - last_gravity_change_time >= gravity_change_interval:
        last_gravity_change_time = current_time
        current_direction_index = (current_direction_index + 1) % 4
        space.gravity = gravity_directions[current_direction_index]
        print(f"重力方向改变: {space.gravity}")

# 游戏循环
running = True
frame_count = 0
max_frames = fps * 30  # 录制30秒的视频，展示多个重力周期

while running and frame_count < max_frames:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                # 按空格键添加一个新球
                mass = 1
                radius = random.randint(15, 25)
                inertia = pymunk.moment_for_circle(mass, 0, radius)
                new_body = pymunk.Body(mass, inertia)
                new_body.position = (random.randint(100, 700), random.randint(100, 500))
                new_body.velocity = (random.uniform(-100, 100), random.uniform(-100, 100))
                new_shape = pymunk.Circle(new_body, radius)
                new_shape.elasticity = 0.8
                new_shape.friction = 0.3
                new_shape.color = (random.randint(50, 200), random.randint(50, 200), random.randint(50, 200), 255)
                space.add(new_body, new_shape)
                balls.append((new_body, new_shape))

    # 更新重力方向
    current_time = pygame.time.get_ticks() / 1000.0
    update_gravity(current_time)

    # 物理步进
    space.step(1/60.0)

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
print(f"视频已保存到 gravity_merry_go_round.mp4 (共{frame_count}帧)")