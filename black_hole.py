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
    'black_hole.mp4',           # 输出文件名
    cv2.VideoWriter_fourcc(*'mp4v'), # 编码格式
    fps,                        # 帧率
    (800, 600)                  # 画面大小
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
for _ in range(8):
    mass = 1
    radius = random.randint(10, 20)
    inertia = pymunk.moment_for_circle(mass, 0, radius)
    body = pymunk.Body(mass, inertia)
    body.position = (random.randint(100, 700), random.randint(100, 500))
    # 给小球随机初始速度
    body.velocity = (random.uniform(-100, 100), random.uniform(-100, 100))

    shape = pymunk.Circle(body, radius)
    shape.elasticity = 0.8
    shape.friction = 0.3
    shape.color = (random.randint(50, 200), random.randint(50, 200), random.randint(50, 200), 255)

    space.add(body, shape)
    balls.append((body, shape))

# 自定义力函数：黑洞吸引力
def apply_black_hole_force():
    for body, shape in balls:
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

while running and frame_count < max_frames:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                # 按空格键添加一个新球
                mass = 1
                radius = random.randint(10, 20)
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

    # 应用黑洞力
    apply_black_hole_force()

    # 物理步进
    space.step(1/60.0)

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
print(f"视频已保存到 black_hole.mp4 (共{frame_count}帧)")