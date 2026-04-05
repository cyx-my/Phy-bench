视频生成要求：
1. **反重力弹跳** - 物体向上掉落，越弹越高

2. **黑洞吸尘器** - 所有物体被吸向屏幕中心，靠近后弹飞

3. **时间减速带** - 屏幕一半物体慢动作，另一半快进

4. **质量震荡球** - 小球周期性地变重变轻，忽快忽慢

5. **反弹加速器** - 每次碰撞速度翻倍，直到飞出屏幕

6. **重力旋转木马** - 重力方向每5秒旋转90度

7. **记忆金属球** - 球体会自动回到5秒前的位置

8. **碰撞分裂术** - 球碰撞时分裂成两个更小的球

9. **弹簧地板** - 地板会随压力下沉然后弹起物体

10. **反牛顿摆** - 两球碰撞后朝同一方向运动

---

## 使用说明

### 环境准备
1. 安装 Python 3.7+
2. 安装依赖包：`pip install -r requirements.txt`

### 现有脚本
已实现以下物理效果的生成脚本：

| 效果 | 脚本文件 | 输出视频 |
|------|----------|----------|
| 反重力弹跳 | `test.py` | `weird_physics.mp4` |
| 黑洞吸尘器 | `black_hole.py` | `black_hole.mp4` |
| 时间减速带 | `time_dilation.py` | `time_dilation.mp4` |
| 质量震荡球 | `mass_oscillation.py` | `mass_oscillation.mp4` |
| 反弹加速器 | `bounce_accelerator.py` | `bounce_accelerator.mp4` |
| 重力旋转木马 | `gravity_merry_go_round.py` | `gravity_merry_go_round.mp4` |
| 记忆金属球 | `memory_metal_ball.py` | `memory_metal_ball.mp4` |
| 碰撞分裂术 | `collision_split.py` | `collision_split.mp4` |
| 弹簧地板 | `spring_floor.py` | `spring_floor.mp4` |
| 反牛顿摆 | `anti_newton_pendulum.py` | `anti_newton_pendulum.mp4` |

### 运行方法
```bash
python test.py                     # 生成反重力弹跳视频
python black_hole.py               # 生成黑洞吸尘器视频
python time_dilation.py            # 生成时间减速带视频
python mass_oscillation.py         # 生成质量震荡球视频
python bounce_accelerator.py       # 生成反弹加速器视频
python gravity_merry_go_round.py   # 生成重力旋转木马视频
python memory_metal_ball.py        # 生成记忆金属球视频
python collision_split.py          # 生成碰撞分裂术视频
python spring_floor.py             # 生成弹簧地板视频
python anti_newton_pendulum.py     # 生成反牛顿摆视频
```

每个脚本会运行约15秒，生成对应的MP4视频文件。

### 交互操作
- 按**空格键**可以添加新的小球
- 按**ESC**或关闭窗口可以提前结束录制

### 自定义参数
可以在脚本中调整以下参数：
- 视频分辨率 (800, 600)
- 帧率 (fps=30)
- 录制时长 (max_frames)
- 物理参数 (重力、弹性、摩擦力等)

### 所有效果已实现
所有10个物理效果均已实现，可以直接运行对应的脚本生成视频。