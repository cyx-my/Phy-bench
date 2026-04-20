#!/usr/bin/env python3
"""测试worlds模块导入问题"""

import sys
import os
import importlib

# 添加项目根目录到sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# 测试导入black_hole
print(f"sys.path: {sys.path[:2]}...")
print(f"Project root: {project_root}")

# 测试1: 直接导入
try:
    import worlds.black_hole as bh
    print("✓ 成功导入 worlds.black_hole")
    print(f"  run_episode 函数: {hasattr(bh, 'run_episode')}")
except ImportError as e:
    print(f"✗ 导入失败: {e}")

# 测试2: 使用import_module
try:
    script_module = "worlds.black_hole"
    mod = importlib.import_module(script_module)
    print(f"✓ import_module 成功: {mod}")
    print(f"  run_episode 函数: {hasattr(mod, 'run_episode')}")
except Exception as e:
    print(f"✗ import_module 失败: {e}")

# 测试3: 测试dataset_builder中的逻辑
from benchmark.world_configs import get_world
cfg = get_world("black_hole")
print(f"\nWorld config for 'black_hole':")
print(f"  script: {cfg.script}")
script_module = cfg.script.replace("/", ".").replace(".py", "")
print(f"  script_module: {script_module}")

try:
    mod = importlib.import_module(script_module)
    print(f"✓ 通过config导入成功")
    print(f"  run_episode 函数: {hasattr(mod, 'run_episode')}")

    # 测试调用
    import tempfile
    tmpdir = tempfile.mkdtemp()
    print(f"\n测试run_episode调用 (输出到: {tmpdir})...")
    result = mod.run_episode(output_dir=tmpdir, seed=42, fps=30)
    print(f"✓ run_episode 调用成功")
    print(f"  返回元数据: {result.keys()}")

    # 检查生成的文件
    import os
    video_file = os.path.join(tmpdir, "video.mp4")
    states_file = os.path.join(tmpdir, "states.jsonl")
    print(f"  视频文件存在: {os.path.exists(video_file)}")
    print(f"  状态文件存在: {os.path.exists(states_file)}")

    # 清理
    import shutil
    shutil.rmtree(tmpdir)

except Exception as e:
    print(f"✗ 失败: {e}")
    import traceback
    traceback.print_exc()