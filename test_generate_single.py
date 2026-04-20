#!/usr/bin/env python3
"""验证单个episode生成：调用worlds脚本生成视频和状态文件，检查文件完整性和格式正确性。"""

import argparse
import importlib
import json
import os
import sys
import tempfile
import shutil
from pathlib import Path

# 添加项目根目录到sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from benchmark.world_configs import WORLDS, get_world


def verify_video_file(video_path: str) -> tuple[bool, str]:
    """验证视频文件存在且非空。"""
    if not os.path.exists(video_path):
        return False, f"视频文件不存在: {video_path}"

    try:
        size = os.path.getsize(video_path)
        if size == 0:
            return False, f"视频文件为空 (大小: {size} 字节)"

        # 基本格式检查（通过文件扩展名）
        if not video_path.lower().endswith('.mp4'):
            return False, f"视频文件扩展名不是.mp4: {video_path}"

        # 可以进一步用OpenCV验证，但这里只做基本检查
        return True, f"视频文件有效，大小: {size} 字节"

    except Exception as e:
        return False, f"验证视频文件时出错: {e}"


def verify_states_file(states_path: str) -> tuple[bool, str]:
    """验证states.jsonl文件格式正确。"""
    if not os.path.exists(states_path):
        return False, f"状态文件不存在: {states_path}"

    try:
        size = os.path.getsize(states_path)
        if size == 0:
            return False, f"状态文件为空 (大小: {size} 字节)"

        frames = []
        line_count = 0
        with open(states_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    frame_data = json.loads(line)
                except json.JSONDecodeError as e:
                    return False, f"第{line_num}行JSON解析失败: {e}\n行内容: {line[:100]}..."

                # 检查必需字段
                required_fields = ["frame", "timestamp", "objects", "world_state", "events"]
                missing = [fld for fld in required_fields if fld not in frame_data]
                if missing:
                    return False, f"第{line_num}行缺少必需字段: {missing}"

                # 检查objects是列表
                if not isinstance(frame_data["objects"], list):
                    return False, f"第{line_num}行objects字段不是列表"

                # 检查world_state是字典
                if not isinstance(frame_data["world_state"], dict):
                    return False, f"第{line_num}行world_state字段不是字典"

                # 检查events是列表
                if not isinstance(frame_data["events"], list):
                    return False, f"第{line_num}行events字段不是列表"

                frames.append(frame_data)
                line_count += 1

        if line_count == 0:
            return False, "状态文件没有包含有效行"

        # 检查frame编号是否连续
        frame_numbers = [f["frame"] for f in frames]
        if sorted(frame_numbers) != list(range(min(frame_numbers), max(frame_numbers) + 1)):
            return False, f"frame编号不连续或不按顺序: {frame_numbers}"

        # 检查时间戳递增
        timestamps = [f["timestamp"] for f in frames]
        for i in range(1, len(timestamps)):
            if timestamps[i] <= timestamps[i - 1]:
                return False, f"时间戳不递增: 第{i-1}帧={timestamps[i-1]}, 第{i}帧={timestamps[i]}"

        return True, f"状态文件有效，包含{line_count}帧，frames: {min(frame_numbers)}-{max(frame_numbers)}"

    except Exception as e:
        return False, f"验证状态文件时出错: {e}"


def verify_episode_meta(meta_dict: dict) -> tuple[bool, str]:
    """验证episode元数据包含必需字段。"""
    required_fields = ["num_objects", "duration_sec", "fps", "width", "height"]
    missing = [fld for fld in required_fields if fld not in meta_dict]
    if missing:
        return False, f"元数据缺少必需字段: {missing}"

    # 检查数值有效性
    if meta_dict["num_objects"] <= 0:
        return False, f"num_objects必须大于0: {meta_dict['num_objects']}"
    if meta_dict["duration_sec"] <= 0:
        return False, f"duration_sec必须大于0: {meta_dict['duration_sec']}"
    if meta_dict["fps"] <= 0:
        return False, f"fps必须大于0: {meta_dict['fps']}"
    if meta_dict["width"] <= 0 or meta_dict["height"] <= 0:
        return False, f"width/height必须大于0: {meta_dict['width']}x{meta_dict['height']}"

    return True, f"元数据有效: {meta_dict.get('num_objects', '?')}个物体，{meta_dict.get('duration_sec', '?')}秒"


def test_world(world_id: str, seed: int = 42, fps: int = 30) -> tuple[bool, str, dict]:
    """
    测试单个世界的episode生成。

    返回: (成功?, 消息, 元数据)
    """
    print(f"\n{'='*60}")
    print(f"测试世界: {world_id}")
    print(f"{'='*60}")

    try:
        cfg = get_world(world_id)
        print(f"配置: {cfg.name}")
        print(f"脚本: {cfg.script}")

        # 检查脚本文件是否存在
        if not os.path.exists(cfg.script):
            print(f"⚠ 脚本文件不存在，跳过测试")
            return True, f"跳过: 脚本文件不存在: {cfg.script}", {}

        # 导入世界模块
        script_module = cfg.script.replace("/", ".").replace(".py", "")
        try:
            mod = importlib.import_module(script_module)
        except ImportError as e:
            return False, f"导入模块失败: {e}", {}

        if not hasattr(mod, "run_episode"):
            return False, f"模块没有run_episode函数", {}

        # 创建临时目录
        tmpdir = tempfile.mkdtemp(prefix=f"phybench_test_{world_id}_")
        print(f"临时目录: {tmpdir}")

        try:
            # 运行episode生成
            print(f"运行run_episode(seed={seed}, fps={fps})...")
            meta = mod.run_episode(output_dir=tmpdir, seed=seed, fps=fps)
            print(f"✓ run_episode成功完成")

            # 验证生成的文件
            video_file = os.path.join(tmpdir, "video.mp4")
            states_file = os.path.join(tmpdir, "states.jsonl")

            # 验证视频文件
            video_ok, video_msg = verify_video_file(video_file)
            if video_ok:
                print(f"✓ {video_msg}")
            else:
                print(f"✗ {video_msg}")
                return False, f"视频验证失败: {video_msg}", meta

            # 验证状态文件
            states_ok, states_msg = verify_states_file(states_file)
            if states_ok:
                print(f"✓ {states_msg}")
            else:
                print(f"✗ {states_msg}")
                return False, f"状态文件验证失败: {states_msg}", meta

            # 验证元数据
            meta_ok, meta_msg = verify_episode_meta(meta)
            if meta_ok:
                print(f"✓ {meta_msg}")
            else:
                print(f"✗ {meta_msg}")
                return False, f"元数据验证失败: {meta_msg}", meta

            # 打印摘要
            print(f"\n生成成功！")
            print(f"  视频: {os.path.getsize(video_file):,} 字节")
            with open(states_file, 'r') as f:
                frame_count = sum(1 for line in f if line.strip())
            print(f"  状态: {frame_count} 帧")
            print(f"  持续时间: {meta.get('duration_sec', '?')} 秒")
            print(f"  物体数量: {meta.get('num_objects', '?')}")

            return True, f"世界'{world_id}'测试通过", meta

        finally:
            # 清理临时目录
            shutil.rmtree(tmpdir, ignore_errors=True)
            print(f"临时目录已清理")

    except Exception as e:
        error_msg = f"测试世界'{world_id}'时出错: {e}"
        import traceback
        traceback.print_exc()
        return False, error_msg, {}


def main():
    parser = argparse.ArgumentParser(description="验证单个episode生成")
    parser.add_argument("--worlds", nargs="+",
                       help="要测试的世界ID列表，默认测试所有世界")
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子 (默认: 42)")
    parser.add_argument("--fps", type=int, default=30,
                       help="帧率 (默认: 30)")
    parser.add_argument("--quick", action="store_true",
                       help="快速模式：只测试第一个世界")

    args = parser.parse_args()

    # 确定要测试的世界
    if args.worlds:
        world_ids = args.worlds
    else:
        world_ids = list(WORLDS.keys())

    if args.quick:
        world_ids = world_ids[:1]
        print(f"快速模式：只测试第一个世界 '{world_ids[0]}'")

    print(f"开始验证 {len(world_ids)} 个世界: {', '.join(world_ids)}")
    print(f"随机种子: {args.seed}, 帧率: {args.fps}")

    results = []
    for world_id in world_ids:
        success, message, meta = test_world(world_id, args.seed, args.fps)

        # 确定状态
        if success and message.startswith("跳过:"):
            status = "skipped"
        elif success:
            status = "success"
        else:
            status = "failed"

        results.append({
            "world_id": world_id,
            "status": status,
            "message": message,
            "meta_keys": list(meta.keys()) if meta else []
        })

    # 总结
    print(f"\n{'='*60}")
    print(f"测试总结")
    print(f"{'='*60}")

    success_count = sum(1 for r in results if r["status"] == "success")
    skipped_count = sum(1 for r in results if r["status"] == "skipped")
    failed_count = sum(1 for r in results if r["status"] == "failed")
    total_count = len(results)

    # 打印统计
    print(f"成功: {success_count}, 跳过: {skipped_count}, 失败: {failed_count} (总计: {total_count})")

    if failed_count == 0:
        if success_count == total_count:
            print(f"✓ 所有 {total_count} 个世界测试通过！")
        else:
            print(f"✓ 所有可测试的世界通过，{skipped_count} 个世界因脚本缺失跳过")
    else:
        print(f"✗ {failed_count} 个世界测试失败")

    # 打印详细信息
    for r in results:
        if r["status"] == "success":
            status_symbol = "✓"
        elif r["status"] == "skipped":
            status_symbol = "⚪"
        else:
            status_symbol = "✗"
        print(f"  {status_symbol} {r['world_id']}: {r['message']}")

    exit_code = 0 if failed_count == 0 else 1
    sys.exit(exit_code)


if __name__ == "__main__":
    main()