"""
Dataset builder for Phy-Bench.

Orchestrates the generation of multiple episodes across all worlds,
organizes outputs into the standard data schema, and writes metadata files.

Usage:
    python -m benchmark.dataset_builder \
        --worlds all \
        --episodes-per-world 20 \
        --data-root data/ \
        --train-ratio 0.7

For a quick smoke-test (2 episodes per world):
    python -m benchmark.dataset_builder --episodes-per-world 2 --data-root data_test/
"""

import argparse
import importlib
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to sys.path to allow importing worlds modules
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from benchmark.world_configs import WORLDS, get_world
from benchmark.data_schema import (
    WorldMeta, EpisodeMeta,
    save_world_meta, save_episode_meta, save_rule_description,
    world_dir, episode_dir,
)
from benchmark import validation


def generate_episode(
    world_id: str,
    episode_id: str,
    data_root: str,
    seed: int,
    fps: int = 30,
) -> Optional[EpisodeMeta]:
    """
    Call the world-specific simulation script to generate one episode.

    Each simulation script in worlds/ is expected to expose a `run_episode` function:

        def run_episode(
            output_dir: str,        # Directory to write video.mp4 and states.jsonl
            seed: int,
            fps: int = 30,
        ) -> dict:                  # Returns episode metadata dict (initial_conditions, etc.)

    If the script does not have `run_episode`, we fall back to running it as __main__
    (legacy behavior from the existing scripts) — but this will not produce states.jsonl.
    """
    cfg = get_world(world_id)
    ep_dir = episode_dir(data_root, world_id, episode_id)
    os.makedirs(ep_dir, exist_ok=True)

    # Try to import run_episode from the world's script module
    script_module = cfg.script.replace("/", ".").replace(".py", "")
    try:
        mod = importlib.import_module(script_module)
    except ImportError as e:
        print(f"  [ERROR] {world_id}: Failed to import module '{script_module}': {e}")
        print(f"          Make sure dependencies (pygame, pymunk, opencv-python, numpy) are installed.")
        print(f"          Run: pip install -r requirements.txt")
        print(f"  Skipping simulation for {world_id}.")
        return None

    try:
        extra = mod.run_episode(output_dir=ep_dir, seed=seed, fps=fps)
    except AttributeError as e:
        print(f"  [ERROR] {world_id}: Module '{script_module}' has no 'run_episode' function: {e}")
        print(f"          Check if the world script implements the required interface.")
        print(f"  Skipping simulation for {world_id}.")
        return None
    except Exception as e:
        print(f"  [ERROR] {world_id}: Simulation failed with error: {e}")
        print(f"          Check the world script for issues.")
        import traceback
        traceback.print_exc()
        print(f"  Skipping simulation for {world_id}.")
        return None

    meta = EpisodeMeta(
        episode_id=episode_id,
        world_id=world_id,
        seed=seed,
        num_objects=cfg.num_objects_range[0],  # Will be overwritten by run_episode if available
        duration_sec=cfg.episode_duration_sec,
        fps=fps,
        width=800,
        height=600,
        initial_conditions=extra.get("initial_conditions", {}),
        extra={k: v for k, v in extra.items() if k != "initial_conditions"},
    )
    if "num_objects" in extra:
        meta.num_objects = extra["num_objects"]

    save_episode_meta(meta, data_root)

    # Validate the generated episode
    states_path = os.path.join(ep_dir, 'states.jsonl')
    if os.path.exists(states_path):
        try:
            valid, details = validation.validate_episode(ep_dir, world_id)
            if not valid:
                print(f"  [WARNING] {world_id}/{episode_id}: Validation failed:")
                for err in details.get('errors', []):
                    print(f"    - {err}")
            elif details.get('warnings'):
                for warn in details.get('warnings', []):
                    print(f"  [WARNING] {world_id}/{episode_id}: {warn}")
            # Print physics check scores if available
            if details.get('physics_checks'):
                for check_name, score in details['physics_checks'].items():
                    status = "✓" if score >= 0.8 else "✗"
                    print(f"    {check_name}: {score:.3f} {status}")
        except Exception as e:
            print(f"  [WARNING] {world_id}/{episode_id}: Validation error: {e}")
    else:
        print(f"  [WARNING] {world_id}/{episode_id}: No states.jsonl generated")

    return meta


def build_dataset(
    worlds: List[str],
    episodes_per_world: int,
    data_root: str,
    train_ratio: float = 0.7,
    fps: int = 30,
    base_seed: int = 42,
) -> None:
    """
    Generate the full dataset for the given worlds.

    Directory structure created:
        data_root/worlds/<world_id>/
            world_meta.json
            rule_description.json
            episodes/
                ep_001/ ... ep_NNN/
    """
    os.makedirs(data_root, exist_ok=True)
    rng = random.Random(base_seed)

    for world_id in worlds:
        cfg = get_world(world_id)
        print(f"\n[{world_id}] Generating {episodes_per_world} episodes...")

        episode_ids = []
        success_count = 0
        for i in range(episodes_per_world):
            ep_id = f"ep_{i+1:03d}"
            seed = rng.randint(0, 2**31)
            print(f"  {ep_id} (seed={seed})")
            meta = generate_episode(world_id, ep_id, data_root, seed=seed, fps=fps)
            if meta:
                episode_ids.append(ep_id)
                success_count += 1

        if not episode_ids:
            print(f"  [WARNING] No episodes generated for {world_id}. Skipping world metadata.")
            continue

        # Train/test split
        n_train = max(1, int(len(episode_ids) * train_ratio))
        shuffled = list(episode_ids)
        rng.shuffle(shuffled)
        split = {
            "train": sorted(shuffled[:n_train]),
            "test": sorted(shuffled[n_train:]),
        }

        # Write world-level metadata
        world_meta = WorldMeta(
            world_id=world_id,
            name=cfg.name,
            num_episodes=len(episode_ids),
            split=split,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        save_world_meta(world_meta, data_root)
        save_rule_description(world_id, cfg.rule_description, data_root)

        print(f"  Done. Generated: {success_count}/{episodes_per_world}. Train: {len(split['train'])}, Test: {len(split['test'])}")

    print(f"\nDataset written to: {data_root}")
    _write_dataset_index(worlds, episodes_per_world, data_root)


def _write_dataset_index(worlds: List[str], episodes_per_world: int, data_root: str) -> None:
    """Write a top-level dataset_index.json for easy discovery."""
    index = {
        "num_worlds": len(worlds),
        "world_ids": worlds,
        "episodes_per_world": episodes_per_world,
        "total_episodes": len(worlds) * episodes_per_world,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    index_path = os.path.join(data_root, "dataset_index.json")
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)
    print(f"Index written: {index_path}")


def main():
    parser = argparse.ArgumentParser(description="Build Phy-Bench dataset")
    parser.add_argument(
        "--worlds", nargs="+", default=["all"],
        help='World IDs to generate, or "all" for all worlds',
    )
    parser.add_argument("--episodes-per-world", type=int, default=20)
    parser.add_argument("--data-root", type=str, default="data/")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    worlds = list(WORLDS.keys()) if "all" in args.worlds else args.worlds
    for w in worlds:
        if w not in WORLDS:
            print(f"Unknown world: {w}. Available: {list(WORLDS.keys())}")
            sys.exit(1)

    build_dataset(
        worlds=worlds,
        episodes_per_world=args.episodes_per_world,
        data_root=args.data_root,
        train_ratio=args.train_ratio,
        fps=args.fps,
        base_seed=args.seed,
    )


if __name__ == "__main__":
    main()
