#!/usr/bin/env python3
"""
Minimal evaluation script for Phy-Bench.

Usage:
    python evaluate_minimal.py --data-root data/ --worlds black_hole gravity_merry_go_round
    python evaluate_minimal.py --episode data/worlds/black_hole/episodes/ep_001/states.jsonl

Outputs a simple report with trajectory MSE and physics consistency scores.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add project root to sys.path to import benchmark modules
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from benchmark.evaluator import (
    compute_trajectory_mse,
    check_upward_acceleration,
    check_centripetal_force,
    check_speed_doubles_on_bounce,
    PHYSICS_CHECK_FUNCTIONS,
)


def load_states(states_path: str) -> List[Dict[str, Any]]:
    """Load a states.jsonl file."""
    states = []
    with open(states_path, 'r') as f:
        for line in f:
            if line.strip():
                states.append(json.loads(line))
    return states


def evaluate_episode(world_id: str, states_path: str, verbose: bool = True) -> Dict[str, Any]:
    """
    Evaluate a single episode.

    Returns a dict with:
        - world_id
        - episode_id (extracted from path)
        - trajectory_mse (placeholder 0.0 for now, needs ground truth)
        - physics_checks: dict of check_name -> pass_rate
        - physics_consistency: average pass rate
        - passed: bool if all checks pass above threshold
    """
    # Extract episode ID from path
    path_parts = Path(states_path).parts
    episode_id = None
    for i, part in enumerate(path_parts):
        if part.startswith('ep_'):
            episode_id = part
            break
    if episode_id is None:
        episode_id = Path(states_path).stem

    # Load states
    states = load_states(states_path)

    # Run physics checks for this world
    checks = {}
    for check_name, check_fn in PHYSICS_CHECK_FUNCTIONS.get(world_id, []):
        try:
            pass_rate = check_fn(states)
            checks[check_name] = pass_rate
        except Exception as e:
            if verbose:
                print(f"  [WARNING] Check '{check_name}' failed: {e}")
            checks[check_name] = 0.0

    physics_consistency = sum(checks.values()) / len(checks) if checks else 0.0

    # Determine if checks pass (threshold 0.8)
    passed = all(rate >= 0.8 for rate in checks.values()) if checks else False

    result = {
        'world_id': world_id,
        'episode_id': episode_id,
        'states_path': states_path,
        'num_frames': len(states),
        'trajectory_mse': 0.0,  # Placeholder: would need ground truth
        'physics_checks': checks,
        'physics_consistency': physics_consistency,
        'passed': passed,
    }

    if verbose:
        print(f"\n[{world_id}/{episode_id}]")
        print(f"  Frames: {len(states)}")
        for check_name, rate in checks.items():
            status = "✓" if rate >= 0.8 else "✗"
            print(f"  {check_name}: {rate:.3f} {status}")
        print(f"  Physics consistency: {physics_consistency:.3f}")
        print(f"  Overall: {'PASS' if passed else 'FAIL'}")

    return result


def evaluate_world(data_root: str, world_id: str, verbose: bool = True) -> List[Dict[str, Any]]:
    """Evaluate all episodes for a given world."""
    world_dir = os.path.join(data_root, 'worlds', world_id, 'episodes')
    if not os.path.exists(world_dir):
        if verbose:
            print(f"[WARNING] World directory not found: {world_dir}")
        return []

    results = []
    for ep_dir in sorted(os.listdir(world_dir)):
        ep_path = os.path.join(world_dir, ep_dir)
        states_path = os.path.join(ep_path, 'states.jsonl')
        if os.path.exists(states_path):
            if verbose:
                print(f"Evaluating {world_id}/{ep_dir}...")
            result = evaluate_episode(world_id, states_path, verbose=verbose)
            results.append(result)

    return results


def evaluate_dataset(data_root: str, world_ids: List[str], verbose: bool = True) -> Dict[str, Any]:
    """Evaluate all specified worlds in the dataset."""
    all_results = {}
    summary = {}

    for world_id in world_ids:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Evaluating world: {world_id}")
            print(f"{'='*60}")

        results = evaluate_world(data_root, world_id, verbose=verbose)
        all_results[world_id] = results

        if results:
            avg_consistency = sum(r['physics_consistency'] for r in results) / len(results)
            pass_rate = sum(1 for r in results if r['passed']) / len(results)
            summary[world_id] = {
                'num_episodes': len(results),
                'avg_physics_consistency': avg_consistency,
                'pass_rate': pass_rate,
            }

            if verbose:
                print(f"\nSummary for {world_id}:")
                print(f"  Episodes evaluated: {len(results)}")
                print(f"  Average physics consistency: {avg_consistency:.3f}")
                print(f"  Pass rate (all checks >= 0.8): {pass_rate:.3f}")
        else:
            summary[world_id] = {'num_episodes': 0, 'error': 'No episodes found'}

    # Overall summary
    overall = {
        'worlds_evaluated': len(world_ids),
        'total_episodes': sum(s.get('num_episodes', 0) for s in summary.values()),
        'summary': summary,
    }

    return overall


def main():
    parser = argparse.ArgumentParser(description='Minimal evaluation for Phy-Bench')
    parser.add_argument('--data-root', type=str, default='data/',
                       help='Root directory of dataset')
    parser.add_argument('--worlds', nargs='+', default=['all'],
                       help='World IDs to evaluate, or "all" for all worlds')
    parser.add_argument('--episode', type=str,
                       help='Evaluate a single episode (provide path to states.jsonl)')
    parser.add_argument('--world-id', type=str,
                       help='World ID for single episode evaluation (required with --episode)')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='Print detailed results')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress output (overrides --verbose)')
    parser.add_argument('--output', type=str,
                       help='Save results to JSON file')

    args = parser.parse_args()

    verbose = args.verbose and not args.quiet

    # Single episode mode
    if args.episode:
        if not args.world_id:
            print("ERROR: --world-id is required when using --episode")
            sys.exit(1)
        if not os.path.exists(args.episode):
            print(f"ERROR: Episode file not found: {args.episode}")
            sys.exit(1)

        result = evaluate_episode(args.world_id, args.episode, verbose=verbose)
        results = {'episode': result}

    # Full dataset mode
    else:
        data_root = args.data_root
        if not os.path.exists(data_root):
            print(f"ERROR: Data root not found: {data_root}")
            sys.exit(1)

        # Determine which worlds to evaluate
        if 'all' in args.worlds:
            # List world directories
            worlds_dir = os.path.join(data_root, 'worlds')
            if os.path.exists(worlds_dir):
                world_ids = [d for d in os.listdir(worlds_dir)
                            if os.path.isdir(os.path.join(worlds_dir, d))]
            else:
                world_ids = []
        else:
            world_ids = args.worlds

        if not world_ids:
            print("ERROR: No worlds found to evaluate")
            sys.exit(1)

        results = evaluate_dataset(data_root, world_ids, verbose=verbose)

    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {args.output}")

    # Return appropriate exit code
    if 'episode' in results:
        sys.exit(0 if results['episode']['passed'] else 1)
    else:
        # Check overall pass rate
        all_passed = True
        for world_summary in results.get('summary', {}).values():
            if world_summary.get('pass_rate', 0) < 1.0:
                all_passed = False
                break
        sys.exit(0 if all_passed else 1)


if __name__ == '__main__':
    main()