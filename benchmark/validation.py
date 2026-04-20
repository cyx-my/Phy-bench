"""
Data validation for Phy-Bench.

Validates generated episodes for correctness and basic physics consistency.
Can be used after dataset generation or as a standalone check.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from .evaluator import (
    check_upward_acceleration,
    check_centripetal_force,
    check_speed_doubles_on_bounce,
    PHYSICS_CHECK_FUNCTIONS,
)


def validate_states_file(states_path: str) -> Tuple[bool, List[str]]:
    """
    Validate the structure and content of a states.jsonl file.

    Returns:
        (is_valid, error_messages)
    """
    errors = []

    if not os.path.exists(states_path):
        return False, [f"File does not exist: {states_path}"]

    try:
        with open(states_path, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        return False, [f"Cannot read file: {e}"]

    if not lines:
        return False, ["File is empty"]

    frames = []
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            frame = json.loads(line)
        except json.JSONDecodeError as e:
            errors.append(f"Line {i+1}: Invalid JSON: {e}")
            continue

        # Required top-level fields
        required = ['frame', 'timestamp', 'objects']
        for field in required:
            if field not in frame:
                errors.append(f"Line {i+1}: Missing required field '{field}'")

        # Validate objects array
        if 'objects' in frame:
            for j, obj in enumerate(frame['objects']):
                obj_required = ['id', 'position_x', 'position_y', 'velocity_x', 'velocity_y']
                for field in obj_required:
                    if field not in obj:
                        errors.append(f"Line {i+1}, object {j}: Missing field '{field}'")

        frames.append(frame)

    # Check frame ordering
    if frames:
        frames_sorted = sorted(frames, key=lambda f: f['frame'])
        if frames != frames_sorted:
            errors.append("Frames are not in ascending order by frame number")

        # Check for consistent object count
        obj_counts = [len(f['objects']) for f in frames]
        if min(obj_counts) != max(obj_counts):
            errors.append(f"Inconsistent object count across frames: {obj_counts}")

    return len(errors) == 0, errors


def validate_episode(episode_dir: str, world_id: str) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate a single episode directory.

    Checks:
    1. Required files exist (states.jsonl, episode_meta.json)
    2. states.jsonl is well-formed
    3. Basic physics checks for the world (if available)

    Returns:
        (is_valid, details_dict)
    """
    details = {
        'world_id': world_id,
        'episode_dir': episode_dir,
        'errors': [],
        'warnings': [],
        'physics_checks': {},
    }

    # Check required files
    states_path = os.path.join(episode_dir, 'states.jsonl')
    meta_path = os.path.join(episode_dir, 'episode_meta.json')

    if not os.path.exists(states_path):
        details['errors'].append(f"Missing states.jsonl")
    if not os.path.exists(meta_path):
        details['warnings'].append(f"Missing episode_meta.json")

    # Validate states file
    if os.path.exists(states_path):
        valid, errors = validate_states_file(states_path)
        if not valid:
            details['errors'].extend(errors)
        else:
            details['states_valid'] = True

            # Load states for physics checks
            try:
                with open(states_path, 'r') as f:
                    states = [json.loads(line) for line in f if line.strip()]
                details['num_frames'] = len(states)

                # Run physics checks if available
                if world_id in PHYSICS_CHECK_FUNCTIONS:
                    for check_name, check_fn in PHYSICS_CHECK_FUNCTIONS[world_id]:
                        try:
                            score = check_fn(states)
                            details['physics_checks'][check_name] = score
                            if score < 0.8:
                                details['warnings'].append(
                                    f"Physics check '{check_name}' score low: {score:.3f}"
                                )
                        except Exception as e:
                            details['warnings'].append(
                                f"Physics check '{check_name}' failed: {e}"
                            )
            except Exception as e:
                details['errors'].append(f"Failed to load states for physics checks: {e}")

    # Validate meta file
    if os.path.exists(meta_path):
        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            details['meta'] = meta
            # Check required meta fields
            required_meta = ['episode_id', 'world_id', 'seed']
            for field in required_meta:
                if field not in meta:
                    details['warnings'].append(f"Meta missing field '{field}'")
        except Exception as e:
            details['warnings'].append(f"Failed to parse episode_meta.json: {e}")

    is_valid = len(details['errors']) == 0
    return is_valid, details


def validate_world(data_root: str, world_id: str) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate all episodes for a given world.

    Returns:
        (all_episodes_valid, summary_dict)
    """
    world_dir = os.path.join(data_root, 'worlds', world_id, 'episodes')
    if not os.path.exists(world_dir):
        return False, {'error': f"World directory not found: {world_dir}"}

    episodes = []
    all_valid = True
    for ep_dir_name in sorted(os.listdir(world_dir)):
        ep_dir = os.path.join(world_dir, ep_dir_name)
        if not os.path.isdir(ep_dir):
            continue

        ep_valid, details = validate_episode(ep_dir, world_id)
        episodes.append({
            'episode_id': ep_dir_name,
            'valid': ep_valid,
            'errors': details['errors'],
            'warnings': details['warnings'],
            'physics_checks': details.get('physics_checks', {}),
        })
        if not ep_valid:
            all_valid = False

    summary = {
        'world_id': world_id,
        'episodes_checked': len(episodes),
        'valid_episodes': sum(1 for ep in episodes if ep['valid']),
        'episodes': episodes,
    }
    return all_valid, summary


def validate_dataset(data_root: str, world_ids: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Validate the entire dataset.

    Args:
        data_root: Root directory of the dataset
        world_ids: List of world IDs to validate, or None for all worlds found

    Returns:
        Summary dictionary
    """
    worlds_dir = os.path.join(data_root, 'worlds')
    if not os.path.exists(worlds_dir):
        return {'error': f"Worlds directory not found: {worlds_dir}"}

    if world_ids is None:
        world_ids = [d for d in os.listdir(worlds_dir)
                    if os.path.isdir(os.path.join(worlds_dir, d))]

    results = {}
    all_valid = True
    for world_id in world_ids:
        world_valid, summary = validate_world(data_root, world_id)
        results[world_id] = summary
        if not world_valid:
            all_valid = False

    # Check for dataset index
    index_path = os.path.join(data_root, 'dataset_index.json')
    if os.path.exists(index_path):
        try:
            with open(index_path, 'r') as f:
                index = json.load(f)
            results['dataset_index'] = index
        except Exception as e:
            results['dataset_index_error'] = str(e)

    overall = {
        'all_worlds_valid': all_valid,
        'worlds_checked': len(world_ids),
        'results': results,
    }
    return overall


def print_validation_summary(summary: Dict[str, Any]) -> None:
    """Print a human-readable validation summary."""
    if 'error' in summary:
        print(f"ERROR: {summary['error']}")
        return

    print(f"\n{'='*70}")
    print(f"Dataset Validation Summary")
    print(f"{'='*70}")

    for world_id, world_summary in summary.get('results', {}).items():
        if 'error' in world_summary:
            print(f"\n[{world_id}] ERROR: {world_summary['error']}")
            continue

        print(f"\n[{world_id}]")
        print(f"  Episodes checked: {world_summary['episodes_checked']}")
        print(f"  Valid episodes: {world_summary['valid_episodes']}")

        for ep in world_summary['episodes']:
            if not ep['valid'] or ep['warnings']:
                status = "✓" if ep['valid'] else "✗"
                print(f"    {ep['episode_id']}: {status}")
                for err in ep['errors']:
                    print(f"      ERROR: {err}")
                for warn in ep['warnings']:
                    print(f"      WARNING: {warn}")
                if ep['physics_checks']:
                    for check, score in ep['physics_checks'].items():
                        print(f"      {check}: {score:.3f}")

    if summary.get('all_worlds_valid'):
        print(f"\n✓ All worlds passed validation!")
    else:
        print(f"\n✗ Some worlds failed validation.")
    print(f"{'='*70}\n")