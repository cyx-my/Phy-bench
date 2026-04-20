"""
Data schema for Phy-Bench dataset.

Directory structure:
  data/
  └── worlds/
      └── <world_id>/
          ├── world_meta.json          # World-level metadata
          ├── rule_description.json    # Natural language rule descriptions
          └── episodes/
              └── ep_<NNN>/
                  ├── episode_meta.json   # Episode-level metadata
                  ├── video.mp4           # Rendered video
                  └── states.jsonl        # Per-frame physics state (one JSON per line)

Each episode contains:
  - video.mp4: The rendered simulation video
  - states.jsonl: Frame-by-frame state log, one dict per line:
      {
        "frame": int,
        "timestamp": float,             # seconds from episode start
        "objects": [
          {
            "id": int,
            "position_x": float,
            "position_y": float,
            "velocity_x": float,
            "velocity_y": float,
            ... (world-specific extra fields like "mass", "radius", etc.)
          }
        ],
        "world_state": {                 # World-level state (e.g., gravity direction)
          ...
        },
        "events": [                      # Discrete events in this frame
          {"type": "collision", "obj_ids": [0, 1]},
          {"type": "teleport", "obj_id": 2},
          ...
        ]
      }
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import json
import os


@dataclass
class EpisodeMeta:
    episode_id: str                # e.g. "ep_001"
    world_id: str
    seed: int
    num_objects: int
    duration_sec: float
    fps: int
    width: int
    height: int
    initial_conditions: Dict[str, Any]   # Snapshot of initial state
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "world_id": self.world_id,
            "seed": self.seed,
            "num_objects": self.num_objects,
            "duration_sec": self.duration_sec,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "initial_conditions": self.initial_conditions,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EpisodeMeta":
        return cls(**d)


@dataclass
class WorldMeta:
    world_id: str
    name: str
    num_episodes: int
    split: Dict[str, List[str]]   # {"train": ["ep_001", ...], "test": [...]}
    created_at: str               # ISO datetime string

    def to_dict(self) -> dict:
        return {
            "world_id": self.world_id,
            "name": self.name,
            "num_episodes": self.num_episodes,
            "split": self.split,
            "created_at": self.created_at,
        }


@dataclass
class FrameState:
    """Single-frame state record (maps to one line in states.jsonl)."""
    frame: int
    timestamp: float
    objects: List[Dict[str, Any]]
    world_state: Dict[str, Any]
    events: List[Dict[str, Any]]

    def to_dict(self) -> dict:
        return {
            "frame": self.frame,
            "timestamp": self.timestamp,
            "objects": self.objects,
            "world_state": self.world_state,
            "events": self.events,
        }

    def to_json_line(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: dict) -> "FrameState":
        return cls(**d)


# ---------------------------------------------------------------------------
# File path helpers
# ---------------------------------------------------------------------------

def world_dir(data_root: str, world_id: str) -> str:
    return os.path.join(data_root, "worlds", world_id)

def episode_dir(data_root: str, world_id: str, episode_id: str) -> str:
    return os.path.join(world_dir(data_root, world_id), "episodes", episode_id)

def video_path(data_root: str, world_id: str, episode_id: str) -> str:
    return os.path.join(episode_dir(data_root, world_id, episode_id), "video.mp4")

def states_path(data_root: str, world_id: str, episode_id: str) -> str:
    return os.path.join(episode_dir(data_root, world_id, episode_id), "states.jsonl")

def episode_meta_path(data_root: str, world_id: str, episode_id: str) -> str:
    return os.path.join(episode_dir(data_root, world_id, episode_id), "episode_meta.json")

def world_meta_path(data_root: str, world_id: str) -> str:
    return os.path.join(world_dir(data_root, world_id), "world_meta.json")

def rule_description_path(data_root: str, world_id: str) -> str:
    return os.path.join(world_dir(data_root, world_id), "rule_description.json")


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def save_episode_meta(meta: EpisodeMeta, data_root: str) -> None:
    path = episode_meta_path(data_root, meta.world_id, meta.episode_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(meta.to_dict(), f, indent=2, ensure_ascii=False)

def load_episode_meta(data_root: str, world_id: str, episode_id: str) -> EpisodeMeta:
    path = episode_meta_path(data_root, world_id, episode_id)
    with open(path) as f:
        return EpisodeMeta.from_dict(json.load(f))

def save_world_meta(meta: WorldMeta, data_root: str) -> None:
    path = world_meta_path(data_root, meta.world_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(meta.to_dict(), f, indent=2, ensure_ascii=False)

def save_rule_description(world_id: str, rule_desc, data_root: str) -> None:
    """Save RuleDescription (from world_configs.py) to JSON."""
    path = rule_description_path(data_root, world_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({
            "world_id": world_id,
            "concise": rule_desc.concise,
            "standard": rule_desc.standard,
            "verbose": rule_desc.verbose,
        }, f, indent=2, ensure_ascii=False)

class StatesWriter:
    """Context manager for writing states.jsonl line by line during simulation."""
    def __init__(self, data_root: str, world_id: str, episode_id: str):
        self.path = states_path(data_root, world_id, episode_id)
        self._file = None

    def __enter__(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._file = open(self.path, "w")
        return self

    def write(self, state: FrameState) -> None:
        self._file.write(state.to_json_line() + "\n")

    def __exit__(self, *args):
        if self._file:
            self._file.close()

def load_states(data_root: str, world_id: str, episode_id: str) -> List[FrameState]:
    path = states_path(data_root, world_id, episode_id)
    states = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                states.append(FrameState.from_dict(json.loads(line)))
    return states
