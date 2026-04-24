#!/usr/bin/env python3
"""
MCQ evaluation script for InternVL3-8B on Phy-Bench.

Usage:
    python eval_internvl3_8b.py --video examples/worlds/black_hole/episodes/ep_001/video.mp4 --world black_hole
    python eval_internvl3_8b.py --data-root examples --world black_hole
    python eval_internvl3_8b.py --data-root examples --all
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from benchmark.questions import WORLD_MCQS, MCQ
from benchmark.tasks import MCQ_INDUCTION_PROMPT

MODEL_PATH = "/home/cyx/.cache/modelscope/hub/models/OpenGVLab/InternVL3-8B"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Image preprocessing (InternVL3 official pipeline)
# ---------------------------------------------------------------------------

def build_transform(input_size: int = 448):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def dynamic_preprocess(image: Image.Image, min_num: int = 1, max_num: int = 1,
                        image_size: int = 448, use_thumbnail: bool = True) -> List[Image.Image]:
    """Tile image into patches matching the closest aspect ratio."""
    orig_w, orig_h = image.size
    aspect = orig_w / orig_h
    target_ratios = sorted(
        {(i, j) for n in range(min_num, max_num + 1)
         for i in range(1, n + 1) for j in range(1, n + 1)
         if min_num <= i * j <= max_num},
        key=lambda x: x[0] * x[1],
    )
    best = min(target_ratios, key=lambda r: abs(aspect - r[0] / r[1]))
    tw, th = image_size * best[0], image_size * best[1]
    resized = image.resize((tw, th))
    tiles = []
    for i in range(best[0] * best[1]):
        col, row = i % best[0], i // best[0]
        box = (col * image_size, row * image_size, (col + 1) * image_size, (row + 1) * image_size)
        tiles.append(resized.crop(box))
    if use_thumbnail and len(tiles) != 1:
        tiles.append(image.resize((image_size, image_size)))
    return tiles


def preprocess_frame(pil_img: Image.Image, input_size: int = 448,
                      max_num: int = 1) -> torch.Tensor:
    """Convert a PIL frame to pixel_values tensor (tiles × 3 × H × W)."""
    transform = build_transform(input_size)
    tiles = dynamic_preprocess(pil_img, image_size=input_size, use_thumbnail=True, max_num=max_num)
    return torch.stack([transform(t) for t in tiles])


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def _patch_tied_weights(model):
    """Patch missing all_tied_weights_keys for InternVL3 custom model class."""
    import torch.nn as nn
    for m in model.modules():
        if not hasattr(m, "all_tied_weights_keys"):
            m.all_tied_weights_keys = {}


def load_model(model_path: str):
    print(f"Loading model from {model_path} ...")
    # Monkey-patch PreTrainedModel.__init__ to inject all_tied_weights_keys before
    # transformers 5.x tries to access it during weight loading.
    from transformers import PreTrainedModel
    original_init = PreTrainedModel.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if not hasattr(self, "all_tied_weights_keys"):
            self.all_tied_weights_keys = {}

    PreTrainedModel.__init__ = patched_init

    try:
        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=False,
            trust_remote_code=True,
            device_map="auto",
        ).eval()
    finally:
        PreTrainedModel.__init__ = original_init

    _patch_tied_weights(model)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    print(f"Model loaded: {type(model).__name__}")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

def extract_frames(video_path: str, num_segments: int = 8) -> List[Image.Image]:
    """Extract evenly spaced frames using cv2 (no decord needed)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = min(num_segments, total)
    seg = total / n
    indices = [int(seg / 2 + seg * i) for i in range(n)]
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    print(f"  Extracted {len(frames)} frames (total={total}, fps={fps:.1f})")
    return frames


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(model, tokenizer, video_path: str, prompt: str,
                  num_segments: int = 8) -> str:
    frames = extract_frames(video_path, num_segments=num_segments)
    if not frames:
        raise ValueError(f"No frames extracted from {video_path}")

    # Stack all frame pixel_values; each frame → max_num=1 tile + thumbnail = 2 tiles
    pv_list = [preprocess_frame(f, max_num=1) for f in frames]
    num_patches_list = [pv.shape[0] for pv in pv_list]
    pixel_values = torch.cat(pv_list).to(torch.bfloat16).to(model.device)

    # Build question with Frame{i}: <image> prefix (InternVL3 video convention)
    video_prefix = "".join(f"Frame{i+1}: <image>\n" for i in range(len(frames)))
    question = video_prefix + prompt

    generation_config = dict(max_new_tokens=512, do_sample=False)
    response = model.chat(
        tokenizer,
        pixel_values,
        question,
        generation_config,
        num_patches_list=num_patches_list,
        history=None,
        return_history=False,
    )
    return response.strip()


# ---------------------------------------------------------------------------
# MCQ helpers (shared logic)
# ---------------------------------------------------------------------------

def build_mcq_prompt(mcqs: List[MCQ]) -> str:
    lines = []
    for i, mcq in enumerate(mcqs, 1):
        lines.append(f"{i}. {mcq.question}")
        for letter in ["A", "B", "C", "D"]:
            lines.append(f"   {letter}. {mcq.options[letter]}")
        lines.append("")
    return MCQ_INDUCTION_PROMPT.format(questions_text="\n".join(lines))


def parse_answers(output: str, num_questions: int) -> Dict[int, str]:
    candidates = {}
    patterns = [
        r'(?:^|\n)\s*(?:Q|问题)?\s*(\d+)\s*[.）:：\)\]\s]+\s*([A-Da-d])\b',
        r'(?:^|\n)\s*[\[\(（]\s*(\d+)\s*[\]\)）]\s*([A-Da-d])\b',
    ]
    for pat in patterns:
        for m in re.finditer(pat, output):
            qnum, letter = int(m.group(1)), m.group(2).upper()
            if 1 <= qnum <= num_questions:
                candidates[qnum] = letter
    return {idx - 1: letter for idx, letter in candidates.items()}


def score_answers(answers: Dict[int, str], mcqs: List[MCQ]) -> Dict[str, Any]:
    per_q = []
    source_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
    for i, mcq in enumerate(mcqs):
        model_ans = answers.get(i)
        correct = model_ans == mcq.correct_answer if model_ans else False
        per_q.append({
            "idx": i,
            "question": mcq.question,
            "correct": correct,
            "model_answer": model_ans,
            "correct_answer": mcq.correct_answer,
            "source": mcq.source,
        })
        source_counts[mcq.source]["total"] += 1
        if correct:
            source_counts[mcq.source]["correct"] += 1

    num_correct = sum(1 for q in per_q if q["correct"])
    return {
        "per_question": per_q,
        "num_correct": num_correct,
        "num_total": len(mcqs),
        "num_answered": sum(1 for q in per_q if q["model_answer"] is not None),
        "accuracy": num_correct / len(mcqs) if mcqs else 0.0,
        "source_breakdown": {
            src: {"correct": v["correct"], "total": v["total"],
                  "accuracy": v["correct"] / v["total"] if v["total"] else 0.0}
            for src, v in source_counts.items()
        },
    }


# ---------------------------------------------------------------------------
# Evaluate one episode
# ---------------------------------------------------------------------------

def evaluate_episode(model, tokenizer, video: str, world_id: str,
                     num_segments: int = 8) -> Dict[str, Any]:
    mcqs = WORLD_MCQS.get(world_id)
    if not mcqs:
        raise ValueError(f"No MCQs found for world: {world_id}")

    print(f"  World: {world_id}, MCQs: {len(mcqs)}, Video: {video}")
    prompt = build_mcq_prompt(mcqs)
    response = run_inference(model, tokenizer, video, prompt, num_segments=num_segments)
    print(f"  Response:\n{response}\n")

    answers = parse_answers(response, len(mcqs))
    scores = score_answers(answers, mcqs)
    print(f"  Score: {scores['num_correct']}/{scores['num_total']} ({scores['accuracy']:.1%})")
    return {
        "video": video,
        "world_id": world_id,
        "model_response": response,
        "parsed_answers": {str(k): v for k, v in answers.items()},
        **scores,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def find_episodes(data_root: str, world_id: str = None) -> List[tuple]:
    episodes = []
    worlds_dir = Path(data_root) / "worlds"
    if not worlds_dir.exists():
        return episodes
    for world_dir in sorted(worlds_dir.iterdir()):
        if not world_dir.is_dir():
            continue
        wid = world_dir.name
        if world_id and wid != world_id:
            continue
        for ep_dir in sorted((world_dir / "episodes").glob("ep_*")):
            vp = ep_dir / "video.mp4"
            if vp.exists():
                episodes.append((str(vp), wid))
    return episodes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", help="Path to a single video file")
    parser.add_argument("--world", help="World ID")
    parser.add_argument("--data-root", default="examples")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--num-segments", type=int, default=8, help="Frames to sample per video")
    parser.add_argument("--model-path", default=MODEL_PATH)
    args = parser.parse_args()

    model, tokenizer = load_model(args.model_path)

    if args.video:
        if not args.world:
            parser.error("--world is required when using --video")
        episodes = [(args.video, args.world)]
    elif args.all or args.world:
        episodes = find_episodes(args.data_root, args.world)
        if not episodes:
            print(f"No episodes found in {args.data_root}")
            sys.exit(1)
    else:
        parser.error("Provide --video or --data-root with --all/--world")

    print(f"\nEvaluating {len(episodes)} episode(s)...\n")

    all_results = []
    for video_path, world_id in episodes:
        print(f"--- {video_path} ---")
        result = evaluate_episode(model, tokenizer, video_path, world_id,
                                  num_segments=args.num_segments)
        all_results.append(result)

    total_correct = sum(r["num_correct"] for r in all_results)
    total_questions = sum(r["num_total"] for r in all_results)
    overall_acc = total_correct / total_questions if total_questions else 0.0

    print(f"\n{'='*50}")
    print(f"Overall: {total_correct}/{total_questions} ({overall_acc:.1%})")

    world_stats: Dict[str, Dict] = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in all_results:
        world_stats[r["world_id"]]["correct"] += r["num_correct"]
        world_stats[r["world_id"]]["total"] += r["num_total"]
    for wid, s in world_stats.items():
        acc = s["correct"] / s["total"] if s["total"] else 0.0
        print(f"  {wid}: {s['correct']}/{s['total']} ({acc:.1%})")

    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    world_tag = args.world or "all"
    out_path = f"results/internvl3_8b_mcq_{world_tag}_{timestamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": args.model_path,
            "num_segments": args.num_segments,
            "overall_accuracy": overall_acc,
            "total_correct": total_correct,
            "total_questions": total_questions,
            "world_breakdown": dict(world_stats),
            "episodes": all_results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
