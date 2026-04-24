#!/usr/bin/env python3
"""
MCQ evaluation script for Qwen2.5-VL-7B-Instruct on Phy-Bench.

Usage:
    python eval_qwen25vl_7b.py --video examples/worlds/black_hole/episodes/ep_001/video.mp4 --world black_hole
    python eval_qwen25vl_7b.py --data-root examples --world black_hole
    python eval_qwen25vl_7b.py --data-root examples --world gravity_merry_go_round
    python eval_qwen25vl_7b.py --data-root examples --all
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
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from benchmark.questions import WORLD_MCQS, MCQ
from benchmark.tasks import MCQ_INDUCTION_PROMPT

MODEL_PATH = "/home/cyx/.cache/modelscope/hub/models/Qwen/Qwen2___5-VL-7B-Instruct"


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_model(model_path: str):
    print(f"Loading model from {model_path} ...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(model_path)
    device = next(model.parameters()).device
    print(f"Model loaded: {type(model).__name__} on {device}")
    return model, processor


# ---------------------------------------------------------------------------
# Frame extraction (cv2-based, avoids broken torchvision/decord backends)
# ---------------------------------------------------------------------------

def extract_frames(video_path: str, fps: float = 1.0) -> List[np.ndarray]:
    """Extract frames at the given fps using cv2. Returns list of uint8 RGB arrays."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, round(video_fps / fps))
    indices = list(range(0, total, step))
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    print(f"  Extracted {len(frames)} frames at {fps} fps (video fps={video_fps:.1f})")
    return frames


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(model, processor, video_path: str, prompt: str, fps: float = 1.0) -> str:
    frames = extract_frames(video_path, fps=fps)
    if not frames:
        raise ValueError(f"No frames extracted from {video_path}")

    # Pass frames as a list of images under "video" type — Qwen2.5-VL supports this
    # Each frame is a PIL Image; processor handles the vision token expansion
    pil_frames = [Image.fromarray(f) for f in frames]

    # Build messages with frames as individual images (interleaved image list)
    content = [{"type": "image", "image": img} for img in pil_frames]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=pil_frames,
        padding=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)

    trimmed = [out[len(inp):] for inp, out in zip(inputs["input_ids"], generated_ids)]
    response = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return response.strip()


# ---------------------------------------------------------------------------
# MCQ helpers
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
    """Extract answer letters from model output. Returns 0-indexed dict."""
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
    num_answered = sum(1 for q in per_q if q["model_answer"] is not None)
    return {
        "per_question": per_q,
        "num_correct": num_correct,
        "num_total": len(mcqs),
        "num_answered": num_answered,
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

def evaluate_episode(model, processor, video: str, world_id: str, fps: float = 1.0) -> Dict[str, Any]:
    mcqs = WORLD_MCQS.get(world_id)
    if not mcqs:
        raise ValueError(f"No MCQs found for world: {world_id}")

    print(f"  World: {world_id}, MCQs: {len(mcqs)}, Video: {video}")
    prompt = build_mcq_prompt(mcqs)
    response = run_inference(model, processor, video, prompt, fps=fps)
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
    """Return list of (video_path, world_id) tuples."""
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
    parser.add_argument("--world", help="World ID (required with --video or --data-root)")
    parser.add_argument("--data-root", default="examples", help="Dataset root directory")
    parser.add_argument("--all", action="store_true", help="Evaluate all episodes in data-root")
    parser.add_argument("--fps", type=float, default=1.0, help="FPS for video sampling")
    parser.add_argument("--model-path", default=MODEL_PATH)
    args = parser.parse_args()

    model, processor = load_model(args.model_path)

    # Collect episodes to evaluate
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
        result = evaluate_episode(model, processor, video_path, world_id, fps=args.fps)
        all_results.append(result)

    # Aggregate
    total_correct = sum(r["num_correct"] for r in all_results)
    total_questions = sum(r["num_total"] for r in all_results)
    overall_acc = total_correct / total_questions if total_questions else 0.0

    print(f"\n{'='*50}")
    print(f"Overall: {total_correct}/{total_questions} ({overall_acc:.1%})")

    # Per-world breakdown
    world_stats: Dict[str, Dict] = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in all_results:
        world_stats[r["world_id"]]["correct"] += r["num_correct"]
        world_stats[r["world_id"]]["total"] += r["num_total"]
    for wid, s in world_stats.items():
        acc = s["correct"] / s["total"] if s["total"] else 0.0
        print(f"  {wid}: {s['correct']}/{s['total']} ({acc:.1%})")

    # Save results
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    world_tag = args.world or "all"
    out_path = f"results/qwen25vl7b_mcq_{world_tag}_{timestamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": args.model_path,
            "fps": args.fps,
            "overall_accuracy": overall_acc,
            "total_correct": total_correct,
            "total_questions": total_questions,
            "world_breakdown": dict(world_stats),
            "episodes": all_results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
