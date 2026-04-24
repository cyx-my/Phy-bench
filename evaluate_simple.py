#!/usr/bin/env python3
"""
Simplified evaluation script for testing U-MARVEL-Qwen3VL-4B-Instruct on Phy-Bench.

Evaluates using multiple-choice questions (MCQ) for automatic, deterministic scoring.
Also supports open-ended induction and prediction tasks.

Usage:
    # MCQ evaluation (default)
    python evaluate_simple.py --video examples/worlds/black_hole/episodes/ep_001/video.mp4 --world black_hole

    # Evaluate all episodes in the examples dataset
    python evaluate_simple.py --data-root examples --all

    # Evaluate specific world
    python evaluate_simple.py --data-root examples --world black_hole

    # Open-ended induction mode (original behavior)
    python evaluate_simple.py --data-root examples --all --eval-mode open_ended
"""

import argparse
import json
import os
import re
import sys
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import cv2
import torch
from PIL import Image
from transformers import AutoProcessor

# Add project root to sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from benchmark.evaluator import (
    evaluate_prediction,
    PredictionResult,
    PHYSICS_CHECK_FUNCTIONS,
    check_upward_acceleration,
    check_centripetal_force,
    check_speed_doubles_on_bounce,
)
from benchmark.tasks import INDUCTION_PROMPT, MCQ_INDUCTION_PROMPT, PREDICTION_PROMPT_ZERO_SHOT
from benchmark.data_schema import load_states, episode_dir, video_path
from benchmark.questions import WORLD_MCQS, MCQ


# ---------------------------------------------------------------------------
# Model loading (from evaluate4.20.py — proven to work)
# ---------------------------------------------------------------------------

def load_model(model_path: str, device_map: str = "auto"):
    """Load U-MARVEL-Qwen3VL-4B-Instruct model and processor."""
    print(f"Loading model from {model_path}...")

    # Try Qwen3VL specific class first
    try:
        from transformers import Qwen3VLForConditionalGeneration
        model_class = Qwen3VLForConditionalGeneration
    except ImportError:
        from transformers import AutoModelForCausalLM
        model_class = AutoModelForCausalLM

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    print(f"Loaded processor: {type(processor).__name__}")

    model = model_class.from_pretrained(
        model_path,
        device_map=device_map,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        trust_remote_code=True,
    )
    model.eval()
    print(f"Loaded model: {type(model).__name__}")
    print(f"Model has generate: {hasattr(model, 'generate')}")
    print(f"Model has chat: {hasattr(model, 'chat')}")
    return model, processor


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

def extract_video_frames(video_path: str, num_frames: int = 8) -> List[Image.Image]:
    """Extract evenly spaced frames from a video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    print(f"  Video: {total_frames} frames, {fps:.1f} fps, {duration:.1f}s")

    if total_frames <= num_frames:
        indices = list(range(total_frames))
    else:
        step = max(1, total_frames // num_frames)
        indices = list(range(0, total_frames, step))[:num_frames]

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))

    cap.release()
    print(f"  Extracted {len(frames)} frames")
    return frames


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(
    model,
    processor,
    frames: List[Image.Image],
    prompt_text: str,
    max_new_tokens: int = 512,
) -> str:
    """Run inference with the model, trying chat then generate."""
    # Prepare messages for Qwen-VL-Chat format
    messages = [{"role": "user", "content": []}]
    for img in frames:
        messages[0]["content"].append({"type": "image", "image": img})
    messages[0]["content"].append({"type": "text", "text": prompt_text})

    # Strategy 1: chat method (most straightforward for Qwen3VL models)
    if hasattr(model, 'chat'):
        try:
            # Use processor.tokenizer if available, else processor
            tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
            response = model.chat(
                messages=messages,
                tokenizer=tokenizer,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
            )
            if isinstance(response, str):
                return response
            elif isinstance(response, dict) and 'text' in response:
                return response['text']
            elif hasattr(response, 'text'):
                return response.text
            return str(response)
        except Exception as e:
            import traceback
            print(f"  Chat method failed: {e}")
            traceback.print_exc()
            print("  Falling back to generate...")

    # Strategy 2: generate method with proper image token handling
    print("  Using generate method for inference")

    # Try to process messages directly (some processors support this)
    try:
        inputs = processor(messages, return_tensors="pt", padding=True)
        print("  Successfully processed messages directly with processor")
    except Exception as e1:
        print(f"  Cannot process messages directly: {e1}")

        # Fallback: Try to apply chat template if available
        apply_chat_template_obj = None
        if hasattr(processor, 'apply_chat_template'):
            apply_chat_template_obj = processor
        elif hasattr(processor, 'tokenizer') and hasattr(processor.tokenizer, 'apply_chat_template'):
            apply_chat_template_obj = processor.tokenizer

        if apply_chat_template_obj:
            try:
                text = apply_chat_template_obj.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                print(f"  Applied chat template, text length: {len(text)}")
                print(f"  Text preview: {repr(text[:200])}")

                # Try to process with images
                try:
                    inputs = processor(text=[text], images=frames, return_tensors="pt", padding=True)
                    print("  Successfully processed text + images")
                except Exception as e2:
                    print(f"  Cannot process text+images: {e2}")
                    # Try without images (last resort)
                    inputs = processor(text=[text], return_tensors="pt", padding=True)
                    print("  Warning: Processing without images - model may fail")
            except Exception as e3:
                print(f"  Chat template failed: {e3}")
                # Final fallback: just text
                inputs = processor(text=[prompt_text], return_tensors="pt", padding=True)
                print("  Using plain text only")
        else:
            # No chat template available, try to process with images
            try:
                inputs = processor(text=[prompt_text], images=frames, return_tensors="pt", padding=True)
                print("  Processed text + images without chat template")
            except Exception as e4:
                print(f"  Cannot process with images: {e4}")
                inputs = processor(text=[prompt_text], return_tensors="pt", padding=True)
                print("  Using plain text only")

    # Move inputs to model device
    inputs = {k: v.to(model.device) for k, v in inputs.items() if hasattr(v, 'to')}

    # Debug: show input keys and shapes
    print(f"  Input keys: {list(inputs.keys())}")
    for key, value in inputs.items():
        if hasattr(value, 'shape'):
            print(f"    {key}: shape {value.shape}, dtype {value.dtype}")
            if key == 'input_ids':
                # Print first 10 token IDs
                token_ids = value[0].tolist()[:10]
                print(f"      First 10 token IDs: {token_ids}")
                # Try to decode them
                if hasattr(processor, 'decode'):
                    try:
                        tokens = processor.decode(value[0][:10], skip_special_tokens=False)
                        print(f"      Decoded tokens: {repr(tokens)}")
                    except:
                        pass
            elif key == 'mm_token_type_ids':
                print(f"      First 10 mm_token_type_ids: {value[0].tolist()[:10]}")
        elif isinstance(value, list):
            print(f"    {key}: list length {len(value)}")

    # Check for image token ID
    if hasattr(processor, 'image_token_id'):
        print(f"  Processor image_token_id: {processor.image_token_id}")
    if hasattr(processor, 'tokenizer') and hasattr(processor.tokenizer, 'image_token_id'):
        print(f"  Tokenizer image_token_id: {processor.tokenizer.image_token_id}")

    with torch.no_grad():
        try:
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
            )
        except Exception as e:
            print(f"  Error during generation with sampling: {e}")
            print("  Trying greedy decoding as fallback...")
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.1,
            )

    # Decode response
    if hasattr(processor, 'decode'):
        response = processor.decode(generated_ids[0], skip_special_tokens=True)
    elif hasattr(processor, 'tokenizer') and hasattr(processor.tokenizer, 'decode'):
        response = processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    else:
        response = str(generated_ids[0])

    # Extract assistant portion if present
    if "assistant" in response.lower():
        parts = response.lower().split("assistant", 1)
        if len(parts) > 1:
            response = response[len("assistant"):].strip() if parts[0] == "" else parts[1].strip()

    return response


# ---------------------------------------------------------------------------
# MCQ evaluation
# ---------------------------------------------------------------------------

def build_mcq_prompt(world_id: str, mcqs: List[MCQ]) -> str:
    """Build a prompt with all MCQs formatted for the model."""
    lines = []
    for i, mcq in enumerate(mcqs, 1):
        lines.append(f"{i}. {mcq.question}")
        for letter in ["A", "B", "C", "D"]:
            lines.append(f"   {letter}. {mcq.options[letter]}")
        lines.append("")  # blank line between questions

    questions_text = "\n".join(lines)
    return MCQ_INDUCTION_PROMPT.format(questions_text=questions_text)


def parse_mcq_answers(model_output: str, num_questions: int) -> Dict[int, str]:
    """
    Parse model output to extract answer letters for each question.

    Returns: dict mapping question_index (0-based) to answer letter ("A"-"D").
    Unparseable questions are omitted (counted as wrong later).
    """
    # Collect all (question_number, letter) candidates using non-overlapping matches
    candidates = []

    # Try each pattern separately with findall (non-overlapping within each pass)
    patterns = [
        # "1. A", "1. A)", "1: A", "1）A", "1. A", "Q1. A", "问题1: A"
        r'(?:^|\n)\s*(?:Q|问题)?\s*(\d+)\s*[.）:：\)\]\s]+\s*([A-Da-d])\b',
        # "[1] A", "(1) A"
        r'(?:^|\n)\s*[\[\(（]\s*(\d+)\s*[\]\)）]\s*([A-Da-d])\b',
        # "问题1：A"
        r'(?:^|\n)\s*问题\s*(\d+)\s*[：:]\s*([A-Da-d])\b',
    ]

    for pat in patterns:
        for m in re.finditer(pat, model_output):
            qnum, letter = int(m.group(1)), m.group(2).upper()
            if 1 <= qnum <= num_questions:
                candidates.append((qnum, letter))

    # If multiple patterns matched, prefer the one with most matches
    # Group by (question_number) -> use last occurrence
    seen = {}
    for qnum, letter in candidates:
        seen[qnum] = letter

    return {idx - 1: letter for idx, letter in seen.items()}


def score_mcq_answers(
    answers: Dict[int, str],
    mcqs: List[MCQ],
) -> Dict[str, Any]:
    """
    Score model answers against ground truth.

    Returns:
        per_question: list of {idx, correct, model_answer, correct_answer, source}
        num_correct: int
        num_total: int
        accuracy: float
        unanswered: int
        source_breakdown: {source: {correct, total}}
    """
    per_question = []
    source_counts = defaultdict(lambda: {"correct": 0, "total": 0})

    for i, mcq in enumerate(mcqs):
        model_ans = answers.get(i)
        correct = model_ans == mcq.correct_answer if model_ans else False

        per_question.append({
            "idx": i,
            "correct": correct,
            "model_answer": model_ans,
            "correct_answer": mcq.correct_answer,
            "source": mcq.source,
        })

        src = mcq.source
        source_counts[src]["total"] += 1
        if correct:
            source_counts[src]["correct"] += 1

    num_correct = sum(1 for pq in per_question if pq["correct"])
    num_total = len(mcqs)
    unanswered = sum(1 for pq in per_question if pq["model_answer"] is None)

    return {
        "per_question": per_question,
        "num_correct": num_correct,
        "num_total": num_total,
        "accuracy": num_correct / num_total if num_total > 0 else 0.0,
        "unanswered": unanswered,
        "source_breakdown": dict(source_counts),
    }


# ---------------------------------------------------------------------------
# Prediction evaluation
# ---------------------------------------------------------------------------

def parse_prediction_json(text: str) -> Optional[List[Dict[str, Any]]]:
    """Try to extract and parse a JSON predictions array from model output."""
    # Try to find a JSON block
    json_match = re.search(r'\{[^}]*(?:".*?"[^}]*)*\}', text, re.DOTALL)
    if json_match:
        candidates = re.findall(r'\{[^}]*"predictions"[^}]*\}', text, re.DOTALL)
        for c in candidates:
            try:
                data = json.loads(c)
                if "predictions" in data:
                    return data["predictions"]
            except json.JSONDecodeError:
                continue

    # Try to find an array of objects with frame/objects keys
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "predictions" in data:
            return data["predictions"]
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    # Try to find anything that looks like a JSON object
    brace_match = re.search(r'\{.*\}', text, re.DOTALL)
    if brace_match:
        try:
            data = json.loads(brace_match.group())
            if isinstance(data, dict):
                if "predictions" in data:
                    return data["predictions"]
                if "frame" in data:
                    return [data]
        except json.JSONDecodeError:
            pass

    return None


def evaluate_prediction_from_response(
    world_id: str,
    episode_id: str,
    response: str,
    ground_truth_states: List[Dict],
) -> Dict[str, Any]:
    """
    Try to parse model's prediction response and evaluate against ground truth.
    Returns a dict with evaluation results or error info.
    """
    predicted = parse_prediction_json(response)

    if predicted is None:
        return {
            "world_id": world_id,
            "episode_id": episode_id,
            "error": "Could not parse JSON prediction from model output",
            "raw_response": response[:500],
        }

    try:
        result = evaluate_prediction(
            world_id=world_id,
            predicted_states=predicted,
            ground_truth_states=ground_truth_states,
            variant="zero_shot",
            episode_id=episode_id,
        )
        return {
            "world_id": result.world_id,
            "episode_id": result.episode_id,
            "variant": result.variant,
            "trajectory_mse": result.trajectory_mse,
            "physics_checks": result.physics_checks,
            "physics_consistency": result.physics_consistency,
            "total_score": result.total_score,
        }
    except Exception as e:
        return {
            "world_id": world_id,
            "episode_id": episode_id,
            "error": f"Evaluation failed: {e}",
            "raw_response": response[:500],
        }


# ---------------------------------------------------------------------------
# Dataset iteration
# ---------------------------------------------------------------------------

def discover_episodes(data_root: str, world_ids: Optional[List[str]] = None) -> List[Tuple[str, str]]:
    """Discover all (world_id, episode_id) pairs in the dataset."""
    episodes = []
    worlds_dir = os.path.join(data_root, "worlds")
    if not os.path.exists(worlds_dir):
        return episodes

    available_worlds = [d for d in os.listdir(worlds_dir) if os.path.isdir(os.path.join(worlds_dir, d))]
    if world_ids:
        available_worlds = [w for w in available_worlds if w in world_ids]

    for world_id in available_worlds:
        ep_dir = os.path.join(worlds_dir, world_id, "episodes")
        if not os.path.exists(ep_dir):
            continue
        for ep_name in sorted(os.listdir(ep_dir)):
            ep_path = os.path.join(ep_dir, ep_name)
            if os.path.isdir(ep_path) and os.path.exists(os.path.join(ep_path, "video.mp4")):
                episodes.append((world_id, ep_name))

    return episodes


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def print_mcq_summary(mcq_results: List[Dict[str, Any]]) -> None:
    """Print MCQ evaluation summary."""
    if not mcq_results:
        return

    print(f"\nMCQ results: {len(mcq_results)} episodes")
    for r in mcq_results:
        print(f"  [{r['world_id']}/{r['episode_id']}] "
              f"accuracy={r['accuracy']:.1%} "
              f"({r['num_correct']}/{r['num_total']}) "
              f"unanswered={r['unanswered']}")

    # Aggregate across all episodes
    total_correct = sum(r["num_correct"] for r in mcq_results)
    total_questions = sum(r["num_total"] for r in mcq_results)
    overall_accuracy = total_correct / total_questions if total_questions > 0 else 0.0

    # Per-world passrate
    world_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in mcq_results:
        world_stats[r["world_id"]]["correct"] += r["num_correct"]
        world_stats[r["world_id"]]["total"] += r["num_total"]

    # Per-source breakdown
    source_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in mcq_results:
        for src, counts in r.get("source_breakdown", {}).items():
            source_stats[src]["correct"] += counts["correct"]
            source_stats[src]["total"] += counts["total"]

    print(f"\n  Overall accuracy: {overall_accuracy:.1%} ({total_correct}/{total_questions})")

    print(f"\n  Per-world passrate:")
    for world_id in sorted(world_stats.keys()):
        s = world_stats[world_id]
        rate = s["correct"] / s["total"] if s["total"] > 0 else 0
        print(f"    {world_id}: {rate:.1%} ({s['correct']}/{s['total']})")

    print(f"\n  Per-source accuracy:")
    for src in ["required_fact", "bonus_fact", "generalization", "confusion", "cross_world"]:
        if src in source_stats:
            s = source_stats[src]
            rate = s["correct"] / s["total"] if s["total"] > 0 else 0
            print(f"    {src}: {rate:.1%} ({s['correct']}/{s['total']})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate U-MARVEL-Qwen3VL-4B-Instruct on Phy-Bench")
    parser.add_argument("--video", type=str, help="Path to single video file")
    parser.add_argument("--world", type=str, default=None, help="World ID")
    parser.add_argument("--data-root", type=str, default="examples", help="Dataset root directory")
    parser.add_argument("--all", action="store_true", help="Evaluate all episodes in data-root")
    parser.add_argument("--model-path", type=str,
                        default="/home/cyx/.cache/modelscope/hub/models/TencentBAC/U-MARVEL-Qwen3VL-4B-Instruct",
                        help="Local path to model directory")
    parser.add_argument("--num-frames", type=int, default=8, help="Number of frames to extract")
    parser.add_argument("--device-map", type=str, default="auto", help="Device map")
    parser.add_argument("--skip-prediction", action="store_true",
                        help="Skip prediction task (induction only)")
    parser.add_argument("--eval-mode", type=str, default="mcq",
                        choices=["mcq", "open_ended"],
                        help="Evaluation mode: mcq (default, automatic scoring) or open_ended (free-text)")

    args = parser.parse_args()

    # Collect episodes to evaluate
    episodes_to_eval = []  # List of (world_id, episode_id, video_path)

    if args.video:
        if not args.world:
            print("Error: --world is required when using --video")
            sys.exit(1)
        if not os.path.exists(args.video):
            print(f"Error: Video not found: {args.video}")
            sys.exit(1)
        ep_name = os.path.splitext(os.path.basename(args.video))[0]
        episodes_to_eval.append((args.world, ep_name, args.video))
    elif args.all or args.world:
        world_ids = [args.world] if args.world else None
        discovered = discover_episodes(args.data_root, world_ids)
        if not discovered:
            print(f"Error: No episodes found in {args.data_root}")
            sys.exit(1)
        for world_id, ep_name in discovered:
            vpath = video_path(args.data_root, world_id, ep_name)
            episodes_to_eval.append((world_id, ep_name, vpath))
    else:
        # Default: single example
        default_video = "examples/worlds/black_hole/episodes/ep_001/video.mp4"
        if not os.path.exists(default_video):
            print("Error: No video specified and default not found. Use --video or --data-root --all")
            sys.exit(1)
        episodes_to_eval.append(("black_hole", "ep_001", default_video))

    # Load model
    print("=" * 60)
    print("Loading model...")
    model, processor = load_model(args.model_path, args.device_map)
    print("Model loaded.\n")

    # Storage for results
    all_results = {
        "induction": [],
        "mcq": [],
        "prediction": [],
        "config": {
            "model_path": args.model_path,
            "num_frames": args.num_frames,
            "eval_mode": args.eval_mode,
        }
    }

    # Evaluate each episode
    for world_id, episode_id, vpath in episodes_to_eval:
        print("\n" + "=" * 60)
        print(f"Evaluating: {world_id} / {episode_id}")
        print(f"  Video: {vpath}")

        # Extract frames
        print("\n[Step 1] Extracting frames...")
        frames = extract_video_frames(vpath, args.num_frames)

        # Load ground truth states (for prediction evaluation)
        gt_states = None
        if not args.skip_prediction:
            try:
                gt_states = load_states(args.data_root, world_id, episode_id)
                gt_dicts = [s.to_dict() for s in gt_states]
                print(f"  Loaded {len(gt_dicts)} ground truth frames")
            except Exception as e:
                print(f"  Warning: Could not load ground truth: {e}")
                gt_dicts = None

        # ------------------------------------------------------------------
        # Induction / MCQ Task
        # ------------------------------------------------------------------
        if args.eval_mode == "mcq":
            print("\n[Step 2] MCQ induction task...")

            mcqs = WORLD_MCQS.get(world_id)
            if mcqs is None:
                print(f"  Warning: No MCQs defined for world '{world_id}', skipping induction task")
                mcq_result = {
                    "world_id": world_id,
                    "episode_id": episode_id,
                    "error": f"No MCQs defined for world '{world_id}'",
                }
            else:
                print(f"  Loaded {len(mcqs)} questions for world '{world_id}'")

                # Build MCQ prompt
                mcq_prompt = build_mcq_prompt(world_id, mcqs)

                # Run inference
                mcq_response = run_inference(model, processor, frames, mcq_prompt, max_new_tokens=256)

                # Parse and score answers
                parsed = parse_mcq_answers(mcq_response, len(mcqs))
                mcq_result = score_mcq_answers(parsed, mcqs)
                mcq_result["world_id"] = world_id
                mcq_result["episode_id"] = episode_id
                mcq_result["raw_response"] = mcq_response

                # Print results
                print(f"  Parsed {len(parsed)}/{len(mcqs)} answers")
                for pq in mcq_result["per_question"]:
                    mark = "✓" if pq["correct"] else "✗"
                    ans = pq["model_answer"] or "?"
                    print(f"    Q{pq['idx']+1}: {mark} model={ans} correct={pq['correct_answer']} [{pq['source']}]")
                print(f"  Accuracy: {mcq_result['accuracy']:.1%} ({mcq_result['num_correct']}/{mcq_result['num_total']})")

            all_results["mcq"].append(mcq_result)

        else:
            # Open-ended induction (original behavior)
            print("\n[Step 2] Induction task (rule description)...")
            induction_prompt = INDUCTION_PROMPT.format(num_episodes=1)
            induction_prompt += f"\n\n世界ID: {world_id}\n请注意观察视频中物体的运动规律。"

            induction_response = run_inference(model, processor, frames, induction_prompt, max_new_tokens=512)

            induction_result = {
                "world_id": world_id,
                "episode_id": episode_id,
                "prompt": induction_prompt[:200],
                "response": induction_response,
            }
            all_results["induction"].append(induction_result)
            print(f"  Induction response ({len(induction_response)} chars):")
            print(f"  {induction_response[:300]}...")

        # ------------------------------------------------------------------
        # Prediction Task (PhysScore)
        # ------------------------------------------------------------------
        if not args.skip_prediction and gt_dicts is not None:
            print("\n[Step 3] Prediction task (trajectory prediction)...")

            # Use first 30 frames (~1 sec) as context, ask to predict next 60 frames (~2 sec)
            context_sec = 1.0
            target_sec = 2.0
            num_objects = len(gt_dicts[0]["objects"]) if gt_dicts else 0

            pred_prompt = PREDICTION_PROMPT_ZERO_SHOT.format(
                context_sec=context_sec,
                target_sec=target_sec,
                num_objects=num_objects,
            )

            pred_response = run_inference(
                model, processor, frames, pred_prompt, max_new_tokens=1024
            )

            # Evaluate prediction
            eval_result = evaluate_prediction_from_response(
                world_id, episode_id, pred_response, gt_dicts
            )
            all_results["prediction"].append(eval_result)

            if "error" in eval_result:
                print(f"  Prediction evaluation: {eval_result['error']}")
            else:
                print(f"  Trajectory MSE: {eval_result['trajectory_mse']:.2f}")
                for check, score in eval_result["physics_checks"].items():
                    print(f"  {check}: {score:.3f}")
                print(f"  Physics consistency: {eval_result['physics_consistency']:.3f}")
                print(f"  Total prediction score: {eval_result['total_score']:.3f}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # MCQ summary
    if args.eval_mode == "mcq":
        mcq_results = [r for r in all_results["mcq"] if "error" not in r]
        print_mcq_summary(mcq_results)

    # Induction summary (open-ended)
    if all_results["induction"]:
        print(f"\nInduction (open-ended) results: {len(all_results['induction'])} episodes")
        for r in all_results["induction"]:
            print(f"  [{r['world_id']}/{r['episode_id']}] {len(r['response'])} chars")

    # Prediction summary
    pred_results = [r for r in all_results["prediction"] if "error" not in r]
    failed_preds = [r for r in all_results["prediction"] if "error" in r]

    if pred_results:
        print(f"\nPrediction results: {len(pred_results)} episodes (passrate & consistency)")
        for r in pred_results:
            checks_str = ", ".join(f"{k}={v:.3f}" for k, v in r["physics_checks"].items())
            print(f"  [{r['world_id']}/{r['episode_id']}] "
                  f"consistency={r['physics_consistency']:.3f} "
                  f"mse={r['trajectory_mse']:.1f} "
                  f"score={r['total_score']:.3f} "
                  f"({checks_str})")

        # Aggregate metrics
        avg_consistency = sum(r["physics_consistency"] for r in pred_results) / len(pred_results)
        avg_mse = sum(r["trajectory_mse"] for r in pred_results) / len(pred_results)
        avg_score = sum(r["total_score"] for r in pred_results) / len(pred_results)
        pass_count = sum(1 for r in pred_results if all(v >= 0.8 for v in r["physics_checks"].values()))
        pass_rate = pass_count / len(pred_results)

        print(f"\n  Aggregate:")
        print(f"    Avg consistency: {avg_consistency:.3f}")
        print(f"    Avg MSE: {avg_mse:.1f}")
        print(f"    Avg total score: {avg_score:.3f}")
        print(f"    Pass rate (all checks >= 0.8): {pass_rate:.1%} ({pass_count}/{len(pred_results)})")

    if failed_preds:
        print(f"\n  Failed predictions (JSON parse errors): {len(failed_preds)}")
        for r in failed_preds:
            print(f"    [{r['world_id']}/{r['episode_id']}] {r['error']}")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(results_dir, f"simple_eval_{timestamp}.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_path}")
    print("Evaluation complete!")


if __name__ == "__main__":
    main()
