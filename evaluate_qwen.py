#!/usr/bin/env python3
"""
Simple evaluation script for testing Qwen-VL on Phy-Bench.

Usage:
    python evaluate_qwen.py --video examples/worlds/black_hole/episodes/ep_001/video.mp4 --world black_hole
    python evaluate_qwen.py --frames-dir path/to/frames --world black_hole

python evaluate_qwen.py \                                                                                                                               
--video examples/worlds/black_hole/episodes/ep_001/video.mp4 \                                                                                        
--world black_hole \                                                                                                                                  
--model "qwen/Qwen-VL-4B-Instruct" \                                                                                                                  
--use-modelscope                                                                                                                                      
                       
"""

import argparse
import os
import sys
import cv2
import base64
from io import BytesIO
from pathlib import Path
from typing import List, Optional

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor

# Add project root to sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from benchmark.tasks import INDUCTION_PROMPT


def extract_video_frames(video_path: str, num_frames: int = 8) -> List[Image.Image]:
    """Extract evenly spaced frames from a video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0

    print(f"Video info: {total_frames} frames, {fps:.1f} fps, {duration:.1f}s")

    # Select frame indices
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
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            frames.append(pil_img)

    cap.release()
    print(f"Extracted {len(frames)} frames from video")
    return frames


def load_qwen_vl_model(model_name: str = "Qwen/Qwen-VL-Chat", device_map: str = "auto",
                       use_modelscope: bool = False):
    """Load Qwen-VL model and tokenizer.

    Args:
        model_name: HuggingFace model name or local path
        device_map: Device mapping strategy
        use_modelscope: Whether to use modelscope for loading (for ModelScope-specific models)
    """
    print(f"Loading model {model_name} (use_modelscope={use_modelscope})...")

    # First try ModelScope if requested
    if use_modelscope:
        try:
            import modelscope
            print("ModelScope available, attempting to load with ModelScope...")

            # Try to load with ModelScope's AutoModel and AutoTokenizer
            from modelscope import AutoModelForCausalLM, AutoTokenizer, AutoProcessor

            try:
                # First try AutoProcessor (for vision-language models)
                processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
                print("Loaded processor via ModelScope AutoProcessor")
            except Exception as e:
                print(f"ModelScope AutoProcessor failed: {e}, trying AutoTokenizer...")
                processor = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                print("Loaded tokenizer via ModelScope AutoTokenizer")

            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device_map,
                torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                trust_remote_code=True
            )
            print("Model loaded successfully via ModelScope")
            return model, processor

        except ImportError:
            print("ModelScope not installed, falling back to transformers...")
            use_modelscope = False
        except Exception as e:
            print(f"Error loading with ModelScope: {e}")
            print("Falling back to transformers...")
            use_modelscope = False

    # Fallback to transformers (HuggingFace)
    print("Using transformers for model loading...")
    try:
        # Try to use AutoProcessor for Qwen-VL models
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        print("Loaded processor via transformers AutoProcessor")
    except Exception as e:
        print(f"Transformers AutoProcessor failed: {e}, trying AutoTokenizer...")
        processor = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        print("Loaded tokenizer via transformers AutoTokenizer")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        trust_remote_code=True
    )
    print("Model loaded successfully via transformers")
    return model, processor


def build_prompt(frames: List[Image.Image], world_id: str, num_episodes: int = 1) -> str:
    """Build the induction prompt with image placeholders."""
    prompt_template = INDUCTION_PROMPT.format(num_episodes=num_episodes)

    # Add world context
    world_context = f"\n\n世界ID: {world_id}\n请注意观察视频中物体的运动规律。"

    return prompt_template + world_context


def run_inference(
    model,
    processor,
    frames: List[Image.Image],
    prompt_text: str,
    max_new_tokens: int = 512
):
    """Run inference with Qwen-VL model."""
    print(f"Running inference with {len(frames)} frames...")

    # Prepare messages for Qwen-VL-Chat format
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text}
            ]
        }
    ]

    # Add images to content
    for img in frames:
        # Resize image to model's expected size (e.g., 448x448 for Qwen-VL)
        img_resized = img.resize((448, 448))
        messages[0]["content"].append({"type": "image", "image": img_resized})

    # Different handling based on processor type
    # Try to find an object with apply_chat_template method
    apply_chat_template_obj = None
    if hasattr(processor, 'apply_chat_template'):
        apply_chat_template_obj = processor
    elif hasattr(processor, 'tokenizer') and hasattr(processor.tokenizer, 'apply_chat_template'):
        apply_chat_template_obj = processor.tokenizer

    if apply_chat_template_obj:
        # Use chat template
        try:
            text = apply_chat_template_obj.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            inputs = processor(text=[text], images=frames, return_tensors="pt")
        except Exception as e:
            print(f"Error with apply_chat_template: {e}")
            # Fallback to simple text
            text = prompt_text
            inputs = processor(text=[text], images=frames, return_tensors="pt")
    else:
        # Simple tokenizer/processor without chat template
        text = prompt_text
        inputs = processor(text=[text], images=frames, return_tensors="pt")

    # Move inputs to GPU
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Generate
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.1,
            top_p=0.9,
        )

    # Decode
    decode_obj = None
    if hasattr(processor, 'decode'):
        decode_obj = processor
    elif hasattr(processor, 'tokenizer') and hasattr(processor.tokenizer, 'decode'):
        decode_obj = processor.tokenizer
    elif hasattr(processor, 'decode'):  # Already checked, but keep structure
        decode_obj = processor
    else:
        # Fallback to model's tokenizer if available
        if hasattr(model, 'tokenizer') and hasattr(model.tokenizer, 'decode'):
            decode_obj = model.tokenizer

    if decode_obj:
        response = decode_obj.decode(generated_ids[0], skip_special_tokens=True)
    else:
        # Last resort: use processor directly (may work for tokenizer)
        response = processor.decode(generated_ids[0], skip_special_tokens=True)

    # Extract only the assistant's response
    if "assistant" in response.lower():
        # Try to find the assistant part
        parts = response.split("assistant", 1)
        if len(parts) > 1:
            response = parts[1].strip()

    return response


def main():
    parser = argparse.ArgumentParser(description="Evaluate Qwen-VL on Phy-Bench")
    parser.add_argument("--video", type=str,
                       default="examples/worlds/black_hole/episodes/ep_001/video.mp4",
                       help="Path to video file")
    parser.add_argument("--world", type=str, default="black_hole",
                       help="World ID (e.g., black_hole, gravity_merry_go_round)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen-VL-Chat",
                       help="Model name from HuggingFace or local path")
    parser.add_argument("--model-path", type=str,
                       help="Local path to model directory (alternative to --model)")
    parser.add_argument("--use-modelscope", action="store_true",
                       help="Use ModelScope for model loading (for ModelScope-specific models)")
    parser.add_argument("--num-frames", type=int, default=8,
                       help="Number of frames to extract from video")
    parser.add_argument("--output", type=str,
                       help="Save results to JSON file")
    parser.add_argument("--device-map", type=str, default="auto",
                       help="Device map for model loading")

    args = parser.parse_args()

    # Check video exists
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)

    print(f"Testing Qwen-VL on Phy-Bench")
    print(f"  Video: {args.video}")
    print(f"  World: {args.world}")
    print(f"  Model: {args.model}")
    if args.model_path:
        print(f"  Model path: {args.model_path}")
    print(f"  Use ModelScope: {args.use_modelscope}")
    print(f"  Frames: {args.num_frames}")

    # Step 1: Extract frames
    print("\n" + "="*60)
    print("Step 1: Extracting frames from video...")
    frames = extract_video_frames(args.video, args.num_frames)

    # Step 2: Load model
    print("\n" + "="*60)
    print("Step 2: Loading model...")
    # Use model-path if provided, otherwise use model name
    model_name = args.model_path if args.model_path else args.model
    model, processor = load_qwen_vl_model(model_name, args.device_map, args.use_modelscope)

    # Step 3: Build prompt
    print("\n" + "="*60)
    print("Step 3: Building prompt...")
    prompt = build_prompt(frames, args.world, num_episodes=1)
    print(f"Prompt length: {len(prompt)} characters")
    print("\nPrompt preview:")
    print(prompt[:500] + "..." if len(prompt) > 500 else prompt)

    # Step 4: Run inference
    print("\n" + "="*60)
    print("Step 4: Running inference...")
    response = run_inference(model, processor, frames, prompt)

    # Step 5: Display results
    print("\n" + "="*60)
    print("Step 5: Results")
    print("\n" + "="*60)
    print("Qwen-VL's response:")
    print(response)
    print("\n" + "="*60)

    # Save results if requested
    if args.output:
        import json
        results = {
            "video": args.video,
            "world": args.world,
            "model": args.model,
            "model_path": args.model_path,
            "use_modelscope": args.use_modelscope,
            "num_frames": args.num_frames,
            "prompt": prompt,
            "response": response,
            "frames_extracted": len(frames),
        }
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {args.output}")

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()