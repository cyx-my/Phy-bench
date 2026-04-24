#!/usr/bin/env python3
"""
Evaluation script for testing U-MARVEL-Qwen3VL-4B-Instruct on Phy-Bench.

Usage:
    python evaluate4.20.py --video examples/worlds/black_hole/episodes/ep_001/video.mp4 --world black_hole
    python evaluate4.20.py --frames-dir path/to/frames --world black_hole

python evaluate4.20.py \\
    --video examples/worlds/black_hole/episodes/ep_001/video.mp4 \\
    --world black_hole \\
    --model-path /home/cyx/.cache/modelscope/hub/models/TencentBAC/U-MARVEL-Qwen3VL-4B-Instruct \\
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
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, AutoProcessor

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


def load_ummarvel_model(model_path: str = "/home/cyx/.cache/modelscope/hub/models/TencentBAC/U-MARVEL-Qwen3VL-4B-Instruct",
                        device_map: str = "auto", use_modelscope: bool = False):
    """Load U-MARVEL-Qwen3VL-4B-Instruct model and tokenizer.

    Args:
        model_path: Local path to model directory
        device_map: Device mapping strategy
        use_modelscope: Whether to use modelscope for loading (for ModelScope-specific models)
    """
    print(f"Loading U-MARVEL-Qwen3VL-4B-Instruct from {model_path} (use_modelscope={use_modelscope})...")

    # First try ModelScope if requested
    if use_modelscope:
        try:
            import modelscope
            print("ModelScope available, attempting to load with ModelScope...")

            # Try to load with ModelScope's AutoModel and AutoTokenizer
            from modelscope import AutoModel, AutoTokenizer, AutoProcessor

            try:
                # Try to import more specific model classes if available
                from modelscope import AutoModelForCausalLM
                model_classes = [AutoModelForCausalLM, AutoModel]
            except ImportError:
                model_classes = [AutoModel]

            try:
                # First try AutoProcessor (for vision-language models)
                processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
                print("Loaded processor via ModelScope AutoProcessor")
            except Exception as e:
                print(f"ModelScope AutoProcessor failed: {e}, trying AutoTokenizer...")
                processor = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                print("Loaded tokenizer via ModelScope AutoTokenizer")

            # Try each model class until one works
            model = None
            last_error = None

            for model_class in model_classes:
                try:
                    print(f"Trying to load with ModelScope {model_class.__name__}...")
                    model = model_class.from_pretrained(
                        model_path,
                        device_map=device_map,
                        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                        trust_remote_code=True
                    )
                    print(f"Model loaded successfully via ModelScope using {model_class.__name__}")
                    # Check if model has generate or chat method
                    has_generate = hasattr(model, 'generate')
                    has_chat = hasattr(model, 'chat')

                    if has_generate:
                        print("Model has generate method - suitable for inference")
                        break
                    elif has_chat:
                        print("Model has chat method (but no generate) - may need different inference approach")
                        break
                    else:
                        print(f"Warning: Model loaded with {model_class.__name__} has neither generate nor chat method")
                        # Continue to try next model class
                        model = None
                        continue
                except Exception as e:
                    last_error = e
                    print(f"Failed with {model_class.__name__}: {e}")
                    continue

            if model is None:
                raise RuntimeError(f"Failed to load model with ModelScope: {last_error}")

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

    # Import transformers classes here to avoid UnboundLocalError
    from transformers import AutoProcessor, AutoTokenizer, AutoModel, AutoModelForCausalLM

    # Try to load processor/tokenizer
    processor = None
    try:
        # Try to use AutoProcessor for Qwen-VL models
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        print("Loaded processor via transformers AutoProcessor")
    except Exception as e:
        print(f"Transformers AutoProcessor failed: {e}, trying AutoTokenizer...")
        try:
            processor = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            print("Loaded tokenizer via transformers AutoTokenizer")
        except Exception as e2:
            print(f"Transformers AutoTokenizer also failed: {e2}")
            raise RuntimeError("Failed to load processor or tokenizer")

    # Try different model classes in order of preference
    model_classes_to_try = []

    # Try to import Qwen3VL specific class
    try:
        from transformers import Qwen3VLForConditionalGeneration
        model_classes_to_try.append(Qwen3VLForConditionalGeneration)
        print("Qwen3VLForConditionalGeneration available")
    except ImportError:
        print("Qwen3VLForConditionalGeneration not available")
        pass

    # Try to import AutoModelForConditionalGeneration
    try:
        from transformers import AutoModelForConditionalGeneration
        model_classes_to_try.append(AutoModelForConditionalGeneration)
        print("AutoModelForConditionalGeneration available")
    except ImportError:
        print("AutoModelForConditionalGeneration not available")
        pass

    # Add fallback classes
    model_classes_to_try.extend([AutoModelForCausalLM, AutoModel])

    # Try each model class
    model = None
    last_error = None

    for model_class in model_classes_to_try:
        try:
            print(f"Trying to load with transformers {model_class.__name__}...")
            model = model_class.from_pretrained(
                model_path,
                device_map=device_map,
                torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                trust_remote_code=True
            )
            print(f"Model loaded successfully via transformers using {model_class.__name__}")

            # Check if model has generate or chat method
            has_generate = hasattr(model, 'generate')
            has_chat = hasattr(model, 'chat')

            if has_generate:
                print("Model has generate method - suitable for inference")
                break
            elif has_chat:
                print("Model has chat method (but no generate) - may need different inference approach")
                # We'll still use this model, but inference will need to use chat
                break
            else:
                print(f"Warning: Model loaded with {model_class.__name__} has neither generate nor chat method")
                # Continue to try next model class
                model = None
                continue
        except Exception as e:
            last_error = e
            print(f"Failed with {model_class.__name__}: {e}")
            continue

    if model is None:
        raise RuntimeError(f"Failed to load model with transformers: {last_error}")

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
    """Run inference with U-MARVEL-Qwen3VL model."""
    print(f"Running inference with {len(frames)} frames...")

    # Debug: print processor type
    print(f"Processor type: {type(processor)}")
    print(f"Processor class: {processor.__class__.__name__ if hasattr(processor, '__class__') else 'Unknown'}")
    print(f"Model type: {type(model)}")
    print(f"Model has generate method: {hasattr(model, 'generate')}")
    print(f"Model has chat method: {hasattr(model, 'chat')}")

    # Debug: check frame properties
    if frames:
        print(f"First frame size: {frames[0].size}, mode: {frames[0].mode}")
        print(f"All frames sizes: {[img.size for img in frames]}")

    # Prepare messages for Qwen-VL-Chat format
    # For Qwen3VL models, images should come before text in content
    messages = [
        {
            "role": "user",
            "content": []
        }
    ]

    # Add images first, then text
    for img in frames:
        # Don't resize manually - let the processor handle it
        messages[0]["content"].append({"type": "image", "image": img})

    # Add text last
    messages[0]["content"].append({"type": "text", "text": prompt_text})

    # Strategy 1: Try chat method first (most straightforward for Qwen3VL models)
    if hasattr(model, 'chat'):
        print("Using chat method for inference")
        try:
            # Try with default parameters
            response = model.chat(
                messages=messages,
                tokenizer=processor,
                max_new_tokens=max_new_tokens,
                do_sample=True,           # Use sampling to avoid repetition
                temperature=0.7,          # Higher temperature for more diverse output
                top_p=0.9,
                repetition_penalty=1.1,   # Penalize repetition
            )
            print("Chat method succeeded")

            # chat method may return text directly or in a dict
            if isinstance(response, str):
                return response
            elif isinstance(response, dict) and 'text' in response:
                return response['text']
            elif hasattr(response, 'text'):  # Some models return object with .text attribute
                return response.text
            else:
                # Try to convert
                return str(response)
        except Exception as e:
            print(f"Chat method failed with error: {e}")
            print("Falling back to generate method...")
            # Continue to generate method

    # Strategy 2: Use generate method with processor
    print("Using generate method for inference")

    # First, let's check if processor can handle the messages format directly
    try:
        # Try to process messages directly (some processors support this)
        inputs = processor(messages, return_tensors="pt", padding=True)
        print("Successfully processed messages directly with processor")
    except Exception as e1:
        print(f"Cannot process messages directly: {e1}")

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
                print(f"Applied chat template, text length: {len(text)}")

                # Try to process with images
                try:
                    inputs = processor(text=[text], images=frames, return_tensors="pt", padding=True)
                    print("Successfully processed text + images")
                except Exception as e2:
                    print(f"Cannot process text+images: {e2}")
                    # Try without images (last resort)
                    inputs = processor(text=[text], return_tensors="pt", padding=True)
                    print("Warning: Processing without images - model may fail")
            except Exception as e3:
                print(f"Chat template failed: {e3}")
                # Final fallback: just text
                inputs = processor(text=[prompt_text], return_tensors="pt", padding=True)
                print("Using plain text only")
        else:
            # No chat template available, try to process with images
            try:
                inputs = processor(text=[prompt_text], images=frames, return_tensors="pt", padding=True)
                print("Processed text + images without chat template")
            except Exception as e4:
                print(f"Cannot process with images: {e4}")
                inputs = processor(text=[prompt_text], return_tensors="pt", padding=True)
                print("Using plain text only")

    # Debug: show input keys and shapes
    print(f"Input keys: {list(inputs.keys())}")
    for key, value in inputs.items():
        if hasattr(value, 'shape'):
            print(f"  {key}: shape {value.shape}, dtype {value.dtype}")
        elif isinstance(value, list):
            print(f"  {key}: list length {len(value)}")

    # Move inputs to GPU
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Initialize generated_ids
    generated_ids = None

    # Generate with better parameters to avoid repetition
    with torch.no_grad():
        try:
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,           # Use sampling
                temperature=0.7,          # Higher temperature
                top_p=0.9,                # Nucleus sampling
                repetition_penalty=1.1,   # Penalize repetition
                no_repeat_ngram_size=3,   # Avoid repeating n-grams
            )
        except Exception as e:
            print(f"Error during generation with sampling: {e}")
            print("Trying greedy decoding as fallback...")
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,          # Greedy decoding
                repetition_penalty=1.1,   # Still use repetition penalty
            )

    # Check if generation succeeded
    if generated_ids is None:
        raise RuntimeError("Generation failed - generated_ids is None")

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
    parser = argparse.ArgumentParser(description="Evaluate U-MARVEL-Qwen3VL-4B-Instruct on Phy-Bench")
    parser.add_argument("--video", type=str,
                       default="examples/worlds/black_hole/episodes/ep_001/video.mp4",
                       help="Path to video file")
    parser.add_argument("--world", type=str, default="black_hole",
                       help="World ID (e.g., black_hole, gravity_merry_go_round)")
    parser.add_argument("--model-path", type=str,
                       default="/home/cyx/.cache/modelscope/hub/models/TencentBAC/U-MARVEL-Qwen3VL-4B-Instruct",
                       help="Local path to model directory")
    parser.add_argument("--use-modelscope", action="store_true",
                       help="Use ModelScope for model loading (for ModelScope-specific models)")
    parser.add_argument("--num-frames", type=int, default=8,
                       help="Number of frames to extract from video")
    parser.add_argument("--device-map", type=str, default="auto",
                       help="Device map for model loading")

    args = parser.parse_args()

    # Check video exists
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)

    print(f"Testing U-MARVEL-Qwen3VL-4B-Instruct on Phy-Bench")
    print(f"  Video: {args.video}")
    print(f"  World: {args.world}")
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
    model, processor = load_ummarvel_model(args.model_path, args.device_map, args.use_modelscope)

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
    print("U-MARVEL-Qwen3VL's response:")
    print(response)
    print("\n" + "="*60)

    # Save results automatically
    import json
    import time
    from datetime import datetime

    # Create results directory if it doesn't exist
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    # Generate output filename with timestamp and world ID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_name = os.path.splitext(os.path.basename(args.video))[0]
    output_filename = f"umarvleval_{args.world}_{video_name}_{timestamp}.json"
    output_path = os.path.join(results_dir, output_filename)

    results = {
        "video": args.video,
        "world": args.world,
        "model_path": args.model_path,
        "use_modelscope": args.use_modelscope,
        "num_frames": args.num_frames,
        "prompt": prompt,
        "response": response,
        "frames_extracted": len(frames),
        "timestamp": timestamp,
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults automatically saved to: {output_path}")

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()