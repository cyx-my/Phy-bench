#!/usr/bin/env python3
"""
Simple test script to verify U-MARVEL-Qwen3VL-4B-Instruct basic multimodal capabilities.
"""

import os
import sys
import cv2
from PIL import Image
import torch

# Add project root to sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def extract_single_frame(video_path: str, frame_idx: int = 0) -> Image.Image:
    """Extract a single frame from video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_idx >= total_frames:
        frame_idx = 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError(f"Cannot read frame {frame_idx} from video")

    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    print(f"Extracted frame {frame_idx} from video, size: {pil_img.size}")
    return pil_img

def test_basic_description(model_path: str, frame: Image.Image, use_modelscope: bool = False):
    """Test basic image description capability."""
    print("\n" + "="*60)
    print("Test 1: Basic Image Description")
    print("="*60)

    try:
        # Use the same loading logic as evaluate4.20.py
        import importlib.util
        import sys

        # Import the evaluate4.20 module
        spec = importlib.util.spec_from_file_location("evaluate4_20", "evaluate4.20.py")
        evaluate_module = importlib.util.module_from_spec(spec)
        sys.modules["evaluate4_20"] = evaluate_module
        spec.loader.exec_module(evaluate_module)

        model, processor = evaluate_module.load_ummarvel_model(model_path, "auto", use_modelscope)
        print(f"Processor type: {type(processor)}")
        print(f"Model type: {type(model)}")

        print(f"Processor type: {type(processor)}")
        print(f"Model type: {type(model)}")
        print(f"Model has chat method: {hasattr(model, 'chat')}")
        print(f"Model has generate method: {hasattr(model, 'generate')}")

        # Simple description task
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": frame},
                {"type": "text", "text": "描述这张图像中有什么物体？"}
            ]
        }]

        print(f"\nPrompt: '描述这张图像中有什么物体？'")

        # Try chat method first
        if hasattr(model, 'chat'):
            print("\nUsing chat method...")
            try:
                response = model.chat(
                    messages=messages,
                    tokenizer=processor,
                    max_new_tokens=50,
                    do_sample=True,
                    temperature=0.7
                )

                if isinstance(response, str):
                    print(f"Response: {response}")
                elif isinstance(response, dict) and 'text' in response:
                    print(f"Response: {response['text']}")
                elif hasattr(response, 'text'):
                    print(f"Response: {response.text}")
                else:
                    print(f"Response (converted): {str(response)}")
                return True
            except Exception as e:
                print(f"Chat method failed: {e}")

        # Fallback to generate
        print("\nUsing generate method...")
        try:
            inputs = processor(messages, return_tensors="pt", padding=True)
            print(f"Input keys: {list(inputs.keys())}")

            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            generated_ids = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )

            # Decode
            decode_obj = None
            if hasattr(processor, 'decode'):
                decode_obj = processor
            elif hasattr(processor, 'tokenizer') and hasattr(processor.tokenizer, 'decode'):
                decode_obj = processor.tokenizer

            if decode_obj:
                response = decode_obj.decode(generated_ids[0], skip_special_tokens=True)
            else:
                response = processor.decode(generated_ids[0], skip_special_tokens=True)

            print(f"Response: {response}")
            return True

        except Exception as e:
            print(f"Generate method failed: {e}")
            return False

    except Exception as e:
        print(f"Model loading failed: {e}")
        return False

def test_simple_qa(model_path: str, frame: Image.Image):
    """Test simple Q&A with visual context."""
    print("\n" + "="*60)
    print("Test 2: Simple Visual Q&A")
    print("="*60)

    try:
        from transformers import AutoModelForCausalLM, AutoProcessor

        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            trust_remote_code=True
        )

        # Simple question about the image
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": frame},
                {"type": "text", "text": "图像中有几个物体？"}
            ]
        }]

        if hasattr(model, 'chat'):
            response = model.chat(
                messages=messages,
                tokenizer=processor,
                max_new_tokens=30,
                do_sample=False  # Greedy for simple answers
            )

            if isinstance(response, str):
                print(f"Response: {response}")
            elif isinstance(response, dict) and 'text' in response:
                print(f"Response: {response['text']}")
            else:
                print(f"Response (converted): {str(response)}")

        return True

    except Exception as e:
        print(f"Simple Q&A failed: {e}")
        return False

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test U-MARVEL basic capabilities")
    parser.add_argument("--video", type=str,
                       default="examples/worlds/black_hole/episodes/ep_001/video.mp4",
                       help="Path to video file")
    parser.add_argument("--model-path", type=str,
                       default="/home/cyx/.cache/modelscope/hub/models/TencentBAC/U-MARVEL-Qwen3VL-4B-Instruct",
                       help="Local path to model directory")
    parser.add_argument("--use-modelscope", action="store_true",
                       help="Use ModelScope for loading")

    args = parser.parse_args()

    print("Testing U-MARVEL-Qwen3VL-4B-Instruct basic capabilities")
    print(f"Video: {args.video}")
    print(f"Model path: {args.model_path}")
    print(f"Use ModelScope: {args.use_modelscope}")

    # Extract a single frame
    try:
        frame = extract_single_frame(args.video, frame_idx=0)
    except Exception as e:
        print(f"Failed to extract frame: {e}")
        # Try to use a simple test image instead
        print("Creating a simple test image...")
        from PIL import Image, ImageDraw
        img = Image.new('RGB', (256, 256), color='white')
        draw = ImageDraw.Draw(img)
        draw.rectangle([50, 50, 150, 150], fill='red', outline='black')
        draw.ellipse([180, 180, 230, 230], fill='blue', outline='black')
        frame = img
        print(f"Created test image with red rectangle and blue circle")

    # Run tests
    success1 = test_basic_description(args.model_path, frame, args.use_modelscope)
    success2 = test_simple_qa(args.model_path, frame)

    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    print(f"Basic description test: {'PASS' if success1 else 'FAIL'}")
    print(f"Simple Q&A test: {'PASS' if success2 else 'FAIL'}")

    if not (success1 or success2):
        print("\n⚠️  Model appears to have fundamental issues with basic tasks")
        print("Possible causes:")
        print("1. Model not loading correctly")
        print("2. Images not being processed")
        print("3. Model weights corrupted")
        print("4. Hardware/compatibility issues")
    else:
        print("\n✅ Model shows basic multimodal capabilities")

if __name__ == "__main__":
    main()