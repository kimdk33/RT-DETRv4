"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torchvision.transforms as T
from openvino.runtime import Core, PartialShape
from PIL import Image, ImageDraw


DEFAULT_INPUT_SIZE = 640


def resize_with_aspect_ratio(image: Image.Image, size: int, interpolation=Image.BILINEAR):
    """Resize an image while preserving aspect ratio and applying padding."""
    original_width, original_height = image.size
    ratio = min(size / original_width, size / original_height)
    new_width = int(original_width * ratio)
    new_height = int(original_height * ratio)
    resized = image.resize((new_width, new_height), interpolation)

    padded = Image.new("RGB", (size, size))
    pad_w = (size - new_width) // 2
    pad_h = (size - new_height) // 2
    padded.paste(resized, (pad_w, pad_h))
    return padded, ratio, pad_w, pad_h


def draw(
    images: List[Image.Image],
    labels: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    ratios: List[float],
    paddings: List[Tuple[int, int]],
    threshold: float = 0.4,
):
    result_images = []
    for i, im in enumerate(images):
        draw_obj = ImageDraw.Draw(im)
        scr = scores[i]
        lab = labels[i][scr > threshold]
        box = boxes[i][scr > threshold]
        scr = scr[scr > threshold]

        ratio = ratios[i]
        pad_w, pad_h = paddings[i]

        for lbl, bb, sc in zip(lab, box, scr):
            adjusted = [
                (bb[0] - pad_w) / ratio,
                (bb[1] - pad_h) / ratio,
                (bb[2] - pad_w) / ratio,
                (bb[3] - pad_h) / ratio,
            ]
            draw_obj.rectangle(adjusted, outline="red")
            draw_obj.text((adjusted[0], adjusted[1]), text=f"{int(lbl)} {float(sc):.2f}", fill="blue")

        result_images.append(im)
    return result_images


def infer_model(
    compiled_model,
    im_data: np.ndarray,
    orig_size: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run OpenVINO inference and return labels, boxes, and scores."""
    input_map: Dict[str, np.ndarray] = {}
    inputs = compiled_model.inputs

    if len(inputs) != 2:
        raise RuntimeError(f"Expected two inputs (images and orig_target_sizes), but got {len(inputs)}")

    input_map[inputs[0].get_any_name()] = im_data
    input_map[inputs[1].get_any_name()] = orig_size

    request = compiled_model.create_infer_request()
    request.infer(input_map)

    outputs = [request.get_output_tensor(i).data for i in range(len(compiled_model.outputs))]
    if len(outputs) != 3:
        raise RuntimeError(f"Expected three outputs (labels, boxes, scores), but got {len(outputs)}")

    labels, boxes, scores = outputs
    return labels, boxes, scores


def process_image(
    compiled_model,
    file_path: str,
    threshold: float = 0.4,
    output_path: str = "openvino_result.jpg",
    input_size: int = DEFAULT_INPUT_SIZE,
):
    im_pil = Image.open(file_path).convert("RGB")
    resized_im, ratio, pad_w, pad_h = resize_with_aspect_ratio(im_pil, input_size)
    orig_size = np.array([[resized_im.size[1], resized_im.size[0]]], dtype=np.int64)

    transforms = T.Compose([T.ToTensor()])
    im_data = transforms(resized_im).unsqueeze(0).numpy()

    labels, boxes, scores = infer_model(compiled_model, im_data, orig_size)

    result_images = draw(
        [im_pil], labels, boxes, scores, [ratio], [(pad_w, pad_h)], threshold
    )
    result_images[0].save(output_path)
    print(f"Image processing complete. Result saved as '{output_path}'.")


def process_video(
    compiled_model,
    video_path: str,
    threshold: float = 0.4,
    output_path: str = "openvino_result.mp4",
    input_size: int = DEFAULT_INPUT_SIZE,
):
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (orig_w, orig_h))

    frame_count = 0
    print("Processing video frames...")
    transforms = T.Compose([T.ToTensor()])

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        resized_frame, ratio, pad_w, pad_h = resize_with_aspect_ratio(frame_pil, input_size)
        orig_size = np.array([[resized_frame.size[1], resized_frame.size[0]]], dtype=np.int64)

        im_data = transforms(resized_frame).unsqueeze(0).numpy()
        labels, boxes, scores = infer_model(compiled_model, im_data, orig_size)

        result_images = draw(
            [frame_pil], labels, boxes, scores, [ratio], [(pad_w, pad_h)], threshold
        )
        frame_with_detections = result_images[0]

        frame_out = cv2.cvtColor(np.array(frame_with_detections), cv2.COLOR_RGB2BGR)
        out.write(frame_out)
        frame_count += 1

        if frame_count % 10 == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()
    out.release()
    print(f"Video processing complete. Result saved as '{output_path}'.")


def main():
    parser = argparse.ArgumentParser(description="Run inference with an OpenVINO RT-DETR model.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the OpenVINO IR (.xml) converted from the exported ONNX model.",
    )
    parser.add_argument("--input", type=str, required=True, help="Path to the input image or video file.")
    parser.add_argument("--device", type=str, default="CPU", help="Target device for inference (e.g., CPU or GPU).")
    parser.add_argument("--threshold", type=float, default=0.4, help="Score threshold for visualizing detections.")
    parser.add_argument(
        "--input-size", type=int, default=DEFAULT_INPUT_SIZE, help="Square input size used during preprocessing."
    )
    args = parser.parse_args()

    core = Core()
    model = core.read_model(args.model)
    # Enable dynamic batch sizes by relaxing the batch dimension to -1 (any)
    images_name = model.inputs[0].get_any_name()
    orig_sizes_name = model.inputs[1].get_any_name()
    model.reshape(
        {
            images_name: PartialShape([-1, 3, args.input_size, args.input_size]),
            orig_sizes_name: PartialShape([-1, 2]),
        }
    )
    compiled_model = core.compile_model(model=model, device_name=args.device)

    try:
        im_pil = Image.open(args.input).convert("RGB")
        im_pil.close()
        process_image(compiled_model, args.input, args.threshold, input_size=args.input_size)
    except (IOError, SyntaxError):
        process_video(compiled_model, args.input, args.threshold, input_size=args.input_size)


if __name__ == "__main__":
    main()
