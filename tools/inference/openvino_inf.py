"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

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


def is_video_file(path: Path) -> bool:
    return path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"}


def chunked(iterable: Iterable[str], batch_size: int) -> Iterable[List[str]]:
    """Yield successive batches from an iterable."""
    batch: List[str] = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


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
    file_paths: List[str],
    threshold: float = 0.4,
    output_dir: str = "openvino_results",
    input_size: int = DEFAULT_INPUT_SIZE,
    batch_size: int = 1,
):
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    transforms = T.Compose([T.ToTensor()])

    for batch in chunked(file_paths, batch_size):
        pil_images: List[Image.Image] = []
        tensors: List[np.ndarray] = []
        orig_sizes: List[np.ndarray] = []
        ratios: List[float] = []
        paddings: List[Tuple[int, int]] = []

        for file_path in batch:
            im_pil = Image.open(file_path).convert("RGB")
            resized_im, ratio, pad_w, pad_h = resize_with_aspect_ratio(im_pil, input_size)
            orig_size = np.array([resized_im.size[1], resized_im.size[0]], dtype=np.int64)

            tensors.append(transforms(resized_im).numpy())
            orig_sizes.append(orig_size)
            ratios.append(ratio)
            paddings.append((pad_w, pad_h))
            pil_images.append(im_pil)

        im_batch = np.stack(tensors, axis=0)
        orig_sizes_batch = np.stack(orig_sizes, axis=0)
        labels, boxes, scores = infer_model(compiled_model, im_batch, orig_sizes_batch)

        result_images = draw(pil_images, labels, boxes, scores, ratios, paddings, threshold)

        for img_path, result_img in zip(batch, result_images):
            stem = Path(img_path).stem
            save_path = output_root / f"{stem}_det.jpg"
            result_img.save(save_path)
            print(f"Saved result for '{img_path}' -> '{save_path}'.")


def process_video(
    compiled_model,
    video_path: str,
    threshold: float = 0.4,
    output_path: str = "openvino_result.mp4",
    input_size: int = DEFAULT_INPUT_SIZE,
    batch_size: int = 1,
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

    def process_batch(frames: List[np.ndarray], start_index: int) -> int:
        pil_images: List[Image.Image] = []
        tensors: List[np.ndarray] = []
        orig_sizes: List[np.ndarray] = []
        ratios: List[float] = []
        paddings: List[Tuple[int, int]] = []

        for frame in frames:
            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            orig_frame_h, orig_frame_w = frame.shape[:2]
            resized_frame, ratio, pad_w, pad_h = resize_with_aspect_ratio(frame_pil, input_size)
            orig_size = np.array([orig_frame_h, orig_frame_w], dtype=np.int64)

            tensors.append(transforms(resized_frame).numpy())
            orig_sizes.append(orig_size)
            ratios.append(ratio)
            paddings.append((pad_w, pad_h))
            pil_images.append(frame_pil)

        im_batch = np.stack(tensors, axis=0)
        orig_sizes_batch = np.stack(orig_sizes, axis=0)
        labels, boxes, scores = infer_model(compiled_model, im_batch, orig_sizes_batch)

        result_images = draw(pil_images, labels, boxes, scores, ratios, paddings, threshold)
        for result_img in result_images:
            frame_out = cv2.cvtColor(np.array(result_img), cv2.COLOR_RGB2BGR)
            out.write(frame_out)

        processed = len(frames)
        total_processed = start_index + processed
        if total_processed % 10 == 0:
            print(f"Processed {total_processed} frames...")
        return processed

    buffer: List[np.ndarray] = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        buffer.append(frame)
        if len(buffer) == batch_size:
            frame_count += process_batch(buffer, frame_count)
            buffer = []

    if buffer:
        frame_count += process_batch(buffer, frame_count)

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
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for image inputs or the number of frames to infer together for video inputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="openvino_results",
        help="Directory to store visualization outputs for image batches.",
    )
    args = parser.parse_args()

    core = Core()
    model = core.read_model(args.model)

    images_input = model.inputs[0]
    orig_sizes_input = model.inputs[1]
    batch_dim = images_input.partial_shape[0]

    if batch_dim.is_dynamic:
        # Enable dynamic batch sizes by relaxing the batch dimension to -1 (any)
        model.reshape(
            {
                images_input.get_any_name(): PartialShape([-1, 3, args.input_size, args.input_size]),
                orig_sizes_input.get_any_name(): PartialShape([-1, 2]),
            }
        )
    else:
        static_batch = int(batch_dim)
        if args.batch_size != static_batch:
            raise ValueError(
                "The loaded OpenVINO model was converted with a static batch size "
                f"of {static_batch}, so --batch-size must be set to {static_batch}. "
                "Reconvert the model with dynamic input shapes to run with other batch sizes."
            )

    # Some IRs keep static batch dimensions baked into output shapes (e.g., reshape constants),
    # which makes runtime batching incompatible even when inputs are dynamic.
    output_batches = {int(out.partial_shape[0]) for out in model.outputs if not out.partial_shape[0].is_dynamic}
    if output_batches:
        if len(output_batches) > 1:
            raise ValueError(
                "Model outputs have inconsistent static batch dimensions: "
                f"{sorted(output_batches)}. Please re-export/convert the model with consistent shapes."
            )
        output_batch = output_batches.pop()
        if args.batch_size != output_batch:
            raise ValueError(
                "This IR was converted with static output batch dimensions, so --batch-size must be "
                f"{output_batch}. Reconvert with dynamic batch axes (e.g., images[?,3,...],orig_target_sizes[?,2]) "
                "to run other batch sizes."
            )
    compiled_model = core.compile_model(model=model, device_name=args.device)

    # Accept a comma-separated list of image files, a single image, or a directory of images.
    if "," in args.input:
        image_paths = [path.strip() for path in args.input.split(",") if path.strip()]
        for path_str in image_paths:
            path = Path(path_str)
            if not path.exists():
                raise FileNotFoundError(f"Input file '{path_str}' does not exist.")
            if is_video_file(path):
                raise ValueError("Video inputs cannot be combined in a comma-separated image list.")
    else:
        input_path = Path(args.input)
        if not input_path.exists():
            raise FileNotFoundError(f"Input path '{args.input}' does not exist.")

        if is_video_file(input_path):
            process_video(
                compiled_model,
                str(input_path),
                args.threshold,
                input_size=args.input_size,
                batch_size=args.batch_size,
            )
            return

        if input_path.is_dir():
            image_paths = sorted(
                [str(p) for p in input_path.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
            )
        else:
            image_paths = [str(input_path)]

    if not image_paths:
        raise ValueError("No image files found for inference.")

    process_image(
        compiled_model,
        image_paths,
        args.threshold,
        output_dir=args.output_dir,
        input_size=args.input_size,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
