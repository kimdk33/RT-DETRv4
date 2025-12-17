# Inference Quickstart

This folder hosts lightweight scripts for running RT-DETRv4 with different runtimes. The steps below describe how to prepare and run the OpenVINO pipeline that backs `openvino_inf.py`.

## 1. Export ONNX from a trained checkpoint

```bash
python tools/deployment/export_onnx.py \
  --config configs/dfine/dfine_hgnetv2_l_coco.yml \
  --resume path/to/ckpt.pth \
  --simplify --check
```

The export produces `model.onnx` (or `<ckpt>.onnx` when `--resume` is given). The OpenVINO script expects models with the two inputs `images` and `orig_target_sizes` and three outputs `labels`, `boxes`, and `scores` produced by this exporter.

## 2. Convert ONNX to OpenVINO IR

Install OpenVINO and use the Model Optimizer (MO) to create an IR package:

```bash
mo --input_model model.onnx \
  --output_dir openvino_ir \
  --input_shape "[1,3,640,640]" \
  --data_type FP16
```

This command generates `openvino_ir/model.xml` and `openvino_ir/model.bin`. The `.xml` path is what `openvino_inf.py` consumes via `--model`.

### Dynamic batch sizes

If you need variable batches (for example, to feed multiple images at once), convert with a dynamic batch dimension and let the
script reshape the model at load time:

```bash
mo --input_model model.onnx \
  --output_dir openvino_ir \
  --input_shape "[?,3,640,640]" \
  --data_type FP16
```

The inference script relaxes the batch axis to `-1` before compilation, so IR files built this way can run with any batch size.

## 3. Run inference (image or video)

```bash
python tools/inference/openvino_inf.py \
  --model openvino_ir/model.xml \
  --input demo.jpg \
  --threshold 0.4 \
  --input-size 640
```

* Use any image or video file for `--input`. Images produce a saved JPEG with drawn boxes; videos produce an MP4 file with per-frame detections.
* Choose `--device GPU` (or another supported device) if OpenVINO is set up for it.

> **Do I need a separate ONNX conversion?** Yes. Export your checkpoint to ONNX first (step 1), then convert that ONNX file to OpenVINO IR with MO (step 2). The inference script only accepts the IR format.
