# Inference Quickstart

This folder hosts lightweight scripts for running RT-DETRv4 with different runtimes. The steps below describe how to prepare and run the OpenVINO pipeline that backs `openvino_inf.py`.

## 1. Export ONNX from a trained checkpoint

```bash
python tools/deployment/export_onnx.py \
  --config configs/dfine/dfine_hgnetv2_l_coco.yml \
  --resume path/to/ckpt.pth \
  --simplify --check
```

The export produces `model.onnx` (or `<ckpt>.onnx` when `--resume` is given). The OpenVINO script expects models with the two inputs `images` and `orig_target_sizes` and three outputs `labels`, `boxes`, and `scores` produced by this exporter. **배치 차원은 기본적으로 `dynamic_axes`로 열려 있으니 유동 배치를 위해 별도 옵션을 줄 필요가 없습니다.**

## 2. Convert ONNX to OpenVINO IR

Install OpenVINO and use the Model Optimizer CLI (`ovc`, formerly `mo`) to create an IR package. The examples below follow the [OpenVINO conversion parameter guide](https://docs.openvino.ai/2025/openvino-workflow/model-preparation/conversion-parameters.html) syntax for `--input` shapes, using the comma-separated `name[dim]` form shown in `ovc -h`. **`ovc` takes the ONNX path as a positional `INPUT_MODEL` argument (no `--input_model` flag).**

```bash
ovc model.onnx \
  --output_dir openvino_ir \
  --input "images[1,3,640,640],orig_target_sizes[1,2]" \
  --compress_to_fp16
```

This command generates `openvino_ir/model.xml` and `openvino_ir/model.bin`. The `.xml` path is what `openvino_inf.py` consumes via `--model`.

### Dynamic batch sizes

If you need variable batches (for example, to feed multiple images at once), convert with a dynamic batch dimension and let the script reshape the model at load time:

```bash
ovc model.onnx \
  --output_dir openvino_ir \
  --input "images[?,3,640,640],orig_target_sizes[?,2]" \
  --compress_to_fp16
```

The inference script relaxes the batch axis to `-1` before compilation, so IR files built this way can run with any batch size.
ONNX 내 입력 축 이름이 `images`와 `orig_target_sizes`로 유지되고, 두 축 모두 batch 축이 열려 있어야 하므로 `export_onnx.py` 기본 설정을 그대로 사용하세요.

> **Static IR?** If you converted with a fixed batch (e.g., `images[1,3,640,640]`), `--batch-size` must match that value. 다른 배치 크기를 사용하려면 `--input "images[?,3,640,640],orig_target_sizes[?,2]"`와 같이 동적 배치로 다시 변환하세요.

> **Static outputs?** 일부 IR은 내부 `reshape` 등으로 출력 배치를 1로 고정합니다. `openvino_inf.py`는 이런 IR에 대해 `--batch-size`가 1이 아니면 즉시 에러로 안내하므로, 여러 프레임/이미지를 한 번에 처리하려면 `ovc --input "images[?,3,...],orig_target_sizes[?,2]"`와 같이 배치 축을 열어 다시 변환해야 합니다.
>
> **강제 배치 크기 고정?** IR에서 입력 또는 출력 배치가 고정되어 있으면 스크립트가 요청한 배치 대신 고정된 배치 크기로 자동 변경하며, 콘솔에 경고를 남깁니다. 온전한 배치 추론을 원한다면 ONNX 내 배치 축이 열려 있는지 확인하고(`ovc --input "images[?,3,640,640],orig_target_sizes[?,2]"`), 필요한 경우 ONNX를 다시 내보낸 뒤 재변환하세요.

## 3. Run inference (image or video)

```bash
python tools/inference/openvino_inf.py \
  --model openvino_ir/model.xml \
  --input demo.jpg \
  --threshold 0.4 \
  --input-size 640
```

* Use any image or video file for `--input`. Images produce a saved JPEG with drawn boxes; videos produce an MP4 file with detections drawn on each processed frame.
* Choose `--device GPU` (or another supported device) if OpenVINO is set up for it.
* 이미지/동영상 모두 `--batch-size`로 묶음 크기를 지정할 수 있습니다. 이미지의 경우 `--input`에 이미지 디렉터리나 콤마로 구분한 여러 경로를 넣어 배치 단위로 처리하며, 결과는 `--output-dir`(기본 `openvino_results/`) 아래에 `<원본이름>_det.jpg`로 저장됩니다. 동영상의 경우 연속된 프레임을 배치 단위로 묶어 한 번의 추론으로 처리합니다.

예) 이미지 4장을 배치 2로 추론

```bash
python tools/inference/openvino_inf.py \
  --model openvino_ir/model.xml \
  --input "img1.jpg,img2.jpg,img3.jpg,img4.jpg" \
  --batch-size 2 \
  --input-size 640 \
  --threshold 0.4
```

> **Do I need a separate ONNX conversion?** Yes. Export your checkpoint to ONNX first (step 1), then convert that ONNX file to OpenVINO IR with MO (step 2). The inference script only accepts the IR format.
