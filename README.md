# Car-part semantic segmentation

Pixel-level segmentation of exterior car parts (doors, bumpers, glass, lights, mirrors,
wheels) in photos of a single vehicle. Inspection and damage-assessment pipelines need
to know which panel a pixel belongs to before they can say anything else about it.
This repo covers the path from a data audit to a TensorRT engine: leak-free splits, a
loss designed for heavy class imbalance, a comparison of several architectures including
an MMSegmentation implementation, and ONNX export with quantization and pruning.

<!-- TODO: qualitative result image, chapter 6 -->

| model | test mIoU | small-parts mIoU | latency |
|---|---|---|---|
| <!-- TODO: chapter 6 --> | | | |

## Status

| # | chapter | state | hardware |
|---|---|---|---|
| 1 | repo bootstrap | done | Mac |
| 2 | data pipeline | not started | Mac |
| 3 | EDA and class weights | not started | Mac |
| 4 | training pipeline, first model | not started | Mac, Colab |
| 5 | imbalance-aware loss | not started | Colab |
| 6 | model comparison A | not started | Colab / Kaggle |
| 7 | model comparison B, leakage experiment | not started | Colab / Kaggle |
| 8 | MMSegmentation | not started | Colab |
| 9 | ONNX export and int8 | not started | Mac |
| 10 | pruning and TensorRT | not started | Colab |

## Quickstart

```bash
git clone https://github.com/IslamFadl/car-part-segmentation.git
cd car-part-segmentation
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python tools/check_env.py
python tools/download_data.py
pytest -q
```

## Data and licenses

Dataset: [Ultralytics carparts-seg](https://docs.ultralytics.com/datasets/segment/carparts-seg/).
`tools/download_data.py` downloads it at run time and verifies it by SHA-256. It is never committed.
<!-- TODO: image and class counts from the download output -->

| what | license | what that means here |
|---|---|---|
| code in this repo | MIT | reuse freely |
| carparts-seg dataset | AGPL-3.0 | not redistributed; trained weights are not published |
| SegFormer pretrained weights (NVIDIA) | NVIDIA Source Code License, non-commercial | used for research and evaluation only; fine-tuned SegFormer weights are not published |
| UperNet + ConvNeXt-T (openmmlab) | MIT | no restriction |
| DINOv2, torchvision and MMSegmentation weights | <!-- TODO: verify in the chapter that first uses them --> | |

## Method
<!-- TODO: chapters 2 to 5 -->

## Results
<!-- TODO: chapters 5 to 10, including what did not work -->

## Design decisions and gaps

- **Environment:** pip and venv, dependency ranges in `pyproject.toml`, exact versions in a
  per-platform lock (`requirements-lock-macos.txt`). Every training run saves its own
  environment report. uv would be faster; Colab and Kaggle are pip-native, so pip won.
- **Data:** downloaded at run time and checksummed, which also keeps AGPL data out of an MIT repo.
- **Reproducibility:** seeded runs repeat on the same machine and software. They are not
  bitwise identical across GPUs or on Apple MPS, so results are reported over seeds.

## Limitations and next steps
<!-- TODO -->