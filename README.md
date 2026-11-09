# Traffic Scene Detector

Detect **people**, **vehicles** and **license plates** in images and video with a YOLO model.
Built as a portfolio project: reproducible training, honest evaluation, and clear documentation.

> **Status:** Part 1 in progress (data pipeline and baseline). No results yet.

## Results

| Model | Image size | mAP@0.5 | mAP@0.5:0.95 | Person AP | Vehicle AP | Plate AP |
|---|---|---|---|---|---|---|
| Baseline | 640 | TBD | TBD | TBD | TBD | TBD |

Recall by object size, error analysis and example predictions will go here. Example images are blurred or cropped to protect people's privacy.

## Classes

`person`, `vehicle` (car, bus, truck, van, motorcycle merged), `license_plate`. See [docs/dataset_card.md](docs/dataset_card.md).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

## Data

Data is not committed. Download and prepare it with the scripts (planned):

```bash
python3 scripts/download_data.py --split validation --max-samples 500
```

## Training and evaluation (planned)

Every run is driven by a config in `configs/`, with a fixed seed and tracked metrics.

## Development

```bash
python3 -m pip install -r requirements-dev.txt
ruff check . && ruff format --check .
pytest -q
```

CI runs the same checks on every push.

## Project structure

```
traffic-scene-detector/
├── configs/                # dataset and training configs
├── docs/                   # dataset card (model card to come)
├── scripts/                # data download and preparation
├── src/traffic_detector/   # class mapping, label conversion, config loading
├── tests/
├── data/                   # not committed
└── runs/                   # training outputs, not committed
```

## Limitations

- License plates are small, so recall drops sharply on distant vehicles.
- Public datasets have geographic and camera biases; results may not transfer to other regions.
- Detection only: no plate reading (OCR) and no tracking.

## Licenses and attribution

- **This repository:** TBD. Note that the Ultralytics library is AGPL-3.0, which may affect how you can license code that uses it; check before choosing.
- **Data:** see [docs/dataset_card.md](docs/dataset_card.md) for sources, licenses and required attribution.

## Privacy

The data contains people and license plates. Use it for training and metrics only, and blur or crop faces and plates in anything you publish.

## Roadmap

- [ ] Part 1: data pipeline, baseline training, evaluation, first README results
- [ ] Part 2: error analysis, improvements (higher resolution), Docker, model card, demo
