# Dataset card

Status: **draft** (fill in as the data pipeline is built).

## Purpose

Training and evaluating a detector for three classes: `person`, `vehicle`, `license_plate`.

## Classes

| Id | Name | Source labels merged into it |
|---|---|---|
| 0 | person | Person |
| 1 | vehicle | Car, Bus, Truck, Van, Motorcycle |
| 2 | license_plate | Vehicle registration plate |

## Sources and licenses

| Source | What was used | Annotation license | Image license | Attribution |
|---|---|---|---|---|
| Open Images V7 | class-filtered subset | TBD (verify) | TBD (verify) | TBD |

CC BY licenses require crediting the original authors. Record exactly what applies to the images and annotations you actually use.

## Size and splits

| Split | Images | Person | Vehicle | License plate |
|---|---|---|---|---|
| train | TBD | TBD | TBD | TBD |
| val | TBD | TBD | TBD | TBD |
| test | TBD | TBD | TBD | TBD |

Splits follow the source dataset's own splits, so no image appears in more than one.

## Known issues and limitations

- License plates are very small relative to the frame, which makes them the hardest class.
- Source images over-represent some regions, camera types and plate formats (TBD: document what you find).
- Merged vehicle classes hide differences between vehicle types.

## Privacy

The images contain people and license plates. Use them for training and metrics only. Do not publish unblurred examples; blur or crop faces and plates in any README image, post or video.
