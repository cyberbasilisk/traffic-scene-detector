# Dataset card

Status: **draft** (fill in as the data pipeline is built).

## Purpose

Training and evaluating a detector for three classes: `person`, `vehicle`, `license_plate`.

## Classes

| Id | Name | Open Images labels merged into it |
|---|---|---|
| 0 | person | Person, Man, Woman, Boy, Girl |
| 1 | vehicle | Car, Bus, Truck, Van, Motorcycle |
| 2 | license_plate | Vehicle registration plate |

## Label policy

- Every other Open Images label on the downloaded images (for example Wheel, Clothing, Tree) is dropped.
- The generic labels `Vehicle` and `Land vehicle` are excluded for now, because they can include boats, aircraft and so on. They are counted separately and can be reviewed later.
- Boxes marked `IsGroupOf` (one box for many objects) or `IsDepiction` (drawings, paintings, statues) are skipped.
- A person labelled both `Person` and `Man` (and so on) would produce two boxes for one object, so same-class boxes that overlap another by more than 0.7 IoU are merged, keeping the larger.
- Boxes are clamped to the image, and boxes with no area after clamping are dropped.

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

### Early sample check

A 50-image sample from the validation split contained 77 `Person`, 31 `Car` and 6 `Vehicle registration plate` boxes before any filtering, so license plates are much rarer than the other classes. The final dataset needs extra images selected for plates.

## Known issues and limitations

- License plates are very small relative to the frame, which makes them the hardest class, and they are scarce.
- Open Images does not label every object in every image, so some people, vehicles and plates have no box. This adds noise to training and to the metrics.
- Source images over-represent some regions, camera types and plate formats (TBD: document what you find).
- Merged vehicle classes hide differences between vehicle types.

## Privacy

The images contain people and license plates. Use them for training and metrics only. Do not publish unblurred examples; blur or crop faces and plates in any README image, post or video.
