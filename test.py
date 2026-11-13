import fiftyone.zoo as foz

ds = foz.load_zoo_dataset(
    "open-images-v7",
    split="validation",
    label_types=["detections"],
    classes=["Person", "Car", "Bus", "Truck", "Van", "Motorcycle", "Vehicle registration plate"],
    max_samples=50,
)
ds.persistent = True

print(ds)
print(ds.count_values("ground_truth.detections.label"))

sample = ds.first()
print(sample.filepath)
dets = sample.ground_truth.detections
print(len(dets), dets[0] if dets else "no detections on the first sample")
