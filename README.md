# bowling-pin-detection-yolov8
Detects when bowling pins fall in a top-down video and renders an annotated output with timestamps and a live score, using a fine-tuned YOLOv8s model and ByteTrack.

Built by **Toka Mokhtar** and **Rana Ahmed** for DSAI 352 (Computer Vision), Spring 2026, Zewail City.

## Pipeline

1. **Preprocessing:** resize the 4K phone recording to 1080p.
2. **Dataset:** extract one frame per second (31 images) and annotate pins in Roboflow (classes: blue, red, yellow).
3. **Detection:** fine-tune YOLOv8s (pretrained on COCO) on the custom dataset.
4. **Tracking:** ByteTrack assigns stable IDs to each pin, keeping the first 2 IDs per color.
5. **Fall detection:** movement-based analysis of each pin's center point.
6. **Rendering:** OpenCV overlay with bounding boxes, fall timestamps, and a pins-down counter.

## The main challenge

The camera is overhead, so standing and fallen pins both look roughly circular. A bounding-box aspect-ratio approach was tested first and discarded because it was unreliable.

Instead, a pin is marked as fallen when its center displacement spikes above 10 px and the average movement over the next 10 frames drops below 8 px (meaning the pin has come to rest).

## Results

| Class  | Precision | Recall | mAP50 | mAP50-95 |
|--------|-----------|--------|-------|----------|
| all    | 0.994     | 0.969  | 0.995 | 0.975    |
| blue   | 0.992     | 1.000  | 0.995 | 0.979    |
| red    | 1.000     | 0.907  | 0.995 | 0.963    |
| yellow | 0.992     | 1.000  | 0.995 | 0.982    |

Trained on 23 images, validated on 5, tested on 3. Fall detection identified 5 of the 6 pins that fell, with timestamps.

These numbers come from a small dataset taken from a single scene, so they should not be read as general performance.

## Limitations

- Small dataset (31 images from one video).
- Track IDs can change when a pin is briefly hidden.
- The pipeline assumes exactly 6 pins (2 per color).
- The ball is not tracked.

See the full technical report in [`docs/`](docs/bowling_report.pdf).

## Tech stack

Python, Ultralytics YOLOv8, ByteTrack, OpenCV, Roboflow, NumPy, Matplotlib, Google Colab.

## How to run

1. Open `bowling_pin_detection.ipynb` in Google Colab.
2. Add your own Roboflow key as a Colab secret named `ROBOFLOW_API_KEY`.
3. Upload your own bowling video when the notebook asks for it, then run the cells in order.
