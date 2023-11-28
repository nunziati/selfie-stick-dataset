# Selfie Stick Dataset Generator
Script for generating a selfie-stick detection dataset, starting from videos and some selfie stick images.

## Instalation
```bash
pip install numpy opencv-python ultralytics
```

## Usage
Before running the `build_dataset.py` script, ensure that you have a text file containing the paths to the videos you want to process. This file should be specified using the `--input_videos` argument.

```bash
python build_dataset.py --input_videos path/to/video_paths.txt --selfie_sticks path/to/selfie_sticks --sampling_rate 0.1 --stick_probability 0.5 --angle 0 --max_compression 0 --format yolov8 --device cpu --output_folder path/to/output --buffer_size 128
```

### Arguments
- --input_videos: Path to the text file containing video paths.
- --selfie_sticks: Path to the folder containing selfie sticks.
- --sampling_rate: Percentage of frames to select (default: 0.1).
- --stick_probability: Probability of having a selfie stick in a skeleton (default: 0.5).
- --angle: Maximum absolute deviation angle for selfie stick tilt (default: 0).
- --max_compression: Maximum compression for perspective distortion (default: 0).
- --format: Format for labeling (e.g., "yolov8") (default: yolov8).
- --device: Device to run the model on (e.g., "cpu", "cuda") (default: cpu).
- --output_folder: Path to the output folder.
- --buffer_size: Size of the buffer for parallel processing (default: 128).
