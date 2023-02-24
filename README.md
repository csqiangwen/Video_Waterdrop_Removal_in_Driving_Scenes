# Video Waterdrop Removal via Spatio-Temporal Fusion in Driving Scenes
Project | [Paper](https://arxiv.org/pdf/2302.05916.pdf)

This is the official PyTorch implementation of "Video Waterdrop Removal via Spatio-Temporal Fusion in Driving Scenes". This works aims at removing various types of waterdrops for driving cars on rainy days. We also provide a large-scale synthetic dataset for the video waterdrop removal task.
## Requirements
- Pytorch 1.9
- OpenCV-Python

If **conda** has been installed, you can directly build the running environment via:
```bash
conda env create -f environment.yaml
```
An environment named "th" will be created.

## Training
Download the training set, and put it under ```./dataset/```
