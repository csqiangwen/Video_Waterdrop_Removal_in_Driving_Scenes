# Video Waterdrop Removal via Spatio-Temporal Fusion in Driving Scenes
[Paper](https://arxiv.org/abs/2302.05916)
<p align='center'>
<img src='visualization/8.gif' width=300>
<img src='visualization/5.gif' width=300>
</p>


This is the official PyTorch implementation. This works aims at removing various types of waterdrops for driving cars on rainy days. We also provide a large-scale synthetic dataset for the video waterdrop removal task.

#### Video Waterdrop Removal via Spatio-Temporal Fusion in Driving Scenes
Qiang Wen, [Yue Wu](https://yuewuhkust.github.io/), [Qifeng Chen](https://cqf.io/) <br />
The Hong Kong University of Science and Technology <br />
IEEE International Conference on Robotics and Automation (ICRA), 2023

## Requirements
- Pytorch 1.9
- OpenCV-Python

If **conda** has been installed, you can directly build the running environment via:
```bash
conda env create -f environment.yaml
```
An environment named "th" will be created.

## Training
- Download the [training dataset](https://hkustconnect-my.sharepoint.com/:f:/g/personal/qwenab_connect_ust_hk/EkaXnD0dkclCvZVk_jqk-IcBGauk7HE-4ZWARfghMiQcPg?e=SuPWoF), and put it in ```dataset/```;
- To train a model:
```
$ bash train.sh
```
You can also use the command ```tensorboard --logdir=runs``` to visually check the training results.
## Testing
- Download the [pretrained model](https://drive.google.com/drive/folders/1c3JYdv64U-OmOyksNK6n51sNwBgy-iQC?usp=sharing), and put it in ```checkpoints_waterdrop/```;
- Download the [test dataset](https://hkustconnect-my.sharepoint.com/:u:/g/personal/qwenab_connect_ust_hk/ESVlboaVRvtLk1xth__yb6QB7t3CSBMaN9G_-SJNKVG8MQ?e=4L45pS), and unzip it in ```dataset/```;
- To test:
```
$ bash test.sh
```
You can choose the test on the synthetic dataset or real-world dataset by specifying ```--data_type```

## Citation
If you find this repository useful for your research, please cite the following work.
```
@inproceedings{wen2023video,
  title={Video Waterdrop Removal via Spatio-Temporal Fusion in Driving Scenes},
  author={Wen, Qiang and Wu, Yue and Chen, Qifeng},
  booktitle={2023 International Conference on Robotics and Automation (ICRA)},
  year={2023},
  organization={IEEE}
}
```
