# Continuous 3D Perception Model with Persistent State
<div align="center">
  <img src="./assets/factory-ezgif.com-video-speed.gif"  alt="CUT3R" />
</div>

<hr>

<br>
Official implementation of <strong>Continuous 3D Perception Model with Persistent State</strong>, CVPR 2025 (Oral)

[*QianqianWang**](https://qianqianwang68.github.io/),
[*Yifei Zhang**](https://forrest-110.github.io/),
[*Aleksander Holynski*](https://holynski.org/),
[*Alexei A Efros*](https://people.eecs.berkeley.edu/~efros/),
[*Angjoo Kanazawa*](https://people.eecs.berkeley.edu/~kanazawa/)


(*: equal contribution)

<div style="line-height: 1;">
  <a href="https://cut3r.github.io/" target="_blank" style="margin: 2px;">
    <img alt="Website" src="https://img.shields.io/badge/Website-CUT3R-536af5?color=536af5&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://arxiv.org/pdf/2501.12387" target="_blank" style="margin: 2px;">
    <img alt="Arxiv" src="https://img.shields.io/badge/Arxiv-CUT3R-red?logo=%23B31B1B" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>


![Example of capabilities](assets/ezgif.com-video-to-gif-converter.gif)

## Table of Contents
- [TODO](#todo)
- [Get Started](#getting-started)
  - [Installation](#installation)
  - [Checkpoints](#download-checkpoints)
  - [Inference](#inference)
- [Datasets](#datasets)
- [Evaluation](#evaluation)
  - [Datasets](#datasets-1)
  - [Evaluation Scripts](#evaluation-scripts)
- [Training and Fine-tuning](#training-and-fine-tuning)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)

## TODO
- [x] Release multi-view stereo results of DL3DV dataset.
- [ ] Online demo integrated with WebCam

## Getting Started

### Installation

1. Clone CUT3R.
```bash
git clone https://github.com/CUT3R/CUT3R.git
cd CUT3R
```

2. Create the environment.
```bash
conda create -n cut3r python=3.11 cmake=3.14.0
conda activate cut3r
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia  # use the correct version of cuda for your system
pip install -r requirements.txt
# issues with pytorch dataloader, see https://github.com/pytorch/pytorch/issues/99625
conda install 'llvm-openmp<16'
# for training logging
pip install git+https://github.com/nerfstudio-project/gsplat.git
# for evaluation
pip install evo
pip install open3d
```

3. Compile the cuda kernels for RoPE (as in CroCo v2).
```bash
cd src/croco/models/curope/
python setup.py build_ext --inplace
cd ../../../../
```

### Download Checkpoints

We currently provide checkpoints on Google Drive:

| Modelname   | Training resolutions | #Views| Head |
|-------------|----------------------|-------|------|
| [`cut3r_224_linear_4.pth`](https://drive.google.com/file/d/11dAgFkWHpaOHsR6iuitlB_v4NFFBrWjy/view?usp=drive_link) | 224x224 | 16 | Linear |
| [`cut3r_512_dpt_4_64.pth`](https://drive.google.com/file/d/1Asz-ZB3FfpzZYwunhQvNPZEUA8XUNAYD/view?usp=drive_link) | 512x384, 512x336, 512x288, 512x256, 512x160, 384x512, 336x512, 288x512, 256x512, 160x512 | 4-64 | DPT |

> `cut3r_224_linear_4.pth` is our intermediate checkpoint and `cut3r_512_dpt_4_64.pth` is our final checkpoint.

To download the weights, run the following commands:
```bash
cd src
# for 224 linear ckpt
gdown --fuzzy https://drive.google.com/file/d/11dAgFkWHpaOHsR6iuitlB_v4NFFBrWjy/view?usp=drive_link 
# for 512 dpt ckpt
gdown --fuzzy https://drive.google.com/file/d/1Asz-ZB3FfpzZYwunhQvNPZEUA8XUNAYD/view?usp=drive_link
cd ..
```

### Inference

To run the inference code, you can use the following command:
```bash
# the following script will run inference offline and visualize the output with viser on port 8080
python demo.py --model_path MODEL_PATH --seq_path SEQ_PATH --size SIZE --vis_threshold VIS_THRESHOLD --output_dir OUT_DIR  # input can be a folder or a video
# Example:
#     python demo.py --model_path src/cut3r_512_dpt_4_64.pth --size 512 \
#         --seq_path examples/001 --vis_threshold 1.5 --output_dir tmp
#
#     python demo.py --model_path src/cut3r_224_linear_4.pth --size 224 \
#         --seq_path examples/001 --vis_threshold 1.5 --output_dir tmp

# the following script will run inference with global alignment and visualize the output with viser on port 8080
python demo_ga.py --model_path MODEL_PATH --seq_path SEQ_PATH --size SIZE --vis_threshold VIS_THRESHOLD --output_dir OUT_DIR
```
Output results will be saved to `output_dir`.

> Currently, we accelerate the feedforward process by processing inputs in parallel within the encoder, which results in linear memory consumption as the number of frames increases.

## Datasets
Our training data includes 32 datasets listed below. We provide processing scripts for all of them. Please download the datasets from their official sources, and refer to [preprocess.md](docs/preprocess.md) for processing scripts and more information about the datasets.

  - [ARKitScenes](https://github.com/apple/ARKitScenes) 
  - [BlendedMVS](https://github.com/YoYo000/BlendedMVS)
  - [CO3Dv2](https://github.com/facebookresearch/co3d)
  - [MegaDepth](https://www.cs.cornell.edu/projects/megadepth/)
  - [ScanNet++](https://kaldir.vc.in.tum.de/scannetpp/) 
  - [ScanNet](http://www.scan-net.org/ScanNet/)
  - [WayMo Open dataset](https://github.com/waymo-research/waymo-open-dataset)
  - [WildRGB-D](https://github.com/wildrgbd/wildrgbd/)
  - [Map-free](https://research.nianticlabs.com/mapfree-reloc-benchmark/dataset)
  - [TartanAir](https://theairlab.org/tartanair-dataset/)
  - [UnrealStereo4K](https://github.com/fabiotosi92/SMD-Nets) 
  - [Virtual KITTI 2](https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/)
  - [3D Ken Burns](https://github.com/sniklaus/3d-ken-burns.git)
  - [BEDLAM](https://bedlam.is.tue.mpg.de/)
  - [COP3D](https://github.com/facebookresearch/cop3d)
  - [DL3DV](https://github.com/DL3DV-10K/Dataset)
  - [Dynamic Replica](https://github.com/facebookresearch/dynamic_stereo)
  - [EDEN](https://lhoangan.github.io/eden/)
  - [Hypersim](https://github.com/apple/ml-hypersim)
  - [IRS](https://github.com/HKBU-HPML/IRS)
  - [Matterport3D](https://niessner.github.io/Matterport/)
  - [MVImgNet](https://github.com/GAP-LAB-CUHK-SZ/MVImgNet)
  - [MVS-Synth](https://phuang17.github.io/DeepMVS/mvs-synth.html)
  - [OmniObject3D](https://omniobject3d.github.io/)
  - [PointOdyssey](https://pointodyssey.com/)
  - [RealEstate10K](https://google.github.io/realestate10k/)
  - [SmartPortraits](https://mobileroboticsskoltech.github.io/SmartPortraits/)
  - [Spring](https://spring-benchmark.org/)
  - [Synscapes](https://synscapes.on.liu.se/)
  - [UASOL](https://osf.io/64532/)
  - [UrbanSyn](https://www.urbansyn.org/)
  - [HOI4D](https://hoi4d.github.io/)


## Evaluation

### Datasets
Please follow [MonST3R](https://github.com/Junyi42/monst3r/blob/main/data/evaluation_script.md) and [Spann3R](https://github.com/HengyiWang/spann3r/blob/main/docs/data_preprocess.md) to prepare **Sintel**, **Bonn**, **KITTI**, **NYU-v2**, **TUM-dynamics**, **ScanNet**, **7scenes** and **Neural-RGBD** datasets.

The datasets should be organized as follows:
```
data/
├── 7scenes
├── bonn
├── kitti
├── neural_rgbd
├── nyu-v2
├── scannetv2
├── sintel
└── tum
```

### Evaluation Scripts
Please refer to the [eval.md](docs/eval.md) for more details.

## Training and Fine-tuning
Please refer to the [train.md](docs/train.md) for more details.

## Acknowledgements
Our code is based on the following awesome repositories:

- [DUSt3R](https://github.com/naver/dust3r)
- [MonST3R](https://github.com/Junyi42/monst3r.git)
- [Spann3R](https://github.com/HengyiWang/spann3r.git)
- [Viser](https://github.com/nerfstudio-project/viser)

We thank the authors for releasing their code!



## Citation

If you find our work useful, please cite:

```bibtex
@article{wang2025continuous,
  title={Continuous 3D Perception Model with Persistent State},
  author={Wang, Qianqian and Zhang, Yifei and Holynski, Aleksander and Efros, Alexei A and Kanazawa, Angjoo},
  journal={arXiv preprint arXiv:2501.12387},
  year={2025}
}
```

