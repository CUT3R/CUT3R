# Preprocess Scripts

Please download all datasets from their <strong>original sources</strong>, except for vKITTI, for which we provide a fully processed version—no need to download the original dataset. For MapFree and DL3DV, we also release depth maps computed using COLMAP Multi-View Stereo (MVS). See the sections below for details on the processing of each dataset. <strong>Please ensure compliance with the respective licensing agreements when downloading.</strong> The total data takes about 25TB of disk space.

> If you encounter issues in the scripts, please feel free to create an issue.

- [ARKitScenes](#arkitscenes) 
- [BlendedMVS](#blendedmvs)
- [CO3Dv2](#co3d)
- [MegaDepth](#megadepth)
- [ScanNet++](#scannet-1) 
- [ScanNet](#scannet)
- [WayMo Open dataset](#waymo)
- [WildRGB-D](#wildrgbd)
- [Map-free](#mapfree)
- [TartanAir](#tartanair)
- [UnrealStereo4K](#unrealstereo4k) 
- [Virtual KITTI 2](#virtual-kitti-2)
- [3D Ken Burns](#3d-ken-burns)
- [BEDLAM](#bedlam)
- [COP3D](#cop3d)
- [DL3DV](#dl3dv)
- [Dynamic Replica](#dynamic-replica)
- [EDEN](#eden)
- [Hypersim](#hypersim)
- [IRS](#irs)
- [Matterport3D](#matterport3d)
- [MVImgNet](#mvimgnet)
- [MVS-Synth](#mvs-synth)
- [OmniObject3D](#omniobject3d)
- [PointOdyssey](#pointodyssey)
- [RealEstate10K](#realestate10k)
- [SmartPortraits](#smartportraits)
- [Spring](#spring)
- [Synscapes](#synscapes)
- [UASOL](#uasol)
- [UrbanSyn](#urbansyn)
- [HOI4D](#hoi4d)

## [ARKitScenes](https://github.com/apple/ARKitScenes)
<div>
<img src="https://img.shields.io/badge/INDOOR-blue" style="display: inline-block; vertical-align: middle;"/>
<!-- <img src="https://img.shields.io/badge/MIXED-blue" style="display: inline-block; vertical-align: middle;"/>
<img src="https://img.shields.io/badge/OUTDOOR-blue" style="display: inline-block; vertical-align: middle;"/>
<img src="https://img.shields.io/badge/ObjectCentric-blue" style="display: inline-block; vertical-align: middle;"/> -->
<img src="https://img.shields.io/badge/METRIC-red" style="display: inline-block; vertical-align: middle;"/>
<img src="https://img.shields.io/badge/RealWorld-orange" style="display: inline-block; vertical-align: middle;"/>
<!-- <img src="https://img.shields.io/badge/Synthetic-orange" style="display: inline-block; vertical-align: middle;"/> -->
<!-- 
<img src="https://img.shields.io/badge/Dynamic-green" style="display: inline-block; vertical-align: middle;"/> -->
<img src="https://img.shields.io/badge/Static-green" style="display: inline-block; vertical-align: middle;"/>
<!-- <img src="https://img.shields.io/badge/CameraOnly-crimson" style="display: inline-block; vertical-align: middle;"/>
<img src="https://img.shields.io/badge/SingleView-crimson" style="display: inline-block; vertical-align: middle;"/> -->
<br>
</div>

First download the pre-computed pairs provided by [DUSt3R](https://download.europe.naverlabs.com/ComputerVision/DUSt3R/arkitscenes_pairs.zip).

Then run the following command,
```
python preprocess_arkitscenes.py --arkitscenes_dir /path/to/your/raw/data --precomputed_pairs /path/to/your/pairs --output_dir /path/to/your/outdir

python generate_set_arkitscenes.py --root /path/to/your/outdir --splits Training Test --max_interval 5.0 --num_workers 8
```

## [ARKitScenes_highres](https://github.com/apple/ARKitScenes)
<div>
<img src="https://img.shields.io/badge/INDOOR-blue" style="display: inline-block; vertical-align: middle;"/>
<!-- <img src="https://img.shields.io/badge/MIXED-blue" style="display: inline-block; vertical-align: middle;"/>
<img src="https://img.shields.io/badge/OUTDOOR-blue" style="display: inline-block; vertical-align: middle;"/>
<img src="https://img.shields.io/badge/ObjectCentric-blue" style="display: inline-block; vertical-align: middle;"/> -->
<img src="https://img.shields.io/badge/METRIC-red" style="display: inline-block; vertical-align: middle;"/>
<img src="https://img.shields.io/badge/RealWorld-orange" style="display: inline-block; vertical-align: middle;"/>
<!-- <img src="https://img.shields.io/badge/Synthetic-orange" style="display: inline-block; vertical-align: middle;"/> -->
<!-- 
<img src="https://img.shields.io/badge/Dynamic-green" style="display: inline-block; vertical-align: middle;"/> -->
<img src="https://img.shields.io/badge/Static-green" style="display: inline-block; vertical-align: middle;"/>
<!-- <img src="https://img.shields.io/badge/CameraOnly-crimson" style="display: inline-block; vertical-align: middle;"/>
<img src="https://img.shields.io/badge/SingleView-crimson" style="display: inline-block; vertical-align: middle;"/> -->
<br>
</div>

This dataset is a subset of ARKitScenes with high resolution depthmaps.

```
python preprocess_arkitscenes_highres.py --arkitscenes_dir /path/to/your/raw/data --output_dir /path/to/your/outdir
```

## [BlendedMVS](https://github.com/YoYo000/BlendedMVS)
<div>
<!-- <img src="https://img.shields.io/badge/INDOOR-blue" style="display: inline-block; vertical-align: middle;"/> -->
<img src="https://img.shields.io/badge/MIXED-blue" style="display: inline-block; vertical-align: middle;"/>
<!-- <img src="https://img.shields.io/badge/OUTDOOR-blue" style="display: inline-block; vertical-align: middle;"/> -->
<!-- <img src="https://img.shields.io/badge/ObjectCentric-blue" style="display: inline-block; vertical-align: middle;"/> -->
<!-- <img src="https://img.shields.io/badge/METRIC-red
" style="display: inline-block; vertical-align: middle;"/> -->
<!-- <img src="https://img.shields.io/badge/RealWorld-orange" style="display: inline-block; vertical-align: middle;"/> -->
<img src="https://img.shields.io/badge/Synthetic-orange" style="display: inline-block; vertical-align: middle;"/>
<!-- 
<img src="https://img.shields.io/badge/Dynamic-green" style="display: inline-block; vertical-align: middle;"/> -->
<img src="https://img.shields.io/badge/Static-green" style="display: inline-block; vertical-align: middle;"/>
<!-- <img src="https://img.shields.io/badge/CameraOnly-crimson" style="display: inline-block; vertical-align: middle;"/>
<img src="https://img.shields.io/badge/SingleView-crimson" style="display: inline-block; vertical-align: middle;"/> -->
<br>
</div>

Follow DUSt3R to generate the processed BlendedMVS data:
```
python preprocess_blendedmvs.py --blendedmvs_dir /path/to/your/raw/data --precomputed_pairs /path/to/your/pairs --output_dir /path/to/your/outdir
```
Then put our [overlap set](https://drive.google.com/file/d/1anBQhF9BgOvgaWgAwWnf70tzspQZHBBB/view?usp=sharing) under `/path/to/your/outdir`.

## [CO3D](https://github.com/facebookresearch/co3d)
<div>
<!-- <img src="https://img.shields.io/badge/INDOOR-blue" style="display: inline-block; vertical-align: middle;"/> -->
<!-- <img src="https://img.shields.io/badge/MIXED-blue" style="display: inline-block; vertical-align: middle;"/>
<img src="https://img.shields.io/badge/OUTDOOR-blue" style="display: inline-block; vertical-align: middle;"/> -->
<img src="https://img.shields.io/badge/ObjectCentric-blue" style="display: inline-block; vertical-align: middle;"/>
<!-- <img src="https://img.shields.io/badge/METRIC-red
" style="display: inline-block; vertical-align: middle;"/> -->
<img src="https://img.shields.io/badge/RealWorld-orange" style="display: inline-block; vertical-align: middle;"/>
<!-- <img src="https://img.shields.io/badge/Synthetic-orange" style="display: inline-block; vertical-align: middle;"/> -->
<!-- 
<img src="https://img.shields.io/badge/Dynamic-green" style="display: inline-block; vertical-align: middle;"/> -->
<img src="https://img.shields.io/badge/Static-green" style="display: inline-block; vertical-align: middle;"/>
<!-- <img src="https://img.shields.io/badge/CameraOnly-crimson" style="display: inline-block; vertical-align: middle;"/>
<img src="https://img.shields.io/badge/SingleView-crimson" style="display: inline-block; vertical-align: middle;"/> -->
<br>
</div>

Follow DUSt3R to generate the processed CO3D data.
```
python3 preprocess_co3d.py --co3d_dir /path/to/your/raw/data --output_dir /path/to/your/outdir
```

## [MegaDepth](https://www.cs.cornell.edu/projects/megadepth/)
<div>
<!-- <img src="https://img.shields.io/badge/INDOOR-blue" style="display: inline-block; vertical-align: middle;"/> -->
<!-- <img src="https://img.shields.io/badge/MIXED-blue" style="display: inline-block; vertical-align: middle;"/> -->
<img src="https://img.shields.io/badge/OUTDOOR-blue" style="display: inline-block; vertical-align: middle;"/>
<!-- <img src="https://img.shields.io/badge/ObjectCentric-blue" style="display: inline-block; vertical-align: middle;"/> -->
<!-- <img src="https://img.shields.io/badge/METRIC-red
" style="display: inline-block; vertical-align: middle;"/> -->
<img src="https://img.shields.io/badge/RealWorld-orange" style="display: inline-block; vertical-align: middle;"/>
<!-- <img src="https://img.shields.io/badge/Synthetic-orange" style="display: inline-block; vertical-align: middle;"/> -->
<!-- 
<img src="https://img.shields.io/badge/Dynamic-green" style="display: inline-block; vertical-align: middle;"/> -->
<img src="https://img.shields.io/badge/Static-green" style="display: inline-block; vertical-align: middle;"/>
<!-- <img src="https://img.shields.io/badge/CameraOnly-crimson" style="display: inline-block; vertical-align: middle;"/>
<img src="https://img.shields.io/badge/SingleView-crimson" style="display: inline-block; vertical-align: middle;"/> -->
<br>
</div>

First download our [precomputed set](https://drive.google.com/file/d/1bU1VqRu1NdW-4J4BQybqQS64mUG1EtF1/view?usp=sharing) under `/path/to/your/outdir`.

Then run
```
python preprocess_megadepth.py --megadepth_dir /path/to/your/raw/data --precomputed_sets /path/to/precomputed_sets  --output_dir /path/to/your/outdir
```

## [Scannet](http://www.scan-net.org/ScanNet/)
<div>
<img src="https://img.shields.io/badge/INDOOR-blue" style="display: inline-block; vertical-align: middle;"/>
<!-- <img src="https://img.shields.io/badge/MIXED-blue" style="display: inline-block; vertical-align: middle;"/>
<img src="https://img.shields.io/badge/OUTDOOR-blue" style="display: inline-block; vertical-align: middle;"/>
<img src="https://img.shields.io/badge/ObjectCentric-blue" style="display: inline-block; vertical-align: middle;"/> -->
<img src="https://img.shields.io/badge/METRIC-red" style="display: inline-block; vertical-align: middle;"/>
<img src="https://img.shields.io/badge/RealWorld-orange" style="display: inline-block; vertical-align: middle;"/>
<!-- <img src="https://img.shields.io/badge/Synthetic-orange" style="display: inline-block; vertical-align: middle;"/> -->
<!-- 
<img src="https://img.shields.io/badge/Dynamic-green" style="display: inline-block; vertical-align: middle;"/> -->
<img src="https://img.shields.io/badge/Static-green" style="display: inline-block; vertical-align: middle;"/>
<!-- <img src="https://img.shields.io/badge/CameraOnly-crimson" style="display: inline-block; vertical-align: middle;"/>
<img src="https://img.shields.io/badge/SingleView-crimson" style="display: inline-block; vertical-align: middle;"/> -->
<br>
</div>

```
python preprocess_scannet.py --scannet_dir /path/to/your/raw/data  --output_dir /path/to/your/outdir

python generate_set_scannet.py --root /path/to/your/outdir \
        --splits scans_test scans_train --max_interval 150 --num_workers 8
```

## [Scannet++](https://kaldir.vc.in.tum.de/scannetpp/)
<div>
<img src="https://img.shields.io/badge/INDOOR-blue" style="display: inline-block; vertical-align: middle;"/>
<!-- <img src="https://img.shields.io/badge/MIXED-blue" style="display: inline-block; vertical-align: middle;"/>
<img src="https://img.shields.io/badge/OUTDOOR-blue" style="display: inline-block; vertical-align: middle;"/>
<img src="https://img.shields.io/badge/ObjectCentric-blue" style="display: inline-block; vertical-align: middle;"/> -->
<img src="https://img.shields.io/badge/METRIC-red" style="display: inline-block; vertical-align: middle;"/>
<img src="https://img.shields.io/badge/RealWorld-orange" style="display: inline-block; vertical-align: middle;"/>
<!-- <img src="https://img.shields.io/badge/Synthetic-orange" style="display: inline-block; vertical-align: middle;"/> -->
<!-- 
<img src="https://img.shields.io/badge/Dynamic-green" style="display: inline-block; vertical-align: middle;"/> -->
<img src="https://img.shields.io/badge/Static-green" style="display: inline-block; vertical-align: middle;"/>
<!-- <img src="https://img.shields.io/badge/CameraOnly-crimson" style="display: inline-block; vertical-align: middle;"/>
<img src="https://img.shields.io/badge/SingleView-crimson" style="display: inline-block; vertical-align: middle;"/> -->
<br>
</div>

First download the pre-computed pairs provided by [DUSt3R](https://download.europe.naverlabs.com/ComputerVision/DUSt3R/scannetpp_pairs.zip).

Then run the following command,
```
python preprocess_scannetpp.py --scannetpp_dir /path/to/your/raw/data --precomputed_pairs /path/to/your/pairs --output_dir /path/to/your/outdir

python generate_set_scannetpp.py --root /path/to/your/outdir \
        --max_interval 150 --num_workers 8
```

## [Waymo](https://github.com/waymo-research/waymo-open-dataset)
<div>
<!-- <img src="https://img.shields.io/badge/INDOOR-blue" style="display: inline-block; vertical-align: middle;"/> -->
<!-- <img src="https://img.shields.io/badge/MIXED-blue" style="display: inline-block; vertical-align: middle;"/> -->
<img src="https://img.shields.io/badge/OUTDOOR-blue" style="display: inline-block; vertical-align: middle;"/>
<!-- <img src="https://img.shields.io/badge/ObjectCentric-blue" style="display: inline-block; vertical-align: middle;"/> -->
<img src="https://img.shields.io/badge/METRIC-red" style="display: inline-block; vertical-align: middle;"/>
<img src="https://img.shields.io/badge/RealWorld-orange" style="display: inline-block; vertical-align: middle;"/>
<!-- <img src="https://img.shields.io/badge/Synthetic-orange" style="display: inline-block; vertical-align: middle;"/> -->
<img src="https://img.shields.io/badge/Dynamic-green" style="display: inline-block; vertical-align: middle;"/>
<!-- <img src="https://img.shields.io/badge/Static-green" style="display: inline-block; vertical-align: middle;"/> -->
<!-- <img src="https://img.shields.io/badge/CameraOnly-crimson" style="display: inline-block; vertical-align: middle;"/>
<img src="https://img.shields.io/badge/SingleView-crimson" style="display: inline-block; vertical-align: middle;"/> -->
<br>
</div>

Follow DUSt3R to generate the processed Waymo data.
```
python3 preprocess_waymo.py --waymo_dir /path/to/your/raw/data -precomputed_pairs /path/to/precomputed_pairs --output_dir /path/to/your/outdir
```
Then download our [invalid_files](https://drive.google.com/file/d/1xI2SHHoXw1Bm7Lqrn7v56x30stCNuhlv/view?usp=sharing) and put it under `/path/to/your/outdir`.

## [WildRGBD](https://github.com/wildrgbd/wildrgbd/)
<div>
<!-- <img src="https://img.shields.io/badge/INDOOR-blue" style="display: inline-block; vertical-align: middle;"/> -->
<!-- <img src="https://img.shields.io/badge/MIXED-blue" style="display: inline-block; vertical-align: middle;"/>
<img src="https://img.shields.io/badge/OUTDOOR-blue" style="display: inline-block; vertical-align: middle;"/>-->
<img src="https://img.shields.io/badge/ObjectCentric-blue" style="display: inline-block; vertical-align: middle;"/> 
<img src="https://img.shields.io/badge/METRIC-red" style="display: inline-block; vertical-align: middle;"/>
<img src="https://img.shields.io/badge/RealWorld-orange" style="display: inline-block; vertical-align: middle;"/>
<!-- <img src="https://img.shields.io/badge/Synthetic-orange" style="display: inline-block; vertical-align: middle;"/> -->
<!-- 
<img src="https://img.shields.io/badge/Dynamic-green" style="display: inline-block; vertical-align: middle;"/> -->
<img src="https://img.shields.io/badge/Static-green" style="display: inline-block; vertical-align: middle;"/>
<!-- <img src="https://img.shields.io/badge/CameraOnly-crimson" style="display: inline-block; vertical-align: middle;"/>
<img src="https://img.shields.io/badge/SingleView-crimson" style="display: inline-block; vertical-align: middle;"/> -->
<br>
</div>

Follow DUSt3R to generate the processed WildRGBD data.
```
python3 preprocess_wildrgbd.py --wildrgbd_dir /path/to/your/raw/data --output_dir  /path/to/your/outdir
```

## [Mapfree](https://research.nianticlabs.com/mapfree-reloc-benchmark/dataset)
<div>
<!-- <img src="https://img.shields.io/badge/INDOOR-blue" style="display: inline-block; vertical-align: middle;"/> -->
<!-- <img src="https://img.shields.io/badge/MIXED-blue" style="display: inline-block; vertical-align: middle;"/>-->
<img src="https://img.shields.io/badge/OUTDOOR-blue" style="display: inline-block; vertical-align: middle;"/>
<!--<img src="https://img.shields.io/badge/ObjectCentric-blue" style="display: inline-block; vertical-align: middle;"/> -->
<img src="https://img.shields.io/badge/METRIC-red" style="display: inline-block; vertical-align: middle;"/>
<img src="https://img.shields.io/badge/RealWorld-orange" style="display: inline-block; vertical-align: middle;"/>
<!-- <img src="https://img.shields.io/badge/Synthetic-orange" style="display: inline-block; vertical-align: middle;"/> -->
<!-- 
<img src="https://img.shields.io/badge/Dynamic-green" style="display: inline-block; vertical-align: middle;"/> -->
<img src="https://img.shields.io/badge/Static-green" style="display: inline-block; vertical-align: middle;"/>
<!-- <img src="https://img.shields.io/badge/CameraOnly-crimson" style="display: inline-block; vertical-align: middle;"/>
<img src="https://img.shields.io/badge/SingleView-crimson" style="display: inline-block; vertical-align: middle;"/> -->
<br>
</div>

First preprocess the colmap results provided by Mapfree:
```
python3 preprocess_mapfree.py --mapfree_dir /path/to/train/data --colmap_dir /path/to/colmap/data --output_dir  /path/to/first/outdir
```

Then re-organize the data structure:
```
python3 preprocess_mapfree2.py --mapfree_dir /path/to/first/outdir --output_dir  /path/to/final/outdir
```

Finally, download our released [depths and masks](https://drive.google.com/file/d/1gJGEAV5e08CR6nK2gH9i71a7_WJ4ANwc/view?usp=drive_link) and combine it with your `/path/to/final/outdir`.
```
rsync -av --update /path/to/our/release /path/to/final/outdir
```

## [TartanAir](https://theairlab.org/tartanair-dataset/)
<div>
<!-- <img src="https://img.shields.io/badge/INDOOR-blue" style="display: inline-block; vertical-align: middle;"/> -->
<img src="https://img.shields.io/badge/MIXED-blue" style="display: inline-block; vertical-align: middle;"/>
<!-- <img src="https://img.shields.io/badge/OUTDOOR-blue" style="display: inline-block; vertical-align: middle;"/> -->
<!--<img src="https://img.shields.io/badge/ObjectCentric-blue" style="display: inline-block; vertical-align: middle;"/> -->
<img src="https://img.shields.io/badge/METRIC-red" style="display: inline-block; vertical-align: middle;"/>
<!-- <img src="https://img.shields.io/badge/RealWorld-orange" style="display: inline-block; vertical-align: middle;"/> -->
<img src="https://img.shields.io/badge/Synthetic-orange" style="display: inline-block; vertical-align: middle;"/>
<img src="https://img.shields.io/badge/Dynamic-green" style="display: inline-block; vertical-align: middle;"/>
<!-- <img src="https://img.shields.io/badge/Static-green" style="display: inline-block; vertical-align: middle;"/> -->
<!-- <img src="https://img.shields.io/badge/CameraOnly-crimson" style="display: inline-block; vertical-align: middle;"/>
<img src="https://img.shields.io/badge/SingleView-crimson" style="display: inline-block; vertical-align: middle;"/> -->
<br>
</div>

```
python3 preprocess_tartanair.py --tartanair_dir /path/to/your/raw/data --output_dir  /path/to/your/outdir
```

## [UnrealStereo4K](https://github.com/fabiotosi92/SMD-Nets)
<div>
<!-- <img src="https://img.shields.io/badge/INDOOR-blue" style="display: inline-block; vertical-align: middle;"/> -->
<img src="https://img.shields.io/badge/MIXED-blue" style="display: inline-block; vertical-align: middle;"/>
<!-- <img src="https://img.shields.io/badge/OUTDOOR-blue" style="display: inline-block; vertical-align: middle;"/> -->
<!--<img src="https://img.shields.io/badge/ObjectCentric-blue" style="display: inline-block; vertical-align: middle;"/> -->
<img src="https://img.shields.io/badge/METRIC-red" style="display: inline-block; vertical-align: middle;"/>
<!-- <img src="https://img.shields.io/badge/RealWorld-orange" style="display: inline-block; vertical-align: middle;"/> -->
<img src="https://img.shields.io/badge/Synthetic-orange" style="display: inline-block; vertical-align: middle;"/>
<!-- 
<img src="https://img.shields.io/badge/Dynamic-green" style="display: inline-block; vertical-align: middle;"/> -->
<img src="https://img.shields.io/badge/Static-green" style="display: inline-block; vertical-align: middle;"/>
<!-- <img src="https://img.shields.io/badge/CameraOnly-crimson" style="display: inline-block; vertical-align: middle;"/>
<img src="https://img.shields.io/badge/SingleView-crimson" style="display: inline-block; vertical-align: middle;"/> -->
<br>
</div>

```
python3 preprocess_unreal4k.py --unreal4k_dir /path/to/your/raw/data --output_dir  /path/to/your/outdir
```

## [Virtual KITTI 2](https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/)
<div>
<!-- <img src="https://img.shields.io/badge/INDOOR-blue" style="display: inline-block; vertical-align: middle;"/> -->
<!-- <img src="https://img.shields.io/badge/MIXED-blue" style="display: inline-block; vertical-align: middle;"/> -->
<img src="https://img.shields.io/badge/OUTDOOR-blue" style="display: inline-block; vertical-align: middle;"/>
<!--<img src="https://img.shields.io/badge/ObjectCentric-blue" style="display: inline-block; vertical-align: middle;"/> -->
<img src="https://img.shields.io/badge/METRIC-red" style="display: inline-block; vertical-align: middle;"/>
<!-- <img src="https://img.shields.io/badge/RealWorld-orange" style="display: inline-block; vertical-align: middle;"/> -->
<img src="https://img.shields.io/badge/Synthetic-orange" style="display: inline-block; vertical-align: middle;"/>
<img src="https://img.shields.io/badge/Dynamic-green" style="display: inline-block; vertical-align: middle;"/> 
<!-- <img src="https://img.shields.io/badge/Static-green" style="display: inline-block; vertical-align: middle;"/> -->
<!-- <img src="https://img.shields.io/badge/CameraOnly-crimson" style="display: inline-block; vertical-align: middle;"/>
<img src="https://img.shields.io/badge/SingleView-crimson" style="display: inline-block; vertical-align: middle;"/> -->
<br>
</div>

As Virtual KITTI 2 is using CC BY-NC-SA 3.0 License, we directly release our [preprocessed data](https://drive.google.com/file/d/1KdAH4ztRkzss1HCkGrPjQNnMg5c-f3aD/view?usp=sharing).

## [3D Ken Burns](https://github.com/sniklaus/3d-ken-burns.git)
<div>
<!-- <img src="https://img.shields.io/badge/INDOOR-blue" style="display: inline-block; vertical-align: middle;"/> -->
<img src="https://img.shields.io/badge/MIXED-blue" style="display: inline-block; vertical-align: middle;"/>
<!-- <img src="https://img.shields.io/badge/OUTDOOR-blue" style="display: inline-block; vertical-align: middle;"/>
<img src="https://img.shields.io/badge/ObjectCentric-blue" style="display: inline-block; vertical-align: middle;"/> -->
<!-- <img src="https://img.shields.io/badge/METRIC-red
" style="display: inline-block; vertical-align: middle;"/> -->
<!-- <img src="https://img.shields.io/badge/RealWorld-orange" style="display: inline-block; vertical-align: middle;"/> -->
<img src="https://img.shields.io/badge/Synthetic-orange" style="display: inline-block; vertical-align: middle;"/>
<!-- 
<img src="https://img.shields.io/badge/Dynamic-green" style="display: inline-block; vertical-align: middle;"/> -->
<img src="https://img.shields.io/badge/Static-green" style="display: inline-block; vertical-align: middle;"/>
<!-- <img src="https://img.shields.io/badge/CameraOnly-crimson" style="display: inline-block; vertical-align: middle;"/> -->
<img src="https://img.shields.io/badge/SingleView-crimson" style="display: inline-block; vertical-align: middle;"/>
<br>
</div>

```
python preprocess_3dkb.py --root /path/to/data_3d_ken_burns \
                           --out_dir /path/to/processed_3dkb \
                           [--num_workers 4] [--seed 42]
```

## [BEDLAM](https://bedlam.is.tue.mpg.de/)
<div>
<!-- <img src="https://img.shields.io/badge/INDOOR-blue" style="display: inline-block; vertical-align: middle;"/> -->
<img src="https://img.shields.io/badge/MIXED-blue" style="display: inline-block; vertical-align: middle;"/>
<!-- <img src="https://img.shields.io/badge/OUTDOOR-blue" style="display: inline-block; vertical-align: middle;"/>
<img src="https://img.shields.io/badge/ObjectCentric-blue" style="display: inline-block; vertical-align: middle;"/> -->
<img src="https://img.shields.io/badge/METRIC-red" style="display: inline-block; vertical-align: middle;"/>
<!-- <img src="https://img.shields.io/badge/RealWorld-orange" style="display: inline-block; vertical-align: middle;"/> -->
<img src="https://img.shields.io/badge/Synthetic-orange" style="display: inline-block; vertical-align: middle;"/>
<img src="https://img.shields.io/badge/Dynamic-green" style="display: inline-block; vertical-align: middle;"/>
<!-- <img src="https://img.shields.io/badge/Static-green" style="display: inline-block; vertical-align: middle;"/> -->
<!-- <img src="https://img.shields.io/badge/CameraOnly-crimson" style="display: inline-block; vertical-align: middle;"/>
<img src="https://img.shields.io/badge/SingleView-crimson" style="display: inline-block; vertical-align: middle;"/> -->
<br>
</div>

```
python preprocess_bedlam.py --root /path/to/extracted_data \
                             --outdir /path/to/processed_bedlam \
                             [--num_workers 4]
```

## [COP3D](https://github.com/facebookresearch/cop3d)
<div>
<!-- <img src="https://img.shields.io/badge/INDOOR-blue" style="display: inline-block; vertical-align: middle;"/> -->
<!-- <img src="https://img.shields.io/badge/MIXED-blue" style="display: inline-block; vertical-align: middle;"/>
<img src="https://img.shields.io/badge/OUTDOOR-blue" style="display: inline-block; vertical-align: middle;"/>-->
<img src="https://img.shields.io/badge/ObjectCentric-blue" style="display: inline-block; vertical-align: middle;"/> 
<!-- <img src="https://img.shields.io/badge/METRIC-red
" style="display: inline-block; vertical-align: middle;"/> -->
<img src="https://img.shields.io/badge/RealWorld-orange" style="display: inline-block; vertical-align: middle;"/>
<!-- <img src="https://img.shields.io/badge/Synthetic-orange" style="display: inline-block; vertical-align: middle;"/> -->
<img src="https://img.shields.io/badge/Dynamic-green" style="display: inline-block; vertical-align: middle;"/>
<!-- <img src="https://img.shields.io/badge/Static-green" style="display: inline-block; vertical-align: middle;"/> -->
<!-- <img src="https://img.shields.io/badge/CameraOnly-crimson" style="display: inline-block; vertical-align: middle;"/>
<img src="https://img.shields.io/badge/SingleView-crimson" style="display: inline-block; vertical-align: middle;"/> -->
<br>
</div>

```
python3 preprocess_cop3d.py --cop3d_dir /path/to/cop3d \
       --output_dir /path/to/processed_cop3d
```

## [DL3DV](https://github.com/DL3DV-10K/Dataset)
<div>
<!-- <img src="https://img.shields.io/badge/INDOOR-blue" style="display: inline-block; vertical-align: middle;"/> -->
<img src="https://img.shields.io/badge/MIXED-blue" style="display: inline-block; vertical-align: middle;"/>
<!-- <img src="https://img.shields.io/badge/OUTDOOR-blue" style="display: inline-block; vertical-align: middle;"/>
<img src="https://img.shields.io/badge/ObjectCentric-blue" style="display: inline-block; vertical-align: middle;"/> -->
<!-- <img src="https://img.shields.io/badge/METRIC-red
" style="display: inline-block; vertical-align: middle;"/> -->
<img src="https://img.shields.io/badge/RealWorld-orange" style="display: inline-block; vertical-align: middle;"/>
<!-- <img src="https://img.shields.io/badge/Synthetic-orange" style="display: inline-block; vertical-align: middle;"/> -->
<!-- 
<img src="https://img.shields.io/badge/Dynamic-green" style="display: inline-block; vertical-align: middle;"/> -->
<img src="https://img.shields.io/badge/Static-green" style="display: inline-block; vertical-align: middle;"/>
<!-- <img src="https://img.shields.io/badge/CameraOnly-crimson" style="display: inline-block; vertical-align: middle;"/>
<img src="https://img.shields.io/badge/SingleView-crimson" style="display: inline-block; vertical-align: middle;"/> -->
<br>
</div>


~~Due to current potential problems with license, you may need to run multi-view stereo on DL3DV by yourself (which is extremely time consuming). If this is done, then you can use our preprocess script:~~

~~```~~
~~python3 preprocess_dl3dv.py --dl3dv_dir /path/to/dl3dv \ 
       --output_dir /path/to/processed_dl3dv~~
~~```~~

**Update: We've released the full version of our processed DL3DV dataset!**

To use our processed DL3DV data, please ensure that you **first** cite the [original DL3DV work](https://github.com/DL3DV-10K/Dataset) and adhere to their licensing terms.

You can then download the following components:

- [RGB images and camera parameters](https://huggingface.co/datasets/zhangify/CUT3R_release/tree/main/processed_dl3dv_ours_rgb_cam_1)

- [Depthmaps and masks](https://drive.google.com/file/d/14E15EG5NJgWH5UVYubrPSSFXmReCIe7f/view?usp=drive_link)

After downloading, merge the components using the provided script:
```
python3 merge_dl3dv.py # remember to change necessary paths
```

## [Dynamic Replica](https://github.com/facebookresearch/dynamic_stereo)
<div>
<img src="https://img.shields.io/badge/INDOOR-blue" style="display: inline-block; vertical-align: middle;"/>
<!-- <img src="https://img.shields.io/badge/MIXED-blue" style="display: inline-block; vertical-align: middle;"/>
<img src="https://img.shields.io/badge/OUTDOOR-blue" style="display: inline-block; vertical-align: middle;"/>
<img src="https://img.shields.io/badge/ObjectCentric-blue" style="display: inline-block; vertical-align: middle;"/> -->
<img src="https://img.shields.io/badge/METRIC-red" style="display: inline-block; vertical-align: middle;"/>
<!-- <img src="https://img.shields.io/badge/RealWorld-orange" style="display: inline-block; vertical-align: middle;"/> -->
<img src="https://img.shields.io/badge/Synthetic-orange" style="display: inline-block; vertical-align: middle;"/>
<img src="https://img.shields.io/badge/Dynamic-green" style="display: inline-block; vertical-align: middle;"/>
<!-- <img src="https://img.shields.io/badge/Static-green" style="display: inline-block; vertical-align: middle;"/> -->
<!-- <img src="https://img.shields.io/badge/CameraOnly-crimson" style="display: inline-block; vertical-align: middle;"/>
<img src="https://img.shields.io/badge/SingleView-crimson" style="display: inline-block; vertical-align: middle;"/> -->
<br>
</div>

```
python preprocess_dynamic_replica.py --root_dir /path/to/data_dynamic_replica \
                                           --out_dir /path/to/processed_dynamic_replica
```

## [EDEN](https://lhoangan.github.io/eden/)
<div>
<!-- <img src="https://img.shields.io/badge/INDOOR-blue" style="display: inline-block; vertical-align: middle;"/> -->
<!-- <img src="https://img.shields.io/badge/MIXED-blue" style="display: inline-block; vertical-align: middle;"/> -->
<img src="https://img.shields.io/badge/OUTDOOR-blue" style="display: inline-block; vertical-align: middle;"/>
<!--<img src="https://img.shields.io/badge/ObjectCentric-blue" style="display: inline-block; vertical-align: middle;"/> -->
<img src="https://img.shields.io/badge/METRIC-red" style="display: inline-block; vertical-align: middle;"/>
<!-- <img src="https://img.shields.io/badge/RealWorld-orange" style="display: inline-block; vertical-align: middle;"/> -->
<img src="https://img.shields.io/badge/Synthetic-orange" style="display: inline-block; vertical-align: middle;"/>
<!-- 
<img src="https://img.shields.io/badge/Dynamic-green" style="display: inline-block; vertical-align: middle;"/> -->
<img src="https://img.shields.io/badge/Static-green" style="display: inline-block; vertical-align: middle;"/>
<!-- <img src="https://img.shields.io/badge/CameraOnly-crimson" style="display: inline-block; vertical-align: middle;"/> -->
<img src="https://img.shields.io/badge/SingleView-crimson" style="display: inline-block; vertical-align: middle;"/>
<br>
</div>

```
python preprocess_eden.py --root /path/to/data_raw_videos/data_eden \
                              --out_dir /path/to/data_raw_videos/processed_eden \
                              [--num_workers N]
```

## [Hypersim](https://github.com/apple/ml-hypersim)
<div>
<img src="https://img.shields.io/badge/INDOOR-blue" style="display: inline-block; vertical-align: middle;"/>
<!-- <img src="https://img.shields.io/badge/MIXED-blue" style="display: inline-block; vertical-align: middle;"/>
<img src="https://img.shields.io/badge/OUTDOOR-blue" style="display: inline-block; vertical-align: middle;"/>
<img src="https://img.shields.io/badge/ObjectCentric-blue" style="display: inline-block; vertical-align: middle;"/> -->
<img src="https://img.shields.io/badge/METRIC-red" style="display: inline-block; vertical-align: middle;"/>
<!-- <img src="https://img.shields.io/badge/RealWorld-orange" style="display: inline-block; vertical-align: middle;"/> -->
<img src="https://img.shields.io/badge/Synthetic-orange" style="display: inline-block; vertical-align: middle;"/>
<!-- 
<img src="https://img.shields.io/badge/Dynamic-green" style="display: inline-block; vertical-align: middle;"/> -->
<img src="https://img.shields.io/badge/Static-green" style="display: inline-block; vertical-align: middle;"/>
<!-- <img src="https://img.shields.io/badge/CameraOnly-crimson" style="display: inline-block; vertical-align: middle;"/>
<img src="https://img.shields.io/badge/SingleView-crimson" style="display: inline-block; vertical-align: middle;"/> -->
<br>
</div>

```
python preprocess_hypersim.py --hypersim_dir /path/to/hypersim \
                                  --output_dir /path/to/processed_hypersim
```

## [IRS](https://github.com/HKBU-HPML/IRS)
<div>
<img src="https://img.shields.io/badge/INDOOR-blue" style="display: inline-block; vertical-align: middle;"/>
<!-- <img src="https://img.shields.io/badge/MIXED-blue" style="display: inline-block; vertical-align: middle;"/>
<img src="https://img.shields.io/badge/OUTDOOR-blue" style="display: inline-block; vertical-align: middle;"/>
<img src="https://img.shields.io/badge/ObjectCentric-blue" style="display: inline-block; vertical-align: middle;"/> -->
<img src="https://img.shields.io/badge/METRIC-red" style="display: inline-block; vertical-align: middle;"/>
<!-- <img src="https://img.shields.io/badge/RealWorld-orange" style="display: inline-block; vertical-align: middle;"/> -->
<img src="https://img.shields.io/badge/Synthetic-orange" style="display: inline-block; vertical-align: middle;"/>
<!-- 
<img src="https://img.shields.io/badge/Dynamic-green" style="display: inline-block; vertical-align: middle;"/> -->
<img src="https://img.shields.io/badge/Static-green" style="display: inline-block; vertical-align: middle;"/>
<!-- <img src="https://img.shields.io/badge/CameraOnly-crimson" style="display: inline-block; vertical-align: middle;"/> -->
<img src="https://img.shields.io/badge/SingleView-crimson" style="display: inline-block; vertical-align: middle;"/>
<br>
</div>

```
python preprocess_irs.py
       --root_dir /path/to/data_irs 
       --out_dir /path/to/processed_irs
```

## [Matterport3D](https://niessner.github.io/Matterport/)
<div>
<img src="https://img.shields.io/badge/INDOOR-blue" style="display: inline-block; vertical-align: middle;"/>
<!-- <img src="https://img.shields.io/badge/MIXED-blue" style="display: inline-block; vertical-align: middle;"/>
<img src="https://img.shields.io/badge/OUTDOOR-blue" style="display: inline-block; vertical-align: middle;"/>
<img src="https://img.shields.io/badge/ObjectCentric-blue" style="display: inline-block; vertical-align: middle;"/> -->
<img src="https://img.shields.io/badge/METRIC-red" style="display: inline-block; vertical-align: middle;"/>
<img src="https://img.shields.io/badge/RealWorld-orange" style="display: inline-block; vertical-align: middle;"/>
<!-- <img src="https://img.shields.io/badge/Synthetic-orange" style="display: inline-block; vertical-align: middle;"/> -->
<!-- 
<img src="https://img.shields.io/badge/Dynamic-green" style="display: inline-block; vertical-align: middle;"/> -->
<img src="https://img.shields.io/badge/Static-green" style="display: inline-block; vertical-align: middle;"/>
<!-- <img src="https://img.shields.io/badge/CameraOnly-crimson" style="display: inline-block; vertical-align: middle;"/>
<img src="https://img.shields.io/badge/SingleView-crimson" style="display: inline-block; vertical-align: middle;"/> -->
<br>
</div>

```
python preprocess_mp3d.py --root_dir /path/to/data_mp3d/v1/scans \
                              --out_dir /path/to/processed_mp3d
```

## [MVImgNet](https://github.com/GAP-LAB-CUHK-SZ/MVImgNet)
<div>
<!-- <img src="https://img.shields.io/badge/INDOOR-blue" style="display: inline-block; vertical-align: middle;"/> -->
<!-- <img src="https://img.shields.io/badge/MIXED-blue" style="display: inline-block; vertical-align: middle;"/>
<img src="https://img.shields.io/badge/OUTDOOR-blue" style="display: inline-block; vertical-align: middle;"/> -->
<img src="https://img.shields.io/badge/ObjectCentric-blue" style="display: inline-block; vertical-align: middle;"/>
<!-- <img src="https://img.shields.io/badge/METRIC-red
" style="display: inline-block; vertical-align: middle;"/> -->
<img src="https://img.shields.io/badge/RealWorld-orange" style="display: inline-block; vertical-align: middle;"/>
<!-- <img src="https://img.shields.io/badge/Synthetic-orange" style="display: inline-block; vertical-align: middle;"/> -->
<!-- 
<img src="https://img.shields.io/badge/Dynamic-green" style="display: inline-block; vertical-align: middle;"/> -->
<img src="https://img.shields.io/badge/Static-green" style="display: inline-block; vertical-align: middle;"/>
<img src="https://img.shields.io/badge/CameraOnly-crimson" style="display: inline-block; vertical-align: middle;"/>
<!--<img src="https://img.shields.io/badge/SingleView-crimson" style="display: inline-block; vertical-align: middle;"/> -->
<br>
</div>

```
python preprocess_mvimgnet.py --data_dir /path/to/MVImgNet_data \
                                --pcd_dir /path/to/MVPNet \
                                --output_dir /path/to/processed_mvimgnet
```

## [MVS-Synth](https://phuang17.github.io/DeepMVS/mvs-synth.html)
<div>
<!-- <img src="https://img.shields.io/badge/INDOOR-blue" style="display: inline-block; vertical-align: middle;"/> -->
<!-- <img src="https://img.shields.io/badge/MIXED-blue" style="display: inline-block; vertical-align: middle;"/>-->
<img src="https://img.shields.io/badge/OUTDOOR-blue" style="display: inline-block; vertical-align: middle;"/>
<!-- <img src="https://img.shields.io/badge/ObjectCentric-blue" style="display: inline-block; vertical-align: middle;"/> -->
<img src="https://img.shields.io/badge/METRIC-red" style="display: inline-block; vertical-align: middle;"/>
<!-- <img src="https://img.shields.io/badge/RealWorld-orange" style="display: inline-block; vertical-align: middle;"/> -->
<img src="https://img.shields.io/badge/Synthetic-orange" style="display: inline-block; vertical-align: middle;"/>
<img src="https://img.shields.io/badge/Dynamic-green" style="display: inline-block; vertical-align: middle;"/>
<!-- <img src="https://img.shields.io/badge/Static-green" style="display: inline-block; vertical-align: middle;"/> -->
<!-- <img src="https://img.shields.io/badge/CameraOnly-crimson" style="display: inline-block; vertical-align: middle;"/>
<img src="https://img.shields.io/badge/SingleView-crimson" style="display: inline-block; vertical-align: middle;"/> -->
<br>
</div>

```
python preprocess_mvs_synth.py --root_dir /path/to/data_mvs_synth/GTAV_720/ \
                                   --out_dir /path/to/processed_mvs_synth \
                                   --num_workers 32
```

## [OmniObject3D](https://omniobject3d.github.io/)
<div>
<!-- <img src="https://img.shields.io/badge/INDOOR-blue" style="display: inline-block; vertical-align: middle;"/> -->
<!-- <img src="https://img.shields.io/badge/MIXED-blue" style="display: inline-block; vertical-align: middle;"/>
<img src="https://img.shields.io/badge/OUTDOOR-blue" style="display: inline-block; vertical-align: middle;"/> -->
<img src="https://img.shields.io/badge/ObjectCentric-blue" style="display: inline-block; vertical-align: middle;"/>
<img src="https://img.shields.io/badge/METRIC-red" style="display: inline-block; vertical-align: middle;"/>
<!-- <img src="https://img.shields.io/badge/RealWorld-orange" style="display: inline-block; vertical-align: middle;"/> -->
<img src="https://img.shields.io/badge/Synthetic-orange" style="display: inline-block; vertical-align: middle;"/>
<!-- 
<img src="https://img.shields.io/badge/Dynamic-green" style="display: inline-block; vertical-align: middle;"/> -->
<img src="https://img.shields.io/badge/Static-green" style="display: inline-block; vertical-align: middle;"/>
<!-- <img src="https://img.shields.io/badge/CameraOnly-crimson" style="display: inline-block; vertical-align: middle;"/>
<img src="https://img.shields.io/badge/SingleView-crimson" style="display: inline-block; vertical-align: middle;"/> -->
<br>
</div>

```
python preprocess_omniobject3d.py --input_dir /path/to/input_root --output_dir /path/to/output_root
```

## [PointOdyssey](https://pointodyssey.com/)
<div>
<!-- <img src="https://img.shields.io/badge/INDOOR-blue" style="display: inline-block; vertical-align: middle;"/> -->
<img src="https://img.shields.io/badge/MIXED-blue" style="display: inline-block; vertical-align: middle;"/>
<!-- <img src="https://img.shields.io/badge/OUTDOOR-blue" style="display: inline-block; vertical-align: middle;"/>
<img src="https://img.shields.io/badge/ObjectCentric-blue" style="display: inline-block; vertical-align: middle;"/> -->
<img src="https://img.shields.io/badge/METRIC-red" style="display: inline-block; vertical-align: middle;"/>
<!-- <img src="https://img.shields.io/badge/RealWorld-orange" style="display: inline-block; vertical-align: middle;"/> -->
<img src="https://img.shields.io/badge/Synthetic-orange" style="display: inline-block; vertical-align: middle;"/>
<img src="https://img.shields.io/badge/Dynamic-green" style="display: inline-block; vertical-align: middle;"/> 
<!-- <img src="https://img.shields.io/badge/Static-green" style="display: inline-block; vertical-align: middle;"/> -->
<!-- <img src="https://img.shields.io/badge/CameraOnly-crimson" style="display: inline-block; vertical-align: middle;"/>
<img src="https://img.shields.io/badge/SingleView-crimson" style="display: inline-block; vertical-align: middle;"/> -->
<br>
</div>

```
python preprocess_point_odyssey.py --input_dir /path/to/input_dataset --output_dir /path/to/output_dataset
```

## [RealEstate10K](https://google.github.io/realestate10k/)
<div>
<img src="https://img.shields.io/badge/INDOOR-blue" style="display: inline-block; vertical-align: middle;"/>
<!-- <img src="https://img.shields.io/badge/MIXED-blue" style="display: inline-block; vertical-align: middle;"/>
<img src="https://img.shields.io/badge/OUTDOOR-blue" style="display: inline-block; vertical-align: middle;"/>
<img src="https://img.shields.io/badge/ObjectCentric-blue" style="display: inline-block; vertical-align: middle;"/> -->
<!-- <img src="https://img.shields.io/badge/METRIC-red
" style="display: inline-block; vertical-align: middle;"/> -->
<img src="https://img.shields.io/badge/RealWorld-orange" style="display: inline-block; vertical-align: middle;"/>
<!-- <img src="https://img.shields.io/badge/Synthetic-orange" style="display: inline-block; vertical-align: middle;"/> -->
<!-- 
<img src="https://img.shields.io/badge/Dynamic-green" style="display: inline-block; vertical-align: middle;"/> -->
<img src="https://img.shields.io/badge/Static-green" style="display: inline-block; vertical-align: middle;"/>
<img src="https://img.shields.io/badge/CameraOnly-crimson" style="display: inline-block; vertical-align: middle;"/>
<!-- <img src="https://img.shields.io/badge/SingleView-crimson" style="display: inline-block; vertical-align: middle;"/> -->
<br>
</div>

```
python preprocess_re10k.py --root_dir /path/to/train \
                             --info_dir /path/to/RealEstate10K/train \
                             --out_dir /path/to/processed_re10k
```

## [SmartPortraits](https://mobileroboticsskoltech.github.io/SmartPortraits/)
<div>
<img src="https://img.shields.io/badge/INDOOR-blue" style="display: inline-block; vertical-align: middle;"/>
<!-- <img src="https://img.shields.io/badge/MIXED-blue" style="display: inline-block; vertical-align: middle;"/>
<img src="https://img.shields.io/badge/OUTDOOR-blue" style="display: inline-block; vertical-align: middle;"/>
<img src="https://img.shields.io/badge/ObjectCentric-blue" style="display: inline-block; vertical-align: middle;"/> -->
<img src="https://img.shields.io/badge/METRIC-red" style="display: inline-block; vertical-align: middle;"/>
<img src="https://img.shields.io/badge/RealWorld-orange" style="display: inline-block; vertical-align: middle;"/>
<!-- <img src="https://img.shields.io/badge/Synthetic-orange" style="display: inline-block; vertical-align: middle;"/> -->
<!-- 
<img src="https://img.shields.io/badge/Dynamic-green" style="display: inline-block; vertical-align: middle;"/> -->
<img src="https://img.shields.io/badge/Static-green" style="display: inline-block; vertical-align: middle;"/>
<!-- <img src="https://img.shields.io/badge/CameraOnly-crimson" style="display: inline-block; vertical-align: middle;"/>-->
<img src="https://img.shields.io/badge/SingleView-crimson" style="display: inline-block; vertical-align: middle;"/> 
<br>
</div>

You need to follow the [official processing pipeline](https://github.com/MobileRoboticsSkoltech/SmartPortraits-toolkit) first. Replace the `convert_to_TUM/utils/convert_to_tum.py` with our `datasets_preprocess/custom_convert2TUM.py` (You may need to change the input path and output path).

Then run
```
python preprocess_smartportraits.py \
        --input_dir /path/to/official/pipeline/output \
        --output_dir /path/to/processed_smartportraits
```

## [Spring](https://spring-benchmark.org/)
<div>
<!-- <img src="https://img.shields.io/badge/INDOOR-blue" style="display: inline-block; vertical-align: middle;"/> -->
<img src="https://img.shields.io/badge/MIXED-blue" style="display: inline-block; vertical-align: middle;"/>
<!--<img src="https://img.shields.io/badge/OUTDOOR-blue" style="display: inline-block; vertical-align: middle;"/>
<img src="https://img.shields.io/badge/ObjectCentric-blue" style="display: inline-block; vertical-align: middle;"/> -->
<img src="https://img.shields.io/badge/METRIC-red" style="display: inline-block; vertical-align: middle;"/>
<!-- <img src="https://img.shields.io/badge/RealWorld-orange" style="display: inline-block; vertical-align: middle;"/> -->
<img src="https://img.shields.io/badge/Synthetic-orange" style="display: inline-block; vertical-align: middle;"/>
<img src="https://img.shields.io/badge/Dynamic-green" style="display: inline-block; vertical-align: middle;"/>
<!-- <img src="https://img.shields.io/badge/Static-green" style="display: inline-block; vertical-align: middle;"/> -->
<!-- <img src="https://img.shields.io/badge/CameraOnly-crimson" style="display: inline-block; vertical-align: middle;"/>
<img src="https://img.shields.io/badge/SingleView-crimson" style="display: inline-block; vertical-align: middle;"/> -->
<br>
</div>

```
python preprocess_spring.py \
        --root_dir /path/to/spring/train \
        --out_dir /path/to/processed_spring \
        --baseline 0.065 \
        --output_size 960 540
```

## [Synscapes](https://synscapes.on.liu.se/)
<div>
<!-- <img src="https://img.shields.io/badge/INDOOR-blue" style="display: inline-block; vertical-align: middle;"/> -->
<!-- <img src="https://img.shields.io/badge/MIXED-blue" style="display: inline-block; vertical-align: middle;"/>-->
<img src="https://img.shields.io/badge/OUTDOOR-blue" style="display: inline-block; vertical-align: middle;"/>
<!-- <img src="https://img.shields.io/badge/ObjectCentric-blue" style="display: inline-block; vertical-align: middle;"/> -->
<img src="https://img.shields.io/badge/METRIC-red" style="display: inline-block; vertical-align: middle;"/>
<!-- <img src="https://img.shields.io/badge/RealWorld-orange" style="display: inline-block; vertical-align: middle;"/> -->
<img src="https://img.shields.io/badge/Synthetic-orange" style="display: inline-block; vertical-align: middle;"/>
<img src="https://img.shields.io/badge/Dynamic-green" style="display: inline-block; vertical-align: middle;"/>
<!-- <img src="https://img.shields.io/badge/Static-green" style="display: inline-block; vertical-align: middle;"/> -->
<!-- <img src="https://img.shields.io/badge/CameraOnly-crimson" style="display: inline-block; vertical-align: middle;"/>
<img src="https://img.shields.io/badge/SingleView-crimson" style="display: inline-block; vertical-align: middle;"/> -->
<br>
</div>

```
python preprocess_synscapes.py \
        --synscapes_dir /path/to/Synscapes/Synscapes \
        --output_dir /path/to/processed_synscapes
```

## [UASOL](https://osf.io/64532/)
<div>
<!-- <img src="https://img.shields.io/badge/INDOOR-blue" style="display: inline-block; vertical-align: middle;"/> -->
<!-- <img src="https://img.shields.io/badge/MIXED-blue" style="display: inline-block; vertical-align: middle;"/>-->
<img src="https://img.shields.io/badge/OUTDOOR-blue" style="display: inline-block; vertical-align: middle;"/>
<!-- <img src="https://img.shields.io/badge/ObjectCentric-blue" style="display: inline-block; vertical-align: middle;"/> -->
<img src="https://img.shields.io/badge/METRIC-red" style="display: inline-block; vertical-align: middle;"/>
<img src="https://img.shields.io/badge/RealWorld-orange" style="display: inline-block; vertical-align: middle;"/>
<!-- <img src="https://img.shields.io/badge/Synthetic-orange" style="display: inline-block; vertical-align: middle;"/> -->
<!-- 
<img src="https://img.shields.io/badge/Dynamic-green" style="display: inline-block; vertical-align: middle;"/> -->
<img src="https://img.shields.io/badge/Static-green" style="display: inline-block; vertical-align: middle;"/>
<!-- <img src="https://img.shields.io/badge/CameraOnly-crimson" style="display: inline-block; vertical-align: middle;"/>
<img src="https://img.shields.io/badge/SingleView-crimson" style="display: inline-block; vertical-align: middle;"/> -->
<br>
</div>

```
python preprocess_uasol.py \
        --input_dir /path/to/data_uasol \
        --output_dir /path/to/processed_uasol
```

## [UrbanSyn](https://www.urbansyn.org/)
<div>
<!-- <img src="https://img.shields.io/badge/INDOOR-blue" style="display: inline-block; vertical-align: middle;"/> -->
<!-- <img src="https://img.shields.io/badge/MIXED-blue" style="display: inline-block; vertical-align: middle;"/>-->
<img src="https://img.shields.io/badge/OUTDOOR-blue" style="display: inline-block; vertical-align: middle;"/>
<!-- <img src="https://img.shields.io/badge/ObjectCentric-blue" style="display: inline-block; vertical-align: middle;"/> -->
<img src="https://img.shields.io/badge/METRIC-red" style="display: inline-block; vertical-align: middle;"/>
<!-- <img src="https://img.shields.io/badge/RealWorld-orange" style="display: inline-block; vertical-align: middle;"/> -->
<img src="https://img.shields.io/badge/Synthetic-orange" style="display: inline-block; vertical-align: middle;"/>
<img src="https://img.shields.io/badge/Dynamic-green" style="display: inline-block; vertical-align: middle;"/>
<!-- <img src="https://img.shields.io/badge/Static-green" style="display: inline-block; vertical-align: middle;"/> -->
<!-- <img src="https://img.shields.io/badge/CameraOnly-crimson" style="display: inline-block; vertical-align: middle;"/>-->
<img src="https://img.shields.io/badge/SingleView-crimson" style="display: inline-block; vertical-align: middle;"/> 
<br>
</div>

```
python preprocess_urbansyn.py \
        --input_dir /path/to/data_urbansyn \
        --output_dir /path/to/processed_urbansyn
```

## [HOI4D](https://hoi4d.github.io/)
<div>
<img src="https://img.shields.io/badge/INDOOR-blue" style="display: inline-block; vertical-align: middle;"/>
<!-- <img src="https://img.shields.io/badge/MIXED-blue" style="display: inline-block; vertical-align: middle;"/>
<img src="https://img.shields.io/badge/OUTDOOR-blue" style="display: inline-block; vertical-align: middle;"/>
<img src="https://img.shields.io/badge/ObjectCentric-blue" style="display: inline-block; vertical-align: middle;"/> -->
<img src="https://img.shields.io/badge/METRIC-red" style="display: inline-block; vertical-align: middle;"/>
<img src="https://img.shields.io/badge/RealWorld-orange" style="display: inline-block; vertical-align: middle;"/>
<!-- <img src="https://img.shields.io/badge/Synthetic-orange" style="display: inline-block; vertical-align: middle;"/> -->
<img src="https://img.shields.io/badge/Dynamic-green" style="display: inline-block; vertical-align: middle;"/>
<!-- <img src="https://img.shields.io/badge/Static-green" style="display: inline-block; vertical-align: middle;"/> -->
<!-- <img src="https://img.shields.io/badge/CameraOnly-crimson" style="display: inline-block; vertical-align: middle;"/> -->
<img src="https://img.shields.io/badge/SingleView-crimson" style="display: inline-block; vertical-align: middle;"/>
<br>
</div>

```
python preprocess_hoi4d.py \
    --root_dir /path/to/HOI4D_release \
    --cam_root /path/to/camera_params \
    --out_dir /path/to/processed_hoi4d
```
