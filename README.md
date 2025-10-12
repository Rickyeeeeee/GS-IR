# <p align="center"> GS-IR Trainer and Viewer </p>



## Installation
(Install conda, only tested on Ubuntu).
First clone the repository, than create the basic environment
```sh
conda env create --file environment.yml
conda activate gsir

pip install kornia
```

install some extensions
```sh
cd gs-ir && python setup.py develop && cd ..

cd submodules
git clone https://github.com/NVlabs/nvdiffrast
pip install ./nvdiffrast

pip install ./simple-knn
pip install ./diff-gaussian-rasterization # or cd ./diff-gaussian-rasterization && python setup.py develop && cd ../..
pip install dearpygui
```

## Demo
### 1. Download dataset and pretrained models
Download contenets for the link: https://gofile.me/7fCAL/XSDwayhMN (password: cgvlab)
- Dowload everything from the `Demo/` folder.
### 2. Installation
Run the above installation.
### 3. Run the viewer
- Modify the paths of `config.json`:
```json
{
  "checkpoint": [
    "${Your folder}Demo/outputs/ITRI_Relighting_Demo/gsir/backdoor_colmap/chkpnt40000.pth",
    "${Your folder}Demo/outputs/ITRI_Relighting_Demo/gsir/owl_colmap_masked/chkpnt40000.pth"],
  "hdri_root": "{Your folder}/Demo/EnvironmentMaps/high_res_envmaps_1k/",
  "m": "",
  "s": "",
  "width": 1200,
  "height": 900,
  "tone": true,
  "gamma": true,
  "metallic": true,
  "eval": true
}

```
- Run the following command:
```bash
python pbr_viewer_demo.py --config config.json
```

## Dataset
Custom dataset can created from colmap, the file structure should be organized in the following ways:
(Using bicycle as an example)
```sh
bicycle/
├── images/
│   ├── 000.jpg
│   └── ...
├── sparse/
│   └── 0/
│       ├── cameras.bin
│       ├── images.bin
│       └── points3D.bin
└── poses_bounds.npy

```

## Training

Take the `bicycle` case as an example.

**Stage1 (Initial Stage)**
```sh
python train.py \
-m outputs/bicycle/ \
-s datasets/MipNerf360/bicycle/ \
--iterations 30000 \
-i images_4 \
-r 1 \
--eval
```
> `-i images_4` for outdoor scenes and `-i images_2` for indoor scenes
> `-r 1` for resolution scaling (1 = not rescale, 2 = rescale to 1/2, 4 = rescale to 1/4)

**Baking**
```sh
python baking.py \
-m outputs/bicycle/ \
--checkpoint outputs/bicycle/chkpnt30000.pth \
--bound 16.0 \
--occlu_res 256 \
--occlusion 0.4
```

**Stage2 (Decomposition Stage)**
```sh
python train.py \
-m outputs/bicycle \
-s datasets/MipNerf360/bicycle/ \
--start_checkpoint outputs/bicycle/chkpnt30000.pth \
--iterations 40000 \
-i images_4 \
-r 1 \
--eval \
--metallic \
--indirect
```

## Open Viewer
Modify the viewer.sh file and change the dataset and output path
```sh
python pbr_viewer.py \
-m outputs/bicycle \
-s datasets/MipNerf360/bicycle/ \
--checkpoint datasets/MipNerf360/bicycle/chkpnt40000.pth \
--hdri datasets/TensoIR_Synthtic/Environment_Maps/high_res_envmaps_1k/bridge.hdr \
--eval \
--tone \
--gamma
```

Environment maps can be found under the `TensoIR_Synthetic/Environment_Maps` folder.


## Acknowledge
- [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- [nvdiffrast](https://github.com/NVlabs/nvdiffrast)
- [nvdiffrec](https://github.com/NVlabs/nvdiffrec)
- [gsir](https://github.com/lzhnb/GS-IR)

