# <p align="center"> GS-IR Trainer and Viewer </p>



## Installation
(Install conda, only tested on Ubuntu 22.04 with cuda 11.8 ).
First clone the repository, than create the basic environment
```sh
apt-get update && apt-get install -y \
    libx11-dev \
    libxrandr-dev \
    libxinerama-dev \
    libxcursor-dev \
    libxi-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    freeglut3-dev

git clone https://github.com/Rickyeeeeee/GS-IR.git --recursive
cd GS-IR
git checkout -b viewer-imgui-bundle origin/viewer-imgui-bundle
conda create --name gsir python=3.9 -y 
conda activate gsir 

conda install -c conda-forge -y plyfile trimesh Ninja matplotlib tqdm tensorboard "numpy<2" scipy=1.10
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118 
cd gs-ir && pip install --no-build-isolation -e . develop && cd .. 

cd submodules 
git clone https://github.com/NVlabs/nvdiffrast 
pip install ./nvdiffrast 
pip install --no-build-isolation ./simple-knn 
cd ./diff-gaussian-rasterization && pip install --no-build-isolation -e .  && cd ../.. 
pip install trimesh imageio open3d kornia opencv-python==4.6.0.66 PyOpenGL
export CMAKE_POLICY_VERSION_MINIMUM=3.5
pip install imgui-bundle
git submodule update --init --recursive
cd ./submodules/pycuda
python configure.py --cuda-enable-gl 
python setup.py install 
```

## Demo
### 1. Download dataset and pretrained models
Download contenets for the link: https://gofile.me/7fCAL/XSDwayhMN (password: cgvlab)
- Dowload everything from the `Demo/` folder.
### 2. Download the example glb files
Download contents from the link: https://gofile.me/7fCAL/XSDwayhMN (password: cgvlab)
-Download everything from the `glb/` folder
### 3. Installation
Run the above installation.
### 4. Run the viewer
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
  "eval": true,
  "mesh": [
    "${Your folder}/glb/SheenChair.glb"
  ]
}

```
- Run the following command:
```bash
python imgui_bundle_viewer.py
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

