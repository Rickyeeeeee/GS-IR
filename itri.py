import os

# === Config ===
prefix = "CUDA_VISIBLE_DEVICES=0"
dataset_root = "/workspace/data/Datasets/ITRI_Relighting_Demo"
output_root = "/workspace/work/Outputs/ITRI_Relighting_Demo"
# light_name_list = ["bridge", "city", "fireplace", "forest", "night"]

scenes = ["backdoor_14", "owl_colmap_masked"]
scenes = ["owl_colmap_masked"]
scenes = ["backdoor_colmap"]

for scene in scenes:
    dataset_name = scene

    scene_path = os.path.join(dataset_root, dataset_name)
    output_path = os.path.join(output_root, 'gsir', dataset_name)
    checkpoint_30k = os.path.join(output_path, "chkpnt30000.pth")
    checkpoint_40k = os.path.join(output_path, "chkpnt40000.pth")
    background = "--white_background" if scene in ["backdoor_colmap"] else ""

    # === Commands ===
    images = "images_2" 
    commands = [
        # f"{prefix} python train.py -m {output_path} -s {scene_path} --iterations 30000 -i {images} --data_device cpu --eval {background}",
        # f"{prefix} python baking.py -m {output_path} --checkpoint {checkpoint_30k} --bound 16.0 --occlu_res 256 --occlusion 0.4 {background}",
        f"{prefix} python train.py -m {output_path} -s {scene_path} --start_checkpoint {checkpoint_30k} --iterations 40000 -i {images} --eval --gamma --metallic --indirect {background}",
        f"{prefix} python render.py -m {output_path} -s {scene_path} --checkpoint {checkpoint_40k} --eval --skip_train --pbr --gamma --indirect {background}",
    ]

    # === Execution ===
    for cmd in commands:
        print(f"\n=== Running: {cmd} ===")
        os.system(cmd)
