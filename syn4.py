import os

dataset_path = "/workspace/data/Synthetic4Relight/Synthetic4Relight"
output_path = "/workspace/work/Outputs/Synthetic4Relight/gsir"
scenes = [
    'air_baloons', 'chair', 'hotdog', 'jugs'
]

for scene in scenes:
    dataset_scene_path = os.path.join(dataset_path, scene)
    output_scene_path = os.path.join(output_path, scene)

    cmd = (
        f"python train.py "
        f"-m {output_scene_path} "
        f"-s {dataset_scene_path} "
        f"--iterations 30000 "
        f"--eval"
    )
    # print(f"\n[Running command] : {cmd}\n")
    # os.system(cmd)

    cmd = (
        f"python baking.py "
        f"-m {output_scene_path} "
        f"--checkpoint {output_scene_path}/chkpnt30000.pth "
        f"--bound 1.5 "
        f"--occlu_res 128 "
        f"--occlusion 0.25 "
    )
    # print(f"\n[Running command] : {cmd}\n")
    # os.system(cmd)

    cmd = (
        f"python train.py "
        f"-m {output_scene_path} "
        f"-s {dataset_scene_path} "
        f"--start_checkpoint {output_scene_path}/chkpnt30000.pth "
        f"--iterations 35000 "
        f"--eval "
        f"--gamma "
        f"--indirect "
    )
    print(f"\n[Running command] : {cmd}\n")
    os.system(cmd)

    cmd = (
        f"python render.py "
        f"-m {output_scene_path} "
        f"-s {dataset_scene_path} "
        f"--checkpoint {output_scene_path}/chkpnt35000.pth "
        f"--eval "
        f"--skip_train "
        f"--pbr "
        f"--gamma "
        f"--indirect "
    )
    print(f"\n[Running command] : {cmd}\n")
    os.system(cmd)
    break