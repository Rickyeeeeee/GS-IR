# python viewer.py \
#     --checkpoint /workspace/work/Outputs/TensoIR_Synthetic/gsir/${scene}/chkpnt35000.pth
    # --hdri /workspace/data/Datasets/TensoIR_Synthtic/Environment_Maps/high_res_envmaps_1k/bridge.hdr \
    # --checkpoint /workspace/work/Relighting/MipNerf360/stump/chkpnt40000.pth
    # --checkpoint /workspace/work/Outputs/Synthetic4Relight/gsir/air_baloons/chkpnt35000.pth

# scene=hotdog

# python pbr_viewer.py \
# -m /workspace/work/Outputs/Synthetic4Relight/gsir/${scene} \
# -s /workspace/data/Datasets/Synthetic4Relight/Synthetic4Relight/${scene}/ \
# --checkpoint /workspace/work/Outputs/Synthetic4Relight/gsir/${scene}/chkpnt35000.pth \
# --hdri /workspace/data/Datasets/TensoIR_Synthtic/Environment_Maps/high_res_envmaps_1k/bridge.hdr \
# --eval \
# --gamma

# scene=ficus
scene=hotdog
scene=lego
# scene=armadillo

python pbr_viewer.py \
-m /workspace/work/Outputs/TensoIR_Synthetic/gsir/${scene} \
-s /workspace/data/Datasets/TensoIR_Synthtic/${scene}/ \
--checkpoint /workspace/work/Outputs/TensoIR_Synthetic/gsir/${scene}/chkpnt35000.pth \
--hdri /workspace/data/Datasets/TensoIR_Synthtic/Environment_Maps/high_res_envmaps_1k/bridge.hdr \
--eval \
--tone \
--gamma

scene=stump

python pbr_viewer.py \
-m /workspace/work/Relighting/MipNerf360/${scene} \
-s /workspace/data/Datasets/MipNerf360/${scene}/ \
--checkpoint /workspace/work/Relighting/MipNerf360/${scene}/chkpnt40000.pth \
--hdri /workspace/data/Datasets/TensoIR_Synthtic/Environment_Maps/high_res_envmaps_1k/bridge.hdr \
--eval \
--tone \
--gamma