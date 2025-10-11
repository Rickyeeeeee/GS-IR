scene=lego
scene=armadillo
scene=ficus
scene=hotdog

python pbr_viewer.py \
-m /workspace/work/Outputs/TensoIR_Synthetic/gsir/${scene} \
-s /workspace/data/Datasets/TensoIR_Synthetic/${scene} \
--checkpoint /workspace/work/Outputs/TensoIR_Synthetic/gsir/${scene}/chkpnt35000.pth \
--hdri /workspace/data/Datasets/TensoIR_Synthetic/Environment_Maps/high_res_envmaps_1k/bridge.hdr \
--eval \
--tone \
--gamma

# scene=stump

# python pbr_viewer.py \
# -m /workspace/work/Relighting/MipNerf360/${scene} \
# -s /workspace/data/Datasets/MipNerf360/${scene}/ \
# --checkpoint /workspace/work/Relighting/MipNerf360/${scene}/chkpnt40000.pth \
# --hdri /workspace/data/Datasets/TensoIR_Synthtic/Environment_Maps/high_res_envmaps_1k/bridge.hdr \
# --eval \
# --tone \
# --gamma