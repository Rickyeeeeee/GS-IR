# python viewer.py \
#     --checkpoint /workspace/work/Outputs/TensoIR_Synthetic/gsir/hotdog/chkpnt35000.pth
    # --hdri /workspace/data/Datasets/TensoIR_Synthtic/Environment_Maps/high_res_envmaps_1k/bridge.hdr \
    # --checkpoint /workspace/work/Relighting/MipNerf360/stump/chkpnt40000.pth
    # --checkpoint /workspace/work/Outputs/Synthetic4Relight/gsir/air_baloons/chkpnt35000.pth

python pbr_viewer.py \
-m /workspace/work/Outputs/TensoIR_Synthetic/gsir/hotdog \
-s /workspace/data/Datasets/TensoIR_Synthtic/hotdog/ \
--checkpoint /workspace/work/Outputs/TensoIR_Synthetic/gsir/hotdog/chkpnt35000.pth \
--hdri /workspace/data/Datasets/TensoIR_Synthtic/Environment_Maps/high_res_envmaps_1k/bridge.hdr \
--eval \
--gamma