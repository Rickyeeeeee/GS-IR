scene=owl_colmap_masked
scene=backdoor_colmap

python pbr_viewer.py \
-m /workspace/work/Outputs/ITRI_Relighting_Demo/gsir/${scene} \
-s /workspace/data/Datasets/ITRI_Relighting_Demo/${scene} \
--checkpoint /workspace/work/Outputs/ITRI_Relighting_Demo/gsir/${scene}/chkpnt40000.pth \
--hdri /workspace/data/Datasets/TensoIR_Synthetic/Environment_Maps/high_res_envmaps_1k/bridge.hdr \
--eval \
--tone \
--gamma