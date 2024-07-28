python3 ./scripts/train_diffusion.py --masked_condition True \
-i data/Task107_hecktor2021/labelsTrain/ \
-t data/Task107_hecktor2021/imagesTrain/ \
-i_val data/Task107_hecktor2021/labelsTest/ \
-t_val data/Task107_hecktor2021/imagesTest/

# continue to train
#--resume_from_checkpoint ./runs/LDM_VQGAN/2024_06_07_115628/epoch=199-step=9999.ckpt