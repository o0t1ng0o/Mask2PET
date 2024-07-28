
from email.mime import audio
from pathlib import Path
from datetime import datetime
import torch 
import torch.nn as nn
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np 
import torchio as tio 

import sys 
import os
sys.path.append("./medfusion_3d") # This should be the absolute path to this folder

print(os.getcwd())
import medical_diffusion
from medical_diffusion.data.datamodules import SimpleDataModule
from medical_diffusion.data.datasets import NiftiPairImageGenerator
from medical_diffusion.models.pipelines import DiffusionPipeline
from medical_diffusion.models.estimators import UNet
from medical_diffusion.external.stable_diffusion.unet_openai import UNetModel
from medical_diffusion.models.noise_schedulers import GaussianNoiseScheduler
from medical_diffusion.models.embedders import Latent_Embedder, TimeEmbbeding
from medical_diffusion.models.embedders.latent_embedders import VAE, VAEGAN, VQVAE, VQGAN
from torchvision.transforms import RandomCrop, Compose, ToPILImage, Resize, ToTensor, Lambda
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import argparse



parser = argparse.ArgumentParser()
parser.add_argument('-i', '--inputfolder', type=str, default="data/Task107_hecktor2021/labelsTrain/")
parser.add_argument('-t', '--targetfolder', type=str, default="data/Task107_hecktor2021/imagesTrain/")

parser.add_argument('-i_val', '--inputfolder_val', type=str, default="data/Task107_hecktor2021/labelsTest/")
parser.add_argument('-t_val', '--targetfolder_val', type=str, default="data/Task107_hecktor2021/imagesTest/")

parser.add_argument('--savefolder', type=str, default="./results")
parser.add_argument('--input_size', type=int, default=128)
parser.add_argument('--depth_size', type=int, default=128)
parser.add_argument('--num_res_blocks', type=int, default=1)
parser.add_argument('--num_class_labels', type=int, default=2)
parser.add_argument('--train_lr', type=float, default=1e-4)
parser.add_argument('--batchsize', type=int, default=4)
parser.add_argument('--epochs', type=int, default=500000)
parser.add_argument('--timesteps', type=int, default=250)
parser.add_argument('--save_and_sample_every', type=int, default=1000)
parser.add_argument('--with_condition', default='True', action='store_true')
parser.add_argument('-r', '--resume_weight', type=str, default="")
parser.add_argument('--masked_condition', type=bool, default=False)
parser.add_argument('--gpu_num', type=int, default=1)

parser.add_argument('--resume_from_checkpoint', type=str, default=None)



args = parser.parse_args()


inputfolder = args.inputfolder
targetfolder = args.targetfolder

inputfolder_val = args.inputfolder_val
targetfolder_val = args.targetfolder_val

input_size = args.input_size
depth_size = args.depth_size
num_res_blocks = args.num_res_blocks
num_class_labels = args.num_class_labels
save_and_sample_every = args.save_and_sample_every
with_condition = args.with_condition
resume_weight = args.resume_weight
train_lr = args.train_lr
batchsize = args.batchsize
epochs = args.epochs
masked_condition = args.masked_condition
gpu_list = [i for i in range(args.gpu_num)]
resume_from_checkpoint = args.resume_from_checkpoint

transform = Compose([
    Lambda(lambda t: torch.tensor(t).float()),
    Lambda(lambda t: (t * 2) - 1),
    Lambda(lambda t: t.transpose(3, 1)),
])

input_transform = Compose([
    Lambda(lambda t: torch.tensor(t).float()),
    Lambda(lambda t: t.transpose(3, 1)),
])


if __name__ == "__main__":

    dataset = NiftiPairImageGenerator(
        inputfolder,
        targetfolder,
        input_size=input_size,
        depth_size=depth_size,
        transform=input_transform if with_condition else transform,
        target_transform=transform,
        full_channel_mask=True
    )

    dataset_val = NiftiPairImageGenerator(
        inputfolder_val,
        targetfolder_val,
        input_size=input_size,
        depth_size=depth_size,
        transform=input_transform if with_condition else transform,
        target_transform=transform,
        full_channel_mask=True
    )


   
    dm = SimpleDataModule(
        ds_train = dataset,
        batch_size=batchsize, 
        ds_val = dataset_val,
        # num_workers=40,
        pin_memory=True
    ) 
    
    current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    path_run_dir = Path.cwd() / 'runs' / 'LDM_VQGAN'/ str(current_time)
    path_run_dir.mkdir(parents=True, exist_ok=True)
    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'



    # ------------ Initialize Model ------------
    # cond_embedder = None 
    cond_embedder = Latent_Embedder


    time_embedder = TimeEmbbeding
    time_embedder_kwargs ={
        'emb_dim': 1024 # stable diffusion uses 4*model_channels (model_channels is about 256)
    }


    noise_estimator = UNet
    if masked_condition:
        noise_estimator_kwargs = {
            'in_ch':8, 
            'out_ch':4, 
            'spatial_dims':3,
            'hid_chs':  [  256, 256, 512, 1024],
            'kernel_sizes':[3, 3, 3, 3],
            'strides':     [1, 2, 2, 2],
            'time_embedder':time_embedder,
            'time_embedder_kwargs': time_embedder_kwargs,
            'cond_embedder':cond_embedder,
            'cond_embedder_kwargs': {
                'in_channels': 5,
                'emb_channels': 5,
                'strides' : [ 1,  1,   1,   1],
                'hid_chs' : [32, 64, 128,  256],
            },
            'deep_supervision': False,
            'use_res_block':True,
            'use_attention':'none',
            'masked_condition': True
        }
    else:
        noise_estimator_kwargs = {
            'in_ch':8, 
            'out_ch':4, 
            'spatial_dims':3,
            'hid_chs':  [  256, 256, 512, 1024],
            'kernel_sizes':[3, 3, 3, 3],
            'strides':     [1, 2, 2, 2],
            'time_embedder':time_embedder,
            'time_embedder_kwargs': time_embedder_kwargs,
            'cond_embedder':cond_embedder,
            'deep_supervision': False,
            'use_res_block':True,
            'use_attention':'none',
        }


    # ------------ Initialize Noise ------------
    noise_scheduler = GaussianNoiseScheduler
    noise_scheduler_kwargs = {
        'timesteps': 1000,
        'beta_start': 0.002, # 0.0001, 0.0015
        'beta_end': 0.02, # 0.01, 0.0195
        'schedule_strategy': 'scaled_linear'
    }
    
    # ------------ Initialize Latent Space  ------------
    # latent_embedder = None 
    # latent_embedder = VQVAE
    latent_embedder = VQGAN # VQVAE: "/home/local/PARTNERS/rh384/runs/VAE/epoch=114-step=23000.ckpt"
    latent_embedder_checkpoint = "./pretrained_models/VQGAN/2024_01_07_090227/epoch=284-step=114000.ckpt"
    # "./runs/VQGAN/2024_01_07_090227/epoch=284-step=114000.ckpt"


    # latent_embedder = VQVAE # VQVAE: "/home/local/PARTNERS/rh384/runs/VAE/epoch=114-step=23000.ckpt"
    # latent_embedder_checkpoint = "./medfusion_3d/runs/VQVAE/2024_01_05_200333/epoch=114-step=23000.ckpt"
   
    # ------------ Initialize Pipeline ------------
    pipeline = DiffusionPipeline(
        noise_estimator=noise_estimator, 
        noise_estimator_kwargs=noise_estimator_kwargs,
        noise_scheduler=noise_scheduler, 
        noise_scheduler_kwargs = noise_scheduler_kwargs,
        latent_embedder=latent_embedder,
        latent_embedder_checkpoint = latent_embedder_checkpoint,
        estimator_objective='x_T',
        estimate_variance=False, 
        use_self_conditioning=False, 
        num_samples = 1,
        use_ema=False,
        classifier_free_guidance_dropout=0.5, # Disable during training by setting to 0
        do_input_centering=False,
        clip_x0=False,
        sample_every_n_steps=save_and_sample_every,
        masked_condition=masked_condition
    )
    
    # pipeline_old = pipeline.load_from_checkpoint('runs/2022_11_27_085654_chest_diffusion/last.ckpt')
    # pipeline.noise_estimator.load_state_dict(pipeline_old.noise_estimator.state_dict(), strict=True)

    # -------------- Training Initialization ---------------
    to_monitor = "train/loss"  # "pl/val_loss" 
    min_max = "min"

    early_stopping = EarlyStopping(
        monitor=to_monitor,
        min_delta=0.0, # minimum change in the monitored quantity to qualify as an improvement
        patience=30, # number of checks with no improvement
        mode=min_max
    )
    checkpointing = ModelCheckpoint(
        dirpath=str(path_run_dir), # dirpath
        monitor=to_monitor,
        every_n_train_steps=save_and_sample_every,
        save_last=False,
        save_top_k=2,
        mode=min_max,
    )
    trainer = Trainer(
        accelerator=accelerator,
        devices=gpu_list, #[0],
        # precision=16,
        # amp_backend='apex',
        # amp_level='O2',
        # gradient_clip_val=0.5,
        default_root_dir=str(path_run_dir),
        callbacks=[checkpointing],
        # callbacks=[checkpointing, early_stopping],
        enable_checkpointing=True,
        check_val_every_n_epoch=1,
        log_every_n_steps=save_and_sample_every, 
        auto_lr_find=False,
        # limit_train_batches=1000,
        limit_val_batches=0, # 0 = disable validation - Note: Early Stopping no longer available 
        min_epochs=100,
        max_epochs=epochs,
        num_sanity_val_steps=2,
        resume_from_checkpoint=resume_from_checkpoint
    )
    
    # ---------------- Execute Training ----------------
    trainer.fit(pipeline, datamodule=dm)

    # ------------- Save path to best model -------------
    pipeline.save_best_checkpoint(trainer.logger.log_dir, checkpointing.best_model_path)


