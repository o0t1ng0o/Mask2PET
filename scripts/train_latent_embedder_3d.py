




from pathlib import Path
from datetime import datetime

import torch 
from torch.utils.data import ConcatDataset
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


from medical_diffusion.data.datamodules import SimpleDataModule
from medical_diffusion.data.datasets import NiftiPairImageGenerator
from medical_diffusion.models.embedders.latent_embedders import VQVAE, VQGAN, VAE, VAEGAN
from torchvision.transforms import RandomCrop, Compose, ToPILImage, Resize, ToTensor, Lambda
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--inputfolder', type=str, default="data/Task107_hecktor2021/labelsTrain/")
parser.add_argument('-t', '--targetfolder', type=str, default="data/Task107_hecktor2021/imagesTrain/")
parser.add_argument('--savefolder', type=str, default="./medfusion_3d/results")
parser.add_argument('--input_size', type=int, default=128)
parser.add_argument('--depth_size', type=int, default=128)
parser.add_argument('--num_channels', type=int, default=64)
parser.add_argument('--num_res_blocks', type=int, default=1)
parser.add_argument('--num_class_labels', type=int, default=2)
parser.add_argument('--train_lr', type=float, default=1e-4)
parser.add_argument('--batchsize', type=int, default=1)
parser.add_argument('--epochs', type=int, default=500000)
parser.add_argument('--timesteps', type=int, default=250)
parser.add_argument('--save_and_sample_every', type=int, default=1000)
parser.add_argument('--with_condition', default='True', action='store_true')
parser.add_argument('-r', '--resume_weight', type=str, default="")
args = parser.parse_args()

inputfolder = args.inputfolder
targetfolder = args.targetfolder
input_size = args.input_size
depth_size = args.depth_size
num_channels = args.num_channels
num_res_blocks = args.num_res_blocks
num_class_labels = args.num_class_labels
save_and_sample_every = args.save_and_sample_every
with_condition = args.with_condition
resume_weight = args.resume_weight
train_lr = args.train_lr


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

    # --------------- Settings --------------------
    current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    path_run_dir = Path.cwd() / 'runs' / 'VQGAN'/ str(current_time)
    path_run_dir.mkdir(parents=True, exist_ok=True)
    gpus = [0] if torch.cuda.is_available() else None

    dataset = NiftiPairImageGenerator(
        inputfolder,
        targetfolder,
        input_size=input_size,
        depth_size=depth_size,
        transform=input_transform if with_condition else transform,
        target_transform=transform,
        full_channel_mask=True
    )

   
    dm = SimpleDataModule(
        ds_train = dataset,
        batch_size=1, 
        # num_workers=0,
        pin_memory=True
    ) 
    

    # ------------ Initialize Model ------------
    # model = VAE(
    #     in_channels=3, 
    #     out_channels=3, 
    #     emb_channels=8,
    #     spatial_dims=2,
    #     hid_chs =    [ 64, 128, 256,  512], 
    #     kernel_sizes=[ 3,  3,   3,    3],
    #     strides =    [ 1,  2,   2,    2],
    #     deep_supervision=1,
    #     use_attention= 'none',
    #     loss = torch.nn.MSELoss,
    #     # optimizer_kwargs={'lr':1e-6},
    #     embedding_loss_weight=1e-6,
    #     sample_every_n_steps = 1000
    # )

    # model.load_pretrained(Path.cwd()/'runs/2022_12_01_183752_patho_vae/last.ckpt', strict=True)

    # model = VAEGAN(
    #     in_channels=1, 
    #     out_channels=1, 
    #     emb_channels=8,
    #     spatial_dims=3,
    #     hid_chs =    [ 64, 128, 256,  512],
    #     deep_supervision=1,
    #     use_attention= 'none',
    #     start_gan_train_step=-1,
    #     embedding_loss_weight=1e-6,
    #     sample_every_n_steps = 1000
    # )

    # model.vqvae.load_pretrained(Path.cwd()/'runs/2022_11_25_082209_chest_vae/last.ckpt')
    # model.load_pretrained(Path.cwd()/'runs/2022_11_25_232957_patho_vaegan/last.ckpt')


    # model = VQVAE(
    #     in_channels=1, 
    #     out_channels=1, 
    #     emb_channels=4,
    #     num_embeddings = 8192,
    #     spatial_dims=3,
    #     hid_chs =    [64, 128, 256, 512],
    #     embedding_loss_weight=1,
    #     beta=1,
    #     loss = torch.nn.L1Loss,
    #     deep_supervision=1,
    #     use_attention = 'none',
    #     sample_every_n_steps = save_and_sample_every
    # )
    

    model = VQGAN(
        in_channels=1, 
        out_channels=1, 
        emb_channels=4,
        num_embeddings = 8192,
        spatial_dims=3,
        hid_chs =    [64, 128, 256, 512],
        embedding_loss_weight=1,
        beta=1,
        start_gan_train_step=-1,
        pixel_loss = torch.nn.L1Loss,
        deep_supervision=1,
        use_attention='none',
    )
    
    # model.vqvae.load_pretrained(Path.cwd()/'runs/2022_12_13_093727_patho_vqvae/last.ckpt')
    

    # -------------- Training Initialization ---------------
    to_monitor = "train/L1"  # "val/loss" 
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
        save_last=True,
        save_top_k=5,
        mode=min_max,
    )
    trainer = Trainer(
        accelerator='gpu',
        devices=[0],
        # precision=16,
        # amp_backend='apex',
        # amp_level='O2',
        # gradient_clip_val=0.5,
        default_root_dir=str(path_run_dir),
        callbacks=[checkpointing],
        # callbacks=[checkpointing, early_stopping],
        enable_checkpointing=True,
        check_val_every_n_epoch=None,
        log_every_n_steps=save_and_sample_every, 
        # limit_train_batches=1000,
        limit_val_batches=0, # 0 = disable validation - Note: Early Stopping no longer available 
        min_epochs=100,
        max_epochs=1001,
        num_sanity_val_steps=2,
    )
    
    # ---------------- Execute Training ----------------
    trainer.fit(model, datamodule=dm)

    # ------------- Save path to best model -------------
    model.save_best_checkpoint(trainer.logger.log_dir, checkpointing.best_model_path)


