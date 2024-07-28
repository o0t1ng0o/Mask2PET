from pathlib import Path 
import logging
from datetime import datetime
from tqdm import tqdm

import numpy as np 
import torch
import torchvision.transforms.functional as tF
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import ImageFolder
from torch.utils.data import TensorDataset, Subset

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from torchmetrics.functional import multiscale_structural_similarity_index_measure as mmssim
from torchvision.transforms import RandomCrop, Compose, ToPILImage, Resize, ToTensor, Lambda

from medical_diffusion.data.datamodules import SimpleDataModule
from medical_diffusion.data.datasets import NiftiPairImageGenerator
from medical_diffusion.models.embedders.latent_embedders import VQVAE
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--inputfolder', type=str, default="/home/local/PARTNERS/rh384/data/Task107_hecktor2021/labelsTrain/")
parser.add_argument('-t', '--targetfolder', type=str, default="/home/local/PARTNERS/rh384/data/Task107_hecktor2021/imagesTrain/")
parser.add_argument('--savefolder', type=str, default="/home/local/PARTNERS/rh384/workspace/med-ddpm/results")
parser.add_argument('--input_size', type=int, default=128)
parser.add_argument('--depth_size', type=int, default=128)
parser.add_argument('--num_channels', type=int, default=64)
parser.add_argument('--num_res_blocks', type=int, default=1)
parser.add_argument('--num_class_labels', type=int, default=2)
parser.add_argument('--train_lr', type=float, default=1e-5)
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
    # Lambda(lambda t: (t * 2) - 1),
    Lambda(lambda t: t.unsqueeze(0)),
    Lambda(lambda t: t.transpose(3, 1)),
])

input_transform = Compose([
    Lambda(lambda t: torch.tensor(t).float()),
    # Lambda(lambda t: (t * 2) - 1),
    Lambda(lambda t: t.permute(3, 0, 1, 2)),
    Lambda(lambda t: t.transpose(3, 1)),
])

# ----------------Settings --------------
batch_size = 1
max_samples = None # set to None for all 
target_class = None # None for no specific class 
# path_out = Path.cwd()/'results'/'MSIvsMSS_2'/'metrics'
# path_out = Path.cwd()/'results'/'AIROGS'/'metrics'
path_out = Path.cwd()/'results'/'metrics'
path_out.mkdir(parents=True, exist_ok=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ----------------- Logging -----------
current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)
logger.addHandler(logging.FileHandler(path_out/f'metrics_{current_time}.log', 'w'))


# ---------------- Dataset/Dataloader ----------------
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

# --------------- Load Model ------------------
model = VQVAE.load_from_checkpoint('/home/local/PARTNERS/rh384/runs/2023_12_27_171409/epoch=9-step=2000.ckpt')
model.to(device)


# ------------- Init Metrics ----------------------
calc_lpips = LPIPS().to(device)


# --------------- Start Calculation -----------------
mmssim_list, mse_list = [], []
for real_batch in tqdm(dm):
    imgs_real_batch = real_batch[0].to(device)

    imgs_real_batch = tF.normalize(imgs_real_batch/255, 0.5, 0.5) # [0, 255] -> [-1, 1]
    with torch.no_grad():
        imgs_fake_batch = model(imgs_real_batch)[0].clamp(-1, 1) 

    # -------------- LPIP -------------------
    calc_lpips.update(imgs_real_batch, imgs_fake_batch) # expect input to be [-1, 1]

    # -------------- MS-SSIM + MSE -------------------
    for img_real, img_fake in zip(imgs_real_batch, imgs_fake_batch):
        img_real, img_fake = (img_real+1)/2, (img_fake+1)/2  # [-1, 1] -> [0, 1]
        mmssim_list.append(mmssim(img_real[None], img_fake[None], normalize='relu')) 
        mse_list.append(torch.mean(torch.square(img_real-img_fake)))


# -------------- Summary -------------------
mmssim_list = torch.stack(mmssim_list)
mse_list = torch.stack(mse_list)

lpips = 1-calc_lpips.compute()
logger.info(f"LPIPS Score: {lpips}")
logger.info(f"MS-SSIM: {torch.mean(mmssim_list)} ± {torch.std(mmssim_list)}")
logger.info(f"MSE: {torch.mean(mse_list)} ± {torch.std(mse_list)}")