
import sys 
import os
sys.path.append("./medfusion_3d")
import torch.nn.functional as F 

from pathlib import Path
import torch 
from torchvision import utils 
import math 
from medical_diffusion.models.pipelines import DiffusionPipeline
import logging
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import RandomCrop, Compose, ToPILImage, Resize, ToTensor, Lambda
from medical_diffusion.data.datasets import NiftiPairImageGenerator
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import nibabel as nib

from medical_diffusion.models.estimators import UNet
from medical_diffusion.models.embedders import Latent_Embedder, TimeEmbbeding
from medical_diffusion.models.embedders.latent_embedders import VAE, VAEGAN, VQVAE, VQGAN
from medical_diffusion.models.noise_schedulers import GaussianNoiseScheduler
from tqdm import tqdm
torch.manual_seed(0)
masked_condition = True

device = torch.device('cuda')
# ----------------------define the model----------------------

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
latent_embedder = VQGAN 
latent_embedder_checkpoint = "./pretrained/VQGAN/2024_01_07_090227/epoch=284-step=114000.ckpt"


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
    # sample_every_n_steps=save_and_sample_every,
    masked_condition=masked_condition
)

# ------------ Load Model ------------
# pipeline = DiffusionPipeline.load_best_checkpoint(path_run_dir)
# pipeline = DiffusionPipeline.load_from_checkpoint("./medfusion_3d/runs/LDM_VQGAN/2024_06_07_115628/epoch=199-step=9999.ckpt") #/home/local/PARTNERS/rh384/runs/LDM/epoch=119-step=24000.ckpt")

ckpt_path = './runs/LDM_VQGAN/2024_06_07_175241/epoch=2759-step=137999.ckpt'
#'./medfusion_3d/runs/LDM_VQGAN/2024_06_07_175241/epoch=1079-step=53999.ckpt'
pipeline.load_pretrained(Path(ckpt_path))

pipeline.to(device)

inputfolder = "data/Task107_hecktor2021/labelsVal/"
targetfolder = "data/Task107_hecktor2021/imagesVal/"
input_size = 128
depth_size = 128
with_condition =  True

transform = Compose([
    Lambda(lambda t: torch.tensor(t).float()),
    Lambda(lambda t: (t * 2) - 1),
    Lambda(lambda t: t.transpose(3, 1)),
])

input_transform = Compose([
    Lambda(lambda t: torch.tensor(t).float()),
    # Lambda(lambda t: (t * 2) - 1),
    Lambda(lambda t: t.transpose(3, 1)),
])

# ----------------Settings --------------
batch_size = 1
max_samples = None # set to None for all 
target_class = None # None for no specific class 
# path_out = Path.cwd()/'results'/'MSIvsMSS_2'/'metrics'
# path_out = Path.cwd()/'results'/'AIROGS'/'metrics'
path_out = Path.cwd()/'results_new'/'metrics'/ 'nocrop'
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


dl = DataLoader(dataset, batch_size = 1, shuffle=False, num_workers=1, pin_memory=True)

# --------- Generate Samples  -------------------
steps = 250
use_ddim = True 
images = {}
n_samples = 1

for i,batch in enumerate(tqdm(dl, total=len(dl))):
# batch = next(iter(dl))
    torch.manual_seed(0)
    x_0 = batch['target']
    condition = batch['input'].cuda()

    condition_save = condition.squeeze(0).squeeze(0).detach().cpu().numpy()
    nifti_img_cond = nib.Nifti1Image(condition_save, affine = np.eye(4))
    nib.save(nifti_img_cond, path_out/f'cond_{i}.nii.gz')  

    print(x_0.max())
    print(x_0.min())
    x_0 = (x_0 + 1) / 2
    target_img1 = x_0.squeeze(0).squeeze(0).detach().cpu().numpy()
    nifti_img_t = nib.Nifti1Image(target_img1, affine = np.eye(4))
    nib.save(nifti_img_t, path_out/f'target_{i}.nii.gz')  
    # --------- Conditioning ---------
    # un_cond = torch.tensor([1-cond]*n_samples, device=device)
    un_cond = None 

    # ----------- Run --------
    if masked_condition:
        masked_x_0 = x_0.cuda() * condition
        pipeline.latent_embedder.eval() 
        with torch.no_grad():
            masked_x_0 = pipeline.latent_embedder.encode(masked_x_0)
        # downscale condition to 16x16x16
        condition = F.interpolate(condition, (16,16,16))
        condition = torch.cat([masked_x_0, condition], dim=1)
    
    results = pipeline.sample(n_samples, (4, 16, 16, 16), condition=condition, guidance_scale=1,  steps=steps, use_ddim=use_ddim )
    # results = pipeline.sample(n_samples, (4, 64, 64), guidance_scale=1, condition=condition, un_cond=un_cond, steps=steps, use_ddim=use_ddim )

    # --------- Save result ---------------
    results = (results+1)/2  # Transform from [-1, 1] to [0, 1]
    results = results.clamp(0, 1)


    path_out = Path(path_out)
    path_out.mkdir(parents=True, exist_ok=True)

    sample_img1 = results.squeeze(0).squeeze(0).detach().cpu().numpy()
    
    nifti_img_s = nib.Nifti1Image(sample_img1, affine = np.eye(4))

    nib.save(nifti_img_s, path_out/f'sample_{i}.nii.gz')  


    img = x_0[0, 0,:,:,:]
    fake = results[0, 0,:,:,:]

    img = img.cpu().numpy()
    fake = fake.cpu().numpy()
    fig, axs = plt.subplots(nrows=1, ncols=3)
    for ax in axs:
        ax.axis("off")
    ax = axs[0]
    ax.imshow(img[..., img.shape[2] // 2], cmap="gray")
    ax = axs[1]
    ax.imshow(img[:, img.shape[1] // 2, ...], cmap="gray")
    ax = axs[2]
    ax.imshow(img[img.shape[0] // 2, ...], cmap="gray")

    fig, axs = plt.subplots(nrows=1, ncols=3)
    for ax in axs:
        ax.axis("off")
    ax = axs[0]
    ax.imshow(fake[..., fake.shape[2] // 2], cmap="gray")
    ax = axs[1]
    ax.imshow(fake[:, fake.shape[1] // 2, ...], cmap="gray")
    ax = axs[2]
    ax.imshow(fake[fake.shape[0] // 2, ...], cmap="gray")