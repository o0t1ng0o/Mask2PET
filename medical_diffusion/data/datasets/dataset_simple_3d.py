
import torch.utils.data as data 
from pathlib import Path 
from torchvision import transforms as T

from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from glob import glob
import matplotlib.pyplot as plt
import nibabel as nib
import torchio as tio
import numpy as np
import torch
import re
import os
import scipy.ndimage as snd
import torchio as tio 

from medical_diffusion.data.augmentation.augmentations_3d import ImageToTensor


class SimpleDataset3D(data.Dataset):
    def __init__(
        self,
        path_root,
        item_pointers =[],
        crawler_ext = ['nii'], # other options are ['nii.gz'],
        transform = None,
        image_resize = None,
        flip = False,
        image_crop = None,
        use_znorm=True, # Use z-Norm for MRI as scale is arbitrary, otherwise scale intensity to [-1, 1]
    ):
        super().__init__()
        self.path_root = path_root
        self.crawler_ext = crawler_ext

        if transform is None: 
            self.transform = T.Compose([
                tio.Resize(image_resize) if image_resize is not None else tio.Lambda(lambda x: x),
                tio.RandomFlip((0,1,2)) if flip else tio.Lambda(lambda x: x),
                tio.CropOrPad(image_crop) if image_crop is not None else tio.Lambda(lambda x: x),
                tio.ZNormalization() if use_znorm else tio.RescaleIntensity((-1,1)),
                ImageToTensor() # [C, W, H, D] -> [C, D, H, W]
            ])
        else:
            self.transform = transform
        
        if len(item_pointers):
            self.item_pointers = item_pointers
        else:
            self.item_pointers = self.run_item_crawler(self.path_root, self.crawler_ext) 

    def __len__(self):
        return len(self.item_pointers)

    def __getitem__(self, index):
        rel_path_item = self.item_pointers[index]
        path_item = self.path_root/rel_path_item
        img = self.load_item(path_item)
        return {'uid':rel_path_item.stem, 'source': self.transform(img)}
    
    def load_item(self, path_item):
        return tio.ScalarImage(path_item) # Consider to use this or tio.ScalarLabel over SimpleITK (sitk.ReadImage(str(path_item)))
    
    @classmethod
    def run_item_crawler(cls, path_root, extension, **kwargs):
        return [path.relative_to(path_root) for path in Path(path_root).rglob(f'*.{extension}')]
    

class NiftiPairImageGenerator(Dataset):
    def __init__(self,
            input_folder: str,
            target_folder: str,
            input_size: int,
            depth_size: int,
            input_channel: int = 2,
            transform=None,
            target_transform=None,
            full_channel_mask=False,
            combine_output=False
        ):
        self.input_folder = input_folder
        self.target_folder = target_folder
        self.pair_files = self.pair_file()
        self.input_size = input_size
        self.depth_size = depth_size
        self.input_channel = input_channel
        self.scaler = MinMaxScaler()
        self.transform = transform
        self.target_transform = target_transform
        self.full_channel_mask = full_channel_mask
        self.combine_output = combine_output

    def pair_file(self):
        input_files = sorted(glob(os.path.join(self.input_folder, '*')))
        target_files = sorted(glob(os.path.join(self.target_folder, '*_0000.nii.gz')))
        pairs = []
        for input_file, target_file in zip(input_files, target_files):
            assert int("".join(re.findall("\d", input_file))) == int("".join(re.findall("\d", target_file))[:-4])
            pairs.append((input_file, target_file))
        return pairs

    def read_image(self, file_path, pass_scaler=False):
        img = nib.load(file_path).get_fdata()
        img = img.clip(min = 0)
        img = np.expand_dims(img,0)
        return img

    def plot(self, index, n_slice=30):
        data = self[index]
        input_img = data['input']
        target_img = data['target']
        plt.subplot(1, 2, 1)
        plt.imshow(input_img[:, :, n_slice])
        plt.subplot(1, 2, 2)
        plt.imshow(target_img[:, :, n_slice])
        plt.show()

    def resize_img(self, img):
        h, w, d = img.shape
        if h != self.input_size or w != self.input_size or d != self.depth_size:
            img = tio.ScalarImage(tensor=img[np.newaxis, ...])
            cop = tio.Resize((self.input_size, self.input_size, self.depth_size))
            img = np.asarray(cop(img))[0]
        return img

    def resize_img_4d(self, input_img):
        c, h, w, d = input_img.shape
        scaled_img = snd.zoom(input_img, [c, self.input_size / h, self.input_size / w, self.depth_size / d])
        return scaled_img.clip(min = 0)

    def resize_img_4d_pad(self, input_img):
        c, h, w, d = input_img.shape
        pad_one_side = (self.input_size - h) // 2
        padding = [(0,0), (pad_one_side,pad_one_side), (pad_one_side,pad_one_side), (pad_one_side,pad_one_side)]
        scaled_img = np.pad(input_img, padding, mode='constant', constant_values=0)
        return scaled_img.clip(min = 0)

    def resize_img_4d_01(self, input_img):
        c, h, w, d = input_img.shape
        scaled_img = snd.zoom(input_img, [c, self.input_size / h, self.input_size / w, self.depth_size / d], order=0)
        scaled_img = np.where(scaled_img > 0.5, 1, 0)
        return scaled_img

    def sample_conditions(self, batch_size: int):
        indexes = np.random.randint(0, len(self), batch_size)
        input_files = [self.pair_files[index][0] for index in indexes]
        input_tensors = []
        for input_file in input_files:
            input_img = self.read_image(input_file, pass_scaler=self.full_channel_mask)
            input_img = np.expand_dims(input_img,0)
            # input_img = self.resize_img_4d(input_img)
            if self.transform is not None:
                input_img = self.transform(input_img).unsqueeze(0)
                input_tensors.append(input_img)
        return torch.cat(input_tensors, 0).cuda()

    def __len__(self):
        return len(self.pair_files)

    def __getitem__(self, index):
        input_file, target_file = self.pair_files[index]
        input_img = self.read_image(input_file, pass_scaler=self.full_channel_mask)
        # input_img = self.resize_img_4d(input_img) # if not self.full_channel_mask else self.resize_img_4d(input_img)
        input_img = self.resize_img_4d_01(input_img)
        target_img = self.read_image(target_file)
        target_img = self.resize_img_4d(target_img)
        target_img = target_img / target_img.max()
        
        if self.transform is not None:
            input_img = self.transform(input_img)
        if self.target_transform is not None:
            target_img = self.target_transform(target_img)

        if self.combine_output:
            return torch.cat([target_img, input_img], 0)

        return {'input':input_img, 'target':target_img}
