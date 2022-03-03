import random
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import glob
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from typing import Optional, Union


def _identity(x):
    return x


def resize(img: np.ndarray, height: Optional[int], width: Optional[int], interpolation: Optional[int]) -> np.ndarray:
    if width is None or height is None or interpolation is None:
        return img
    if img.shape[0] == height and img.shape[1] == width:
        return img
    return cv2.resize(img, (width, height), interpolation=interpolation)


class RandomCrop(object):
    def __init__(self, size, inter_flag=False):
        self.size = size
        self.inter_flag = inter_flag

    def __call__(self, im_lb):
        im = im_lb['im']
        lb = im_lb['lb']
        if self.inter_flag:
            inter = im_lb['inter']
        # assert im.size == lb.size
        W, H = self.size
        w, h = im.size

        if (W, H) == (w, h): return dict(im=im, lb=lb, inter=inter)
        if w < W or h < H:
            scale = float(W) / w if w < h else float(H) / h
            w, h = int(scale * w + 1), int(scale * h + 1)
            im = im.resize((w, h), Image.BILINEAR)
            lb = lb.resize((w, h), Image.NEAREST)
            if self.inter_flag:
                inter = cv2.resize((w, h), Image.NEAREST)
        sw, sh = random.random() * (w - W), random.random() * (h - H)
        crop = int(sw), int(sh), int(sw) + W, int(sh) + H
        if self.inter_flag:
            return dict(im=im.crop(crop), lb=lb.crop(crop), inter=inter.crop(crop))
        else:
            return dict(im=im.crop(crop), lb=lb.crop(crop))


class SceneNetDataLoader(Dataset):
    def __init__(self, root_path, mode='train', crop_size=(192, 256)):
        # super(SceneNetDataLoader, self).__init__()
        assert mode in ('train', 'val', 'test')
        self.average_depth = 3.7584526086847587
        self.images_one_seq = int(3e5)
        if mode == 'train':
            self.total_seq = 1
        else:
            self.total_seq = 1
        self.mode = mode
        self.crop_size = (crop_size[1], crop_size[0])
        self.data_root = os.path.join(root_path, mode)
        image_seq = os.listdir(self.data_root)
        self.interpolation = 1
        # self.to_tensor = transforms.Compose([transforms.RandomCrop(crop_size), transforms.ToTensor()])
        self.random_crop = RandomCrop(self.crop_size)

    def __len__(self):
        return self.total_seq * self.images_one_seq

    def proximity_normilize(self, depth):
        '''
        This method trans the depth to [0, 1] according CodeSLAM introduce in Training Setup
        p = a/(d + a), a isi average depth
        :param depth: [C,W,H]
        :return: depth whitch value belong [0, 1]
        a = 3.7584526086847587
        '''
        p = np.divide(self.average_depth, (depth + self.average_depth))
        return p

    def __getitem__(self, item):
        seq_num, img_num = divmod(item, self.images_one_seq)
        folder, img_id = divmod(img_num, 300)  # each image fold contain 300 images
        source_dir = self.data_root + '/' + str(int(seq_num)) + '/' + str(int(1000 * seq_num + folder))
        photo_dir = source_dir + '/photo/' + str(int(25 * img_id)) + '.jpg'
        depth_dir = source_dir + '/depth/' + str(int(25 * img_id)) + '.png'
        photo = Image.open(photo_dir).convert('RGB')
        depth = Image.open(depth_dir)
        img_lb = dict(im=photo, lb=depth)
        img_lb = self.random_crop(img_lb)
        photo = img_lb['im']
        depth = np.array(img_lb['lb'], dtype=np.float32) / 1000.
        depth_norm = self.proximity_normilize(depth)

        depth_out_stage3 = np.copy(depth)
        depth_out_stage2 = resize(depth_out_stage3, height=self.crop_size[1] // 2, width=self.crop_size[0] // 2,
                                  interpolation=self.interpolation)
        depth_out_stage1 = resize(depth_out_stage3, height=self.crop_size[1] // 4, width=self.crop_size[0] // 4,
                                  interpolation=self.interpolation)

        # mask depth
        mask_out_stage3 = (depth_out_stage3 > 0).astype(np.float32)
        mask_out_stage2 = (depth_out_stage2 > 0).astype(np.float32)
        mask_out_stage1 = (depth_out_stage1 > 0).astype(np.float32)

        photo = transforms.ToTensor()(photo)
        depth_norm = transforms.ToTensor()(depth_norm)

        item = {
            'depth': {
                'stage3': depth_out_stage3,
                'stage2': depth_out_stage2,
                'stage1': depth_out_stage1
            },
            'mask': {
                'stage3': mask_out_stage3,
                'stage2': mask_out_stage2,
                'stage1': mask_out_stage1
            },
            'image': photo,
            'gt_norm': depth_norm
        }

        return item


def make_dataloader(hparams: dict, split: str, truncate=None):
    if hparams['DATA.NAME'] == 'replica' or hparams['DATA.NAME'] == 'dji':
        ds = SceneNetDataLoader(root_path=hparams["DATA.ROOT_DIR"], mode=split)
        batch_fun = _identity
    else:
        raise NotImplementedError(f"Dataset {hparams['DATA.NAME']} not implemented.")

    dataloader = DataLoader(
        dataset=ds,
        batch_size=hparams["TRAIN.BATCH_SIZE"],
        # shuffle=hparams["TRAIN.SHUFFLE"] and split == 'train',
        num_workers=hparams["TRAIN.NUM_WORKERS"],
        drop_last=hparams["TRAIN.DROP_LAST"] and split == 'train'
    )

    return dataloader, batch_fun
