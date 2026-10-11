from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image

import random
import h5py
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from scipy import ndimage
from PIL import Image
from torchvision import transforms

class NPY_datasets(Dataset):
    def __init__(self, path_Data, config=None, train=True, *, split=None,
                 list_dir=None, output_size=(256, 256)):
        """Original tuple interface, or binary RGB dict interface with split set.

        Example: NPY_datasets(root, split='train'). TXT mode uses its own
        preprocessing; config transformers apply only to the original mode.
        """
        super(NPY_datasets, self)
        self.binary_dataset = None
        if split is not None:
            self.binary_dataset = RGBBinaryDataset(
                path_Data, split, list_dir=list_dir, output_size=output_size)
            return
        if train:
            images_list = sorted(os.listdir(path_Data+'train/images/'))
            masks_list = sorted(os.listdir(path_Data+'train/masks/'))
            self.data = []
            for i in range(len(images_list)):
                img_path = path_Data+'train/images/' + images_list[i]
                mask_path = path_Data+'train/masks/' + masks_list[i]
                self.data.append([img_path, mask_path])
            self.transformer = config.train_transformer
        else:
            images_list = sorted(os.listdir(path_Data+'val/images/'))
            masks_list = sorted(os.listdir(path_Data+'val/masks/'))
            self.data = []
            for i in range(len(images_list)):
                img_path = path_Data+'val/images/' + images_list[i]
                mask_path = path_Data+'val/masks/' + masks_list[i]
                self.data.append([img_path, mask_path])
            self.transformer = config.test_transformer
        
    def __getitem__(self, indx):
        if self.binary_dataset is not None:
            return self.binary_dataset[indx]
        img_path, msk_path = self.data[indx]
        img = np.array(Image.open(img_path).convert('RGB'))
        msk = np.expand_dims(np.array(Image.open(msk_path).convert('L')), axis=2) / 255
        img, msk = self.transformer((img, msk))
        return img, msk

    def __len__(self):
        if self.binary_dataset is not None:
            return len(self.binary_dataset)
        return len(self.data)
    


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


class RGBBinaryTransform:
    """Shared train-only rotations/flips; deterministic resize for val/test.

    output_size is (height, width). RGB uses bilinear resize and [0, 1]
    normalization; binary class labels use nearest resize.
    """

    def __init__(self, output_size=(256, 256), augment=False):
        if (len(output_size) != 2 or
                any(not isinstance(size, int) or size <= 0 for size in output_size)):
            raise ValueError('output_size must contain two positive integers')
        if augment and output_size[0] != output_size[1]:
            raise ValueError('rotation augmentation requires a square output_size')
        self.output_size = tuple(output_size)
        self.augment = augment

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        size = (self.output_size[1], self.output_size[0])
        image = np.asarray(Image.fromarray(image).resize(size, Image.Resampling.BILINEAR))
        # Merge instance IDs before resize; never divide the mask by 255.
        label = np.asarray(Image.fromarray((label > 0).astype(np.uint8)).resize(
            size, Image.Resampling.NEAREST))
        if self.augment:
            image, label = random_rot_flip(image, label)
        result = dict(sample)
        result['image'] = torch.from_numpy(
            np.ascontiguousarray(image.transpose(2, 0, 1), dtype=np.float32)) / 255.0
        result['label'] = torch.from_numpy(np.ascontiguousarray(label, dtype=np.int64))
        return result


class RGBBinaryDataset(Dataset):
    """GlaS/CVC PNG pairs selected by train.txt, val.txt or test.txt.

    Layout: root/images/<ID>.png and root/masks/<ID>.png. Split files
    default to root; list_dir may specify a different directory. Nonzero
    mask values are foreground; raw instance IDs are preserved when read.
    """

    def __init__(self, base_dir, split, list_dir=None, output_size=(256, 256)):
        if split not in ('train', 'val', 'test'):
            raise ValueError("split must be 'train', 'val' or 'test'")
        self.split = split
        self.data_dir = os.fspath(base_dir)
        self.transform = RGBBinaryTransform(output_size, augment=split == 'train')
        split_path = os.path.join(list_dir if list_dir is not None else base_dir,
                                  split + '.txt')
        with open(split_path, encoding='utf-8-sig') as handle:
            self.sample_list = [line.strip() for line in handle if line.strip()]
        if not self.sample_list:
            raise ValueError(f'Empty split file: {split_path}')
        if len(set(self.sample_list)) != len(self.sample_list):
            raise ValueError(f'Duplicate IDs in split file: {split_path}')
        self.data = []
        for case_name in self.sample_list:
            if (case_name in ('.', '..') or '/' in case_name or
                    '\\' in case_name or ':' in case_name):
                raise ValueError(f'Expected a plain sample ID, got {case_name!r}')
            image_path = os.path.join(self.data_dir, 'images', case_name + '.png')
            mask_path = os.path.join(self.data_dir, 'masks', case_name + '.png')
            for path in (image_path, mask_path):
                if not os.path.isfile(path):
                    raise FileNotFoundError(f'Missing {split} sample {case_name}: {path}')
            self.data.append((image_path, mask_path))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, mask_path = self.data[idx]
        with Image.open(image_path) as image_file:
            image = np.array(image_file.convert('RGB'))
        with Image.open(mask_path) as mask_file:
            label = np.array(mask_file)
        # CVC stores grayscale binary masks in three identical RGB channels.
        if label.ndim == 3 and label.shape[2] == 3:
            if not np.all(label == label[..., :1]):
                raise ValueError(f'Expected identical RGB mask channels: {mask_path}')
            label = label[..., 0]
        if label.ndim != 2:
            raise ValueError(f'Expected a 2D class/instance mask: {mask_path}')
        if image.shape[:2] != label.shape:
            raise ValueError(f'Image/mask dimensions differ: {image_path}, {mask_path}')
        return self.transform({'image': image, 'label': label,
                               'case_name': self.sample_list[idx]})


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample

class Generator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample

class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        
        '''
        # ================== 新增代码：小数据调试模式 ==================
        if self.split == "train"or self.split == "val_vol_2d":
            # 比如只取前 100 张切片来训练 (原量是2000+)
            self.sample_list = self.sample_list[:30] 
            print(f"⚠️ 调试模式开启：仅使用 {len(self.sample_list)} 张切片进行训练！")
            
        else:
            # 测试集本来就只有12个，可以不改，或者只取前2个测测速度
            self.sample_list = self.sample_list[:1]
            print(f"⚠️ 调试模式开启：仅使用 {len(self.sample_list)} 个病例进行测试！")
        # ============================================================

        '''
        if os.getenv("LOCAL_DEBUG", "0") == "1":
            if self.split == "train" or self.split == "val_vol_2d":
                self.sample_list = self.sample_list[:20]
            else:
                self.sample_list = self.sample_list[:1]

            print(
                f"⚠️ LOCAL_DEBUG：{self.split} 当前仅使用 "
                f"{len(self.sample_list)} 个样本"
            )
        
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train"or self.split == "val_vol_2d":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath)
            image, label = data['image'][:], data['label'][:]

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample
    
if __name__ == "__main__":
    
    data_path = '../../Synapse/train_npz/'
    list_dir = '../../Synapse/lists/lists_Synapse/'
    volume_path = '../../Synapse/test_vol_h5/'
    img_size=256
    train_dataset = Synapse_dataset(base_dir=data_path, list_dir=list_dir, split="train",
                            transform=transforms.Compose(
                                [RandomGenerator(output_size=[img_size, img_size])]))

    val_dataset = Synapse_dataset(base_dir=volume_path, split="test_vol", list_dir=list_dir)
    
        
    