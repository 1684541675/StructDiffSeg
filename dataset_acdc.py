import itertools
import os
import random
import re
from glob import glob

import cv2
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from skimage import io
import cv2

from torchvision import transforms

class BaseDataSets(Dataset):
    def __init__(self, base_dir=None, split='train', list_dir=None, transform=None):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        train_ids, val_ids, test_ids = self._get_ids()
        if self.split.find('train') != -1:
            self.all_slices = os.listdir(
                self._base_dir + "/ACDC_training_slices")
            self.sample_list = []
            for ids in train_ids:
                new_data_list = list(filter(lambda x: re.match('{}.*'.format(ids), x) != None, self.all_slices))
                self.sample_list.extend(new_data_list)

        elif self.split.find('val') != -1:
            self.all_volumes = os.listdir(
                self._base_dir + "/ACDC_training_volumes")
            self.sample_list = []
            for ids in val_ids:
                new_data_list = list(filter(lambda x: re.match('{}.*'.format(ids), x) != None, self.all_volumes))
                self.sample_list.extend(new_data_list)

        elif self.split.find('test') != -1:
            self.all_volumes = os.listdir(
                self._base_dir + "/ACDC_training_volumes")
            self.sample_list = []
            for ids in test_ids:
                new_data_list = list(filter(lambda x: re.match('{}.*'.format(ids), x) != None, self.all_volumes))
                self.sample_list.extend(new_data_list)

        # if num is not None and self.split == "train":
        #     self.sample_list = self.sample_list[:num]
        # ================== 新增代码：小数据调试模式 ==================
        
        '''
        if self.split == "train":
            # 比如只取前 100 张切片来训练 (原量是2000+)
            self.sample_list = self.sample_list[:100] 
            print(f"⚠️ 调试模式开启：仅使用 {len(self.sample_list)} 张切片进行训练！")
            
        else:
            # 测试集本来就只有12个，可以不改，或者只取前2个测测速度
            self.sample_list = self.sample_list[:1]
            print(f"⚠️ 调试模式开启：仅使用 {len(self.sample_list)} 个病例进行测试！")
        # ============================================================

        print("total {} samples".format(len(self.sample_list)))
        '''
        

    def _get_ids(self):
        # 1. 生成所有 1-100 的 ID
        all_cases_set = ["patient{:0>3}".format(i) for i in range(1, 101)]
        
        # 2. 【关键修改】使用随机种子打乱
        # 建议使用一个固定的种子（比如 1234, 42），这样你下次跑代码，训练集还是这些人
        # 如果你不固定种子，每次跑训练集都不一样，这在科研中是大忌！
        seed = 1234 
        rng = random.Random(seed) # 创建一个独立的随机生成器，不影响全局
        rng.shuffle(all_cases_set) # 原地打乱
        
        # 3. 按比例切分 (ACDC 标准通常是: 20测试, 10验证, 70训练)
        test_num = 20
        val_num = 10
        
        # 切片操作
        testing_set = all_cases_set[:test_num]
        validation_set = all_cases_set[test_num : test_num + val_num]
        training_set = all_cases_set[test_num + val_num :]
        
        # 打印一下，让你心里有数这次分了谁去测试
        #if self.split == 'test':
        #    print(f"⚠️ 当前随机种子 {seed} 下的测试集: {testing_set[:5]} ...")
            
        return [training_set, validation_set, testing_set]
    
    '''
    #第一次的结果是这样划分的,跑的是diffunetspm,结果存在acdc1
    def _get_ids(self):
        all_cases_set = ["patient{:0>3}".format(i) for i in range(1, 101)]
        testing_set = ["patient{:0>3}".format(i) for i in range(1, 21)]
        validation_set = ["patient{:0>3}".format(i) for i in range(21, 31)]
        training_set = [i for i in all_cases_set if i not in testing_set+validation_set]

        return [training_set, validation_set, testing_set]
    '''
    
    

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]

        # image = h5f['image'][:]
        # label = h5f['label'][:]
        # sample = {'image': image, 'label': label}
        if self.split == "train":
            h5f = h5py.File(self._base_dir + "/ACDC_training_slices/{}".format(case), 'r')
            image = h5f['image'][:]
            label = h5f['label'][:]  # fix sup_type to label
            sample = {'image': image, 'label': label}
            sample = self.transform(sample)
        else:
            h5f = h5py.File(self._base_dir + "/ACDC_training_volumes/{}".format(case), 'r')
            image = h5f['image'][:]
            label = h5f['label'][:]
            sample = {'image': image, 'label': label}
        sample["idx"] = idx
        sample['case_name'] = case.replace('.h5', '')
        return sample


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


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
        # ind = random.randrange(0, img.shape[0])
        # image = img[ind, ...]
        # label = lab[ind, ...]
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)  # the default is 0
            label = zoom( label, (self.output_size[0] / x, self.output_size[1] / y), order=0)

        assert (image.shape[0] == self.output_size[0]) and (image.shape[1] == self.output_size[1])
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        sample = {'image': image, 'label': label}
        return sample


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)



if __name__ == "__main__":
    
    root_path="../ACDC"
    img_size=256
    db_train = BaseDataSets(base_dir=root_path, split="train", transform=transforms.Compose([
        RandomGenerator([img_size, img_size])]))
    db_val = BaseDataSets(base_dir=root_path, split="val")
    db_test = BaseDataSets(base_dir=root_path, split="test_vol")