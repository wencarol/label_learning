from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
import torch
import torchvision
import torch.nn as nn
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'

class CIFAR10(torchvision.datasets.CIFAR10):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    """

    base_folder = "cifar-10-batches-py"
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = "c58f30108f718f92721af3b95e74349a"
    train_list = [
        ["data_batch_1", "c99cafc152244af753f735de768cd75f"],
        ["data_batch_2", "d4bba439e000b95fd0a9bffe97cbabec"],
        ["data_batch_3", "54ebc095f3ab1f0389bbae665268c751"],
        ["data_batch_4", "634d18415352ddfa80567beed471001a"],
        ["data_batch_5", "482c414d41f54cd18b22e5b47cb7c3cb"],
    ]

    test_list = [
        ["test_batch", "40351d587109b95175f43aff81a1287e"],
    ]
    meta = {
        "filename": "batches.meta",
        "key": "label_names",
        "md5": "5ff9c542aee3614f3951f8cda6e48888",
    }
    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False, quantized_data=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.dataset = 'cifar10'
        self._subsampled = False
        self._quantized = False
        self.num_classes = 10

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []
        self.soft_labels = []
        
        # load the generated dataset from file
        if quantized_data is not None and os.path.exists(os.path.join(self.root, self.base_folder, quantized_data)):
            quantized_data = os.path.join(self.root, self.base_folder, quantized_data)
            print('Load quantized dataset from %s.'%quantized_data)
            with open(quantized_data, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data = entry['data'] 
                self.targets = entry['labels']
                self.soft_labels = entry['soft_labels']
                self._quantized = True

        # or load the picked numpy arrays
        else:
            for file_name, checksum in downloaded_list:
                file_path = os.path.join(self.root, self.base_folder, file_name)
                with open(file_path, 'rb') as f:
                    entry = pickle.load(f, encoding='latin1')
                    self.data.append(entry['data'])
                    if 'labels' in entry:
                        self.targets.extend(entry['labels'])
                    else:
                        self.targets.extend(entry['fine_labels'])
            self.targets = self.targets if isinstance(self.targets, np.ndarray) else np.array(self.targets).astype(np.int64)
            self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
            self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

    def subsample(self, seed: int, fraction: float):
        """Subsample the dataset."""

        if self._subsampled:
            raise ValueError('Cannot subsample more than once.')

        self._subsampled = True
        examples_to_retain = np.ceil(len(self.targets) * fraction).astype(int)
        examples_to_retain = np.random.RandomState(seed=seed).permutation(len(self.targets))[:examples_to_retain]
        self.targets = self.targets[examples_to_retain]
        self.data = self.data[examples_to_retain]

    def label_quantization(self, seed: int, mix_mode: str = 'mixup', mixing_complexity: int = 2,
                           trail_num: int = 100,
                           save_path: str = None):
        """Mix inputs using mixing-based methods, and quantize soft labels to hard labels,
           e.g., quantize labels to y_1 or y_2 with probability p(y_1)=lam, p(y_2)=1-lam.

        Args:
            seed (int): random seed to control reproducibility.
            mix_mode (str, optional): mix mode, either mixup or patchmix mode. Default: 'mixup'.
            mixing_complexity (int, optional): complexity of mixed inputs, either 2 or 4. Default: 2.
            trail_num (int, optional): number of trails for the multinominal distribution of lam. Default: 100.
            save_path (str, optional): path to save the quantized dataset. Default: None.
        """
        if self._quantized:
            print('Cannot mix inputs and quantize labels more than once, skip this process.')
            return

        self._quantized = True
        dataset_size = len(self.targets)
        shuffled_data_list, shuffled_targets_list = [], []
        for n in range(mixing_complexity):
            shuffled_index = np.random.RandomState(seed=seed+n+1).permutation(dataset_size)
            shuffled_data_list.append(self.data[shuffled_index])
            shuffled_targets_list.append(self.targets[shuffled_index])

        multinom_distribution = [1/float(mixing_complexity) for n in range(mixing_complexity)]
        for idx in range(dataset_size):
            lam = np.random.RandomState(seed=seed+idx).multinomial(trail_num, multinom_distribution, 
                                        size=1)[0] / float(trail_num)

            if mix_mode == 'mixup':
                self.data[idx] = lam[0] * shuffled_data_list[0][idx]
                for n in range(1, mixing_complexity):
                    self.data[idx] = self.data[idx] + lam[n] * shuffled_data_list[n][idx]

            elif mix_mode == 'patchmix+':
                if mixing_complexity == 4:
                    H, W = self.data.shape[-3:-1] # H,W,C
                    # cuting and splicing 4 images with non-equal size
                    cutted_img_list, targets_list, cut_rate = [], [], []
                    dividing_x = H * np.random.RandomState(seed=seed+idx+2).multinomial(trail_num, [0.5,0.5], 
                                                                                    size=1)[0][0] / float(trail_num)
                    dividing_y = W * np.random.RandomState(seed=seed+idx+3).multinomial(trail_num, [0.5,0.5], 
                                                                                    size=1)[0][0] / float(trail_num)
                    dividing_x, dividing_y = round(dividing_x), round(dividing_y)
                    patch_size = [(dividing_y,dividing_x), (dividing_y,W-dividing_x), (H-dividing_y,dividing_x), (H-dividing_y,W-dividing_x)]
                    anchor_point = [(0,0), (0,dividing_x), (dividing_y,0), (dividing_y,dividing_x)]
                    for n in range(mixing_complexity):
                        bby1, bby2, bbx1, bbx2 = anchor_point[n][0], anchor_point[n][0]+patch_size[n][0], \
                                                 anchor_point[n][1], anchor_point[n][1]+patch_size[n][1]
                        self.data[idx, bby1:bby2, bbx1:bbx2, :] = shuffled_data_list[n][idx, bby1:bby2, bbx1:bbx2, :]
                        lam[n] = (bbx2 - bbx1) * (bby2 - bby1) / (H * W)
                else:
                    raise ValueError('Method to mix inputs between %d classes is not implemented for %s mode'
                                     %(mixing_complexity, mix_mode))
            else:
                raise ValueError('%s for input is not supported in CIFAR-Q dataset'%(mix_mode))

            soft_label = np.zeros(self.num_classes)
            for n in range(mixing_complexity):
                soft_label += lam[n] * np.eye(self.num_classes)[shuffled_targets_list[n][idx]]
            self.soft_labels.append(soft_label)
            quantized_target = np.random.RandomState(seed=seed+idx+3).multinomial(1, soft_label, size=1)[0]
            self.targets[idx] = np.where(quantized_target == 1)[0]
        self.soft_labels = np.array(self.soft_labels).astype(np.float32)

        if save_path is not None:
            self.save_data(save_path)

        return

    def save_data(self, save_path):
        # save_path = 'train_' + save_path if self.train else 'test_' + save_path
        file = os.path.join(self.root, self.base_folder, save_path)
        print("save quantized dataset to %s ..."%file)
        entry = {'data': self.data, 'labels': self.targets, 'soft_labels': self.soft_labels}
        with open(file, 'wb') as f:
            pickle.dump(entry, f, protocol=pickle.HIGHEST_PROTOCOL)

    def download(self):
        with open(os.devnull, 'w') as fp:
            sys.stdout = fp
            super(CIFAR10, self).download()
            sys.stdout = sys.__stdout__

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


    def __len__(self):
        return len(self.targets)



class CIFAR100(CIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    """
    base_folder = "cifar-100-python"
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = "eb9058c3a382ffc7106e4002c42a8d85"
    train_list = [
        ["train", "16019d7e3df5f24257cddd939b257f8d"],
    ]

    test_list = [
        ["test", "f0ef6b0ae62326f3e7ffdfab6717acfc"],
    ]
    meta = {
        "filename": "meta",
        "key": "fine_label_names",
        "md5": "7973b15100ade9c7d40fb424638fde48",
    }

    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False, quantized_data=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.dataset = 'cifar100'
        self._subsampled = False
        self._quantized = False
        self.num_classes = 100
        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []
        self.soft_labels = []
        
        # load the generated dataset from file
        if quantized_data is not None and os.path.exists(os.path.join(self.root, self.base_folder, quantized_data)):
            quantized_data = os.path.join(self.root, self.base_folder, quantized_data)
            print('Load quantized dataset from %s.'%quantized_data)
            with open(quantized_data, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data = entry['data'] 
                self.targets = entry['labels']
                self.soft_labels = entry['soft_labels']
                self._quantized = True

        # or load the picked numpy arrays
        else:
            for file_name, checksum in downloaded_list:
                file_path = os.path.join(self.root, self.base_folder, file_name)
                with open(file_path, 'rb') as f:
                    entry = pickle.load(f, encoding='latin1')
                    self.data.append(entry['data'])
                    if 'labels' in entry:
                        self.targets.extend(entry['labels'])
                    else:
                        self.targets.extend(entry['fine_labels'])
            self.targets = self.targets if isinstance(self.targets, np.ndarray) else np.array(self.targets).astype(np.int64)
            self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
            self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

    def download(self):
        with open(os.devnull, 'w') as fp:
            sys.stdout = fp
            super(CIFAR100, self).download()
            sys.stdout = sys.__stdout__


if __name__ == '__main__':
    mix_mode = 'patchmix'
    mixing_complexity = 2
    trail_num = 100
    seed = 12345
    args_dict = {
                'mix_mode': mix_mode,
                'mixing_complexity': mixing_complexity,
                'trail_num': trail_num,
                'seed': seed,
                }
    save_path = [k + '_' + str(v) for k, v in args_dict.items()]
    save_path = 'train' + '_' + '-'.join(quantized_data)
    dataset = CIFAR10(root='/workspace/data/cifar10', train=False)
    dataset.label_quantization(mix_mode=mix_mode, mixing_complexity=mixing_complexity, trail_num=trail_num, 
                               seed=seed, save_path=save_path)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True, num_workers=2)