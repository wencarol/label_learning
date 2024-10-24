from __future__ import print_function
from PIL import Image
from torch.utils.data import Dataset
import os


afhq_targets = {'cat':0, 'dog':1, 'wild':2}

class AFHQ(Dataset):

    def __init__(self, train_root, test_root, val_root, train=True, transform=None):
        self.train_imgs_path = []
        self.test_imgs_path = []
        # self.val_imgs_path = []
        self.train_labels = []
        self.test_labels = []

        self.transform = transform
        self.train = train

        train_files = [f for f in os.listdir(train_root) if 'quantized' in f]
        test_files = [f for f in os.listdir(test_root) if 'jpg' in f]
        val_files = [f for f in os.listdir(val_root) if 'jpg' in f]

        for f in train_files:
            img_path = os.path.join(train_root, f)
            label = afhq_targets[img_path.split('_')[-2]]
            self.train_imgs_path.append(img_path)
            self.train_labels.append(label)
        
        for f in test_files:
            img_path = os.path.join(test_root, f)
            label = afhq_targets[img_path.split('_')[-2]]
            self.test_imgs_path.append(img_path)
            self.test_labels.append(label)

        for f in val_files:
            img_path = os.path.join(val_root, f)
            label = afhq_targets[img_path.split('_')[-2]]
            self.test_imgs_path.append(img_path)
            self.test_labels.append(label)


    # We need to override __len__ special method
    def __len__(self):
        if self.train:
            return len(self.train_imgs_path)
        else:
            return len(self.test_imgs_path)
        # elif self.mode=='val':
        #     return len(self.val_imgs_path)


    # Also we need to override __getitem__ special method
    # This method should return the image and its label from index given.
    def __getitem__(self, index):
        if self.train:
            img_path = self.train_imgs_path[index]
            target = self.train_labels[index]
        else:
            img_path = self.test_imgs_path[index]
            target = self.test_labels[index]
        # elif self.mode=='val':
        #     img_path = self.val_imgs_path[index]
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        return img, target


    def classes_num(self):
        train_cat_num = self.train_labels.count(0)
        train_dog_num = self.train_labels.count(1)
        train_wild_num = self.train_labels.count(2)

        test_cat_num = self.test_labels.count(0)
        test_dog_num = self.test_labels.count(1)
        test_wild_num = self.test_labels.count(2)

        print(train_cat_num, train_dog_num, train_wild_num, train_cat_num+train_dog_num+train_wild_num)
        print(test_cat_num, test_dog_num, test_wild_num, test_cat_num+test_dog_num+test_wild_num)

if __name__ == '__main__':
    train_root = '/root/paddlejob/workspace/env_run/expr/local_editing/afhq/synthesized_image/'
    val_root = '/root/paddlejob/workspace/env_run/data/afhq/raw_images/val/images/'
    test_root = '/root/paddlejob/workspace/env_run/data/afhq/raw_images/test/images/'
    dataset = AFHQ(train_root, test_root, val_root, train=True, transform=None)
    dataset.classes_num()