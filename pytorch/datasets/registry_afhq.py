
import torchvision.transforms as transforms
from datasets.cifar import CIFAR10, CIFAR100
from datasets.afhq import AFHQ

train_cifar10_transform = transforms.Compose([
    transforms.RandomCrop(28, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

test_cifar10_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_cifar100_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

test_cifar100_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])


train_afhq_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.1, 1.0), ratio=(0.8, 1.25)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

test_afhq_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


def get(data_name, data_path, train: bool, quantized_data=None,
        mixing_complexity: int = 2, mix_mode: str = 'mixup', 
        trail_num: int = 100, seed: int = None, test_root: str=None, val_root: str=None):

    cifar10_transform = train_cifar10_transform if train else test_cifar10_transform
    cifar100_transform = train_cifar100_transform if train else test_cifar100_transform
    afhq_transform = train_afhq_transform if train else test_afhq_transform

    seed = seed if seed is not None else 0
    _dataset_type = 'train' if train else 'test'

    # quantized data name
    if quantized_data is None:
        args_dict = {
                    'mix_mode': mix_mode,
                    'mixing_complexity': mixing_complexity,
                    'trail_num': trail_num,
                    'seed': seed,
                    }
        quantized_data = [str(v) for k, v in args_dict.items()]
        quantized_data = _dataset_type + '_' + '_'.join(quantized_data)

    if data_name == 'cifar10':
        dataset = CIFAR10(root=data_path,
                            download=True,
                            train=train,
                            transform=cifar10_transform,
                            quantized_data=quantized_data,
                            )
        num_classes = 10
    elif data_name == 'cifar100':
        dataset = CIFAR100(root=data_path,
                            download=True,
                            train=train,
                            transform=cifar100_transform,
                            quantized_data=quantized_data,
                            )
        num_classes = 100
    elif data_name == 'afhq':
        dataset = AFHQ(train_root=data_path,
                       test_root=test_root,
                       val_root=val_root,
                       train=train,
                       transform=afhq_transform,
                        )
        num_classes = 3
    else:
        raise ValueError('%s dataset is not supported.'%(data_name))

    if data_name != 'afhq':
        print('Mix inputs and quantize labels with random seed %i for %s dataset'%(seed, _dataset_type))
        dataset.label_quantization(seed=seed, mix_mode=mix_mode, mixing_complexity=mixing_complexity,
                                    trail_num=trail_num, save_path=quantized_data)


    return dataset, num_classes
