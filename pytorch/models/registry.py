
from models import resnet, wide_resnet, wrn_madry, imagenet_resnet
import torch

def get(model_name, num_classes=10, depth=None, width_factor=None, drop_rate=None):

    if model_name == "resnet18":
        model = resnet.ResNet18(num_classes)
        net = "resnet18"
    if model_name == "resnet34":
        model = resnet.ResNet34(num_classes)
        net = "resnet34"
    if model_name == "imagenet_resnet50":
        model = imagenet_resnet.ResNet50(num_classes)
        net = "resnet50 for imagenet"
    if model_name == "WRN":
        # WRN-34-10
        model = wide_resnet.Wide_ResNet(depth=depth, num_classes=num_classes, widen_factor=width_factor, dropRate=drop_rate)
        net = "WRN{}-{}-dropout{}".format(depth,width_factor,drop_rate)
    if model_name == 'WRN_madry':
        # WRN-32-10
        model = wrn_madry.Wide_ResNet_Madry(depth=depth, num_classes=num_classes, widen_factor=width_factor, dropRate=drop_rate)
        net = "WRN_madry{}-{}-dropout{}".format(depth, width_factor, drop_rate)

    print("   ",net)

    return model


def load(model, state_dict_path):
    """
    load model with parameters
    """
    state_dict = torch.load(state_dict_path)['state_dict']
    model.load_state_dict(state_dict)

    return model
