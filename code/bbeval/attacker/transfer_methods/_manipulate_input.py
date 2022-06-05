import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from PIL import Image


# TODO: figure out where this function should belong
def save_img(save_path, img):
    Image.fromarray(np.array(img * 255).astype('uint8')
                    ).save(save_path, quality=95)


def input_diversity(input_tensor, image_width, image_resize, prob, interpol_dim = None, mode='nearest', **kwargs):
    if interpol_dim == None:
        interpol_dim = image_resize
    rnd = torch.randint(image_width, image_resize, ())
    rescaled = F.interpolate(
        input_tensor, size=[rnd, rnd], mode='bilinear', align_corners=True)
    h_rem = image_resize - rnd
    w_rem = image_resize - rnd
    pad_top = torch.randint(0, h_rem, ())
    pad_bottom = h_rem - pad_top
    pad_left = torch.randint(0, w_rem, ())
    pad_right = w_rem - pad_left
    pad_list = (pad_left, pad_right, pad_top, pad_bottom)
    padded = nn.ConstantPad2d(pad_list, 0.)(rescaled)
    padded = nn.functional.interpolate(padded, [image_resize, image_resize],mode=mode)
    return padded if torch.rand(()) < prob else input_tensor

def ensemble_input_diversity(input_tensor, image_width, image_resize, prob=1.0, interpol_dim=None, mode='bilinear', **kwargs):
    """
    # below is the original implementation, which has high overlap with the method described before
    # [560,620,680,740,800] --> [575, 650, 725, 800]
    image_width = kwargs.get('image_width')
    image_resizes = kwargs.get('image_resizes')

    # TODO: manually set these values for testing.
    image_width = 500
    # check what the values for these resized images should be for regular imagenet dataset
    # >= 575 dims are for imagenet challenges (https://github.com/qilong-zhang/CVPR2021-Competition-Unrestricted-Adversarial-Attacks-on-ImageNet/blob/main/run.py),
    # not for the regular imagenet dataset we test, so need to figure out possible values for these.
    image_resizes = [330, 575, 650, 725, 800] 

    rnd = torch.randint(image_width, image_resizes[idx], ())
    rescaled = F.interpolate(
        input_tensor, size=[rnd, rnd], mode='bilinear', align_corners=True)
    h_rem = image_resizes[idx] - rnd
    w_rem = image_resizes[idx] - rnd
    pad_top = torch.randint(0, h_rem, ())
    pad_bottom = h_rem - pad_top
    pad_left = torch.randint(0, w_rem, ())
    pad_right = w_rem - pad_left
    pad_list = (pad_left, pad_right, pad_top, pad_bottom)
    padded = nn.ConstantPad2d(pad_list, 0.)(rescaled)
    padded = nn.functional.interpolate(padded, [256, 256], mode='bilinear')
    """
    padded = input_diversity(input_tensor, image_width, image_resize, prob, interpol_dim, mode)
    return padded

def clip_by_tensor(t, t_min, t_max):
    """
    clip_by_tensor
    :param t: tensor
    :param t_min: min
    :param t_max: max
    :return: cliped tensor
    """
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result
