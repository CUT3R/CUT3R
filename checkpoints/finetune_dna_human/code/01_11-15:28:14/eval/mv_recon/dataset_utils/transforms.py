# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------

import torchvision.transforms as tvf
from dust3r.utils.image import ImgNorm


ColorJitter = tvf.Compose([tvf.ColorJitter(0.5, 0.5, 0.5, 0.1), ImgNorm])


def _check_input(value, center=1, bound=(0, float("inf")), clip_first_on_zero=True):
    if isinstance(value, (int, float)):
        if value < 0:
            raise ValueError(f"If  is a single number, it must be non negative.")
        value = [center - float(value), center + float(value)]
        if clip_first_on_zero:
            value[0] = max(value[0], 0.0)
    elif isinstance(value, (tuple, list)) and len(value) == 2:
        value = [float(value[0]), float(value[1])]
    else:
        raise TypeError(f"should be a single number or a list/tuple with length 2.")

    if not bound[0] <= value[0] <= value[1] <= bound[1]:
        raise ValueError(f"values should be between {bound}, but got {value}.")

    if value[0] == value[1] == center:
        return None
    else:
        return tuple(value)


import torch
import torchvision.transforms.functional as F


def SeqColorJitter():
    """
    Return a color jitter transform with same random parameters
    """
    brightness = _check_input(0.5)
    contrast = _check_input(0.5)
    saturation = _check_input(0.5)
    hue = _check_input(0.1, center=0, bound=(-0.5, 0.5), clip_first_on_zero=False)

    fn_idx = torch.randperm(4)
    brightness_factor = (
        None
        if brightness is None
        else float(torch.empty(1).uniform_(brightness[0], brightness[1]))
    )
    contrast_factor = (
        None
        if contrast is None
        else float(torch.empty(1).uniform_(contrast[0], contrast[1]))
    )
    saturation_factor = (
        None
        if saturation is None
        else float(torch.empty(1).uniform_(saturation[0], saturation[1]))
    )
    hue_factor = None if hue is None else float(torch.empty(1).uniform_(hue[0], hue[1]))

    def _color_jitter(img):
        for fn_id in fn_idx:
            if fn_id == 0 and brightness_factor is not None:
                img = F.adjust_brightness(img, brightness_factor)
            elif fn_id == 1 and contrast_factor is not None:
                img = F.adjust_contrast(img, contrast_factor)
            elif fn_id == 2 and saturation_factor is not None:
                img = F.adjust_saturation(img, saturation_factor)
            elif fn_id == 3 and hue_factor is not None:
                img = F.adjust_hue(img, hue_factor)
        return ImgNorm(img)

    return _color_jitter
