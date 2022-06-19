import numpy as np
import random
import torch
import math
from math import exp
from PIL import Image
import random

def construct_transformation_matrix(limits):
    # limits is a list of [(y_min, y_max), (x_min, x_max)]
    # in normalized coordinates from -1 to 1
    x_limits = limits[1]
    y_limits = limits[0]
    theta = torch.zeros(2, 3)
    tx = np.sum(x_limits) / 2
    ty = np.sum(y_limits) / 2
    s = x_limits[1] - tx
    assert(np.abs((x_limits[1] - tx) - (y_limits[1] - ty)) < 1e-9)
    theta[0, 0] = s
    theta[1, 1] = s
    theta[0, 2] = (tx) / 2
    theta[1, 2] = (ty) / 2
    transform = torch.zeros(3, 3)
    transform[:2, :] = theta
    transform[2, 2] = 1.0
    return transform

class PatchSampler(object):
    def __init__(self, patch_size, random_shift=True, random_scale=True, scale_anneal=-1,
                 max_scale=None, min_scale=None, **kwargs):
        self.patch_size = patch_size # variable p in paper
        self.random_shift = random_shift
        self.random_scale = random_scale
        # full image range is [-1, 1]
        self.w = np.array([-1, 1])
        self.h = np.array([-1, 1])
        # image size = 1/scale
        self.max_scale = max_scale if max_scale is not None else 1.0
        self.min_scale = min_scale
        self.iterations = 0
        self.scale_anneal = scale_anneal
        self.initial_min = 1.0 # 0.9

    def sample_patch(self, im):
        im_w, im_h  = im.size
        assert(im_w == im_h) # crop to square image before patch sampling

        # minimum scale bound based on image size
        min_scale = self.patch_size / im_h
        if self.min_scale is not None:
            min_scale = max(min_scale, self.min_scale)
        params = {'min_scale_absolute': min_scale}

        #  adjust min scale if annealing
        if self.scale_anneal > 0:
            k_iter = (self.iterations)// 1000 * 3
            # decays min_scale between self.min_scale and initial_min
            min_scale = max(min_scale, self.max_scale * exp(-k_iter*self.scale_anneal))
            min_scale = min(self.initial_min, min_scale)
        params['min_scale_anneal'] = min_scale

        scale = 1.0
        if self.random_scale:
            # this samples is size uniformly from min_size to max_size
            max_size = self.patch_size / min_scale
            min_size = self.patch_size / self.max_scale
            random_size = random.uniform(min_size, max_size)
            scale = self.patch_size / random_size
        params['sampled_scale'] = scale

        # resize the image to a random new size and take a crop
        new_size = int(np.round(self.patch_size / scale))
        crop_size = self.patch_size
        im_resized = im.resize((new_size, new_size), Image.LANCZOS)
        assert(new_size <= im_h) # do not upsample
        x = random.randint(0, np.maximum(0, new_size - crop_size)) # inclusive [low, high]
        y = random.randint(0, np.maximum(0, new_size - crop_size))
        im_crop = im_resized.crop((x, y, x+crop_size, y+crop_size))

        # normalized limits
        limits = [(y/(new_size)*2-1, (y+crop_size) /(new_size)*2-1),
                  (x/(new_size)*2-1, (x+crop_size) /(new_size)*2-1)]
        params['limits'] = limits
        params['x'] = x
        params['y'] = y
        params['new_size'] = new_size
        params['orig_size'] = im_w

        # calculate the transformation matrix
        transform = construct_transformation_matrix(limits)
        params['transform'] = transform

        return im_crop, params

def generate_full_from_patches(new_size, patch_size=256):
    # returns the bounding boxes and transformations needed to 
    # piece together patches of size patch_size into a 
    # full image of size new_size
    patch_params = []
    for y in range(0, new_size, patch_size):
        for x in range(0, new_size, patch_size):
            if y + patch_size > new_size:
                y = new_size - patch_size
            if x + patch_size > new_size:
                x = new_size - patch_size
            limits = [(y/(new_size)*2-1, (y+patch_size) /(new_size)*2-1),
              (x/(new_size)*2-1, (x+patch_size) /(new_size)*2-1)]
            transform = construct_transformation_matrix(limits)
            patch_params.append(((y, y+patch_size, x, x+patch_size), transform))
    return patch_params

def compute_scale_inputs(G, w, transform):
    if transform is None:
        scale = torch.ones(w.shape[0], 1).to(w.device)
    else:
        scale = 1/transform[:, [0], 0]
    scale = G.scale_norm(scale)
    mapped_scale = G.scale_mapping(scale, None)
    return scale, mapped_scale

def scale_condition_wrapper(G, w, transform, **kwargs):
    # convert transformation matrix into scale input
    # and pass through scale mapping network
    if not G.scale_mapping_kwargs:
        img = G.synthesis(w, transform=transform, **kwargs)
        return img
    scale, mapped_scale = compute_scale_inputs(G, w, transform)
    img = G.synthesis(w, mapped_scale=mapped_scale, transform=transform, **kwargs)
    return img
