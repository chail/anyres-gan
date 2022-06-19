# Anyres-GAN
[Project Page](https://chail.github.io/anyres-gan/) | [Paper](https://arxiv.org/abs/2204.07156) | [Bibtex](https://chail.github.io/anyres-gan/bibtex.txt)

Any-resolution Training for High-resolution Image Synthesis.\
[Lucy Chai](http://people.csail.mit.edu/lrchai/), [Michaël Gharbi](http://mgharbi.com/), [Eli Shechtman](https://research.adobe.com/person/eli-shechtman/), [Phillip Isola](http://web.mit.edu/phillipi/), [Richard Zhang](https://richzhang.github.io/)

## Prerequisites
- Linux
- gcc-7
- Python 3
- NVIDIA GPU + CUDA CuDNN

**Table of Contents:**<br>
1. [Colab](#colab) - run it in your browser without installing anything locally<br>
2. [Setup](#setup) - download pretrained models and resources
3. [Pretrained Models](#pretrained) - quickstart with pretrained models<br>
3. [Notebooks](#notebooks) - jupyter notebooks for interactive composition<br>
4. [Training](#training) - pipeline for training encoders<br>
5. [Evaluation](#evaluation) - evaluation script<br>


<img src='img/github_loop.gif'>


<a name="colab"/>

## Colab

[Interactive Demo](TODO): Try our interactive demo here! Does not require local installations. 

<a name="setup"/>

## Setup

- Clone this repo:
```bash
git clone https://github.com/chail/anyres-gan.git
```

- Install dependencies:
	- gcc-7 or above is required for installation. Update gcc following [these steps](https://gist.github.com/jlblancoc/99521194aba975286c80f93e47966dc5).
	- We provide a Conda `environment.yml` file listing the dependencies. You can create a Conda environment with the dependencies using:
```bash
conda env create -f environment.yml
```

- Download resources: we provide a script for downloading associated resources and pretrained models. Fetch these by running:
```bash
bash download_resources.sh
```

<a name="pretrained"/>

## Quickstart with pretrained models

Pretrained models are downloaded from the above `download_resources.sh` script. Any-resolution images can be constructed by specifying the appropriate transformation matrices. The following code snippet provides a basic example; additional examples can be found in the notebook. 

```python
import pickle
import torch
import numpy as np
from util import patch_util, renormalize
torch.set_grad_enabled(False)

PATH = 'pretrained/bird_pretrained_final.pkl'

with open(PATH, 'rb') as f:
    G_base = pickle.load(f)['G_ema'].cuda()  # torch.nn.Module
    
full_size = 500
seed = 0

rng = np.random.RandomState(seed)
z = torch.from_numpy(rng.standard_normal(G_base.z_dim)).float()
z = z[None].cuda()
c = None

ws = G_base.mapping(z, c, truncation_psi=0.5, truncation_cutoff=8)
full = torch.zeros([1, 3, full_size, full_size])
patches = patch_util.generate_full_from_patches(full_size, G_base.img_resolution)
for bbox, transform in patches:
    img = patch_util.scale_condition_wrapper(G_base, ws, transform[None].cuda(), noise_mode='const', force_fp32=True)
    full[:, :, bbox[0]:bbox[1], bbox[2]:bbox[3]] = img
renormalize.as_image(full[0])
```

<a name="notebooks"/>

## Notebooks

Note: remember to add the conda environment to jupyter kernels:
```bash
python -m ipykernel install --user --name anyres-gan
```

We provide example notebook `notebook-demo.ipynb` for running inference on pretrained models.

<a name="training"/>

## Training

See the script `train.sh` for training examples.

Training notes:
- patch-based training is run in two stages: first global fixed-resolution pretraining, then patch training
- arguments `--batch-gpu` and `--gamma` are taken from Stylegan 3 [recommended configurations](https://github.com/NVlabs/stylegan3/blob/main/docs/configs.md#recommended-configurations)
- arguments `--random_crop=True` and `--patch_crop=True` performs random cropping on fixed-resolution and variable resolution datasets respectively. 
- `--scale_max` and `--scale_min` correspond to the largest and smallest sampled image scales for patch training (size = 1/scale * g_size). `--scale_max` should correspond to the smallest image size in the patch dataset (for example, if the smallest image is 512px and the generator size is 256, then `--scale_max=0.5`).  Omitting `--scale_min` will use the smallest possible scale as the minimum bound (the image native size). 
- `--scale_mapping_min` and `--scale_mapping_max` correspond to normalization limits in the scale mapping branch; the min can be kept at 1 and the max can be set to an approximate zoom factor between the fixed-resolution dataset and the size of the HR images.
- for patch training, metrics are evaluated offline, hence `--metrics=none` should be specified for training. See below for more details on evaluation.


Training progress can be visualized using:
```bash
tensorboard --logdir training-runs/
```

Note on datasets: beyond the standard FFHQ and LSUN Church datasets, we train on datasets scraped from flickr. Due to licensing we cannot release this images directly but can provide the image IDs used to construct the datasets.

<a name="evalution"/>

## Evaluations

See `custom_metrics.sh` for an example on running FID variations and pFID on the patch models. 
- pFID can be specified using a string such as `fid-patch256-min256max0`: this samples 50k patches of size 256, with minimum image size 256 and maximum image size as the max size allowable by a given real image. 
- The max sampled size could also be specified with a number; for example `fid-patch256-min256max1024`. 
- For larger models (e.g. mountains), FID by default downsamples images to 299 width; therefore we use a variant that further takes a crop of the image: `fid-subpatch1024-min1024max0`. 
- Note that _these metrics are implemented to run on a single gpu._

Note: the released pretrained models are reimplementations of the models used in the current paper version, so the evaluation numbers are slightly different.


### Acknowledgements

Our code is largely based on the [Stylegan3](https://github.com/NVlabs/stylegan3) repository ([license](./LICENSE_stylegan.txt)). Changes to the StyleGAN3 code are documented in [diff](./diff.txt). Some additional utilities are from David Bau and Taesung Park. Remaining changes are covered under [Adobe Research License](./LICENSE.txt).

<a name="citation"/>

### Citation
If you use this code for your research, please cite our paper:
```
@article{chai2022anyresolution,
  title={Any-resolution training for high-resolution image synthesis.},
  author={Chai, Lucy and Gharbi, Michael and Shechtman, Eli and Isola, Phillip and Zhang, Richard},
  journal={arXiv preprint arXiv:2204.07156},
  year={2022}
}
```

<p align="center">
<img src='img/pano010-2.gif' width=600px>
<img src='img/pano010.png' width=600px>
</p>

