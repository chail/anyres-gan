# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Loss functions."""

import numpy as np
import torch
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import upfirdn2d

# added imports
from metrics import equivariance
from util import losses, util, patch_util
import random

#----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg): # to be overridden by subclass
        raise NotImplementedError()

#----------------------------------------------------------------------------

def apply_affine_batch(img, transform):
    # hacky .. apply affine transformation with cuda kernel in batch form
    crops = []
    masks = []
    for i, t in zip(img, transform):
        crop, mask = equivariance.apply_affine_transformation(
            i[None], t.inverse())
        crops.append(crop)
        masks.append(mask)
    crops = torch.cat(crops, dim=0)
    masks = torch.cat(masks, dim=0)
    return crops, masks

class StyleGAN2Loss(Loss):
    def __init__(self, device, G, D, augment_pipe=None, r1_gamma=10, style_mixing_prob=0, pl_weight=0, pl_batch_shrink=2,
                 pl_decay=0.01, pl_no_weight_grad=False, blur_init_sigma=0,
                 blur_fade_kimg=0, teacher=None, added_kwargs=None):
        super().__init__()
        self.device             = device
        self.G                  = G
        self.D                  = D
        self.augment_pipe       = augment_pipe
        self.r1_gamma           = r1_gamma
        self.style_mixing_prob  = style_mixing_prob
        self.pl_weight          = pl_weight
        self.pl_batch_shrink    = pl_batch_shrink
        self.pl_decay           = pl_decay
        self.pl_no_weight_grad  = pl_no_weight_grad
        self.pl_mean            = torch.zeros([], device=device)
        self.blur_init_sigma    = blur_init_sigma
        self.blur_fade_kimg     = blur_fade_kimg

        self.teacher = teacher
        self.added_kwargs = added_kwargs
        self.training_mode = self.G.training_mode
        if self.teacher is not None:
            self.loss_l1 = losses.Masked_L1_Loss().to(device)
            self.loss_lpips = losses.Masked_LPIPS_Loss(net='alex', device=device)
            util.set_requires_grad(False, self.loss_lpips)
            util.set_requires_grad(False, self.teacher)

    def style_mix(self, z, c, ws):
        if self.style_mixing_prob > 0:
            with torch.autograd.profiler.record_function('style_mixing'):
                cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]
        return ws

    def run_G(self, z, c, transform, update_emas=False):
        mapped_scale = None
        crop_fn = None
        if 'patch' in self.training_mode:
            ws = self.G.mapping(z, c, update_emas=update_emas)
            scale, mapped_scale = patch_util.compute_scale_inputs(self.G, ws, transform)
            ws = self.style_mix(z, c, ws)
            img = self.G.synthesis(ws, mapped_scale=mapped_scale, transform=transform, update_emas=update_emas)
        elif '360' in self.training_mode:
            ws = self.G.mapping(z, c, update_emas=update_emas)
            ws = self.style_mix(z, c, ws)
            input_layer = self.G.synthesis.input
            crop_start = random.randint(0, 360 // input_layer.fov * input_layer.frame_size[0] - 1)
            crop_fn = lambda grid : grid[:, :, crop_start:crop_start+input_layer.size[0], :]
            img_base = self.G.synthesis(ws, crop_fn=crop_fn, update_emas=update_emas)
            crop_shift = crop_start + input_layer.frame_size[0]
            # generate shifted frame for cross-frame discriminator
            crop_fn_shift = lambda grid : grid[:, :, crop_shift:crop_shift+input_layer.size[0], :]
            img_shifted = self.G.synthesis(ws, crop_fn=crop_fn_shift, update_emas=update_emas)
            img_splice = torch.cat([img_base, img_shifted], dim=3)
            img_size = img_base.shape[-1]
            splice_start = random.randint(0, img_size)
            img = img_splice[:, :, :, splice_start:splice_start+img_size]
        elif 'global' in self.training_mode:
            ws = self.G.mapping(z, c, update_emas=update_emas)
            ws = self.style_mix(z, c, ws)
            assert(transform is None)
            img = self.G.synthesis(ws, transform=transform, update_emas=update_emas)
        return img, ws

    def run_D(self, img, c, blur_sigma=0, update_emas=False):
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            with torch.autograd.profiler.record_function('blur'):
                f = torch.arange(-blur_size, blur_size + 1, device=img.device).div(blur_sigma).square().neg().exp2()
                img = upfirdn2d.filter2d(img, f / f.sum())
        if self.augment_pipe is not None:
            img = self.augment_pipe(img)
        logits = self.D(img, c, update_emas=update_emas)
        return logits

    def accumulate_gradients(self, phase, real_img, real_c, transform, gen_z,
                             gen_c, gain, cur_nimg, min_scale, max_scale):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        if self.pl_weight == 0:
            phase = {'Greg': 'none', 'Gboth': 'Gmain'}.get(phase, phase)
        if self.r1_gamma == 0:
            phase = {'Dreg': 'none', 'Dboth': 'Dmain'}.get(phase, phase)
        blur_sigma = max(1 - cur_nimg / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma if self.blur_fade_kimg > 0 else 0

        # Gmain: Maximize logits for generated images.
        if phase in ['Gmain', 'Gboth']:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, transform)
                # vutils.save_image(gen_img, 'out_fake_patch.png', range=(-1, 1),
                #                   normalize=True, nrow=4)
                gen_logits = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Gmain = torch.nn.functional.softplus(-gen_logits) # -log(sigmoid(gen_logits))
                training_stats.report('Loss/G/loss', loss_Gmain)
                training_stats.report('Scale/G/min_scale', min_scale)
                training_stats.report('Scale/G/max_scale', max_scale)
                if self.teacher is not None and self.added_kwargs.teacher_lambda > 0:
                    teacher_img = self.teacher(gen_z, gen_c)
                    if self.added_kwargs.teacher_mode == 'forward':
                        teacher_crop, teacher_mask = apply_affine_batch(teacher_img, transform)
                        # removes the border around the above mask 
                        # (mask should be all ones bc zooming in)
                        teacher_mask = torch.ones_like(teacher_mask)
                        l1_loss = self.loss_l1(gen_img, teacher_crop,
                                               teacher_mask[:, :1])
                        lpips_loss = self.loss_lpips(
                            losses.adaptive_downsample256(gen_img),
                            losses.adaptive_downsample256(teacher_crop),
                            losses.adaptive_downsample256(teacher_mask[:, :1],
                                                       mode='nearest')
                        )
                    elif self.added_kwargs.teacher_mode == 'inverse':
                        out_crop, out_mask = apply_affine_batch(gen_img, transform.inverse())
                        l1_loss = self.loss_l1(out_crop, teacher_img,
                                               out_mask[:, :1])
                        lpips_loss = self.loss_lpips(
                            losses.adaptive_downsample256(out_crop),
                            losses.adaptive_downsample256(teacher_img),
                            losses.adaptive_downsample256(out_mask[:, :1],
                                                       mode='nearest')
                        )
                    else:
                        assert(False)
                    teacher_loss = (l1_loss + lpips_loss)[:, None]
                    loss_Gmain = (loss_Gmain + self.added_kwargs.teacher_lambda
                                  * teacher_loss)
                    training_stats.report('Loss/G/loss_teacher_l1', l1_loss)
                    training_stats.report('Loss/G/loss_teacher_lpips', lpips_loss)
                    training_stats.report('Loss/G/loss_total', loss_Gmain)
            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.mean().mul(gain).backward()

        # Gpl: Apply path length regularization.
        if phase in ['Greg', 'Gboth']:
            with torch.autograd.profiler.record_function('Gpl_forward'):
                batch_size = gen_z.shape[0] // self.pl_batch_shrink
                gen_img, gen_ws = self.run_G(gen_z[:batch_size],
                                             gen_c[:batch_size],
                                             transform[:batch_size])
                pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
                with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients(self.pl_no_weight_grad):
                    pl_grads = torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws], create_graph=True, only_inputs=True)[0]
                pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
                pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
                self.pl_mean.copy_(pl_mean.detach())
                pl_penalty = (pl_lengths - pl_mean).square()
                training_stats.report('Loss/pl_penalty', pl_penalty)
                loss_Gpl = pl_penalty * self.pl_weight
                training_stats.report('Loss/G/reg', loss_Gpl)
            with torch.autograd.profiler.record_function('Gpl_backward'):
                loss_Gpl.mean().mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if phase in ['Dmain', 'Dboth']:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, transform, update_emas=True)
                gen_logits = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma, update_emas=True)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_logits) # -log(1 - sigmoid(gen_logits))
            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if phase in ['Dmain', 'Dreg', 'Dboth']:
            name = 'Dreal' if phase == 'Dmain' else 'Dr1' if phase == 'Dreg' else 'Dreal_Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp = real_img.detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                real_logits = self.run_D(real_img_tmp, real_c, blur_sigma=blur_sigma)
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                loss_Dreal = 0
                if phase in ['Dmain', 'Dboth']:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits) # -log(sigmoid(real_logits))
                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                if phase in ['Dreg', 'Dboth']:
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                    r1_penalty = r1_grads.square().sum([1,2,3])
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (loss_Dreal + loss_Dr1).mean().mul(gain).backward()

#----------------------------------------------------------------------------
