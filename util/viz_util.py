from PIL import ImageDraw, Image
from util import renormalize, patch_util
import numpy as np
import torch

def init_bbox(params):
    patch_size = params['patch_size']
    display_size = params['display_size']
    zoom_rate = params['zoom_rate']
    if patch_size <= display_size:
        # add a small zoom factor
        full_size = patch_size + 2*zoom_rate
        bbox_info = {
            'display': np.array([[zoom_rate, full_size-zoom_rate],
                                 [zoom_rate, full_size-zoom_rate]]), # [(y_min, y_max), (x_min, x_max)]
            'generated': np.array([[zoom_rate, full_size-zoom_rate],
                                 [zoom_rate, full_size-zoom_rate]]),
            'full_size': patch_size + 2*zoom_rate
        }
    else:
        bbox_info = {
            'display': np.array([[patch_size//2-display_size//2, patch_size//2+ display_size//2],
                                 [patch_size//2-display_size//2, patch_size//2+ display_size//2]]),
            'generated': np.array([[0, patch_size], [0, patch_size]]),
            'full_size': patch_size
        }
    return bbox_info

def draw_bbox(bbox_info, base_img, size=256):
    pil_img = renormalize.as_image(base_img).resize((size, size), Image.ANTIALIAS)
    draw = ImageDraw.Draw(pil_img)
    limits = bbox_info['display'] / bbox_info['full_size'] * 2 - 1
    scaled_limits = ((np.array(limits) + 1) / 2 * size).astype(np.int64)
    draw.rectangle((scaled_limits[1][0], scaled_limits[0][0], scaled_limits[1][1], scaled_limits[0][1]))
    return pil_img

def get_bbox_center(bbox):
    return np.mean(bbox, axis=1).astype(np.int64)

def shift_bbox_inbounds(bbox, full_size):
    # ymin, xmin, ymax, xmax
    if bbox[0,0] < 0: # ymin
        shift = bbox[0,0]
        bbox[0] -= shift
    if bbox[1,0] < 0: # xmin
        shift = bbox[1,0]
        bbox[1] -= shift
    if bbox[0,1] > full_size: # ymax
        shift = bbox[0,1]-full_size
        bbox[0] -= shift
    if bbox[1,1] > full_size: # xmax
        shift = bbox[1,1]-full_size
        bbox[1] -= shift
    assert(np.min(bbox) >= 0)
    assert(np.max(bbox) <= full_size)
    return bbox

def update_bbox(key, bbox_info, generated_img, display_img, G_base, ws, params):
    shift_rate = params['shift_rate']
    zoom_rate = params['zoom_rate']
    display_size = params['display_size']
    patch_size = params['patch_size']
    if key == 'ArrowDown':
        new_display = np.copy(bbox_info['display'])
        new_display[0] = new_display[0] + shift_rate
        bbox_info, update_transformation, update_display = check_bbox_bounds(bbox_info, new_display, patch_size)
    elif key == 'ArrowUp':
        new_display = np.copy(bbox_info['display'])
        new_display[0] = new_display[0] - shift_rate
        bbox_info, update_transformation, update_display = check_bbox_bounds(bbox_info, new_display, patch_size)
    elif key == 'ArrowLeft':
        new_display = np.copy(bbox_info['display'])
        new_display[1] = new_display[1] - shift_rate
        bbox_info, update_transformation, update_display = check_bbox_bounds(bbox_info, new_display, patch_size)
    elif key == 'ArrowRight':
        new_display = np.copy(bbox_info['display'])
        new_display[1] = new_display[1] + shift_rate
        bbox_info, update_transformation, update_display = check_bbox_bounds(bbox_info, new_display, patch_size)
    elif key == 'i':
        # need to update scale and regenerate transform (center it)
        bbox_info['full_size'] += zoom_rate
        bbox_center = get_bbox_center(bbox_info['display']) + zoom_rate // 2
        new_display = np.stack([bbox_center - display_size//2, bbox_center+display_size//2], axis=1)
        new_generated = np.stack([bbox_center - patch_size//2, bbox_center+patch_size//2], axis=1)
        bbox_info['display'] = new_display
        bbox_info['generated'] = new_generated
        update_transformation = True
        update_display = True
    elif key == 'o':
        if bbox_info['full_size'] - zoom_rate >= patch_size:
            bbox_info['full_size'] -= zoom_rate
            assert(bbox_info['full_size'] >= patch_size)
            bbox_center = get_bbox_center(bbox_info['display']) - zoom_rate // 2
            new_display = np.stack([bbox_center - display_size//2, bbox_center+display_size//2], axis=1)
            new_generated = np.stack([bbox_center - patch_size//2, bbox_center+patch_size//2], axis=1)
            # check the bounds of display and generated
            bbox_info['display'] = shift_bbox_inbounds(new_display,  bbox_info['full_size'])
            bbox_info['generated'] = shift_bbox_inbounds(new_generated,  bbox_info['full_size'])
            update_transformation = True
            update_display = True
        else:
            # max zoom out
            update_transformation = False
            update_display = False
    elif key == 'c':
        # make this a function
        bbox_info = init_bbox(params)
        update_transformation = True
        update_display = True
    else:
        pass
    generated_img, display_img = update_img(
        bbox_info, update_transformation, update_display, generated_img,
        display_img, params, G_base, ws)
    return bbox_info, generated_img, display_img

def update_img(bbox_info, update_transformation, update_display, 
               generated_img, display_img, params, G_base, ws):
    display_size = params['display_size']
    if update_transformation:
        transform = bbox_to_transformation(bbox_info)[None].to(ws.device)
        generated_img = patch_util.scale_condition_wrapper(
            G_base, ws, transform=transform, noise_mode='const', force_fp32=True)[0]
    if update_display:
        offset_x = bbox_info['display'][1, 0] - bbox_info['generated'][1, 0]
        assert (offset_x >= 0)
        offset_y = bbox_info['display'][0, 0] - bbox_info['generated'][0, 0]
        assert (offset_y >= 0)
        display_img = generated_img[:, offset_y:offset_y+display_size, offset_x:offset_x+display_size]
    return generated_img, display_img

def check_bbox_bounds(bbox_info, new_display, patch_size):
    full_size = bbox_info['full_size']
    if (new_display[:, 0] < 0).any() or (new_display[:, 1] > full_size).any():
        # shift it to the edge
        new_display = shift_bbox_inbounds(new_display, full_size)
    if ((new_display[:, 0] >= bbox_info['generated'][:, 0]).all() and
          (new_display[:, 1] <= bbox_info['generated'][:, 1]).all()):
        # within generated bbox, don't need to regenerate
        bbox_info['display'] = new_display
        update_transformation = False
        update_display = True
    else:
        # need to regenerate, recenter the generated bbox
        bbox_center = get_bbox_center(new_display)
        new_generated = np.stack([bbox_center - patch_size//2, bbox_center+patch_size//2], axis=1)
        new_generated = shift_bbox_inbounds(new_generated, full_size)
        bbox_info['generated'] = new_generated
        bbox_info['display'] = new_display
        update_transformation = True
        update_display = True
    return bbox_info, update_transformation, update_display

def bbox_to_transformation(bbox_info):
    # normalize it by full_size and call contruct_transformation_matrix 
    limits = bbox_info['generated'] / bbox_info['full_size'] * 2 - 1
    transform = patch_util.construct_transformation_matrix(limits)
    return transform

### panorama utils
def make_grid(G_pano):
    input_layer = G_pano.synthesis.input

    # make secondary grid
    theta = torch.eye(2, 3, device='cuda')
    theta[0, 0] = 0.5 * input_layer.size[0] / input_layer.sampling_rate
    theta[1, 1] = 0.5 * input_layer.size[1] / input_layer.sampling_rate
    grid_width = (input_layer.size[0] - 2*input_layer.margin_size) * 360 // input_layer.fov + 2*input_layer.margin_size
    grids = torch.nn.functional.affine_grid(theta.unsqueeze(0),
                                            [1, 1, input_layer.size[1], grid_width],
                                            align_corners=False)
    # ensure that the x coordinate completes a full circle without padding
    base_width = grid_width - 2*input_layer.margin_size
    new_x = torch.arange(-input_layer.margin_size, base_width*2+input_layer.margin_size, device=grids.device) / base_width  * 2 - 1
    new_y = grids[0, :, 0, 1]
    new_grids = torch.cat([new_x.view(1, 1, -1, 1).repeat(1, input_layer.size[1], 1, 1),
                                 new_y.view(1, -1, 1, 1).repeat(1, 1, grid_width+base_width*1, 1)], dim=3)
    return new_grids

def generate_pano_transform(G_pano, z, grid):
    input_layer = G_pano.synthesis.input
    num_frames = int(360 / input_layer.fov)

    images = []
    with torch.no_grad():
        for tx in range(num_frames):
            transform = torch.eye(3)[None].to(z.device)
            transform[0, 0, 2] = tx
            crop_fn = lambda x: grid  # fix the coord grid, and shift using transform matr
            out = G_pano(z, None, transform = transform, crop_fn = crop_fn, truncation_psi=0.5)
            images.append(out[0].cpu())
    pano = torch.cat(images, dim=2)
    return pano
