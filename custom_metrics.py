import os
import argparse
from util import pidfile, util
import pickle
from metrics import metric_utils
from metrics import frechet_inception_distance
import json
import dnnlib
import numpy as np
import tempfile
import torch
from torch_utils.ops import conv2d_gradfix

def save_metric(output_folder, metric, value):
    np.savez(os.path.join(output_folder, metric + '.npz'), value=value)
    with open(os.path.join(output_folder, metric + '.txt'), 'w') as f:
        f.write('%f\n' % value)
    print("%s: %f" % (metric, value))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--input', type=str, required=True, help='G pkl')
    parser.add_argument('--metrics', type=str, required=True, help='which metrics to calculate')
    parser.add_argument('--add_transform', action='store_true',
                        help='add transformation input to the model '
                        '(useful for models downloaded from original sgan3 repo')
    parser.add_argument('--training_options', type=str)

    args = parser.parse_args()
    metrics_list = args.metrics.split(',')
    output_folder = args.input.replace('.pkl', '_metrics')
    os.makedirs(output_folder, exist_ok=True)
    pidfile.exit_if_job_done(output_folder, redo=True) # exits if the job is currently locked

    # check which evaluations have not been computed
    prev_expts = set(os.listdir(output_folder))
    if all([m + '.npz' in prev_expts for m in metrics_list]):
        print("All metrics computed!")
        exit()
    new_expts = [m for m in metrics_list if m + '.npz' not in prev_expts]
    print("Running new metrics")
    print(new_expts)
    assert(len(set(new_expts)) == len(new_expts)) # check no duplicates

    # required for multiple workers loading from same dataset.zip file
    torch.multiprocessing.set_start_method('spawn')

    device = torch.device('cuda', 0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    conv2d_gradfix.enabled = True

    with open(args.input, 'rb') as f:
        G_ema = pickle.load(f)['G_ema']

    if args.add_transform:
        # map G_ema state_dict to updated code that allows insertable transform input
        import copy
        from training import networks_stylegan3
        G_copy = networks_stylegan3.Generator(**G_ema.init_kwargs)
        G_base_sd = G_ema.state_dict()
        G_copy.load_state_dict(G_base_sd)
        G_ema = G_copy
        assert(args.training_options is not None)
        with open(args.training_options) as f:
            training_options = json.load(f)
    elif args.training_options is not None:
        print("Using specified training options")
        with open(args.training_options) as f:
            training_options = json.load(f)
    else:
        # use the training_options.json file saved 
        # in the snapshot directory
        with open(os.path.join(os.path.dirname(args.input), 'training_options.json')) as f:
            training_options = json.load(f)

    for metric in new_expts:
        if metric.startswith('fid-'):
            if metric.startswith('fid-full'):
                splits = metric.split('-') # fid-full1024
                target_resolution = int(util.remove_prefix(splits[1], 'full'))
                print("Computing metric %s at resolution %d" % (metric, target_resolution))
                dataset_kwargs = dnnlib.EasyDict(training_options['training_set_kwargs'])
                if dataset_kwargs.resolution < target_resolution:
                    # evaluate at a higher resolution using smaller HR
                    # dataset, resizing everything to the same size
                    dataset_kwargs.resolution = target_resolution
                    dataset_kwargs.max_size = training_options['patch_kwargs']['max_size']
                    dataset_kwargs.path = training_options['patch_kwargs']['path']
                    dataset_kwargs.crop_image = True
                print(dataset_kwargs)
                opts = metric_utils.MetricOptions(G=G_ema, dataset_kwargs=dataset_kwargs,
                                                  num_gpus=1, rank=0)
                mode = 'full'
                fid = frechet_inception_distance.compute_fid_full(
                    opts, target_resolution, None, 50000, mode=mode)
                save_metric(output_folder, metric, fid)
            elif metric.startswith('fid-up'): # fid-up1024
                target_resolution = int(util.remove_prefix(metric, 'fid-up'))
                print("Computing metric %s at resolution %d" % (metric, target_resolution))
                dataset_kwargs = dnnlib.EasyDict(training_options['training_set_kwargs'])
                if dataset_kwargs.resolution < target_resolution:
                    # evaluate at a higher resolution using smaller HR
                    # dataset, resizing everything to the same size
                    dataset_kwargs.resolution = target_resolution
                    dataset_kwargs.max_size = training_options['patch_kwargs']['max_size']
                    dataset_kwargs.path = training_options['patch_kwargs']['path']
                    dataset_kwargs.crop_image = True
                print(dataset_kwargs)
                opts = metric_utils.MetricOptions(G=G_ema, dataset_kwargs=dataset_kwargs,
                                                  num_gpus=1, rank=0)
                fid = frechet_inception_distance.compute_fid_full(
                    opts, target_resolution, None, 50000, mode='up')
                save_metric(output_folder, metric, fid)
            elif metric.startswith('fid-patch'): # fid-patch256-minXmaxY
                patch_options = training_options
                dataset_kwargs = dnnlib.EasyDict(patch_options['patch_kwargs'])
                patch_size = int(util.remove_prefix(metric.split('-')[1], 'patch'))
                assert(patch_size == dataset_kwargs.resolution)
                size_min = int(util.remove_prefix(metric.split('-')[2].split('max')[0], 'min'))
                size_max = int(metric.split('-')[2].split('max')[1])
                scale_min = patch_size / size_max if size_max > 0 else None
                scale_max = patch_size / size_min
                # adjust dataset kwargs using desired scale min and max
                dataset_kwargs.scale_min = scale_min
                dataset_kwargs.scale_max = scale_max
                dataset_kwargs.scale_anneal = -1
                print("Computing metric %s" % (metric))
                print("Patch size %d, size_min: %d size_max: %d scale_min: %s scale_max %f"
                      % (patch_size, size_min, size_max, str(scale_min), scale_max))
                print(dataset_kwargs)
                target_resolution = G_ema.init_kwargs.img_resolution
                opts = metric_utils.MetricOptions(G=G_ema, dataset_kwargs=dataset_kwargs,
                                                  num_gpus=1, rank=0)
                splits = metric.split('-')
                mode = 'patch'
                print("FID mode: %s" % mode)
                fid = frechet_inception_distance.compute_fid_patch(
                    opts, target_resolution, 50000, 50000, mode=mode)
                np.savez(os.path.join(output_folder, metric + '.npz'), value=fid)
                save_metric(output_folder, metric, fid)
            elif metric.startswith('fid-subpatch'): # fid-subpatch1024-minXmaxY
                patch_options = training_options
                dataset_kwargs = dnnlib.EasyDict(patch_options['patch_kwargs'])
                patch_size = int(util.remove_prefix(metric.split('-')[1], 'subpatch'))
                assert(patch_size == dataset_kwargs.resolution)
                size_min = int(util.remove_prefix(metric.split('-')[2].split('max')[0], 'min'))
                size_max = int(metric.split('-')[2].split('max')[1])
                scale_min = patch_size / size_max if size_max > 0 else None
                scale_max = patch_size / size_min
                # adjust dataset kwargs using desired scale min and max
                dataset_kwargs.scale_min = scale_min
                dataset_kwargs.scale_max = scale_max
                dataset_kwargs.scale_anneal = -1
                print("Computing metric %s" % (metric))
                print("Patch size %d, size_min: %d size_max: %d scale_min: %s scale_max %f"
                      % (patch_size, size_min, size_max, str(scale_min), scale_max))
                print(dataset_kwargs)
                target_resolution = G_ema.init_kwargs.img_resolution
                opts = metric_utils.MetricOptions(G=G_ema, dataset_kwargs=dataset_kwargs,
                                                  num_gpus=1, rank=0)
                splits = metric.split('-')
                mode = 'subpatch'
                print("FID mode: %s" % mode)
                fid = frechet_inception_distance.compute_fid_patch(
                    opts, target_resolution, 50000, 50000, mode=mode)
                save_metric(output_folder, metric, fid)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
    pidfile.mark_job_done(output_folder)
