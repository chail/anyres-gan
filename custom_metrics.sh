# flickr bird model example
pkl=pretrained/bird_pretrained_final.pkl
# Options can be the training_options.json file saved with each experiment run.
# The important thing used here is training_set_kwargs and patch_kwargs, 
# make sure those are adjusted appropriately if training and evaluation settings are different. 
# scale_min, scale_max, and scale_anneal in patch_kwargs are overridden based on the metrics specified here.
# The following options files are provided as examples, but the dataset paths need to be adjusted accordingly.
options=pretrained/patch-bird-final-options.json
echo RUNNING $pkl
python custom_metrics.py --input $pkl --metrics fid-full256,fid-full512,fid-patch256-min256max0 --training_options $options

# lsun church model example
pkl=pretrained/church_pretrained_final.pkl
options=pretrained/patch-church-final-options.json
echo RUNNING $pkl
python custom_metrics.py --input $pkl --metrics fid-full256,fid-full1024,fid-patch256-min256max0 --training_options $options

# ffhq example
# note: it is trained on 6k varied size images, but evaluated on full 70K ffhq ground-truth
pkl=pretrained/ffhq6k_pretrained_final.pkl
options=pretrained/patch-ffhq6k-final-options.json
echo RUNNING $pkl
python custom_metrics.py --input $pkl --metrics fid-full256,fid-patch256-min256max1024,fid-full512,fid-full1024 --training_options $options

# # mountain example
# # note: subpatch fid avoids downsampling the larger generated patch
# pkl=pretrained/mountain_pretrained_final.pkl
# options=pretrained/patch-mountain-final-options.json
# echo RUNNING $pkl
# python custom_metrics.py --input $pkl --metrics fid-full1024,fid-subpatch1024-min1024max0 --training_options $options
