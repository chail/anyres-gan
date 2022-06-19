# example run: stage 1 global pretraining
python train.py --cfg=stylegan3-t --gpus=4 --batch=32 --gamma=2 \
	--mirror=1 --aug=noaug --kimg 25000 --batch-gpu 4 \
	--outdir training-runs/bird-global-training \
	--data ./datasets/flickr_birds_LR.zip \
	--training_mode=global --g_size 256  --data_resolution 256 --random_crop=True

# example run: stage 2 patch training
# the teacher can be any sgan3 pretrained model, e.g. from the stylegan3 official repository also works
python train.py --cfg=stylegan3-t --gpus=4 --batch=32 --gamma=2 \
	--mirror=1 --aug=noaug --kimg 25000 --batch-gpu 4 \
	--outdir training-runs/bird-patch-training \
	--data ./datasets/flickr_birds_LR.zip \
	--training_mode=patch --g_size 256 --data_resolution 256 --random_crop=True \
	--teacher training-runs/bird-global-training/00000-stylegan3-t-flickr_birds_LR-gpus4-batch32-gamma2/network-snapshot-XXXXXX.pkl \
	--data_hr ./datasets/flickr_birds_HR.zip \
	--metrics none --teacher_lambda 5.0 --teacher_mode inverse --scale_max 0.5 --scale_anneal -1 \
	--scale_mapping_min 1 --scale_mapping_max 8 --patch_crop=True 

# example run: 360 training
python train.py \
        --cfg=stylegan3-t --gpus=4 --batch=32 --gamma=2 \
        --mirror=1 --aug=noaug --kimg 25000 --batch-gpu 8 \
        --outdir=training-runs/mountain-360 \
        --data ./datasets/mountain256_jpeg95/train \
        --training_mode=global-360 --data_resolution 256 --fov 60 --random_crop=True 
