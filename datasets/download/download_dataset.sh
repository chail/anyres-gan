### birds HR: min = 512, max = 2048
python download_dataset.py --filelist flickr_birds_HR.txt --minsize 512 --maxsize 2048 --download_HR

### birds LR: smallest img above 256
# python download_dataset.py --filelist flickr_birds_LR.txt --minsize 256

### church HR: min = 1024, no max
### NOTE: these images were then center cropped for training
# python download_dataset.py --filelist flickr_church_exteriors_HR.txt --minsize 1024 --download_HR
