import os
import numpy as np
import flickr_api as f
import argparse
from tqdm import tqdm
from PIL import Image

# file list
# min size
# max size


# python data/trace_data.py [[PATH]]
parser = argparse.ArgumentParser()
parser.add_argument('--filelist', type=str, required=True)
parser.add_argument('--minsize', type=int)
parser.add_argument('--maxsize', type=int)
parser.add_argument('--outputpath', type=str)
parser.add_argument('--download_HR', action='store_true')
opt = parser.parse_args()

f.set_keys(api_key = 'XXXX', api_secret = 'XXXX')

if opt.outputpath is None:
    opt.outputpath = '../%s' % os.path.basename(opt.filelist).split('.')[0]
if(not os.path.exists(opt.outputpath)):
    os.makedirs(opt.outputpath, exist_ok=True)
    print('Making output [%s]'%opt.outputpath)
else:
    print('Output [%s] already exists' % opt.outputpath)

with open(opt.filelist, 'r') as r:
    files = [l.strip() for l in r.readlines()]

exception_ids = []
for filename in files:
    license, idnum = filename.split('_')
    idnum, filetype = idnum.split('.')
    if os.path.exists(os.path.join(opt.outputpath, filename)):
        print('[%s] already exists'%filename)
        continue
    try:
        photo = f.Photo(id=idnum)
        sizes = photo.getSizes()
        if opt.download_HR:
            # download highest resolution between minsize and maxsize
            downloaded = False
            for s in ['Original', 'X-Large 5K', 'X-Large 4K', 'X-Large 3K', 'Large 2048',
                      'Large 1600', 'Large', 'Medium 800', 'Medium 640', 'Medium']:
                if s in sizes:
                    height = sizes[s]['height']
                    width = sizes[s]['width']
                    if opt.maxsize is not None and min(height, width) > opt.maxsize:
                        # above max size, go to next size
                        continue
                    if opt.minsize is not None and min(height, width) < opt.minsize:
                        # below min size skip
                        continue
                    photo_size = sizes[s]
                    print('%s: trying %s [%sx%s]'%(idnum, photo_size['label'], photo_size['height'], photo_size['width']))
                    os.system('wget -q %s -O %s/%s'%(photo_size['source'], opt.outputpath, filename))
                    downloaded = True
                    break
            if not downloaded:
                # did not find suitable size
                exception_ids.append(idnum)
        else:
            # download the smallest img larger than minsize for LR dataset
            downloaded = False
            for k, v in sizes.items(): # smallest to largest
                height = v['height']
                width = v['width']
                if min(height, width) >= opt.minsize:
                    photo_size = sizes[k]
                    print('%s: trying %s [%sx%s]'%(idnum, photo_size['label'], photo_size['height'], photo_size['width']))
                    os.system('wget -q %s -O %s/%s' %(photo_size['source'], opt.outputpath, filename))
                    downloaded=True
                    break
            if not downloaded:
                # did not find suitable size
                exception_ids.append(idnum)
    except Exception as ex:
        # import pdb; pdb.set_trace()
        print('[%s] not processed'%filename)
        exception_ids.append(id_num)

if len(exception_ids) > 0:
    with open(opt.filelist.replace('.txt', '_exception.txt'), 'w') as r:
        [r.write('%s\n' % idnum) for idnum in exception_ids]

