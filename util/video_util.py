import torch
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import skvideo.io
import numpy as np

def draw_text(img, text, size=None, position=None, fill=(255,255,255),
              fontpath='suppmat/arial.ttf'):
    if(size is None):
        size = img.size[0]/10.
    font = ImageFont.truetype(fontpath, size=int(size))
    draw = ImageDraw.Draw(img)
    if position is None:
        position = ((img.size[0]-int(size*2.2)), int(img.size[1]-int(size)),)

    draw.text(position, text, align='right', fill=fill, font=font)

def get_writer(file_name, frame_rate=24, pix_fmt='yuv420p', vcodec='libx264'):
    rate = str(frame_rate)
    inputdict = {
        '-r': rate
    }
    outputdict = {
        '-pix_fmt': pix_fmt,
        '-r': rate,
        '-vcodec': vcodec,
    }
    writer = skvideo.io.FFmpegWriter(file_name, inputdict, outputdict)
    return writer

def interpolate_ws(start_ws, target_ws, interpolation_frames):
    ws_list = []
    for i in np.linspace(0, 1, interpolation_frames):
        ws_list.append((start_ws * (1-i) + target_ws * i).to('cpu'))
    return ws_list
