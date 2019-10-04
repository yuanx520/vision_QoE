import torch
import torch.nn as nn
import numpy as np
import cv2
import os
from model.SalEMA import *
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import yaml
from easydict import EasyDict as edict
import subprocess
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='benchmark a detection model')
    parser.add_argument('--config', dest='config',
                      help='benchmark_config',
                      default='config.yaml', type=str)
    args = parser.parse_args()

    return args
def load_model(pt_model, new_model):

    temp = torch.load(pt_model)['state_dict']
    # Because of dataparallel there is contradiction in the name of the keys so we need to remove part of the string in the keys:.
    from collections import OrderedDict
    checkpoint = OrderedDict()
    for key in temp.keys():
        new_key = key.replace("module.","")
        checkpoint[new_key]=temp[key]

    new_model.load_state_dict(checkpoint, strict=True)

    return new_model

def main():
    ''' read frames in path_indata and generate frame-wise saliency maps in path_output '''
    # optional two command-line arguments
    args = parse_args()
    config_path = args.config
    config = edict(yaml.load(open(config_path)))
    # path_indata = '/data2/yuanx/QoEData/sailency-models/TASED-Net/example/'
    # path_output = '/data2/yuanx/QoEData/sailency-models/TASED-Net/output/'
    # model_path = '/data2/yuanx/QoEData/sailency-models/TASED-Net/models/'
    path_output = config.output_path
    model_path = config.model_path
    len_temporal = 10

    if not os.path.isdir(path_output):
        os.makedirs(path_output)

    model = SalEMA(alpha = config.model.alpha, \
                    residual=config.model.residual, \
                    dropout = config.model.dropout, \
                    ema_loc=config.model.ema_loc)
    model = load_model(model_path, model)
    torch.backends.cudnn.benchmark = False
    model = model.cuda()
    model.eval()
    list_video = [os.path.join(config.data.base_path, v) for v in config.data.video_list]
    # list_indata.sort()
    for vname in list_video:
        print ('processing ' + vname)
        # list_frames = [f for f in os.listdir(os.path.join(path_indata, vname)) if os.path.isfile(os.path.join(path_indata, vname, f))]
        # list_frames.sort()
        vname = vname + config.data.quailty_index + '.' + config.data.video_suffix
        capture = cv2.VideoCapture(vname)
        read_flag, img = capture.read()
        i = 0
        path_outdata = os.path.join(path_output, vname.split('/')[-1].split('.')[0])
        encoded_vid_path = os.path.join(path_outdata, "sailency.mp4")

        if not os.path.isdir(path_outdata):
            os.makedirs(path_outdata)

        # for i in range(len(list_frames)):
        state = None
        while (read_flag):
            # process this clip
            state = process(model, path_outdata, img, i, state)
            if (i + 1) % len_temporal == 0:
                state = repackage_hidden(state)
            read_flag, img = capture.read()
            i += 1

        capture.release()
        encoding_result = subprocess.run(["ffmpeg", "-y",
                                          "-start_number", '0',
                                          "-i", f"{path_outdata}/%06d.jpg",
                                          "-loglevel", "error",
                                          "-vcodec", "libx264",
                                          "-pix_fmt", "yuv420p",
                                          "-crf", "23",
                                          encoded_vid_path],
                                         stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE,
                                         universal_newlines=True)

        if encoding_result.returncode != 0:
            # Encoding failed
            print("ENCODING FAILED")
            print(encoding_result.stdout)
            print(encoding_result.stderr)
            # exit()
            continue
        #
        # else:
        #     print (' more frames are needed')


def process(model, path_outdata, image, idx, prev_state):
    X, image_resized = image_preprocess(image)
    X = torch.autograd.Variable(X.unsqueeze(0)).cuda()

    with torch.no_grad():
        # saliency_map = model(X).cpu().data[0,0]
        state, saliency_map = model.forward(input_ = X, prev_state = prev_state)
    saliency_map = saliency_map.cpu().data[0,0]
    saliency_map = (saliency_map.numpy() * 255).astype(np.int)/255.

    # resize back to original size
    # saliency_map = cv2.resize(saliency_map, (orig_width, orig_height), interpolation=cv2.INTER_CUBIC)
    # blur
    # saliency_map = cv2.GaussianBlur(saliency_map, (5, 5), 0)
    saliency_map = gaussian_filter(saliency_map, sigma=7)
    # clip again
    # saliency_map = np.clip(saliency_map, 0, 255)
    plt.axis("off")
    plt.imshow(image_resized)
    plt.imshow((saliency_map/np.max(saliency_map)*255.).astype(np.uint8), cmap='jet', alpha=0.5)
    # plt.show()
    # plt.pause(0.01)
    # plt.close()
    plt.savefig(os.path.join(path_outdata, '%06d.jpg'%(idx)))
    plt.cla()
    return state

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

if __name__ == '__main__':
    main()
