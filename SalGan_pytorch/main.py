import torch
import torch.nn as nn
import numpy as np
import cv2
import os
from model import *
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

    if not os.path.isdir(path_output):
        os.makedirs(path_output)

    model = SalGan()
    model.load_state_dict(torch.load(model_path))
    model.cuda()
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
        # process in a sliding window fashion
        # suppose list_frames always > 2*len_temporal
        # if len(list_frames) >= 2*len_temporal-1:
        path_outdata = os.path.join(path_output, vname.split('/')[-1].split('.')[0])
        encoded_vid_path = os.path.join(path_outdata, "sailency.mp4")

        if not os.path.isdir(path_outdata):
            os.makedirs(path_outdata)

        # for i in range(len(list_frames)):
        while (read_flag):
            process(model, path_outdata, img, i)
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


def process(model, path_outdata, image, idx):
    X, image_resized = image_preprocess(image)
    X = torch.autograd.Variable(X.unsqueeze(0)).cuda()

    with torch.no_grad():
        saliency_map = model(X).cpu().data[0,0]

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

if __name__ == '__main__':
    main()



    # root_dir = 'images/'
    # result_dir = 'results/'
    # if not os.path.exists(result_dir):
    #     os.makedirs(result_dir)
    # images = os.listdir(root_dir)
    #
    #
    # for image_path in images:
    #     image = cv2.imread(os.path.join(root_dir, image_path), cv2.IMREAD_COLOR)
    #     size = image.shape[:2]
    #
    #     X = image_preprocess(image)
    #     X = torch.autograd.Variable(X.unsqueeze(0))
    #
    #     result = gan(X)
    #
    #     saliency_map = post_process(result.data.numpy()[0, 0], size[0], size[1])
    #     cv2.imwrite(os.path.join(result_dir, image_path), saliency_map)
