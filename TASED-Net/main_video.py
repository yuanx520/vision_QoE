import sys
import os
import numpy as np
import cv2
import torch
from model import TASED_v2
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('TkAgg')
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

    len_temporal = 32
    file_weight = os.path.join(model_path, 'TASED_v2.pt')

    model = TASED_v2()
    # load the weight file and copy the parameters
    if os.path.isfile(file_weight):
        print ('loading weight file')
        weight_dict = torch.load(file_weight)
        model_dict = model.state_dict()
        for name, param in weight_dict.items():
            if 'module' in name:
                name = '.'.join(name.split('.')[1:])
            if name in model_dict:
                if param.size() == model_dict[name].size():
                    model_dict[name].copy_(param)
                else:
                    print (' size? ' + name, param.size(), model_dict[name].size())
            else:
                print (' name? ' + name)

        print (' loaded')
    else:
        print ('weight file?')

    model = model.cuda()
    torch.backends.cudnn.benchmark = False
    model.eval()

    # iterate over the path_indata directory
    # list_indata = [d for d in os.listdir(path_indata) if os.path.isdir(os.path.join(path_indata, d))]
    list_video = [os.path.join(config.data.base_path, v) for v in config.data.video_list]
    # list_indata.sort()
    for i_vname, vname in enumerate(list_video):
        # import pdb; pdb.set_trace()
        for dash_idx in range(config.data.quailty_num[i_vname]):
            smaps = []
            v_name = vname + str((dash_idx)).zfill(2)
            print ('processing ' + v_name)
            # import pdb; pdb.set_trace()
            # list_frames = [f for f in os.listdir(os.path.join(path_indata, vname)) if os.path.isfile(os.path.join(path_indata, vname, f))]
            # list_frames.sort()
            v_name = v_name + '.' + config.data.video_suffix
            # vname = vname + config.data.quailty_index + '.' + config.data.video_suffix
            capture = cv2.VideoCapture(v_name)
            read_flag, img = capture.read()
            i = 0
            # process in a sliding window fashion
            # suppose list_frames always > 2*len_temporal
            # if len(list_frames) >= 2*len_temporal-1:
            path_outdata = os.path.join(path_output, v_name.split('/')[-1].split('.')[0])
            encoded_vid_path = os.path.join(path_outdata, "sailency.mp4")
            saliency_map_path = os.path.join(path_outdata, "sailency")


            if not os.path.isdir(path_outdata):
                os.makedirs(path_outdata)

            f, axarr = plt.subplots(1,2, figsize=(10,3))
            snippet = []
            # for i in range(len(list_frames)):
            while (read_flag):
                # print(i)
                # img = cv2.imread(os.path.join(path_indata, vname, list_frames[i]))
                img = cv2.resize(img, (384, 224))
                img = img[...,::-1]
                snippet.append(img)

                if i >= len_temporal-1:
                    clip = transform(snippet)

                    smaps.append(process(model, clip, path_outdata, i, snippet[-1], f, axarr))

                    # process first (len_temporal-1) frames
                    if i < 2*len_temporal-2:
                        smaps.append(process(model, torch.flip(clip, [1]), path_outdata, i-len_temporal+1, snippet[-1], f, axarr))

                    del snippet[0]
                read_flag, img = capture.read()
                i += 1
            capture.release()
            smaps = np.asarray(smaps)
            np.save(saliency_map_path, smaps)
            # encoding_result = subprocess.run(["ffmpeg", "-y",
            #                                   "-start_number", '0',
            #                                   "-i", f"{path_outdata}/%06d.jpg",
            #                                   "-loglevel", "error",
            #                                   "-vcodec", "libx264",
            #                                   "-pix_fmt", "yuv420p",
            #                                   "-crf", "23",
            #                                   encoded_vid_path],
            #                                  stdout=subprocess.PIPE,
            #                                  stderr=subprocess.PIPE,
            #                                  universal_newlines=True)
            #
            # if encoding_result.returncode != 0:
            #     # Encoding failed
            #     print("ENCODING FAILED")
            #     print(encoding_result.stdout)
            #     print(encoding_result.stderr)
            #     # exit()
            #     continue
            #
            # else:
            #     print (' more frames are needed')


def transform(snippet):
    ''' stack & noralization '''
    snippet = np.concatenate(snippet, axis=-1)
    snippet = torch.from_numpy(snippet).permute(2, 0, 1).contiguous().float()
    snippet = snippet.mul_(2.).sub_(255).div(255)

    return snippet.view(1,-1,3,snippet.size(1),snippet.size(2)).permute(0,2,1,3,4)


def process(model, clip, path_outdata, idx, ori, f, axarr):
    ''' process one clip and save the predicted saliency map '''
    with torch.no_grad():
        smap = model(clip.cuda()).cpu().data[0]
    # fig=plt.figure()
    # smap
    smap = (smap.numpy()*255.).astype(np.int)/255.
    smap = gaussian_filter(smap, sigma=7)
    return smap

    # axarr[0].axis("off")
    # axarr[0].imshow(ori)
    # axarr[0].imshow(smap, cmap='jet', alpha=0.5, vmin=0, vmax=1)
    # weights = np.ones_like(smap.reshape(-1))/float(len(smap.reshape(-1)))
    # axarr[1].hist(smap.reshape(-1), bins=100, histtype='step', weights = weights)
    # axarr[1].set_xlim([0., 1.])
    # axarr[1].set_ylim([0., 1.])
    # f.savefig(os.path.join(path_outdata, '%06d.jpg'%(idx)))
    # axarr[0].clear()
    # axarr[1].clear()

    # f.clf()
    # f.cla()
    # plt.close()
    # cv2.imwrite(os.path.join(path_outdata, '%06d.jpg'%(idx+1)), (smap/np.max(smap)*255.).astype(np.uint8))


if __name__ == '__main__':
    main()
