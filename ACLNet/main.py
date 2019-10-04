from __future__ import division
from keras.layers import Input
from keras.models import Model
import os
import numpy as np
from config import *
from utilities import preprocess_images, postprocess_predictions
from models import acl_vgg
from imageio import imread, imwrite
from math import ceil


def get_test(video_test_path):
    images = [video_test_path + frames_path + f for f in os.listdir(video_test_path + frames_path) if
              f.endswith(('.jpg', '.jpeg', '.png'))]
    images.sort()
    start = 0
    while True:
        Xims = np.zeros((1, num_frames, shape_r, shape_c, 3))
        X = preprocess_images(images[start:min(start + num_frames, len(images))], shape_r, shape_c)
        Xims[0, 0:min(len(images)-start, num_frames), :] = np.copy(X)
        yield Xims  #
        start = min(start + num_frames, len(images))


if __name__ == '__main__':
    phase = 'test'
    if phase == 'train':
        x = Input(batch_shape=(None, None, shape_r, shape_c, 3))
        stateful = False
    else:
        x = Input(batch_shape=(1, None, shape_r, shape_c, 3))
        stateful = True

    if phase == "test":
        # videos_test_path = './video_data/'
        # videos = [videos_test_path + f for f in os.listdir(videos_test_path) if os.path.isdir(videos_test_path + f)]
        # videos.sort()
        # nb_videos_test = len(videos)

        m = Model(inputs=x, outputs=acl_vgg(x, stateful))
        print("Loading ACL weights")
        m.load_weights('/data2/yuanx/QoEData/sailency-models/ACLNet/models/ACL.h5')
        import pdb; pdb.set_trace()

        for i in range(nb_videos_test):

            output_folder = videos[i]+'/saliency/'
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            images_names = [f for f in os.listdir(videos[i] + frames_path) if
                          f.endswith(('.jpg', '.jpeg', '.png'))]
            images_names.sort()

            print("Predicting saliency maps for " + videos[i])
            prediction = m.predict_generator(get_test(video_test_path=videos[i]), max(ceil(len(images_names)/num_frames),2))
            predictions = prediction[0]


            for j in range(len(images_names)):
                original_image = imread(videos[i] + frames_path + images_names[j])
                x, y = divmod(j, num_frames)
                res = postprocess_predictions(predictions[x, y, :, :, 0], original_image.shape[0],
                                                  original_image.shape[1])

                imwrite(output_folder + '%s' % images_names[j], res.astype(int))
            m.reset_states()
    else:
        raise NotImplementedError
