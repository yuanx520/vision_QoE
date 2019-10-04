from __future__ import division
from keras.layers import Reshape, TimeDistributed, Flatten, RepeatVector, Permute, Multiply, Add, UpSampling2D
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from dcn_vgg import dcn_vgg


def schedule_vgg(epoch):
    lr = [1e-4, 1e-4, 1e-4, 1e-5, 1e-5,
          1e-5, 1e-6, 1e-6, 1e-7, 1e-7]
    return lr[epoch]

def acl_vgg(data, stateful):
    dcn = dcn_vgg()
    outs = TimeDistributed(dcn)(data)
    attention = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), padding='same'))(outs)
    attention = TimeDistributed(Conv2D(64, (1, 1), padding='same', activation='relu'))(attention)
    attention = TimeDistributed(Conv2D(128, (3, 3), padding='same', activation='relu'))(attention)
    attention = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), padding='same'))(attention)
    attention = TimeDistributed(Conv2D(64, (1, 1), padding='same', activation='relu'))(attention)
    attention = TimeDistributed(Conv2D(128, (3, 3), padding='same', activation='relu'))(attention)
    attention = TimeDistributed(Conv2D(1, (1, 1), padding='same', activation='sigmoid'))(attention)
    attention = TimeDistributed(UpSampling2D(4))(attention)

    # attention = TimeDistributed(Conv2D(256, (3, 3), padding='same', activation='relu'))(outs)
    # attention = TimeDistributed(Conv2D(128, (3, 3), padding='same', activation='relu'))(attention)
    # attention = TimeDistributed(Conv2D(1, (1, 1), padding='same', activation='sigmoid'))(attention)

    f_attention = TimeDistributed(Flatten())(attention)
    f_attention = TimeDistributed(RepeatVector(512))(f_attention)
    f_attention = TimeDistributed(Permute((2, 1)))(f_attention)
    f_attention = TimeDistributed(Reshape((32, 40, 512)))(f_attention)#30
    m_outs = Multiply()([outs, f_attention])
    outs = Add()([outs, m_outs])

    outs = (ConvLSTM2D(filters=256, kernel_size=(3, 3),
                       padding='same', return_sequences=True, stateful=stateful, dropout=0.4))(outs)

    outs = TimeDistributed(Conv2D(1, (1, 1), padding='same', activation='sigmoid'))(outs)
    outs = TimeDistributed(UpSampling2D(4))(outs)
    attention = TimeDistributed(UpSampling2D(2))(attention)
    return [outs, outs, outs, attention, attention, attention]#

