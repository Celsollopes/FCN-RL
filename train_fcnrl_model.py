"""[options]

Options:
   -t --train-folder FOLDER      specify output file       [default: ]
   -v --validation-folder FOLDER      specify output file [default: ]
   -m --model NAME      specify output file [default: fcn_rl]
   --gpu F specify gpu allocation                         [default: 1.0]
   --bs F batch size                         [default: 12]
   --valid-steps F specify gpu allocation                         [default: 10]
   --train-steps F specify gpu allocation                         [default: 10]
   --train-samples NUMBER
   --valid-samples NUMBER
   --lr FLOAT [default: 1e-5]
   --no-aug
   
"""
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, Callback
from utils.global_config import __DEF_HEIGHT, __DEF_WIDTH, __CHANNEL
from utils.metrics_functions import dice_coef, dice_coef_loss
from utils.models import get_encoder_decoder, build_refinement
from utils.data_generator import generator_batch
from keras.optimizers import Adam
from keras import backend as K
from utils.docopt import docopt
import tensorflow as tf
import os.path as P
import numpy as np
import keras
import glob
import cv2

arguments = docopt(__doc__, version='FIXME')
print(arguments)
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
config.gpu_options.per_process_gpu_memory_fraction = float(arguments['--gpu'])
sess = tf.Session(config=config)
try:
    K.set_session(sess)
except:
    pass
    
train_steps = int(arguments['--train-steps'])
valid_steps = int(arguments['--valid-steps'])
train_samples = arguments['--train-samples']
valid_samples = arguments['--valid-samples']
bs = int(arguments['--bs'])

train_fns = sorted(glob.glob(P.join(arguments['--train-folder'], "*.png")))
train_fns = [k for k in train_fns if '_gt' not in k]

valid_fns = sorted(glob.glob(P.join(arguments['--validation-folder'], "*.png")))
valid_fns = [k for k in valid_fns if '_gt' not in k]
	
def main(train_steps=10, valid_steps=10, train_samples, valid_samples, bs=12, train_fns, valid_fns):
    """
    Load the parameters for training the refinement layer model.
    """

    if train_samples:
        train_fns = train_fns[:int(train_samples)]

    if valid_samples:
        valid_fns = valid_fns[:int(valid_samples)]
        
    train_samples = len(train_fns) - (len(train_fns) % bs)
    valid_samples = len(valid_fns) - (len(valid_fns) % bs)
    train_fns = train_fns[:train_samples]
    valid_fns = valid_fns[:valid_samples]

    np.random.seed(0)
    np.random.shuffle(train_fns)
    np.random.shuffle(valid_fns)
    np.random.seed()

    callbacks = []
    monitor = 'val_loss'
    monitor_mode = 'min'
	
    # pretrain_path: local path of the encoder-decoder model pre-train in your machine.
    pretrain_path = "./model_path_hdf/pretrain_autoencoder_model.hdf5"
    encoder_decoder = keras.models.load_model(pretrain_path, custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})
    encoder_decoder.load_weights(pretrain_path)

    # freezes the weights of the encoder-decoder layers.
    for layer in encoder_decoder.layers:
            layer.trainable = False

    model = build_refinement(encoder_decoder) # load the encoder-decoder model as a parameter for the refinement layers
    model.compile(optimizer=Adam(lr=float(arguments['--lr'])), loss=dice_coef_loss, metrics=[dice_coef])
    print(model.summary())

    checkpoint_model = ModelCheckpoint('./FCN_RL_hdf/%s.hdf5' % arguments['--model'],
                                     monitor=monitor, save_best_only=True,
                                     verbose=1, mode=monitor_mode,
                                     )

    callbacks.append(EarlyStopping(
        monitor=monitor, patience=30, verbose=1, mode=monitor_mode,
    ))

    exitlog = CSVLogger('training-fcnrl.txt')

    train_gen = generator_batch(train_fns, bs=bs, stroke=False)
    valid_gen = generator_batch(valid_fns, bs=bs, validation=True)

    class SaveImageCallback(Callback):
        def __init__(self, stroke=False):
            super(SaveImageCallback, self ).__init__()

        def on_epoch_end(self, epoch, logs={}):
            """
            data: input images
            gt:   ground-truth images
            mask: predicted image, output image result
            """
            data, gt = next(train_gen)
            mask = self.model.predict_on_batch(data)
            for i in range(mask.shape[0]):
                if (epoch%10==0):
                    cv2.imwrite(r'./output_images/%d-%d-out.png' % (epoch, i), mask[i,:,:,0]*255)# save mask image - optional
                    cv2.imwrite(r'./output_images/%d-%d-gt.png' % (epoch, i), gt[i,:,:,0]*255)# sava ground truth image - optional
                    #cv2.imwrite(r'./output_images\%d-%d-2.png' % (epoch, i), data[i,:,:,0]*255)# optional
            
    save_net = SaveImageCallback()

    # Final Callbacks
    Callbacks = [save_net, checkpoint_model, exitlog]

    model.fit_generator(
        generator=train_gen, steps_per_epoch=train_steps,
        epochs=3,
        verbose=1,
        validation_data=valid_gen, validation_steps=valid_steps,
        callbacks=Callbacks
    )
if __name__ == "__main__":
    import os, sys
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    main(train_steps, valid_steps, train_samples, valid_samples, bs, train_fns, valid_fns)
