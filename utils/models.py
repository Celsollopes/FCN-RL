"""
Deep Learning models.
Stores model architectural functions.

"""
from keras.layers import Input, Conv2D, UpSampling2D, BatchNormalization, ZeroPadding2D, MaxPooling2D, Concatenate, Reshape, Lambda
from utils.global_config import __CHANNEL, __DEF_HEIGHT, __DEF_WIDTH, INIT_CHANNELS
from keras.layers import *
from keras.models import Model

def get_encoder_decoder():
    """
   Fully convolutional encoder-decoder model - Training for the first stage.
    Network-based on the U-net model.
    """
    inputs = Input((__DEF_WIDTH, __DEF_HEIGHT, __CHANNEL))

    conv1 = Conv2D(INIT_CHANNELS, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(inputs)
    conv1 = Conv2D(INIT_CHANNELS, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(conv1)
    # conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(2*INIT_CHANNELS, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(pool1)
    conv2 = Conv2D(2*INIT_CHANNELS, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(conv2)
    # conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(4*INIT_CHANNELS, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(pool2)
    conv3 = Conv2D(4*INIT_CHANNELS, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(conv3)
    # conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(8*INIT_CHANNELS, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(pool3)
    conv4 = Conv2D(8*INIT_CHANNELS, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(conv4)
    # conv4 = BatchNormalization()(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(16*INIT_CHANNELS, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(pool4)
    conv5 = Conv2D(16*INIT_CHANNELS, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(conv5)
    # conv5 = BatchNormalization()(conv5)

    up6 = concatenate([Conv2DTranspose(8*INIT_CHANNELS, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(8*INIT_CHANNELS, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(up6)
    conv6 = Conv2D(8*INIT_CHANNELS, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(conv6)
    # conv6 = BatchNormalization()(conv6)

    up7 = concatenate([Conv2DTranspose(4*INIT_CHANNELS, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(4*INIT_CHANNELS, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(up7)
    conv7 = Conv2D(4*INIT_CHANNELS, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(conv7)
    # conv7 = BatchNormalization()(conv7)

    up8 = concatenate([Conv2DTranspose(2*INIT_CHANNELS, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(2*INIT_CHANNELS, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(up8)
    conv8 = Conv2D(2*INIT_CHANNELS, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(conv8)
    # conv8 = BatchNormalization()(conv8)

    up9 = concatenate([Conv2DTranspose(INIT_CHANNELS, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(INIT_CHANNELS, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(up9)
    conv9 = Conv2D(INIT_CHANNELS, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(conv9)
    # conv9 = BatchNormalization()(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid', kernel_initializer='he_uniform')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])
    
    #model.compile(optimizer=Adam(lr=float(arguments['--lr'])), loss=dice_coef_loss, metrics=['acc'])
    #model.compile(optimizer=Adam(lr=float(arguments['--lr'])), loss=dice_coef_loss, metrics=[dice_coef])
    return model

def build_refinement(encoder_decoder):
    """
    The build_refinement () function represents the block of the refinement layers.
    It receives the model of the network hide-decoder as a parameter.
    
    Receives the image predicted by the autoencoder model concatenated with the original gray image.
    input channels: binary image and the original image (in grayscale).
    """
    input_tensor = encoder_decoder.input # input of the enconder-decoder
    
    input = Lambda(lambda i: i[:, :, :, 0:3])(input_tensor)

    conv_ref1 = Concatenate(axis=3)([input, encoder_decoder.output]) # input and output concatenation
    
    conv_ref2 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', bias_initializer='zeros')(conv_ref1)
    conv_ref2 = BatchNormalization()(conv_ref2)

    conv_ref3 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', bias_initializer='zeros')(conv_ref2)
    conv_ref3 = BatchNormalization()(conv_ref3)
    
    conv_ref4 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', bias_initializer='zeros')(conv_ref3)
    conv_ref4 = BatchNormalization()(conv_ref4)
    
    conv_ref_out = Conv2D(1, (3, 3), activation='sigmoid', padding='same', name='refinement_pred', kernel_initializer='he_normal', bias_initializer='zeros')(conv_ref4)

    model = Model(inputs=input_tensor, outputs=conv_ref_out)
    return model