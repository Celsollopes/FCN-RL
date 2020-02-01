"""
Gerador de dados - Gera os dados para treinamento e validação.

"""
#import dataaug # se invocado, cria o dataaumentation online durante o treino
from utils.global_config import __CHANNEL, __DEF_WIDTH, __DEF_HEIGHT
import numpy as np
import cv2

def generator_batch(fns, bs, validation=False, stroke=True):
    batches = []
    for i in range(0, len(fns), bs):
        batches.append(fns[i: i + bs])

    print("Batching {} batches of size {} each for {} total files".format(len(batches), bs, len(fns)))
    while True:
        for fns in batches:
            imgs_batch = []
            masks_batch = []
            bounding_batch = []
            for fn in fns:
                _img = cv2.imread(fn, cv2.IMREAD_GRAYSCALE)
                if _img is None:
                  print(fn)
                  continue
                
                _img = cv2.resize(_img, (__DEF_WIDTH, __DEF_HEIGHT), interpolation=cv2.INTER_CUBIC)
                _img = _img.astype('float32')
                if stroke:
                    gt = "gt.png"
                else:
                    gt = "gt.png"

                mask = cv2.imread(fn.replace("in.png", gt), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                  print(fn)
                  continue
                
                mask = cv2.resize(mask, (__DEF_WIDTH, __DEF_HEIGHT), interpolation=cv2.INTER_CUBIC)
                # cv2.imwrite('/home/victormelo/playground/sanity/%s.png' % P.basename(fn), _img)
                # cv2.imwrite('/home/victormelo/playground/sanity/%s-mask.png' % P.basename(fn), mask)
                '''
                if not validation and not arguments['--no-aug']:
                    # _img = dataaug.invert_channel(_img)
                    _img, mask = dataaug.flip_h(_img, mask)
                    _img, mask = dataaug.flip_v(_img, mask)
                    _img, mask = dataaug.rotate_90_clockwise(_img, mask)
                    _img, mask = dataaug.rotate(_img, mask)
                    _img = dataaug.sp_noise(_img, chance=0.1)
                '''   
                _img = 1 - (_img.reshape((__DEF_WIDTH, __DEF_HEIGHT, 1)) / 255)
                mask = mask.reshape((__DEF_WIDTH, __DEF_HEIGHT, 1)) / 255
                mask = mask > 0.3

                mask = mask.astype('float32')
                imgs_batch.append(_img)
                masks_batch.append(mask)

            imgs_batch = np.asarray(imgs_batch).reshape((bs, __DEF_WIDTH, __DEF_HEIGHT, 1)).astype('float32')
            masks_batch = np.asarray(masks_batch).reshape((bs, __DEF_WIDTH, __DEF_HEIGHT, 1)).astype('float32')

            yield imgs_batch, masks_batch