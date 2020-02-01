"""
Funções de métricas, ativações e de perda (loss functions).

"""
import keras.backend as K

#Adicionado Dice coeficient
def dice_coef(y_true, y_pred, smooth=1000.0):
    """ {2 * (Intersection over union)}
        Gives a value between 0 and 1 """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_hard(y_true, y_pred):
    """ Round the values of y_pred to be 0 or 1 to
        compute and then compute IoU
        y_true is also rounded so it will change the
        labels if using soft targets in knowledge distilation """
    y_true_round = K.round(y_true)
    y_pred_round = K.round(y_pred)
    return dice_coef(y_true_round, y_pred_round)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def dice_coef_distance_loss(y_true, y_pred):
    """ Dice coefficient is between 0 and 1, so for loss use
        distance from 1 """
    return 1 - dice_coef(y_true, y_pred)


def jaccard_coef(y_true, y_pred):
    """ Intersection over union
        Gives a value between 0 and 1 """
    intersection = K.sum(y_true * y_pred, axis=-1)
    summation = K.sum(y_true + y_pred, axis=-1)
    jaccard = intersection / (summation - intersection)
    return jaccard


def jaccard_coef_hard(y_true, y_pred):
    """ Round the values of y_pred to be 0 or 1 to
        compute and then compute IoU
        y_true is also rounded so it will change the
        labels if using soft targets in knowledge distilation """
    y_true_round = K.round(y_true)
    y_pred_round = K.round(y_pred)
    return jaccard_coef(y_true_round, y_pred_round)


def jaccard_distance_loss(y_true, y_pred):
    """ Jaccard coefficient is between 0 and 1, so for loss use
        distance from 1 """
    return 1 - jaccard_coef(y_true, y_pred)

