import numpy as np
from keras.layers import Flatten
from tensorflow import reduce_sum
import keras.backend as back

smooth=1e-15
def dice_coef(y_true,y_pred):
    y_true=Flatten()(y_true)
    y_pred=Flatten()(y_pred)
    intersec=reduce_sum(y_true*y_pred)
    return (2.* intersec+smooth)/(reduce_sum(y_true)+reduce_sum(y_pred)+smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

if __name__=="__main__":
    pass