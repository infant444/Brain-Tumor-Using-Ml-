import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from keras.layers import Conv2D,Activation,MaxPooling2D,Dense,Dropout,Conv2DTranspose,BatchNormalization,Input,Concatenate,MaxPool2D
from keras.models import Model
from keras.utils import plot_model
import pydot
from matplotlib import pyplot as plt

#conv_block
def con_block(input,no_filter):
    #part1
    x=Conv2D(no_filter,3,padding="same")(input)
    x=BatchNormalization()(x)
    x=Activation('relu')(x)

    #part2
    x = Conv2D(no_filter, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

#encoder block
def encode_block(input,no_filter):
    y=con_block(input,no_filter)
    p=MaxPool2D((2,2))(y)
    return y,p

#decoder block
def decode_block(input,skip_feature,no_filters):
    # print(input.shape)
    x=Conv2DTranspose(no_filters,2,strides=2,padding="same")(input)
    x=Concatenate()([x,skip_feature])
    x=con_block(x,no_filters)
    # print(x.shape)
    return x


#build u-net Architecture
def build_unet(input_shape):
    input=Input(input_shape)

    s1,p1=encode_block(input,64)
    s2,p2=encode_block(p1,128)
    s3,p3=encode_block(p2,256)
    s4,p4=encode_block(p3,512)

    # print(s1.shape,s2.shape,s3.shape,s4.shape)
    # print(p1.shape,p2.shape,p3.shape,p4.shape)

    b1=con_block(p4,1024)


    d1=decode_block(b1,s4,512)
    d2=decode_block(d1,s3,256)
    d3=decode_block(d2,s2,128)
    d4=decode_block(d3,s1,64)

    res=Conv2D(1,kernel_size=(1,1),padding='same',activation='sigmoid')(d4)
    model=Model(input,res,name="UNET")
    return model
    print(res.shape)


if __name__=='__main__':
    input_shape=(256,256,3)
    model=build_unet(input_shape)
    model.summary()
    # plot_model(model, show_shapes=True, show_layer_names=True, dpi=120)