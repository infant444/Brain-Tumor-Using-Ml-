import os
import cv2
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from keras.models import model_from_json
import numpy as np
from keras.preprocessing import image

def load_model():
    json_file = open('classification/file/model2.json', 'r')
    # json_file = open('file/model2.json', 'r')

    file = json_file.read()
    json_file.close()
    file = model_from_json(file)
    file.load_weights('classification/file/model2.h5')
    # file.load_weights('file/model2.h5')
    return file

# test_image = image.image_utils.load_img("dataset1/Testing/pituitary/Te-pi_0017.jpg")

# test_image = image.image_utils.load_img("sample/4.jpg")



def classfy(test_image):
    file=load_model()
    # test_image = test_image.resize((128, 128))
    test_image=cv2.resize(test_image,(128, 128))
    lable = ['glioma', 'meningioma', 'notumor', 'pituitary']
    test_image = image.image_utils.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    res = file.predict(test_image)
    sum=np.sum(res[0]);
    predicted_class_probabilit=res[0][res.argmax()];
    confidence_level=(predicted_class_probabilit/sum)*100
    confidence_level = format(confidence_level, '.2f')

    reslut=lable[res.argmax()]
    return reslut,confidence_level;

if __name__=="__main__":
    test_image = image.image_utils.load_img("../../src/segmentation/dataset2/images/50.png",)
    print(classfy(test_image))
