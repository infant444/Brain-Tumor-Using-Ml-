import os
from glob import glob

from sklearn.model_selection import train_test_split

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import tensorflow as ts
import cv2
import pandas as pd
from tqdm import tqdm
from keras.utils import CustomObjectScope
from sklearn.metrics import f1_score, precision_score, recall_score, jaccard_score

# from src.segmentation.Unet.metrics import dice_coef, dice_loss

# from Unet.metrics import dice_coef, dice_loss

H = 256
W = 256
def load_dataset(path, split=0.2):
    images = sorted(glob(os.path.join(path, "images", "*.png")))
    mask = sorted(glob(os.path.join(path, "masks", "*.png")))
    print(images)
    split_size = int(len(images) * split)

    train_x, valid_x = train_test_split(images, test_size=split_size, random_state=42)
    train_y, valid_y = train_test_split(mask, test_size=split_size, random_state=42)
    train_x, test_x = train_test_split(train_x, test_size=split_size, random_state=42)
    train_y, test_y = train_test_split(train_y, test_size=split_size, random_state=42)
    return (test_x, test_y), (valid_x, valid_y), (train_x, train_y)


def image_process(img):
    img = cv2.resize(img, (H, W))
    x = img / 255.0
    x = np.expand_dims(x, axis=0)
    return x

def read_img(path):
    x=path
    img = cv2.imread(x, cv2.IMREAD_COLOR)
    x=image_process(img)
    return x




def prediction(x):
    from src.segmentation.Unet.metrics import dice_coef, dice_loss
    with CustomObjectScope({"dice_coef": dice_coef, "dice_loss": dice_loss}):
        model = ts.keras.models.load_model("segmentation/files/model04.h5")
    y_pred = model.predict(x, verbose=0)[0]
    y_pred = np.squeeze(y_pred, axis=-1)
    y_pred = y_pred >= 0.5
    y_pred = y_pred.astype(np.int32)
    return y_pred

def predicated_data_process(y_pred):
    y_pred = np.expand_dims(y_pred, axis=-1)
    y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=-1)
    y_pred = y_pred * 255.0
    return y_pred
def save_result(img, img1, y_pred, save_image_path):
    img1 = np.expand_dims(img1, axis=-1)
    img1 = np.concatenate([img1, img1, img1], axis=-1)
    y_pred=predicated_data_process(y_pred)
    concat_img=np.concatenate([img,img1,y_pred],axis=1)
    cv2.imwrite(save_image_path,concat_img)


def save_csv(mask,y_pred):
    global SCORE

    """ Flatten the array """
    mask = mask / 255.0
    mask = (mask > 0.5).astype(np.int32).flatten()
    y_pred = y_pred.flatten()

    """ Calculating the metrics values """
    f1_value = f1_score(mask, y_pred, labels=[0, 1], average="binary")
    jac_value = jaccard_score(mask, y_pred, labels=[0, 1], average="binary")
    recall_value = recall_score(mask, y_pred, labels=[0, 1], average="binary", zero_division=0)
    precision_value = precision_score(mask, y_pred, labels=[0, 1], average="binary", zero_division=0)
    SCORE.append([name, f1_value, jac_value, recall_value, precision_value])



def tumor_segmentation(img):
    img=image_process(img)
    y_pred = prediction(img)
    y_pred = predicated_data_process(y_pred)
    cv2.imwrite("x.png",y_pred)
    return y_pred


if __name__ == '__main__':
    """ Seeding """
    np.random.seed(42)
    ts.random.set_seed(42)

    """Load the model """
    with CustomObjectScope({"dice_coef": dice_coef, "dice_loss": dice_loss}):
        model = ts.keras.models.load_model(os.path.join("files", "model04.h5"))
    """Dataset insert"""
    (test_x, test_y), (valid_x, valid_y), (train_x, train_y) = load_dataset("dataset2")

    """Prediction and evaluation"""
    SCORE = []
    for x, y in tqdm(zip(test_x, test_y), total=len(test_x)):
        """Exteacting the test name"""

        name = x.split('\\')[-1]
        # print(name)
        """Read the image"""
        img = cv2.imread(x, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (H, W))
        x=read_img(x)
        # print(x.shape)
        # tumor_segmentation(x)
        """reading the mask"""
        img1 = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
        img1 = cv2.resize(img1, (H, W))
        """Prediction"""
        y_pred = prediction(x)

        """Save the prediction"""
        save_image_path = os.path.join("result", name)
        save_result(img, img1, y_pred, save_image_path)
        save_csv(img1,y_pred)

        break


    """ Metrics values """
    score = [s[1:] for s in SCORE]
    score = np.mean(score, axis=0)
    print(f"F1: {score[0]:0.5f}")
    print(f"Jaccard: {score[1]:0.5f}")
    print(f"Recall: {score[2]:0.5f}")
    print(f"Precision: {score[3]:0.5f}")

    df = pd.DataFrame(SCORE, columns=["Image", "F1", "Jaccard", "Recall", "Precision"])
    df.to_csv("files/score.csv", index=None)



