import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import cv2
from glob import glob
import tensorflow as ts
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from src.segmentation.Unet.unet import build_unet
from src.segmentation.Unet.metrics import dice_coef, dice_loss

H = 256
W = 256
lr = 1e-4


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


"""" Image load and preprocessing"""


def read_image(path):
    path = path.decode()
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (H, W))
    img = img / 255.0
    img = img.astype(np.float32)
    return img


def read_mask(path):
    path = path.decode()
    img = cv2.imwrite(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (H, W))
    img = img / 255.0
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=-1)
    return img


def tensorflow_parse(image, mask):
    def _parse(image, mask):
        image = read_image(image)
        mask = read_mask(mask)
        return image, mask
    image, mask = ts.numpy_function(func=_parse,inp= [image, mask],Tout=[ts.float32, ts.float32])
    image.set_shape([H,W,3])
    mask.set_shape([H,W,1])
    return image, mask


def tf_dataset(image, mask, batch=2):
    dataset = ts.data.Dataset.from_tensor_slices((image, mask))
    dataset = dataset.map(tensorflow_parse)
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(10)
    return dataset


if __name__ == '__main__':
    """ Seeding """
    np.random.seed(42)
    ts.random.set_seed(42)
    """" Directory for storing files """
    (test_x, test_y), (valid_x, valid_y), (train_x, train_y) = load_dataset("dataset2")
    print("Train_datas : ", str(len(train_x)), "images\t", str(len(train_y)), "masks")
    print("Validate_datas : ", str(len(valid_x)), "images\t", str(len(valid_y)), "masks")
    print("Test_datas : ", str(len(test_x)), "images\t", str(len(test_y)), "masks")
    train_dataset = tf_dataset(train_x, train_y, batch=16)
    valid_dataset = tf_dataset(valid_x, valid_y, batch=16)
    print(len(train_dataset))
    for image,mask in train_dataset:
        print(image.shape)
    model = build_unet((H, W, 3))
    model.compile(loss=dice_loss, optimizer=Adam(lr), metrics=[dice_coef])
    callbacks = [
        ModelCheckpoint("files/model1.h5", verbose=1, save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
        CSVLogger("files/model1.csv"),
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=False),
    ]
    model.fit(train_dataset,epochs=6, callbacks=callbacks,
              validation_data=valid_dataset)
